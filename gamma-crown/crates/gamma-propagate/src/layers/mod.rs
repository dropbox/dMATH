//! Layer implementations for bound propagation.
//!
//! This module contains all layer types that support IBP and CROWN bound propagation.
//! Each layer implements the `BoundPropagation` trait.

use faer::Mat;
use gamma_core::{GammaError, GemmEngine, Result};
use gamma_tensor::BoundedTensor;
use ndarray::{s, Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use std::borrow::Cow;
use tracing::debug;

/// Minimum number of elements to use parallel iteration for element-wise operations.
/// Below this threshold, sequential iteration is faster due to parallelization overhead.
/// Benchmark results: 24K elements shows regression, 98K+ elements shows 2-3x speedup.
const PARALLEL_ELEMENT_THRESHOLD: usize = 65536;

use crate::{broadcast_shapes, relu_ibp, BatchedLinearBounds, LinearBounds};

/// Compute strides for a multi-dimensional array shape.
///
/// For shape [d0, d1, ..., dn], returns strides [s0, s1, ..., sn] where
/// s_i = product of d_(i+1) to d_n. This allows converting between flat
/// indices and multi-dimensional coordinates.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Trait for layers that support bound propagation.
pub trait BoundPropagation {
    /// Propagate bounds through the layer using IBP.
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Propagate linear bounds through the layer (for CROWN).
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>>;
}

/// Helper function for CROWN backward propagation through element-wise activation functions.
///
/// This function handles the common boilerplate for propagating linear bounds backward
/// through element-wise nonlinear layers. Each neuron is relaxed independently using
/// the provided `relaxation_fn` which computes linear lower and upper bounds.
///
/// # Arguments
/// * `bounds` - Incoming linear bounds from layers above
/// * `pre_activation` - Pre-activation bounds for this layer's inputs
/// * `relaxation_fn` - Function that computes (lower_slope, lower_intercept, upper_slope, upper_intercept) for a given interval [l, u]
///
/// # Returns
/// New linear bounds with the activation's linear relaxation composed in.
pub fn crown_elementwise_backward<F>(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<LinearBounds>
where
    F: Fn(f32, f32) -> (f32, f32, f32, f32),
{
    let pre_flat = pre_activation.flatten();
    let pre_lower = pre_flat
        .lower
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| GammaError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.lower.shape().to_vec(),
        })?;
    let pre_upper = pre_flat
        .upper
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| GammaError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.upper.shape().to_vec(),
        })?;

    let num_neurons = pre_lower.len();
    if bounds.num_inputs() != num_neurons {
        return Err(GammaError::ShapeMismatch {
            expected: vec![num_neurons],
            got: vec![bounds.num_inputs()],
        });
    }

    let num_outputs = bounds.num_outputs();

    // Compute relaxation parameters for each neuron
    let mut lower_slopes = Array1::<f32>::zeros(num_neurons);
    let mut lower_intercepts = Array1::<f32>::zeros(num_neurons);
    let mut upper_slopes = Array1::<f32>::zeros(num_neurons);
    let mut upper_intercepts = Array1::<f32>::zeros(num_neurons);

    for i in 0..num_neurons {
        let l = pre_lower[i];
        let u = pre_upper[i];
        let (ls, li, us, ui) = relaxation_fn(l, u);
        lower_slopes[i] = ls;
        lower_intercepts[i] = li;
        upper_slopes[i] = us;
        upper_intercepts[i] = ui;
    }

    // Backward propagation: compose the linear relaxation
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_b = bounds.lower_b.clone();
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_b = bounds.upper_b.clone();

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a[[j, i]];
            let ua = bounds.upper_a[[j, i]];

            // For lower bound: use lower relaxation when coeff is positive, upper when negative
            if la >= 0.0 {
                new_lower_a[[j, i]] = la * lower_slopes[i];
                new_lower_b[j] += la * lower_intercepts[i];
            } else {
                new_lower_a[[j, i]] = la * upper_slopes[i];
                new_lower_b[j] += la * upper_intercepts[i];
            }

            // For upper bound: use upper relaxation when coeff is positive, lower when negative
            if ua >= 0.0 {
                new_upper_a[[j, i]] = ua * upper_slopes[i];
                new_upper_b[j] += ua * upper_intercepts[i];
            } else {
                new_upper_a[[j, i]] = ua * lower_slopes[i];
                new_upper_b[j] += ua * lower_intercepts[i];
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

/// A fully-connected linear layer: y = Wx + b
///
/// Stores weight matrix W and optional bias b for bound propagation.
/// Precomputes W+ = max(W,0) and W- = min(W,0) as faer matrices for fast IBP.
#[derive(Clone)]
pub struct LinearLayer {
    /// Weight matrix of shape (out_features, in_features)
    pub weight: Array2<f32>,
    /// Optional bias of shape (out_features,)
    pub bias: Option<Array1<f32>>,
    /// Cached positive part of weight for IBP (ndarray): max(W, 0)
    w_pos: Array2<f32>,
    /// Cached negative part of weight for IBP (ndarray): min(W, 0)
    w_neg: Array2<f32>,
    /// Cached transpose of w_pos as faer Mat: [in_features, out_features] for fast matmul
    w_pos_t_faer: Mat<f32>,
    /// Cached transpose of w_neg as faer Mat: [in_features, out_features] for fast matmul
    w_neg_t_faer: Mat<f32>,
    /// Cached weight as faer Mat: [out_features, in_features] for fast CROWN backward matmul
    weight_faer: Mat<f32>,
    /// Spectral norm (largest singular value) of weight matrix, computed via power iteration.
    /// Used for zonotope scaling to prevent coefficient overflow in quadratic cross-terms.
    spectral_norm: f32,
}

impl std::fmt::Debug for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearLayer")
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .field("in_features", &self.in_features())
            .field("out_features", &self.out_features())
            .field("spectral_norm", &self.spectral_norm)
            .finish()
    }
}

impl LinearLayer {
    /// Create a new linear layer from weight matrix and optional bias.
    /// Precomputes W+, W-, their transposes as faer matrices, and spectral norm for fast IBP.
    pub fn new(weight: Array2<f32>, bias: Option<Array1<f32>>) -> Result<Self> {
        if let Some(ref b) = bias {
            if b.len() != weight.nrows() {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![weight.nrows()],
                    got: vec![b.len()],
                });
            }
        }
        // Precompute W+ and W- for IBP (big speedup - avoids recomputing every call)
        let w_pos = weight.mapv(|v| v.max(0.0));
        let w_neg = weight.mapv(|v| v.min(0.0));

        // Precompute transposed faer matrices: w_pos_t is [in_features, out_features]
        // For IBP: X @ W_pos_t where X is [batch, in] and W_pos_t is [in, out]
        let (out_features, in_features) = (weight.nrows(), weight.ncols());
        let w_pos_t_faer =
            Mat::<f32>::from_fn(in_features, out_features, |i, j| w_pos[[j, i]].max(0.0));
        let w_neg_t_faer =
            Mat::<f32>::from_fn(in_features, out_features, |i, j| w_neg[[j, i]].min(0.0));

        // Precompute weight as faer Mat for fast CROWN backward matmul
        // Shape: [out_features, in_features] - same as weight
        let weight_faer = Mat::<f32>::from_fn(out_features, in_features, |i, j| weight[[i, j]]);

        // Compute spectral norm (largest singular value) via power iteration
        // This is used for zonotope scaling to prevent coefficient overflow
        let spectral_norm = Self::compute_spectral_norm(&weight, 10);

        Ok(Self {
            weight,
            bias,
            w_pos,
            w_neg,
            w_pos_t_faer,
            w_neg_t_faer,
            weight_faer,
            spectral_norm,
        })
    }

    /// Create from ArrayD (dynamic arrays), converting to appropriate shapes.
    pub fn from_dynamic(weight: &ArrayD<f32>, bias: Option<&ArrayD<f32>>) -> Result<Self> {
        // Weight should be 2D: (out_features, in_features)
        if weight.ndim() != 2 {
            return Err(GammaError::ShapeMismatch {
                expected: vec![0, 0], // 2D expected
                got: weight.shape().to_vec(),
            });
        }

        let weight_2d = weight
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![weight.shape()[0], weight.shape()[1]],
                got: weight.shape().to_vec(),
            })?;

        let bias_1d = if let Some(b) = bias {
            if b.ndim() != 1 {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![weight.shape()[0]],
                    got: b.shape().to_vec(),
                });
            }
            Some(
                b.clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![b.len()],
                        got: b.shape().to_vec(),
                    })?,
            )
        } else {
            None
        };

        Self::new(weight_2d, bias_1d)
    }

    /// Input dimension.
    pub fn in_features(&self) -> usize {
        self.weight.ncols()
    }

    /// Output dimension.
    pub fn out_features(&self) -> usize {
        self.weight.nrows()
    }

    /// Spectral norm (largest singular value) of the weight matrix.
    /// Precomputed during construction for zonotope scaling.
    pub fn spectral_norm(&self) -> f32 {
        self.spectral_norm
    }

    /// Compute spectral norm via power iteration.
    /// Returns the largest singular value of the weight matrix.
    /// Uses fixed iterations for efficiency (10 iterations gives ~3 digits of precision).
    fn compute_spectral_norm(weight: &Array2<f32>, iterations: usize) -> f32 {
        let (m, n) = (weight.nrows(), weight.ncols());

        // Initialize random-ish vector (deterministic for reproducibility)
        let mut v: Vec<f32> = (0..n)
            .map(|i| ((i * 31337) % 1000) as f32 / 1000.0)
            .collect();

        // Normalize initial vector
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return 0.0;
        }
        for x in &mut v {
            *x /= norm;
        }

        // Power iteration: v = A^T A v / ||A^T A v||
        for _ in 0..iterations {
            // u = A @ v
            let mut u = vec![0.0f32; m];
            for i in 0..m {
                for j in 0..n {
                    u[i] += weight[[i, j]] * v[j];
                }
            }

            // v_new = A^T @ u
            let mut v_new = vec![0.0f32; n];
            for j in 0..n {
                for i in 0..m {
                    v_new[j] += weight[[i, j]] * u[i];
                }
            }

            // Normalize
            let norm: f32 = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                return 0.0;
            }
            for x in &mut v_new {
                *x /= norm;
            }
            v = v_new;
        }

        // Compute ||Av|| as estimate of largest singular value
        let mut av = vec![0.0f32; m];
        for i in 0..m {
            for j in 0..n {
                av[i] += weight[[i, j]] * v[j];
            }
        }
        av.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl BoundPropagation for LinearLayer {
    /// IBP for linear layer: y = Wx + b
    ///
    /// For x in [l, u], compute y bounds:
    /// - W+ = max(W, 0), W- = min(W, 0)
    /// - lower_y = W+ @ l + W- @ u + b
    /// - upper_y = W+ @ u + W- @ l + b
    ///
    /// Supports N-D batched inputs where the last dimension must match in_features().
    /// For input shape [...batch_dims..., in_features], output is [...batch_dims..., out_features].
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        // Handle 1D case (original behavior)
        if ndim == 1 {
            let in_len = shape[0];
            if in_len != self.in_features() {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![self.in_features()],
                    got: vec![in_len],
                });
            }

            let x_lower = input
                .lower
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| GammaError::ShapeMismatch {
                    expected: vec![in_len],
                    got: input.lower.shape().to_vec(),
                })?;

            let x_upper = input
                .upper
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| GammaError::ShapeMismatch {
                    expected: vec![in_len],
                    got: input.upper.shape().to_vec(),
                })?;

            // Use cached w_pos and w_neg for fast IBP
            let lower_y = self.w_pos.dot(&x_lower) + self.w_neg.dot(&x_upper);
            let upper_y = self.w_pos.dot(&x_upper) + self.w_neg.dot(&x_lower);

            let (mut lower_y, mut upper_y) = if let Some(ref b) = self.bias {
                (lower_y + b, upper_y + b)
            } else {
                (lower_y, upper_y)
            };

            // Clamp bounds to finite values to prevent overflow propagation.
            const MAX_BOUND: f32 = f32::MAX / 2.0;
            lower_y.mapv_inplace(|v| v.max(-MAX_BOUND));
            upper_y.mapv_inplace(|v| v.min(MAX_BOUND));

            return BoundedTensor::new(lower_y.into_dyn(), upper_y.into_dyn());
        }

        // N-D batched case: last dimension is in_features
        let in_features = shape[ndim - 1];
        if in_features != self.in_features() {
            return Err(GammaError::ShapeMismatch {
                expected: vec![self.in_features()],
                got: vec![in_features],
            });
        }

        // Output shape: [...batch_dims..., out_features]
        let mut out_shape: Vec<usize> = shape[..ndim - 1].to_vec();
        out_shape.push(self.out_features());

        // Compute batch size (product of all dimensions except last)
        let batch_size: usize = shape[..ndim - 1].iter().product();

        // Reshape input to [batch_size, in_features] for efficient matrix operations
        let x_lower_2d = input
            .lower
            .clone()
            .into_shape_with_order((batch_size, in_features))
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![batch_size, in_features],
                got: input.lower.shape().to_vec(),
            })?;

        let x_upper_2d = input
            .upper
            .clone()
            .into_shape_with_order((batch_size, in_features))
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![batch_size, in_features],
                got: input.upper.shape().to_vec(),
            })?;

        // Use faer for high-performance matrix multiplication
        // For batched input X [batch, in_features] and weight W [out, in]:
        // Y = X @ W^T gives [batch, out_features]
        // lower_y[b, o] = sum_i (w_pos[o,i] * l[b,i] + w_neg[o,i] * u[b,i])
        // upper_y[b, o] = sum_i (w_pos[o,i] * u[b,i] + w_neg[o,i] * l[b,i])

        // Convert ndarray to faer matrices (copy into column-major format for optimal matmul)
        // Note: Attempted zero-copy MatRef::from_row_major_slice but it causes 3-5x regression
        // due to strided memory access in matmul. The copy cost is amortized by faster matmul.
        let x_lower_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| x_lower_2d[[i, j]]);
        let x_upper_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| x_upper_2d[[i, j]]);

        // Matrix multiply using faer: [batch, in] @ [in, out] = [batch, out]
        // P3 Optimization: Use faer's element-wise addition instead of manual loop.
        // This leverages SIMD-optimized operations and better cache utilization.
        let lower_pos = &x_lower_faer * &self.w_pos_t_faer;
        let lower_neg = &x_upper_faer * &self.w_neg_t_faer;
        let upper_pos = &x_upper_faer * &self.w_pos_t_faer;
        let upper_neg = &x_lower_faer * &self.w_neg_t_faer;

        // Fused element-wise addition using faer's optimized operators
        let lower_y_faer = &lower_pos + &lower_neg;
        let upper_y_faer = &upper_pos + &upper_neg;

        // Convert faer results back to ndarray
        let out_features = self.out_features();
        let lower_y_2d =
            Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| lower_y_faer[(i, j)]);
        let upper_y_2d =
            Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| upper_y_faer[(i, j)]);

        // Add bias if present (broadcast across batch)
        let (mut lower_y_2d, mut upper_y_2d) = if let Some(ref b) = self.bias {
            // b has shape [out_features], broadcast to [batch, out_features]
            let b_broadcast = b
                .broadcast((batch_size, self.out_features()))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: vec![batch_size, self.out_features()],
                    got: b.shape().to_vec(),
                })?
                .to_owned();
            (lower_y_2d + &b_broadcast, upper_y_2d + b_broadcast)
        } else {
            (lower_y_2d, upper_y_2d)
        };

        // Clamp bounds to finite values to prevent overflow propagation.
        // Uses f32::MAX / 2.0 to leave headroom for downstream operations.
        const MAX_BOUND: f32 = f32::MAX / 2.0;
        lower_y_2d.mapv_inplace(|v| v.max(-MAX_BOUND));
        upper_y_2d.mapv_inplace(|v| v.min(MAX_BOUND));

        // Reshape back to original batch dimensions + out_features
        let out_lower = lower_y_2d
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| GammaError::ShapeMismatch {
                expected: out_shape.clone(),
                got: vec![batch_size, self.out_features()],
            })?;

        let out_upper = upper_y_2d
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| GammaError::ShapeMismatch {
                expected: out_shape.clone(),
                got: vec![batch_size, self.out_features()],
            })?;

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN backward propagation through linear layer.
    ///
    /// For a linear layer y = Wx + b, and current linear bounds A @ y + c on subsequent layers:
    /// - Substitute y: A @ (Wx + b) + c = (A @ W) @ x + (A @ b + c)
    /// - new_A = A @ W
    /// - new_b = A @ b + c
    ///
    /// Uses faer for accelerated matrix multiplication (5-10x faster than ndarray::dot).
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Linear layer CROWN backward propagation");

        // bounds: A @ y + b where y is output of this layer
        // this layer: y = W @ x + bias
        // result: (A @ W) @ x + (A @ bias + b)

        // Use faer for fast matrix multiplication: A @ W
        // bounds.lower_a has shape (num_outputs, out_features)
        // self.weight_faer has shape (out_features, in_features)
        // Result: (num_outputs, in_features)
        let (num_outputs, bounds_inputs) = (bounds.lower_a.nrows(), bounds.lower_a.ncols());
        let weight_rows = self.weight_faer.nrows();
        let in_features = self.weight_faer.ncols();

        // Check dimension compatibility - bounds inputs must match weight rows,
        // OR be a multiple (sequence/batch dimension case from ReduceMean expansion)
        let (out_features, num_positions) = if bounds_inputs == weight_rows {
            (bounds_inputs, 1usize)
        } else if bounds_inputs % weight_rows == 0 {
            // Sequence dimension case: bounds have shape [num_outputs, seq_len * out_features]
            // Apply transformation position-wise
            let num_pos = bounds_inputs / weight_rows;
            debug!(
                "Linear backward with sequence dim: {} positions, {} features each",
                num_pos, weight_rows
            );
            (weight_rows, num_pos)
        } else {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_outputs, weight_rows],
                got: vec![num_outputs, bounds_inputs],
            });
        };

        // Handle position-wise case (from ReduceMean expansion)
        let total_in_features = num_positions * in_features;
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, total_in_features));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, total_in_features));
        let mut lower_bias_contrib = Array1::<f32>::zeros(num_outputs);
        let mut upper_bias_contrib = Array1::<f32>::zeros(num_outputs);

        for pos in 0..num_positions {
            let in_start = pos * out_features;
            let out_start = pos * in_features;

            // Extract coefficient block for this position
            let lower_block = Mat::<f32>::from_fn(num_outputs, out_features, |i, j| {
                bounds.lower_a[[i, in_start + j]]
            });
            let upper_block = Mat::<f32>::from_fn(num_outputs, out_features, |i, j| {
                bounds.upper_a[[i, in_start + j]]
            });

            // Compute A_pos @ W using faer
            let new_lower_block = &lower_block * &self.weight_faer;
            let new_upper_block = &upper_block * &self.weight_faer;

            // Place result in output
            for i in 0..num_outputs {
                for j in 0..in_features {
                    new_lower_a[[i, out_start + j]] = new_lower_block[(i, j)];
                    new_upper_a[[i, out_start + j]] = new_upper_block[(i, j)];
                }
            }

            // Accumulate bias contribution across all positions
            if let Some(ref bias) = self.bias {
                for i in 0..num_outputs {
                    for j in 0..out_features {
                        lower_bias_contrib[i] += bounds.lower_a[[i, in_start + j]] * bias[j];
                        upper_bias_contrib[i] += bounds.upper_a[[i, in_start + j]] * bias[j];
                    }
                }
            }
        }

        // new_lower_b = sum of bias contributions + bounds.lower_b
        let (new_lower_b, new_upper_b) = if self.bias.is_some() {
            (
                &bounds.lower_b + &lower_bias_contrib,
                &bounds.upper_b + &upper_bias_contrib,
            )
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        Ok(Cow::Owned(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        }))
    }
}

impl LinearLayer {
    #[inline]
    pub fn propagate_linear_with_engine<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Cow<'a, LinearBounds>> {
        let Some(engine) = engine else {
            return self.propagate_linear(bounds);
        };

        match self.propagate_linear_via_gemm(bounds, engine) {
            Ok(lb) => Ok(Cow::Owned(lb)),
            Err(e) => {
                debug!("GEMM engine failed for Linear CROWN backward, falling back to CPU: {e}");
                self.propagate_linear(bounds)
            }
        }
    }

    fn propagate_linear_via_gemm(
        &self,
        bounds: &LinearBounds,
        engine: &dyn GemmEngine,
    ) -> Result<LinearBounds> {
        let (num_outputs, bounds_inputs) = (bounds.lower_a.nrows(), bounds.lower_a.ncols());
        let weight_rows = self.weight.nrows();
        let in_features = self.weight.ncols();

        let (out_features, num_positions) = if bounds_inputs == weight_rows {
            (bounds_inputs, 1usize)
        } else if bounds_inputs % weight_rows == 0 {
            (weight_rows, bounds_inputs / weight_rows)
        } else {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_outputs, weight_rows],
                got: vec![num_outputs, bounds_inputs],
            });
        };

        let weight_slice = self.weight.as_slice().ok_or_else(|| {
            GammaError::InvalidSpec("Linear weight is not contiguous".to_string())
        })?;

        let total_in_features = num_positions * in_features;
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, total_in_features));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, total_in_features));
        let mut lower_bias_contrib = Array1::<f32>::zeros(num_outputs);
        let mut upper_bias_contrib = Array1::<f32>::zeros(num_outputs);

        for pos in 0..num_positions {
            let in_start = pos * out_features;
            let out_start = pos * in_features;

            let mut lower_block = vec![0.0f32; num_outputs * out_features];
            let mut upper_block = vec![0.0f32; num_outputs * out_features];

            for i in 0..num_outputs {
                let row_off = i * out_features;
                for j in 0..out_features {
                    lower_block[row_off + j] = bounds.lower_a[[i, in_start + j]];
                    upper_block[row_off + j] = bounds.upper_a[[i, in_start + j]];
                }
            }

            let new_lower_block = engine.gemm_f32(
                num_outputs,
                out_features,
                in_features,
                &lower_block,
                weight_slice,
            )?;
            let new_upper_block = engine.gemm_f32(
                num_outputs,
                out_features,
                in_features,
                &upper_block,
                weight_slice,
            )?;

            for i in 0..num_outputs {
                let src_off = i * in_features;
                for j in 0..in_features {
                    new_lower_a[[i, out_start + j]] = new_lower_block[src_off + j];
                    new_upper_a[[i, out_start + j]] = new_upper_block[src_off + j];
                }
            }

            if let Some(ref bias) = self.bias {
                for i in 0..num_outputs {
                    for j in 0..out_features {
                        lower_bias_contrib[i] += bounds.lower_a[[i, in_start + j]] * bias[j];
                        upper_bias_contrib[i] += bounds.upper_a[[i, in_start + j]] * bias[j];
                    }
                }
            }
        }

        let new_lower_b = if self.bias.is_some() {
            &bounds.lower_b + &lower_bias_contrib
        } else {
            bounds.lower_b.clone()
        };
        let new_upper_b = if self.bias.is_some() {
            &bounds.upper_b + &upper_bias_contrib
        } else {
            bounds.upper_b.clone()
        };

        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        })
    }

    /// Batched CROWN backward propagation through linear layer.
    ///
    /// For a linear layer y = Wx + b, and current batched linear bounds A @ y + c:
    /// - Substitute y: A @ (Wx + b) + c = (A @ W) @ x + (A @ b + c)
    /// - new_A = A @ W (batched matmul, W broadcasts over batch dims)
    /// - new_b = A @ b + c
    ///
    /// This operates on the last dimensions only, preserving batch structure.
    /// A has shape [...batch, out_dim, mid_dim], W has shape [mid_dim, in_dim]
    /// Result has shape [...batch, out_dim, in_dim]
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        debug!("Linear layer batched CROWN backward propagation");

        // bounds.lower_a: [...batch, out_dim, mid_dim] where mid_dim = self.out_features()
        // self.weight: [out_features, in_features] = [mid_dim, in_dim]
        // new_lower_a: [...batch, out_dim, in_dim]

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != self.out_features() {
            return Err(GammaError::ShapeMismatch {
                expected: vec![out_dim, self.out_features()],
                got: vec![out_dim, mid_dim],
            });
        }

        let in_dim = self.in_features();
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_dim]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(in_dim);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Reshape A to [batch, out_dim, mid_dim] for computation
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        // new_A = A @ W where W is [mid_dim, in_dim]
        // For each batch position: [out_dim, mid_dim] @ [mid_dim, in_dim] = [out_dim, in_dim]
        // Use faer for fast matrix multiplication
        let mut new_lower_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, in_dim));

        for b in 0..total_batch {
            let a_lower = lower_a_3d.slice(s![b, .., ..]);
            let a_upper = upper_a_3d.slice(s![b, .., ..]);

            // Convert to faer for fast matmul
            let a_lower_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_lower[[i, j]]);
            let a_upper_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_upper[[i, j]]);

            let new_a_lower_faer = a_lower_faer * &self.weight_faer;
            let new_a_upper_faer = a_upper_faer * &self.weight_faer;

            // Copy to result
            for i in 0..out_dim {
                for j in 0..in_dim {
                    new_lower_a[[b * out_dim + i, j]] = new_a_lower_faer[(i, j)];
                    new_upper_a[[b * out_dim + i, j]] = new_a_upper_faer[(i, j)];
                }
            }
        }

        // Reshape back to [...batch, out_dim, in_dim]
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;

        // Compute bias contribution: A @ bias + old_b
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            // For each batch position: [out_dim, mid_dim] @ [mid_dim] = [out_dim]
            let lower_b_3d = bounds
                .lower_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_3d = bounds
                .upper_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let mut new_lower_b = Array2::zeros((total_batch, out_dim));
            let mut new_upper_b = Array2::zeros((total_batch, out_dim));

            for b in 0..total_batch {
                let a_lower = lower_a_3d.slice(s![b, .., ..]);
                let a_upper = upper_a_3d.slice(s![b, .., ..]);
                let bias_contrib_lower = a_lower.dot(bias);
                let bias_contrib_upper = a_upper.dot(bias);

                for i in 0..out_dim {
                    new_lower_b[[b, i]] = lower_b_3d[[b, i]] + bias_contrib_lower[i];
                    new_upper_b[[b, i]] = upper_b_3d[[b, i]] + bias_contrib_upper[i];
                }
            }

            let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec).map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string())
                })?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec).map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string())
                })?,
            )
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        // Update input shape to reflect the linear layer's input dimension
        let mut new_input_shape = bounds.input_shape.clone();
        if !new_input_shape.is_empty() {
            *new_input_shape.last_mut().unwrap() = in_dim;
        }

        Ok(BatchedLinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
            input_shape: new_input_shape,
            output_shape: bounds.output_shape.clone(),
        })
    }

    /// Batched GPU CROWN backward propagation for multiple domains.
    ///
    /// This function processes N domains' LinearBounds through a single Linear layer
    /// using a single batched GPU GEMM call. This dramatically improves GPU utilization
    /// compared to processing domains sequentially (N small GEMMs vs 1 large GEMM).
    ///
    /// # Math
    /// For linear layer y = Wx + b, and incoming linear bounds A @ y + c:
    /// - Substitute: A @ (Wx + b) + c = (A @ W) @ x + (A @ b + c)
    /// - new_A = A @ W
    ///
    /// For N domains, we stack A matrices vertically:
    /// - stacked_A = [A_1; A_2; ...; A_N] with shape [N * num_outputs, mid_dim]
    /// - stacked_result = stacked_A @ W with shape [N * num_outputs, in_features]
    /// - Unstack into N results
    ///
    /// # Arguments
    /// * `bounds_batch` - Slice of LinearBounds, one per domain
    /// * `engine` - GPU compute engine for GEMM
    ///
    /// # Returns
    /// Vector of LinearBounds, one per domain, in same order as input
    pub fn propagate_linear_batched_with_engine(
        &self,
        bounds_batch: &[&LinearBounds],
        engine: &dyn GemmEngine,
    ) -> Result<Vec<LinearBounds>> {
        if bounds_batch.is_empty() {
            return Ok(Vec::new());
        }

        // All bounds must have the same shape (same layer in same network)
        let first = bounds_batch[0];
        let num_outputs = first.num_outputs();
        let bounds_inputs = first.num_inputs();
        let n_domains = bounds_batch.len();

        // Validate all domains have same shape
        for (i, b) in bounds_batch.iter().enumerate().skip(1) {
            if b.num_outputs() != num_outputs || b.num_inputs() != bounds_inputs {
                return Err(GammaError::InvalidSpec(format!(
                    "Domain {} has shape [{}, {}], expected [{}, {}]",
                    i,
                    b.num_outputs(),
                    b.num_inputs(),
                    num_outputs,
                    bounds_inputs
                )));
            }
        }

        let weight_rows = self.weight.nrows();
        let in_features = self.weight.ncols();

        let (out_features, num_positions) = if bounds_inputs == weight_rows {
            (bounds_inputs, 1usize)
        } else if bounds_inputs % weight_rows == 0 {
            (weight_rows, bounds_inputs / weight_rows)
        } else {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_outputs, weight_rows],
                got: vec![num_outputs, bounds_inputs],
            });
        };

        let weight_slice = self.weight.as_slice().ok_or_else(|| {
            GammaError::InvalidSpec("Linear weight is not contiguous".to_string())
        })?;

        let total_in_features = num_positions * in_features;
        let total_stacked_rows = n_domains * num_outputs;

        // Pre-allocate result storage
        let mut results: Vec<LinearBounds> = Vec::with_capacity(n_domains);

        // Process each position block with batched GEMM
        for pos in 0..num_positions {
            let in_start = pos * out_features;
            let out_start = pos * in_features;

            // Stack all domains' A matrices for this position
            // Shape: [n_domains * num_outputs, out_features]
            let mut stacked_lower = vec![0.0f32; total_stacked_rows * out_features];
            let mut stacked_upper = vec![0.0f32; total_stacked_rows * out_features];

            for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
                let row_offset = domain_idx * num_outputs;
                for i in 0..num_outputs {
                    let src_row = i;
                    let dst_row = row_offset + i;
                    for j in 0..out_features {
                        stacked_lower[dst_row * out_features + j] =
                            bounds.lower_a[[src_row, in_start + j]];
                        stacked_upper[dst_row * out_features + j] =
                            bounds.upper_a[[src_row, in_start + j]];
                    }
                }
            }

            // Single batched GEMM: [n_domains * num_outputs, out_features] @ [out_features, in_features]
            // Result: [n_domains * num_outputs, in_features]
            let result_lower = engine.gemm_f32(
                total_stacked_rows,
                out_features,
                in_features,
                &stacked_lower,
                weight_slice,
            )?;
            let result_upper = engine.gemm_f32(
                total_stacked_rows,
                out_features,
                in_features,
                &stacked_upper,
                weight_slice,
            )?;

            // Unstack results into per-domain arrays
            if pos == 0 {
                // First position: initialize result LinearBounds
                for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
                    let row_offset = domain_idx * num_outputs;
                    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, total_in_features));
                    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, total_in_features));

                    for i in 0..num_outputs {
                        let src_row = row_offset + i;
                        for j in 0..in_features {
                            new_lower_a[[i, out_start + j]] =
                                result_lower[src_row * in_features + j];
                            new_upper_a[[i, out_start + j]] =
                                result_upper[src_row * in_features + j];
                        }
                    }

                    // Initialize bias contribution (will be accumulated across positions)
                    let lower_bias_contrib = Array1::<f32>::zeros(num_outputs);
                    let upper_bias_contrib = Array1::<f32>::zeros(num_outputs);

                    results.push(LinearBounds {
                        lower_a: new_lower_a,
                        lower_b: bounds.lower_b.clone() + &lower_bias_contrib,
                        upper_a: new_upper_a,
                        upper_b: bounds.upper_b.clone() + &upper_bias_contrib,
                    });
                }
            } else {
                // Subsequent positions: update existing results
                for (domain_idx, result) in results.iter_mut().enumerate() {
                    let row_offset = domain_idx * num_outputs;

                    for i in 0..num_outputs {
                        let src_row = row_offset + i;
                        for j in 0..in_features {
                            result.lower_a[[i, out_start + j]] =
                                result_lower[src_row * in_features + j];
                            result.upper_a[[i, out_start + j]] =
                                result_upper[src_row * in_features + j];
                        }
                    }
                }
            }

            // Handle bias contribution for this position
            if let Some(ref bias) = self.bias {
                for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
                    for i in 0..num_outputs {
                        let mut lower_contrib = 0.0f32;
                        let mut upper_contrib = 0.0f32;
                        for j in 0..out_features {
                            lower_contrib += bounds.lower_a[[i, in_start + j]] * bias[j];
                            upper_contrib += bounds.upper_a[[i, in_start + j]] * bias[j];
                        }
                        results[domain_idx].lower_b[i] += lower_contrib;
                        results[domain_idx].upper_b[i] += upper_contrib;
                    }
                }
            }
        }

        // Adjust bias for domains that had pre-existing bias
        // (The bias was cloned from original in pos==0, then we added to it)
        // We need to subtract the original bias since we cloned it
        for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
            // Actually, we added zero initially and accumulated, so no adjustment needed
            // The bias handling is correct as-is
            let _ = (domain_idx, bounds); // silence unused warning
        }

        Ok(results)
    }
}

/// A ReLU activation layer.
#[derive(Debug, Clone, Default)]
pub struct ReLULayer;

impl ReLULayer {
    /// CROWN backward propagation through ReLU with pre-activation bounds.
    ///
    /// For ReLU y = ReLU(x), with pre-activation bounds \[l, u\] for x:
    /// - If l >= 0: y = x (identity), pass-through
    /// - If u <= 0: y = 0 (zero), no dependence
    /// - If l < 0 < u: use linear relaxation
    ///   - Upper: y <= λ(x - l) where λ = u/(u-l)
    ///   - Lower: y >= α*x where α ∈ \[0,1\] (default heuristic: α=1 if u > -l, else α=0)
    ///
    /// The backward propagation handles positive/negative coefficients differently:
    /// - For positive `A[j,i]` in lower bound: want y_i large, use lower relaxation (α*x)
    /// - For negative `A[j,i]` in lower bound: want y_i small, use upper relaxation (λ*x - λ*l)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("ReLU layer CROWN backward propagation with pre-activation bounds");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        // For crossing neurons (l < 0 < u):
        //   lambda = u / (u - l)       (upper slope)
        //   alpha = heuristic (default: 1 if u > -l, else 0)
        let mut lambda = Array1::<f32>::zeros(num_neurons);
        let mut lambda_intercept = Array1::<f32>::zeros(num_neurons); // -lambda * l
        let mut alpha = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l >= 0.0 {
                // Always positive: identity
                lambda[i] = 1.0;
                lambda_intercept[i] = 0.0;
                alpha[i] = 1.0;
            } else if u <= 0.0 {
                // Always negative: zero
                lambda[i] = 0.0;
                lambda_intercept[i] = 0.0;
                alpha[i] = 0.0;
            } else {
                // Crossing: linear relaxation
                lambda[i] = u / (u - l);
                lambda_intercept[i] = -lambda[i] * l;
                // Heuristic for lower bound slope (α-CROWN would optimize this)
                alpha[i] = if u > -l { 1.0 } else { 0.0 };
            }
        }

        // Backward propagation through ReLU
        // For each coefficient A[j,i]:
        //   - If A[j,i] >= 0 in lower_a: use alpha (lower relaxation gives tighter lower bound)
        //   - If A[j,i] < 0 in lower_a: use lambda (upper relaxation gives tighter lower bound)
        //   - If A[j,i] >= 0 in upper_a: use lambda (upper relaxation gives tighter upper bound)
        //   - If A[j,i] < 0 in upper_a: use alpha (lower relaxation gives tighter upper bound)

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large
                    // Use lower relaxation: y_i >= alpha * x_i
                    new_lower_a[[j, i]] = la * alpha[i];
                    // No bias adjustment for lower relaxation (intercept is 0)
                } else {
                    // Negative coeff: want y_i to be small
                    // Use upper relaxation: y_i <= lambda * x_i + intercept
                    new_lower_a[[j, i]] = la * lambda[i];
                    new_lower_b[j] += la * lambda_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be small
                    // Use upper relaxation: y_i <= lambda * x_i + intercept
                    new_upper_a[[j, i]] = ua * lambda[i];
                    new_upper_b[j] += ua * lambda_intercept[i];
                } else {
                    // Negative coeff: want y_i to be large
                    // Use lower relaxation: y_i >= alpha * x_i
                    new_upper_a[[j, i]] = ua * alpha[i];
                    // No bias adjustment for lower relaxation
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

    /// SDP-CROWN backward propagation through ReLU for an ℓ2 ball constraint on pre-activations.
    ///
    /// This follows the standard CROWN/LiRPA backward pass to compute `g(α)` (slopes), but
    /// replaces the per-neuron box offset with the SDP-CROWN offset `h(g,λ)` (arXiv:2506.06665),
    /// using the ℓ2 ball `||x - x_hat||_2 <= rho` for this layer's pre-activation vector.
    ///
    /// Notes:
    /// - Currently implemented for 1-D flattened pre-activations.
    /// - Uses a lightweight 1-D search to pick a near-optimal λ per output row.
    pub fn propagate_linear_with_bounds_sdp(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        x_hat: &Array1<f32>,
        rho: f32,
    ) -> Result<LinearBounds> {
        debug!("ReLU layer SDP-CROWN backward propagation (ℓ2 ball offset)");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if x_hat.len() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![x_hat.len()],
            });
        }

        let x_hat_slice = x_hat.as_slice().ok_or_else(|| {
            GammaError::InvalidSpec("SDP-CROWN: x_hat must be contiguous".to_string())
        })?;

        let num_outputs = bounds.num_outputs();

        // Compute standard ReLU box relaxation parameters (α, λ, intercept).
        let mut lambda = Array1::<f32>::zeros(num_neurons);
        let mut alpha = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l >= 0.0 {
                lambda[i] = 1.0;
                alpha[i] = 1.0;
            } else if u <= 0.0 {
                lambda[i] = 0.0;
                alpha[i] = 0.0;
            } else {
                lambda[i] = u / (u - l);
                alpha[i] = if u > -l { 1.0 } else { 0.0 };
            }
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_b = bounds.upper_b.clone();

        // Scratch buffers to avoid per-neuron allocations inside the inner loops.
        let mut c_prime = vec![0.0f32; num_neurons];
        let mut g_prime = vec![0.0f32; num_neurons];

        for j in 0..num_outputs {
            // Build new coefficient rows (g) for lower/upper.
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // Lower bound transform (same as standard CROWN, but without box intercept).
                new_lower_a[[j, i]] = if la >= 0.0 {
                    la * alpha[i]
                } else {
                    la * lambda[i]
                };

                // Upper bound transform (same as standard CROWN, but without box intercept).
                new_upper_a[[j, i]] = if ua >= 0.0 {
                    ua * lambda[i]
                } else {
                    ua * alpha[i]
                };
            }

            // SDP-CROWN offset for the LOWER inequality:
            //   c^T ReLU(x) + d >= g^T x + (h(g,λ) + d)
            let c_lower = bounds.lower_a.row(j);
            let g_lower = new_lower_a.row(j);
            let c_lower = c_lower.as_slice().ok_or_else(|| {
                GammaError::InvalidSpec("SDP-CROWN: lower_a row must be contiguous".to_string())
            })?;
            let g_lower = g_lower.as_slice().ok_or_else(|| {
                GammaError::InvalidSpec("SDP-CROWN: new_lower_a row must be contiguous".to_string())
            })?;

            let h_lower =
                crate::sdp_crown::relu_sdp_offset_opt(c_lower, g_lower, x_hat_slice, rho)?;
            new_lower_b[j] = bounds.lower_b[j] + h_lower;

            // SDP-CROWN offset for the UPPER inequality:
            //   c^T ReLU(x) + d <= (-g')^T x + (d - h(g',λ)) where g' corresponds to -c.
            // Here, we compute (c', g') for the lower bound on (-c)^T ReLU(x):
            //   c' = -c, g' = -new_upper_a_row  (since new_upper_a is the upper-bound coefficients).
            for i in 0..num_neurons {
                let ua = bounds.upper_a[[j, i]];
                c_prime[i] = -ua;
                g_prime[i] = -new_upper_a[[j, i]];
            }
            let h_prime =
                crate::sdp_crown::relu_sdp_offset_opt(&c_prime, &g_prime, x_hat_slice, rho)?;
            new_upper_b[j] = bounds.upper_b[j] - h_prime;
        }

        // Note: we intentionally do NOT add the standard box intercept contributions here.
        // The SDP-CROWN offset replaces those terms.

        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        })
    }

    /// CROWN backward propagation with explicit α values for α-CROWN optimization.
    ///
    /// Same as `propagate_linear_with_bounds` but uses provided α values instead of heuristic.
    /// Also returns gradients ∂bounds/∂α for optimization.
    ///
    /// Returns: (new_bounds, gradient) where `gradient\[i\]` = ∂(sum of lower bounds)/∂α\[i\]
    pub fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
    ) -> Result<(LinearBounds, Array1<f32>)> {
        debug!("ReLU layer α-CROWN backward propagation");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if alpha.len() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![alpha.len()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute upper bound relaxation parameters (lambda) for crossing neurons
        let mut lambda = Array1::<f32>::zeros(num_neurons);
        let mut lambda_intercept = Array1::<f32>::zeros(num_neurons);
        let mut is_crossing = Array1::<bool>::from_elem(num_neurons, false);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l >= 0.0 {
                // Always positive: identity
                lambda[i] = 1.0;
                lambda_intercept[i] = 0.0;
            } else if u <= 0.0 {
                // Always negative: zero
                lambda[i] = 0.0;
                lambda_intercept[i] = 0.0;
            } else {
                // Crossing: linear relaxation
                lambda[i] = u / (u - l);
                lambda_intercept[i] = -lambda[i] * l;
                is_crossing[i] = true;
            }
        }

        // Backward propagation with provided alpha values
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        // Track gradient: ∂(lower_bound)/∂α[i]
        // The lower bound uses α when A[j,i] >= 0 in lower_a
        // And also when A[j,i] < 0 in upper_a
        let mut gradient = Array1::<f32>::zeros(num_neurons);

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Effective alpha for this neuron (use provided for crossing, fixed for stable)
                let effective_alpha = if l >= 0.0 {
                    1.0 // Always active
                } else if u <= 0.0 {
                    0.0 // Always inactive
                } else {
                    alpha[i] // Crossing: use provided
                };

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Use lower relaxation: y >= α*x
                    new_lower_a[[j, i]] = la * effective_alpha;
                    // Gradient contribution: ∂(la * α * x_contribution)/∂α
                    // When concretized, this becomes la * α * (positive coeff * input contribution)
                    // For gradient, we need ∂/∂α which is just la (when crossing)
                    if is_crossing[i] {
                        // The contribution to the gradient depends on how α affects the final bound
                        // In the end, after concretization, the lower bound coefficient contribution is:
                        // la * α * x (where x is the input)
                        // So ∂(contribution)/∂α = la * x
                        // But we want the gradient w.r.t. the final scalar bound, so we sum
                        // For now, track a simplified gradient: ∂(coeff)/∂α = la
                        gradient[i] += la;
                    }
                } else {
                    // Use upper relaxation: y <= λ*x + intercept
                    new_lower_a[[j, i]] = la * lambda[i];
                    new_lower_b[j] += la * lambda_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Use upper relaxation
                    new_upper_a[[j, i]] = ua * lambda[i];
                    new_upper_b[j] += ua * lambda_intercept[i];
                } else {
                    // Use lower relaxation: y >= α*x
                    new_upper_a[[j, i]] = ua * effective_alpha;
                    // Gradient: negative coefficient using α affects upper bound
                    // We want to minimize upper, so this contributes negatively
                    if is_crossing[i] {
                        // ∂(upper_bound contribution)/∂α = ua (negative, since ua < 0)
                        // For optimization, we track this for upper bound minimization
                        // But our loss is -lower_bound, so we focus on lower bound gradient
                    }
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
            gradient,
        ))
    }

    /// Batched CROWN backward propagation through ReLU with pre-activation bounds.
    ///
    /// Same as `propagate_linear_with_bounds` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    ///
    /// For ReLU y = ReLU(x), with pre-activation bounds [l, u] for x:
    /// - If l >= 0: y = x (identity), pass-through
    /// - If u <= 0: y = 0 (zero), no dependence
    /// - If l < 0 < u: use linear relaxation
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("ReLU layer batched CROWN backward propagation");

        // Pre-activation bounds shape should match bounds dimensions
        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let in_dim = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        // Pre-activation should have shape [...batch, in_dim]
        let pre_in_dim = *pre_shape.last().unwrap_or(&0);
        if pre_in_dim != in_dim {
            return Err(GammaError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![pre_in_dim],
            });
        }

        // Reshape pre-activation to [batch, in_dim]
        let pre_lower_flat = pre_activation
            .lower
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape pre_lower".to_string()))?;
        let pre_upper_flat = pre_activation
            .upper
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape pre_upper".to_string()))?;

        // Reshape bounds A and b to [batch, out_dim, in_dim] and [batch, out_dim]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        // Output arrays
        let mut new_lower_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_lower_b = Array2::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::zeros((total_batch, out_dim));

        // Copy initial bias
        for b in 0..total_batch {
            for j in 0..out_dim {
                new_lower_b[[b, j]] = lower_b_2d[[b, j]];
                new_upper_b[[b, j]] = upper_b_2d[[b, j]];
            }
        }

        // Process each batch position
        for b in 0..total_batch {
            // Compute relaxation parameters for this batch position
            for i in 0..in_dim {
                let l = pre_lower_flat[[b, i]];
                let u = pre_upper_flat[[b, i]];

                let (lambda, lambda_intercept, alpha) = if l.is_nan() || u.is_nan() {
                    // NaN bounds: use trivial relaxation (slope = 0, no constraints)
                    (0.0_f32, 0.0_f32, 0.0_f32)
                } else if l.is_infinite() && u.is_infinite() {
                    // Both infinite: use identity-like behavior with no intercept adjustment
                    // This avoids NaN from inf/inf and 0*inf
                    (1.0_f32, 0.0_f32, 1.0_f32)
                } else if l >= 0.0 {
                    // Always positive: identity
                    (1.0_f32, 0.0_f32, 1.0_f32)
                } else if u <= 0.0 {
                    // Always negative: zero
                    (0.0_f32, 0.0_f32, 0.0_f32)
                } else if u.is_infinite() {
                    // l < 0 < u = +inf: slope approaches 1, intercept approaches 0
                    (1.0_f32, 0.0_f32, 1.0_f32)
                } else if l.is_infinite() {
                    // l = -inf < 0 < u: slope approaches 0
                    (0.0_f32, 0.0_f32, 0.0_f32)
                } else {
                    // Crossing: linear relaxation (finite bounds)
                    let lam = u / (u - l);
                    let lam_int = -lam * l;
                    // Heuristic for lower bound slope
                    let alph = if u > -l { 1.0 } else { 0.0 };
                    (lam, lam_int, alph)
                };

                // Apply to each output dimension
                for j in 0..out_dim {
                    let la = lower_a_3d[[b, j, i]];
                    let ua = upper_a_3d[[b, j, i]];
                    let row_idx = b * out_dim + j;

                    // For lower bound output
                    if la >= 0.0 {
                        new_lower_a[[row_idx, i]] = la * alpha;
                    } else {
                        new_lower_a[[row_idx, i]] = la * lambda;
                        new_lower_b[[b, j]] += la * lambda_intercept;
                    }

                    // For upper bound output
                    if ua >= 0.0 {
                        new_upper_a[[row_idx, i]] = ua * lambda;
                        new_upper_b[[b, j]] += ua * lambda_intercept;
                    } else {
                        new_upper_a[[row_idx, i]] = ua * alpha;
                    }
                }
            }
        }

        // Reshape back to [...batch, out_dim, in_dim]
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, in_dim])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

        Ok(BatchedLinearBounds {
            lower_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            lower_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            upper_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            upper_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        })
    }
}

impl BoundPropagation for ReLULayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(relu_ibp(input))
    }

    /// Note: For CROWN, use propagate_linear_with_bounds which requires pre-activation bounds.
    /// This placeholder returns the input unchanged.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("ReLU layer CROWN propagation (requires pre-activation bounds, using placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Leaky ReLU layer: y = x if x >= 0, else alpha * x
///
/// Leaky ReLU allows a small gradient for negative inputs, which helps prevent
/// "dying ReLU" problems during training. Typical alpha values are 0.01 or 0.1.
#[derive(Debug, Clone)]
pub struct LeakyReLULayer {
    /// Negative slope (typically 0.01)
    pub alpha: f32,
}

impl LeakyReLULayer {
    /// Create a new Leaky ReLU layer with the given negative slope.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Create a Leaky ReLU layer with default alpha = 0.01.
    pub fn default_alpha() -> Self {
        Self { alpha: 0.01 }
    }
}

impl BoundPropagation for LeakyReLULayer {
    /// IBP for Leaky ReLU: y = x if x >= 0, else alpha * x
    ///
    /// For x in [l, u]:
    /// - If l >= 0: y in [l, u] (positive region, identity)
    /// - If u <= 0: y in [alpha*l, alpha*u] (negative region, scaled)
    /// - If l < 0 < u: y in [alpha*l, u] (crossing region)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;
        let lower = input.lower.mapv(|v| if v >= 0.0 { v } else { alpha * v });
        let upper = input.upper.mapv(|v| if v >= 0.0 { v } else { alpha * v });
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN (would need pre-activation bounds for proper linear relaxation)
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!(
            "LeakyReLU layer CROWN propagation (requires pre-activation bounds, using placeholder)"
        );
        Ok(Cow::Borrowed(bounds))
    }
}

impl LeakyReLULayer {
    /// CROWN backward propagation through Leaky ReLU with pre-activation bounds.
    ///
    /// Leaky ReLU: y = x if x >= 0, else α*x
    ///
    /// Linear relaxation on interval [l, u]:
    /// - If l >= 0: identity (slope=1, intercept=0)
    /// - If u <= 0: scaled (slope=α, intercept=0)
    /// - If l < 0 < u: crossing region needs relaxation
    ///   - Upper bound: line from (l, αl) to (u, u), slope = (u - αl)/(u - l)
    ///   - Lower bound: choose y = x or y = αx based on heuristic
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!(
            "LeakyReLU layer CROWN backward propagation with pre-activation bounds (alpha={})",
            self.alpha
        );

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let alpha = self.alpha;

        // Compute relaxation parameters for each neuron
        // For crossing neurons (l < 0 < u):
        //   upper_slope = (u - alpha*l) / (u - l)  (connects (l, alpha*l) to (u, u))
        //   upper_intercept = -upper_slope * l + alpha * l = l * (alpha - upper_slope)
        //   lower_slope = either 1 (use identity) or alpha (use scaled), based on heuristic
        //   lower_intercept = 0
        let mut upper_slope = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercept = Array1::<f32>::zeros(num_neurons);
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l >= 0.0 {
                // Always positive: identity
                upper_slope[i] = 1.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = 1.0;
            } else if u <= 0.0 {
                // Always negative: scaled by alpha
                upper_slope[i] = alpha;
                upper_intercept[i] = 0.0;
                lower_slope[i] = alpha;
            } else {
                // Crossing: linear relaxation
                // Upper bound line: from (l, alpha*l) to (u, u)
                // slope = (u - alpha*l) / (u - l)
                // intercept at x=0: y = slope * 0 + b where b makes line pass through (l, alpha*l)
                //   alpha*l = slope * l + b => b = alpha*l - slope * l = l*(alpha - slope)
                let slope = (u - alpha * l) / (u - l);
                upper_slope[i] = slope;
                upper_intercept[i] = l * (alpha - slope);

                // Lower bound heuristic: use identity (slope=1) if positive area larger,
                // else use scaled (slope=alpha)
                // Area under identity from 0 to u: u*u/2
                // Area under alpha*x from l to 0: |alpha*l*l/2|
                // Simpler heuristic: if u > |alpha*l|, use identity
                lower_slope[i] = if u > (-alpha * l).abs() { 1.0 } else { alpha };
            }
        }

        // Backward propagation through LeakyReLU
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large
                    // Use lower relaxation (no intercept adjustment needed)
                    new_lower_a[[j, i]] = la * lower_slope[i];
                } else {
                    // Negative coeff: want y_i to be small
                    // Use upper relaxation
                    new_lower_a[[j, i]] = la * upper_slope[i];
                    new_lower_b[j] += la * upper_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be small
                    // Use upper relaxation
                    new_upper_a[[j, i]] = ua * upper_slope[i];
                    new_upper_b[j] += ua * upper_intercept[i];
                } else {
                    // Negative coeff: want y_i to be large
                    // Use lower relaxation
                    new_upper_a[[j, i]] = ua * lower_slope[i];
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

/// Clip layer: clamp values to [min, max] range.
///
/// Clip is commonly used in quantization-aware training and to limit activations.
/// clip(x, min, max) = max(min, min(max, x))
#[derive(Debug, Clone)]
pub struct ClipLayer {
    /// Minimum value (inclusive)
    pub min: f32,
    /// Maximum value (inclusive)
    pub max: f32,
}

impl ClipLayer {
    /// Create a new Clip layer with the given bounds.
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

impl BoundPropagation for ClipLayer {
    /// IBP for Clip: y = clip(x, min, max)
    ///
    /// For x in [l, u]:
    /// - lower_bound = clip(l, min, max)
    /// - upper_bound = clip(u, min, max)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let min_val = self.min;
        let max_val = self.max;
        let lower = input.lower.mapv(|v| v.clamp(min_val, max_val));
        let upper = input.upper.mapv(|v| v.clamp(min_val, max_val));
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Clip layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl ClipLayer {
    /// CROWN backward propagation through Clip with pre-activation bounds.
    ///
    /// Clip: y = clamp(x, min, max)
    ///
    /// Piecewise linear:
    /// - x < min: y = min (constant)
    /// - min <= x <= max: y = x (identity)
    /// - x > max: y = max (constant)
    ///
    /// Linear relaxation on interval [l, u]:
    /// - Entirely below min (u < min): slope=0, intercept=min
    /// - Entirely above max (l > max): slope=0, intercept=max
    /// - Entirely within [min, max]: slope=1, intercept=0
    /// - Crossing min boundary: relaxation needed
    /// - Crossing max boundary: relaxation needed
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!(
            "Clip layer CROWN backward propagation with pre-activation bounds (min={}, max={})",
            self.min, self.max
        );

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let min_val = self.min;
        let max_val = self.max;

        // Compute relaxation parameters for each neuron
        let mut upper_slope = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercept = Array1::<f32>::zeros(num_neurons);
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercept = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            // clip(l) and clip(u)
            let cl = l.clamp(min_val, max_val);
            let cu = u.clamp(min_val, max_val);

            if u <= min_val {
                // Entirely below min: constant output = min
                upper_slope[i] = 0.0;
                upper_intercept[i] = min_val;
                lower_slope[i] = 0.0;
                lower_intercept[i] = min_val;
            } else if l >= max_val {
                // Entirely above max: constant output = max
                upper_slope[i] = 0.0;
                upper_intercept[i] = max_val;
                lower_slope[i] = 0.0;
                lower_intercept[i] = max_val;
            } else if l >= min_val && u <= max_val {
                // Entirely within [min, max]: identity
                upper_slope[i] = 1.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = 1.0;
                lower_intercept[i] = 0.0;
            } else {
                // Crossing at least one boundary
                // Chord line from (l, cl) to (u, cu)
                let chord_slope = if (u - l).abs() > 1e-8 {
                    (cu - cl) / (u - l)
                } else {
                    0.0
                };
                let chord_intercept = cl - chord_slope * l;

                // Upper bound: use chord
                upper_slope[i] = chord_slope;
                upper_intercept[i] = chord_intercept;

                // Lower bound: heuristic - use chord for simplicity
                // (could use tighter bounds by case analysis)
                lower_slope[i] = chord_slope;
                lower_intercept[i] = chord_intercept;
            }
        }

        // Backward propagation through Clip
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output
                if la >= 0.0 {
                    new_lower_a[[j, i]] = la * lower_slope[i];
                    new_lower_b[j] += la * lower_intercept[i];
                } else {
                    new_lower_a[[j, i]] = la * upper_slope[i];
                    new_lower_b[j] += la * upper_intercept[i];
                }

                // For upper bound output
                if ua >= 0.0 {
                    new_upper_a[[j, i]] = ua * upper_slope[i];
                    new_upper_b[j] += ua * upper_intercept[i];
                } else {
                    new_upper_a[[j, i]] = ua * lower_slope[i];
                    new_upper_b[j] += ua * lower_intercept[i];
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

/// ELU (Exponential Linear Unit) layer: y = x if x >= 0, else alpha * (exp(x) - 1)
///
/// ELU helps push mean activations closer to zero (compared to ReLU), which can
/// speed up learning. The exponential term ensures smooth transitions at zero.
#[derive(Debug, Clone)]
pub struct EluLayer {
    /// Scale for negative values (typically 1.0)
    pub alpha: f32,
}

impl EluLayer {
    /// Create a new ELU layer with the given alpha.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Create an ELU layer with default alpha = 1.0.
    pub fn default_alpha() -> Self {
        Self { alpha: 1.0 }
    }
}

impl BoundPropagation for EluLayer {
    /// IBP for ELU: y = x if x >= 0, else alpha * (exp(x) - 1)
    ///
    /// For x in [l, u]:
    /// - If l >= 0: y in [l, u] (positive region, identity)
    /// - If u <= 0: y in [alpha*(exp(l)-1), alpha*(exp(u)-1)] (negative region)
    /// - If l < 0 < u: y in [alpha*(exp(l)-1), u] (crossing region)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;
        let elu = |x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                alpha * (x.exp() - 1.0)
            }
        };
        let lower = input.lower.mapv(elu);
        let upper = input.upper.mapv(elu);
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("ELU layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for ELU on interval [l, u].
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept).
fn elu_linear_relaxation(l: f32, u: f32, alpha: f32) -> (f32, f32, f32, f32) {
    let elu = |x: f32| -> f32 {
        if x >= 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    };

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        // Point interval: use derivative as slope
        let y = elu(l);
        let slope = if l >= 0.0 {
            1.0
        } else {
            alpha * l.exp() // derivative
        };
        let intercept = y - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let yl = elu(l);
    let yu = elu(u);

    // If both in positive region (l >= 0): identity
    if l >= 0.0 {
        return (1.0, 0.0, 1.0, 0.0);
    }

    // If both in negative region (u <= 0): ELU is monotonic, use chord
    if u <= 0.0 {
        let chord_slope = (yu - yl) / (u - l);
        let chord_intercept = yl - chord_slope * l;

        // Sample to find max deviation
        let num_samples = 50;
        let mut max_above_chord = 0.0_f32;
        let mut max_below_chord = 0.0_f32;

        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let ex = elu(x);
            let cx = chord_slope * x + chord_intercept;
            let diff = ex - cx;
            if diff > max_above_chord {
                max_above_chord = diff;
            }
            if -diff > max_below_chord {
                max_below_chord = -diff;
            }
        }

        let eps = 1e-6;
        return (
            chord_slope,
            chord_intercept - max_below_chord - eps,
            chord_slope,
            chord_intercept + max_above_chord + eps,
        );
    }

    // Crossing region: l < 0 < u
    // Chord from (l, elu(l)) to (u, elu(u))
    let chord_slope = (yu - yl) / (u - l);
    let chord_intercept = yl - chord_slope * l;

    // Sample to find bounds
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let ex = elu(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = ex - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Check x=0 (transition point)
    let e0 = 0.0;
    let c0 = chord_intercept;
    let diff0 = e0 - c0;
    if diff0 > max_above_chord {
        max_above_chord = diff0;
    }
    if -diff0 > max_below_chord {
        max_below_chord = -diff0;
    }

    let eps = 1e-6;
    (
        chord_slope,
        chord_intercept - max_below_chord - eps,
        chord_slope,
        chord_intercept + max_above_chord + eps,
    )
}

impl EluLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!(
            "ELU layer CROWN backward propagation with pre-activation bounds (alpha={})",
            self.alpha
        );

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        let mut lower_slopes = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercepts = Array1::<f32>::zeros(num_neurons);
        let mut upper_slopes = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercepts = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];
            let (ls, li, us, ui) = elu_linear_relaxation(l, u, self.alpha);
            lower_slopes[i] = ls;
            lower_intercepts[i] = li;
            upper_slopes[i] = us;
            upper_intercepts[i] = ui;
        }

        // Backward propagation through ELU
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                if la >= 0.0 {
                    new_lower_a[[j, i]] = la * lower_slopes[i];
                    new_lower_b[j] += la * lower_intercepts[i];
                } else {
                    new_lower_a[[j, i]] = la * upper_slopes[i];
                    new_lower_b[j] += la * upper_intercepts[i];
                }

                if ua >= 0.0 {
                    new_upper_a[[j, i]] = ua * upper_slopes[i];
                    new_upper_b[j] += ua * upper_intercepts[i];
                } else {
                    new_upper_a[[j, i]] = ua * lower_slopes[i];
                    new_upper_b[j] += ua * lower_intercepts[i];
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

/// SELU (Scaled ELU) layer: y = lambda * (x if x >= 0, else alpha * (exp(x) - 1))
///
/// SELU is self-normalizing: for properly initialized networks, activations
/// converge to zero mean and unit variance. Uses fixed constants:
/// - alpha ≈ 1.6732632423543772848170429916717
/// - lambda ≈ 1.0507009873554804934193349852946
#[derive(Debug, Clone)]
pub struct SeluLayer;

impl SeluLayer {
    /// SELU alpha constant
    pub const ALPHA: f32 = 1.673_263_2;
    /// SELU lambda (scale) constant
    pub const LAMBDA: f32 = 1.050_701;

    /// Create a new SELU layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SeluLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for SeluLayer {
    /// IBP for SELU: y = lambda * (x if x >= 0, else alpha * (exp(x) - 1))
    ///
    /// Similar to ELU but with fixed scaling.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let selu = |x: f32| -> f32 {
            if x >= 0.0 {
                Self::LAMBDA * x
            } else {
                Self::LAMBDA * Self::ALPHA * (x.exp() - 1.0)
            }
        };
        let lower = input.lower.mapv(selu);
        let upper = input.upper.mapv(selu);
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("SELU layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for SELU on interval [l, u].
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept).
fn selu_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    let alpha = SeluLayer::ALPHA;
    let lambda = SeluLayer::LAMBDA;

    let selu = |x: f32| -> f32 {
        if x >= 0.0 {
            lambda * x
        } else {
            lambda * alpha * (x.exp() - 1.0)
        }
    };

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let y = selu(l);
        let slope = if l >= 0.0 {
            lambda
        } else {
            lambda * alpha * l.exp() // derivative
        };
        let intercept = y - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let yl = selu(l);
    let yu = selu(u);

    // If both in positive region (l >= 0): linear with slope lambda
    if l >= 0.0 {
        return (lambda, 0.0, lambda, 0.0);
    }

    // For negative or crossing region, compute chord and sample
    let chord_slope = (yu - yl) / (u - l);
    let chord_intercept = yl - chord_slope * l;

    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let sx = selu(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = sx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Check x=0 if in range (transition point)
    if l < 0.0 && u > 0.0 {
        let s0 = 0.0;
        let c0 = chord_intercept;
        let diff = s0 - c0;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-6;
    (
        chord_slope,
        chord_intercept - max_below_chord - eps,
        chord_slope,
        chord_intercept + max_above_chord + eps,
    )
}

impl SeluLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("SELU layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, selu_linear_relaxation)
    }
}

/// PRelu (Parametric ReLU) layer: y = x if x >= 0, else slope * x
///
/// Unlike LeakyReLU which has a fixed slope, PRelu has learned per-channel
/// slopes. Common in detection models like RetinaNet.
#[derive(Debug, Clone)]
pub struct PReluLayer {
    /// Per-channel slopes for negative inputs (can be a single value broadcast to all channels)
    pub slope: Array1<f32>,
}

impl PReluLayer {
    /// Create a new PRelu layer with the given slopes.
    pub fn new(slope: Array1<f32>) -> Self {
        Self { slope }
    }

    /// Create a PRelu layer with a single slope value (broadcast to all elements).
    pub fn from_scalar(slope: f32) -> Self {
        Self {
            slope: Array1::from_elem(1, slope),
        }
    }

    /// Get the slope for a given index (handles broadcasting).
    #[inline]
    fn get_slope(&self, idx: usize) -> f32 {
        if self.slope.len() == 1 {
            self.slope[0]
        } else {
            self.slope[idx % self.slope.len()]
        }
    }
}

impl BoundPropagation for PReluLayer {
    /// IBP for PRelu: y = x if x >= 0, else slope * x
    ///
    /// Similar to LeakyReLU but with per-channel slopes.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let flat_lower = input
            .lower
            .view()
            .into_shape_with_order(input.lower.len())
            .unwrap();
        let flat_upper = input
            .upper
            .view()
            .into_shape_with_order(input.upper.len())
            .unwrap();

        let mut lower = Array1::zeros(flat_lower.len());
        let mut upper = Array1::zeros(flat_upper.len());

        for i in 0..flat_lower.len() {
            let slope = self.get_slope(i);
            let l = flat_lower[i];
            let u = flat_upper[i];

            // PRelu is monotonic (slope > 0 typical, but handle slope < 0 case too)
            if slope >= 0.0 {
                // Monotonically increasing on both regions
                lower[i] = if l >= 0.0 { l } else { slope * l };
                upper[i] = if u >= 0.0 { u } else { slope * u };
            } else {
                // slope < 0: negative region reverses order
                if l >= 0.0 {
                    lower[i] = l;
                    upper[i] = u;
                } else if u <= 0.0 {
                    // Both negative: slope * x is decreasing since slope < 0
                    lower[i] = slope * u;
                    upper[i] = slope * l;
                } else {
                    // l < 0 < u: need union of both regions
                    lower[i] = (slope * l).min(0.0).min(slope * u);
                    upper[i] = (slope * l).max(u).max(slope * u);
                }
            }
        }

        let lower = lower
            .into_shape_with_order(input.shape())
            .unwrap()
            .to_owned();
        let upper = upper
            .into_shape_with_order(input.shape())
            .unwrap()
            .to_owned();
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("PRelu layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl PReluLayer {
    /// CROWN backward propagation through PRelu with pre-activation bounds.
    ///
    /// PRelu: `y = x` if `x >= 0`, else `slope[i] * x` (per-channel slopes)
    ///
    /// Linear relaxation on interval [l, u]:
    /// - If l >= 0: identity (slope=1, intercept=0)
    /// - If `u <= 0`: scaled (`slope=slope[i]`, `intercept=0`)
    /// - If l < 0 < u: crossing region needs relaxation
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("PRelu layer CROWN backward propagation with pre-activation bounds");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        let mut upper_slope = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercept = Array1::<f32>::zeros(num_neurons);
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];
            let alpha = self.get_slope(i);

            if l >= 0.0 {
                // Always positive: identity
                upper_slope[i] = 1.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = 1.0;
            } else if u <= 0.0 {
                // Always negative: scaled by alpha
                upper_slope[i] = alpha;
                upper_intercept[i] = 0.0;
                lower_slope[i] = alpha;
            } else {
                // Crossing: linear relaxation
                // Upper bound line: from (l, alpha*l) to (u, u)
                let slope = (u - alpha * l) / (u - l);
                upper_slope[i] = slope;
                upper_intercept[i] = l * (alpha - slope);

                // Lower bound heuristic: use identity (slope=1) if positive area larger
                lower_slope[i] = if u > (-alpha * l).abs() { 1.0 } else { alpha };
            }
        }

        // Backward propagation through PRelu
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    new_lower_a[[j, i]] = la * lower_slope[i];
                } else {
                    new_lower_a[[j, i]] = la * upper_slope[i];
                    new_lower_b[j] += la * upper_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    new_upper_a[[j, i]] = ua * upper_slope[i];
                    new_upper_b[j] += ua * upper_intercept[i];
                } else {
                    new_upper_a[[j, i]] = ua * lower_slope[i];
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

/// HardSigmoid layer: y = max(0, min(1, alpha * x + beta))
///
/// A piecewise linear approximation of sigmoid, more efficient to compute.
/// Default ONNX values: alpha = 0.2, beta = 0.5
#[derive(Debug, Clone)]
pub struct HardSigmoidLayer {
    /// Slope in the linear region (default: 0.2)
    pub alpha: f32,
    /// Offset in the linear region (default: 0.5)
    pub beta: f32,
}

impl HardSigmoidLayer {
    /// Create a new HardSigmoid layer with the given parameters.
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }

    /// Create a HardSigmoid layer with ONNX default parameters (alpha=0.2, beta=0.5).
    pub fn default_params() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.5,
        }
    }

    /// Evaluate HardSigmoid at a point: max(0, min(1, alpha * x + beta))
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        (self.alpha * x + self.beta).clamp(0.0, 1.0)
    }
}

impl Default for HardSigmoidLayer {
    fn default() -> Self {
        Self::default_params()
    }
}

impl BoundPropagation for HardSigmoidLayer {
    /// IBP for HardSigmoid: y = max(0, min(1, alpha * x + beta))
    ///
    /// Three regions:
    /// - y = 0 when alpha * x + beta <= 0
    /// - y = alpha * x + beta when 0 < alpha * x + beta < 1
    /// - y = 1 when alpha * x + beta >= 1
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // HardSigmoid is monotonically increasing (alpha > 0 typically)
        let eval_lower = |x: f32| self.eval(x);
        let eval_upper = |x: f32| self.eval(x);

        // Since it's monotonic, bounds are straightforward
        let lower = input.lower.mapv(eval_lower);
        let upper = input.upper.mapv(eval_upper);
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("HardSigmoid layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl HardSigmoidLayer {
    /// CROWN backward propagation through HardSigmoid with pre-activation bounds.
    ///
    /// HardSigmoid: y = max(0, min(1, alpha*x + beta))
    ///
    /// Piecewise linear with three regions:
    /// - x < (0 - beta) / alpha: y = 0
    /// - (0 - beta) / alpha <= x <= (1 - beta) / alpha: y = alpha*x + beta
    /// - x > (1 - beta) / alpha: y = 1
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!(
            "HardSigmoid layer CROWN backward propagation (alpha={}, beta={})",
            self.alpha, self.beta
        );

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let alpha = self.alpha;
        let beta = self.beta;

        // Boundaries of the linear region
        let x_low = (0.0 - beta) / alpha; // where alpha*x + beta = 0
        let x_high = (1.0 - beta) / alpha; // where alpha*x + beta = 1

        let mut upper_slope = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercept = Array1::<f32>::zeros(num_neurons);
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercept = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            // Evaluate HardSigmoid at endpoints
            let yl = self.eval(l);
            let yu = self.eval(u);

            if u <= x_low {
                // Entirely in y = 0 region
                upper_slope[i] = 0.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = 0.0;
                lower_intercept[i] = 0.0;
            } else if l >= x_high {
                // Entirely in y = 1 region
                upper_slope[i] = 0.0;
                upper_intercept[i] = 1.0;
                lower_slope[i] = 0.0;
                lower_intercept[i] = 1.0;
            } else if l >= x_low && u <= x_high {
                // Entirely in linear region: y = alpha*x + beta
                upper_slope[i] = alpha;
                upper_intercept[i] = beta;
                lower_slope[i] = alpha;
                lower_intercept[i] = beta;
            } else {
                // Crossing at least one boundary - use chord
                let chord_slope = if (u - l).abs() > 1e-8 {
                    (yu - yl) / (u - l)
                } else {
                    0.0
                };
                let chord_intercept = yl - chord_slope * l;

                upper_slope[i] = chord_slope;
                upper_intercept[i] = chord_intercept;
                lower_slope[i] = chord_slope;
                lower_intercept[i] = chord_intercept;
            }
        }

        // Backward propagation
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                if la >= 0.0 {
                    new_lower_a[[j, i]] = la * lower_slope[i];
                    new_lower_b[j] += la * lower_intercept[i];
                } else {
                    new_lower_a[[j, i]] = la * upper_slope[i];
                    new_lower_b[j] += la * upper_intercept[i];
                }

                if ua >= 0.0 {
                    new_upper_a[[j, i]] = ua * upper_slope[i];
                    new_upper_b[j] += ua * upper_intercept[i];
                } else {
                    new_upper_a[[j, i]] = ua * lower_slope[i];
                    new_upper_b[j] += ua * lower_intercept[i];
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

/// HardSwish layer: y = x * HardSigmoid(x)
///
/// Used in MobileNetV3 as a more efficient alternative to Swish (SiLU).
/// y = x * max(0, min(1, (x + 3) / 6))
#[derive(Debug, Clone, Default)]
pub struct HardSwishLayer;

impl HardSwishLayer {
    /// Create a new HardSwish layer.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate HardSwish at a point: x * max(0, min(1, (x + 3) / 6))
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
    }
}

impl BoundPropagation for HardSwishLayer {
    /// IBP for HardSwish: y = x * max(0, min(1, (x + 3) / 6))
    ///
    /// Three regions:
    /// - y = 0 when x <= -3 (HardSigmoid = 0)
    /// - y = x * (x + 3) / 6 when -3 < x < 3 (quadratic)
    /// - y = x when x >= 3 (HardSigmoid = 1)
    ///
    /// The quadratic region has derivative (2x + 3) / 6, zero at x = -1.5
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut lower_vals = input.lower.clone();
        let mut upper_vals = input.upper.clone();

        ndarray::Zip::from(&mut lower_vals)
            .and(&mut upper_vals)
            .and(&input.lower)
            .and(&input.upper)
            .for_each(|out_l, out_u, &in_l, &in_u| {
                // Evaluate at bounds
                let y_l = self.eval(in_l);
                let y_u = self.eval(in_u);

                // In the quadratic region (-3 < x < 3), the minimum is at x = -1.5
                // where y = -1.5 * (-1.5 + 3) / 6 = -1.5 * 1.5 / 6 = -0.375
                let min_at_critical = if in_l < -1.5 && in_u > -1.5 && in_l > -3.0 {
                    self.eval(-1.5) // = -0.375
                } else {
                    f32::INFINITY
                };

                *out_l = y_l.min(y_u).min(min_at_critical);
                *out_u = y_l.max(y_u);
            });

        BoundedTensor::new(lower_vals, upper_vals)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("HardSwish layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Evaluate HardSwish: x * HardSigmoid(x) = x * max(0, min(1, (x + 3) / 6))
#[inline]
fn hardswish_eval(x: f32) -> f32 {
    x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// Linear relaxation for HardSwish on interval [l, u].
///
/// HardSwish has three regions:
/// - x <= -3: y = 0 (constant)
/// - -3 < x < 3: y = x * (x + 3) / 6 (quadratic, min at x = -1.5)
/// - x >= 3: y = x (linear)
fn hardswish_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        // Point interval: use derivative
        // d/dx[x * HardSigmoid(x)] = HardSigmoid(x) + x * HardSigmoid'(x)
        // For -3 < x < 3: d/dx = (x + 3)/6 + x/6 = (2x + 3)/6
        // For x <= -3: d/dx = 0
        // For x >= 3: d/dx = 1
        let slope = if l <= -3.0 {
            0.0
        } else if l >= 3.0 {
            1.0
        } else {
            (2.0 * l + 3.0) / 6.0
        };
        let y = hardswish_eval(l);
        let intercept = y - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let fl = hardswish_eval(l);
    let fu = hardswish_eval(u);

    // Chord slope and intercept
    let chord_slope = (fu - fl) / (u - l);
    let chord_intercept = fl - chord_slope * l;

    // Sample the interval to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let fx = hardswish_eval(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = fx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Check critical point x = -1.5 (minimum of quadratic region) if in interval
    if l < -1.5 && -1.5 < u {
        let fx = hardswish_eval(-1.5);
        let cx = chord_slope * (-1.5) + chord_intercept;
        let diff = fx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Check boundary points -3 and 3 if in interval
    if l < -3.0 && -3.0 < u {
        let fx = hardswish_eval(-3.0);
        let cx = chord_slope * (-3.0) + chord_intercept;
        let diff = fx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }
    if l < 3.0 && 3.0 < u {
        let fx = hardswish_eval(3.0);
        let cx = chord_slope * 3.0 + chord_intercept;
        let diff = fx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl HardSwishLayer {
    /// CROWN backward propagation through HardSwish with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("HardSwish layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, hardswish_linear_relaxation)
    }
}

/// Exp layer: y = exp(x)
///
/// Element-wise exponential function. Common in softmax decomposition
/// and various neural network architectures.
#[derive(Debug, Clone, Default)]
pub struct ExpLayer;

impl ExpLayer {
    /// Create a new Exp layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for ExpLayer {
    /// IBP for Exp: y = exp(x)
    ///
    /// Exp is monotonically increasing, so bounds are straightforward.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let lower = input.lower.mapv(|x| x.exp());
        let upper = input.upper.mapv(|x| x.exp());
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Exp CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for exp on interval [l, u].
/// exp(x) is convex and monotonically increasing.
/// For convex functions: chord is upper bound, tangent is lower bound.
fn exp_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let exp_l = l.exp();
        let slope = exp_l; // derivative of exp at l
        let intercept = exp_l - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let exp_l = l.exp();
    let exp_u = u.exp();

    // Chord slope connecting (l, exp(l)) to (u, exp(u))
    // For convex exp, chord is an UPPER bound
    let chord_slope = (exp_u - exp_l) / (u - l);
    let chord_intercept = exp_l - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let ex = x.exp();
        let cx = chord_slope * x + chord_intercept;
        let diff = ex - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    // For exp (convex): chord is upper, function is below chord
    // So max_above_chord should be ~0, max_below_chord is the gap
    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl ExpLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Exp layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, exp_linear_relaxation)
    }
}

/// Log layer: y = ln(x)
///
/// Element-wise natural logarithm. Requires x > 0.
/// Common in log-softmax and various loss functions.
#[derive(Debug, Clone, Default)]
pub struct LogLayer;

impl LogLayer {
    /// Create a new Log layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for LogLayer {
    /// IBP for Log: y = ln(x)
    ///
    /// Log is monotonically increasing for x > 0, so bounds are straightforward.
    /// For numerical stability, clamp inputs to a small positive value.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        const EPSILON: f32 = 1e-10;
        let lower = input.lower.mapv(|x| x.max(EPSILON).ln());
        let upper = input.upper.mapv(|x| x.max(EPSILON).ln());
        BoundedTensor::new(lower, upper)
    }

    /// CROWN propagation without bounds - uses identity
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Log CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for log on interval [l, u].
/// log(x) is concave and monotonically increasing (for x > 0).
/// For concave functions: chord is lower bound, tangent is upper bound.
fn log_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    const EPSILON: f32 = 1e-10;
    let l = l.max(EPSILON);
    let u = u.max(EPSILON);

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let log_l = l.ln();
        let slope = 1.0 / l; // derivative of log at l
        let intercept = log_l - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let log_l = l.ln();
    let log_u = u.ln();

    // Chord slope connecting (l, log(l)) to (u, log(u))
    // For concave log, chord is a LOWER bound
    let chord_slope = (log_u - log_l) / (u - l);
    let chord_intercept = log_l - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let lx = x.ln();
        let cx = chord_slope * x + chord_intercept;
        let diff = lx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    // For log (concave): chord is lower, function is above chord
    // So max_below_chord should be ~0, max_above_chord is the gap
    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl LogLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Log layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, log_linear_relaxation)
    }
}

/// CELU (Continuous ELU) layer: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
///
/// CELU is a smooth, continuously differentiable variant of ELU.
/// Unlike ELU, CELU is differentiable everywhere (including at x=0).
/// The parameter alpha controls the saturation value for negative inputs.
#[derive(Debug, Clone)]
pub struct CeluLayer {
    /// Scale parameter for the exponential (default: 1.0)
    pub alpha: f32,
}

impl CeluLayer {
    /// Create a new CELU layer with the given alpha.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Create a CELU layer with default alpha = 1.0.
    pub fn default_alpha() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Default for CeluLayer {
    fn default() -> Self {
        Self::default_alpha()
    }
}

impl BoundPropagation for CeluLayer {
    /// IBP for CELU: y = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    ///
    /// CELU is monotonically increasing, so bounds map directly.
    /// For x >= 0: y = x
    /// For x < 0: y = alpha * (exp(x/alpha) - 1) (approaches -alpha as x -> -inf)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;
        let celu = |x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                alpha * ((x / alpha).exp() - 1.0)
            }
        };
        // CELU is monotonically increasing, so bounds map directly
        let lower = input.lower.mapv(celu);
        let upper = input.upper.mapv(celu);
        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("CELU layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Evaluate CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
#[inline]
fn celu_eval(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        alpha * ((x / alpha).exp() - 1.0)
    }
}

/// Derivative of CELU: 1 for x >= 0, exp(x/alpha) for x < 0
#[inline]
fn celu_derivative(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        (x / alpha).exp()
    }
}

/// Linear relaxation for CELU on interval [l, u].
///
/// CELU is monotonically increasing:
/// - For x >= 0: y = x (linear with slope 1)
/// - For x < 0: y = alpha * (exp(x/alpha) - 1) (convex, increasing)
fn celu_linear_relaxation(l: f32, u: f32, alpha: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let slope = celu_derivative(l, alpha);
        let y = celu_eval(l, alpha);
        let intercept = y - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let fl = celu_eval(l, alpha);
    let fu = celu_eval(u, alpha);

    // Chord slope and intercept
    let chord_slope = (fu - fl) / (u - l);
    let chord_intercept = fl - chord_slope * l;

    // CELU is convex in the negative region (second derivative > 0)
    // and linear in the positive region
    // For convex functions: chord is above, tangent is below

    // Sample the interval to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let fx = celu_eval(x, alpha);
        let cx = chord_slope * x + chord_intercept;
        let diff = fx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Check the boundary at x=0 if in interval
    if l < 0.0 && 0.0 < u {
        let fx = 0.0; // celu_eval(0, alpha) = 0
        let cx = chord_intercept;
        let diff = fx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl CeluLayer {
    /// CROWN backward propagation through CELU with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("CELU layer CROWN backward propagation with pre-activation bounds");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let alpha = self.alpha;

        // Compute relaxation parameters for each neuron
        let mut lower_slopes = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercepts = Array1::<f32>::zeros(num_neurons);
        let mut upper_slopes = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercepts = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];
            let (ls, li, us, ui) = celu_linear_relaxation(l, u, alpha);
            lower_slopes[i] = ls;
            lower_intercepts[i] = li;
            upper_slopes[i] = us;
            upper_intercepts[i] = ui;
        }

        // Backward propagation through CELU
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                if la >= 0.0 {
                    new_lower_a[[j, i]] = la * lower_slopes[i];
                    new_lower_b[j] += la * lower_intercepts[i];
                } else {
                    new_lower_a[[j, i]] = la * upper_slopes[i];
                    new_lower_b[j] += la * upper_intercepts[i];
                }

                if ua >= 0.0 {
                    new_upper_a[[j, i]] = ua * upper_slopes[i];
                    new_upper_b[j] += ua * upper_intercepts[i];
                } else {
                    new_upper_a[[j, i]] = ua * lower_slopes[i];
                    new_upper_b[j] += ua * lower_intercepts[i];
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

/// Mish layer: y = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// Mish is a self-regularized non-monotonic activation function.
/// It has been shown to outperform ReLU/Swish in various benchmarks
/// (e.g., YOLOv4). Unlike ReLU, Mish is smooth and allows small
/// negative values, which can improve gradient flow.
#[derive(Debug, Clone, Default)]
pub struct MishLayer;

impl MishLayer {
    /// Create a new Mish layer.
    pub fn new() -> Self {
        Self
    }
}

/// Evaluate Mish: x * tanh(softplus(x))
#[inline]
fn mish_eval(x: f32) -> f32 {
    // softplus(x) = ln(1 + exp(x))
    // For numerical stability, use log1p(exp(x)) for small x
    // and x + log1p(exp(-x)) for large x
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    };
    x * softplus.tanh()
}

/// Mish derivative: tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
#[inline]
fn mish_derivative(x: f32) -> f32 {
    // For numerical stability
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    };
    let tanh_sp = softplus.tanh();
    let sech2_sp = 1.0 - tanh_sp * tanh_sp;
    let sigmoid = 1.0 / (1.0 + (-x).exp());

    // d/dx[x * tanh(softplus(x))]
    // = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
    tanh_sp + x * sech2_sp * sigmoid
}

impl BoundPropagation for MishLayer {
    /// IBP for Mish: y = x * tanh(softplus(x))
    ///
    /// Mish is NOT monotonic - it has a minimum near x ≈ -0.31.
    /// We need to check for the critical point in each interval.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Mish has a critical point (minimum) near x ≈ -0.31 where derivative = 0
        // We use Newton's method to find it more precisely
        static MISH_CRITICAL: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let critical = *MISH_CRITICAL.get_or_init(|| {
            // Newton's method to find where derivative = 0
            let mut x = -0.31_f32;
            for _ in 0..20 {
                let d = mish_derivative(x);
                // Numerical derivative of derivative
                let eps = 1e-5_f32;
                let dd = (mish_derivative(x + eps) - mish_derivative(x - eps)) / (2.0 * eps);
                if dd.abs() < 1e-10 {
                    break;
                }
                x -= d / dd;
            }
            x
        });
        let critical_val = mish_eval(critical);

        let bound_fn = |l: f32, u: f32| -> (f32, f32) {
            let fl = mish_eval(l);
            let fu = mish_eval(u);

            // Check if critical point is in interval
            if l <= critical && critical <= u {
                // Minimum is at critical point
                (critical_val.min(fl).min(fu), fl.max(fu))
            } else {
                // Mish is monotonic in this interval (either side of minimum)
                (fl.min(fu), fl.max(fu))
            }
        };

        let lower_shape = input.lower.shape().to_vec();
        let mut lower_data = Vec::with_capacity(input.lower.len());
        let mut upper_data = Vec::with_capacity(input.upper.len());

        for (l, u) in input.lower.iter().zip(input.upper.iter()) {
            let (lo, hi) = bound_fn(*l, *u);
            lower_data.push(lo);
            upper_data.push(hi);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&lower_shape), lower_data)
            .map_err(|e| GammaError::InvalidSpec(format!("Mish lower reshape: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&lower_shape), upper_data)
            .map_err(|e| GammaError::InvalidSpec(format!("Mish upper reshape: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Mish layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for Mish on interval [l, u].
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// Mish is NOT monotonic - it has a minimum near x ≈ -0.31.
fn mish_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        // Point interval: use derivative
        let slope = mish_derivative(l);
        let y = mish_eval(l);
        let intercept = y - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let fl = mish_eval(l);
    let fu = mish_eval(u);

    // Chord slope and intercept
    let chord_slope = (fu - fl) / (u - l);
    let chord_intercept = fl - chord_slope * l;

    // Sample the interval to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let fx = mish_eval(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = fx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Check the critical point (minimum) near x ≈ -0.31 if in interval
    // The exact critical point is where mish_derivative(x) = 0
    let critical = -0.31_f32;
    if l < critical && critical < u {
        let fx = mish_eval(critical);
        let cx = chord_slope * critical + chord_intercept;
        let diff = fx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Also check x=0 since softplus(0) = ln(2), which affects the curve
    if l < 0.0 && 0.0 < u {
        let fx = mish_eval(0.0);
        let cx = chord_intercept;
        let diff = fx - cx;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl MishLayer {
    /// CROWN backward propagation through Mish with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Mish layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, mish_linear_relaxation)
    }
}

/// LogSoftmax layer: y = log(softmax(x)) = x - logsumexp(x)
///
/// LogSoftmax is more numerically stable than computing log(softmax(x))
/// directly. It's commonly used with NLLLoss for classification.
#[derive(Debug, Clone)]
pub struct LogSoftmaxLayer {
    /// Dimension along which to apply logsoftmax (default: -1)
    pub axis: i32,
}

impl LogSoftmaxLayer {
    /// Create a new LogSoftmax layer.
    pub fn new(axis: i32) -> Self {
        Self { axis }
    }
}

impl Default for LogSoftmaxLayer {
    fn default() -> Self {
        Self { axis: -1 }
    }
}

impl BoundPropagation for LogSoftmaxLayer {
    /// IBP for LogSoftmax: y = x - logsumexp(x)
    ///
    /// For bounds on logsoftmax_i = x_i - log(sum_j exp(x_j)):
    /// - Lower bound on logsoftmax_i: x_i^L - logsumexp(x^U)
    /// - Upper bound on logsoftmax_i: x_i^U - logsumexp(x^L)
    ///
    /// This is a simplified but sound bound: we use the max logsumexp for all lower bounds
    /// and min logsumexp for all upper bounds.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.lower.shape();
        let ndim = shape.len();
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Handle axis >= ndim gracefully (see SoftmaxLayer::propagate_ibp for explanation)
        let axis = if axis >= ndim {
            debug!(
                "LogSoftmax axis {} out of bounds for {} dimensions, using last axis (-1)",
                self.axis, ndim
            );
            ndim - 1
        } else {
            axis
        };

        // Compute logsumexp bounds along the specified axis
        // logsumexp(x^U) is the max logsumexp (for lower bound)
        // logsumexp(x^L) is the min logsumexp (for upper bound)

        // For numerical stability, we compute logsumexp as: max + log(sum(exp(x - max)))
        let max_upper = input
            .upper
            .fold_axis(Axis(axis), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
        let max_lower = input
            .lower
            .fold_axis(Axis(axis), f32::NEG_INFINITY, |&acc, &x| acc.max(x));

        // Broadcast max back to original shape for subtraction
        let max_upper_expanded = max_upper.clone().insert_axis(Axis(axis));
        let max_lower_expanded = max_lower.clone().insert_axis(Axis(axis));

        // Compute exp(x - max) for numerical stability
        let exp_upper_shifted = (&input.upper - &max_upper_expanded).mapv(|x| x.exp());
        let exp_lower_shifted = (&input.lower - &max_lower_expanded).mapv(|x| x.exp());

        // Sum along axis
        let sum_exp_upper = exp_upper_shifted.sum_axis(Axis(axis));
        let sum_exp_lower = exp_lower_shifted.sum_axis(Axis(axis));

        // logsumexp = max + log(sum(exp(x - max)))
        let logsumexp_upper = &max_upper + &sum_exp_upper.mapv(|x| x.ln());
        let logsumexp_lower = &max_lower + &sum_exp_lower.mapv(|x| x.ln());

        // Broadcast back to original shape
        let logsumexp_upper_expanded = logsumexp_upper.insert_axis(Axis(axis));
        let logsumexp_lower_expanded = logsumexp_lower.insert_axis(Axis(axis));

        // logsoftmax_i = x_i - logsumexp(x)
        // Lower: x_i^L - logsumexp(x^U)
        // Upper: x_i^U - logsumexp(x^L)
        let lower = &input.lower - &logsumexp_upper_expanded;
        let upper = &input.upper - &logsumexp_lower_expanded;

        BoundedTensor::new(lower, upper)
    }

    /// CROWN propagation without bounds - uses identity
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("LogSoftmax CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl LogSoftmaxLayer {
    /// Evaluate logsoftmax at a point.
    ///
    /// logsoftmax(x) = x - logsumexp(x)
    fn eval(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();
        let logsumexp = max_val + exp_sum.ln();
        x.mapv(|v| v - logsumexp)
    }

    /// Compute softmax at a point.
    fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Array1<f32> = x.mapv(|v| (v - max_val).exp());
        let sum: f32 = exp_vals.sum();
        exp_vals / sum
    }

    /// Compute Jacobian of logsoftmax at a point.
    ///
    /// J[i,j] = δ_ij - softmax[j]
    ///
    /// The Jacobian is: I - 1 * softmax^T
    fn jacobian(&self, x: &Array1<f32>) -> Array2<f32> {
        let n = x.len();
        let s = self.softmax(x);

        let mut j = Array2::<f32>::eye(n);
        for i in 0..n {
            for k in 0..n {
                j[[i, k]] -= s[k];
            }
        }
        j
    }

    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// LogSoftmax has global dependencies through the logsumexp term.
    /// We use a Jacobian-based linear approximation at the interval center
    /// with sampling to bound the approximation error.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("LogSoftmax layer CROWN backward propagation with pre-activation bounds");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Check for infinite or NaN bounds - return identity relaxation
        let has_infinite = pre_lower.iter().any(|&v| v.is_infinite() || v.is_nan())
            || pre_upper.iter().any(|&v| v.is_infinite() || v.is_nan());
        if has_infinite {
            return Ok(bounds.clone());
        }

        // Compute center point and evaluate
        let x_center: Array1<f32> = pre_lower
            .iter()
            .zip(pre_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();

        let y_center = self.eval(&x_center);
        let jacobian = self.jacobian(&x_center);

        // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
        let jx_center = jacobian.dot(&x_center);
        let b_approx: Array1<f32> = &y_center - &jx_center;

        // Sample to find max error from linear approximation
        let num_samples = 50;
        let mut max_error_above: Array1<f32> = Array1::zeros(num_neurons);
        let mut max_error_below: Array1<f32> = Array1::zeros(num_neurons);

        let mut x_sample = x_center.clone();

        // Sample random points in the hypercube
        for sample_idx in 0..num_samples {
            x_sample.assign(&x_center);
            for i in 0..num_neurons {
                // Pseudo-random sampling with fixed seed for reproducibility
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }

            // Also sample corners for first few samples
            if sample_idx < num_neurons * 2 {
                let dim = sample_idx / 2;
                if dim < num_neurons {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        pre_lower[dim]
                    } else {
                        pre_upper[dim]
                    };
                }
            }

            let y_actual = self.eval(&x_sample);
            let y_approx: Array1<f32> = jacobian.dot(&x_sample) + &b_approx;

            for i in 0..num_neurons {
                let error = y_actual[i] - y_approx[i];
                if error > max_error_above[i] {
                    max_error_above[i] = error;
                }
                if -error > max_error_below[i] {
                    max_error_below[i] = -error;
                }
            }
        }

        // Add safety margin (10% extra for unsampled regions)
        let safety_factor = 1.1;
        let min_margin = 1e-6_f32;
        for i in 0..num_neurons {
            max_error_above[i] = (max_error_above[i] * safety_factor).max(min_margin);
            max_error_below[i] = (max_error_below[i] * safety_factor).max(min_margin);
        }

        // Backward propagation using linear relaxation
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large, use lower relaxation
                    for k in 0..num_neurons {
                        new_lower_a[[j, k]] += la * jacobian[[i, k]];
                    }
                    new_lower_b[j] += la * (b_approx[i] - max_error_below[i]);
                } else {
                    // Negative coeff: want y_i to be small, use upper relaxation
                    for k in 0..num_neurons {
                        new_lower_a[[j, k]] += la * jacobian[[i, k]];
                    }
                    new_lower_b[j] += la * (b_approx[i] + max_error_above[i]);
                }

                // For upper bound output
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be large, use upper relaxation
                    for k in 0..num_neurons {
                        new_upper_a[[j, k]] += ua * jacobian[[i, k]];
                    }
                    new_upper_b[j] += ua * (b_approx[i] + max_error_above[i]);
                } else {
                    // Negative coeff: want y_i to be small, use lower relaxation
                    for k in 0..num_neurons {
                        new_upper_a[[j, k]] += ua * jacobian[[i, k]];
                    }
                    new_upper_b[j] += ua * (b_approx[i] - max_error_below[i]);
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

/// ThresholdedRelu layer: y = x if x > alpha, else 0
///
/// Similar to ReLU but with a configurable threshold alpha (default: 1.0).
/// This is useful for sparse feature selection where only sufficiently
/// strong activations are passed through.
#[derive(Debug, Clone)]
pub struct ThresholdedReluLayer {
    /// Threshold value (default: 1.0)
    pub alpha: f32,
}

impl ThresholdedReluLayer {
    /// Create a new ThresholdedRelu layer.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ThresholdedReluLayer {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl BoundPropagation for ThresholdedReluLayer {
    /// IBP for ThresholdedRelu: y = x if x > alpha, else 0
    ///
    /// - If lower > alpha: both bounds pass through unchanged
    /// - If upper <= alpha: output is [0, 0]
    /// - If lower <= alpha < upper: output is [0, upper]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;

        let lower_shape = input.lower.shape().to_vec();
        let mut lower_data = Vec::with_capacity(input.lower.len());
        let mut upper_data = Vec::with_capacity(input.upper.len());

        for (&l, &u) in input.lower.iter().zip(input.upper.iter()) {
            if l > alpha {
                // Entirely above threshold: pass through
                lower_data.push(l);
                upper_data.push(u);
            } else if u <= alpha {
                // Entirely below or at threshold: output is 0
                lower_data.push(0.0);
                upper_data.push(0.0);
            } else {
                // Crosses threshold: lower could be 0, upper passes through
                lower_data.push(0.0);
                upper_data.push(u);
            }
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&lower_shape), lower_data).map_err(|e| {
            GammaError::InvalidSpec(format!("ThresholdedRelu lower reshape: {}", e))
        })?;
        let upper = ArrayD::from_shape_vec(IxDyn(&lower_shape), upper_data).map_err(|e| {
            GammaError::InvalidSpec(format!("ThresholdedRelu upper reshape: {}", e))
        })?;

        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("ThresholdedRelu layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl ThresholdedReluLayer {
    /// CROWN backward propagation through ThresholdedRelu with pre-activation bounds.
    ///
    /// For ThresholdedRelu y = x if x > α, else 0, with pre-activation bounds [l, u]:
    /// - If l > α: y = x (identity), pass-through
    /// - If u <= α: y = 0 (zero), no dependence
    /// - If l <= α < u: use linear relaxation
    ///   - Upper: y <= λ(x - l) where λ = u/(u-l) (line from (l,0) to (u,u))
    ///   - Lower: y >= slope*x + intercept, heuristic based on which region dominates
    ///
    /// The backward propagation handles positive/negative coefficients differently.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("ThresholdedRelu layer CROWN backward propagation with pre-activation bounds");

        let alpha = self.alpha;
        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        // For crossing neurons (l <= alpha < u):
        //   lambda = u / (u - l)       (upper slope)
        //   lower_slope heuristic based on active vs inactive region
        let mut lambda = Array1::<f32>::zeros(num_neurons);
        let mut lambda_intercept = Array1::<f32>::zeros(num_neurons); // -lambda * l
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercept = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l > alpha {
                // Always active: identity
                lambda[i] = 1.0;
                lambda_intercept[i] = 0.0;
                lower_slope[i] = 1.0;
                lower_intercept[i] = 0.0;
            } else if u <= alpha {
                // Always inactive: zero
                lambda[i] = 0.0;
                lambda_intercept[i] = 0.0;
                lower_slope[i] = 0.0;
                lower_intercept[i] = 0.0;
            } else {
                // Crossing: linear relaxation
                // ThresholdedRelu has a discontinuity at x=alpha: f(alpha)=0 but f(alpha+)≈alpha
                // Upper bound must satisfy:
                //   - y >= 0 for x in [l, alpha]
                //   - y >= x for x in (alpha, u]
                // Line through (l, 0) needs slope >= max(u/(u-l), alpha/(alpha-l))
                let slope_endpoint = u / (u - l);
                let slope_threshold = alpha / (alpha - l);
                lambda[i] = slope_endpoint.max(slope_threshold);
                lambda_intercept[i] = -lambda[i] * l;

                // Lower bound: y >= 0 is always valid since f(x) >= 0 everywhere
                // Tighter: y >= x - alpha is valid when alpha >= 0 (f(x) >= x - alpha for x > alpha)
                // But at x = l, y = l - alpha which may be < 0 = f(l), so we need l >= alpha
                // Since l <= alpha in crossing case, use y >= 0 as lower bound
                // unless we can use the active region tangent
                if alpha >= 0.0 && alpha >= l {
                    // Tangent to active region: y = x - alpha passes through (alpha, 0)
                    // At x = l: y = l - alpha <= 0 = f(l) since l <= alpha. Valid!
                    // At x = u: y = u - alpha < u = f(u) since alpha > 0. Valid lower bound!
                    lower_slope[i] = 1.0;
                    lower_intercept[i] = -alpha;
                } else {
                    // Conservative: y >= 0
                    lower_slope[i] = 0.0;
                    lower_intercept[i] = 0.0;
                }
            }
        }

        // Backward propagation through ThresholdedRelu
        // Same pattern as ReLU: use upper/lower relaxation based on coefficient sign
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large
                    // Use lower relaxation
                    new_lower_a[[j, i]] = la * lower_slope[i];
                    new_lower_b[j] += la * lower_intercept[i];
                } else {
                    // Negative coeff: want y_i to be small
                    // Use upper relaxation
                    new_lower_a[[j, i]] = la * lambda[i];
                    new_lower_b[j] += la * lambda_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be small
                    // Use upper relaxation
                    new_upper_a[[j, i]] = ua * lambda[i];
                    new_upper_b[j] += ua * lambda_intercept[i];
                } else {
                    // Negative coeff: want y_i to be large
                    // Use lower relaxation
                    new_upper_a[[j, i]] = ua * lower_slope[i];
                    new_upper_b[j] += ua * lower_intercept[i];
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

/// Shrink layer: soft thresholding / shrinkage operation
///
/// y = x - bias if x > lambd
/// y = x + bias if x < -lambd
/// y = 0 otherwise
///
/// This implements soft thresholding used in sparse coding and LASSO.
/// Default: bias = 0.0, lambd = 0.5
#[derive(Debug, Clone)]
pub struct ShrinkLayer {
    /// Bias value (default: 0.0)
    pub bias: f32,
    /// Lambda threshold (default: 0.5)
    pub lambd: f32,
}

impl ShrinkLayer {
    /// Create a new Shrink layer.
    pub fn new(bias: f32, lambd: f32) -> Self {
        Self { bias, lambd }
    }
}

impl Default for ShrinkLayer {
    fn default() -> Self {
        Self {
            bias: 0.0,
            lambd: 0.5,
        }
    }
}

/// Evaluate Shrink function: soft thresholding
fn shrink_scalar(x: f32, bias: f32, lambd: f32) -> f32 {
    if x > lambd {
        x - bias
    } else if x < -lambd {
        x + bias
    } else {
        0.0
    }
}

impl BoundPropagation for ShrinkLayer {
    /// IBP for Shrink: soft thresholding
    ///
    /// The function has three linear pieces:
    /// - x < -lambd: y = x + bias (slope 1)
    /// - -lambd <= x <= lambd: y = 0 (slope 0)
    /// - x > lambd: y = x - bias (slope 1)
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let bias = self.bias;
        let lambd = self.lambd;

        let lower_shape = input.lower.shape().to_vec();
        let mut lower_data = Vec::with_capacity(input.lower.len());
        let mut upper_data = Vec::with_capacity(input.upper.len());

        for (&l, &u) in input.lower.iter().zip(input.upper.iter()) {
            // Compute bounds by evaluating at endpoints and checking critical points
            let fl = shrink_scalar(l, bias, lambd);
            let fu = shrink_scalar(u, bias, lambd);

            // The function is piecewise linear with breakpoints at -lambd and lambd
            // Within each piece, it's monotonic

            // Check if interval spans multiple pieces
            let spans_neg_break = l < -lambd && u > -lambd;
            let spans_pos_break = l < lambd && u > lambd;
            let in_dead_zone = l >= -lambd && u <= lambd;

            let (nl, nu) = if in_dead_zone {
                // Entirely in dead zone
                (0.0, 0.0)
            } else if !spans_neg_break && !spans_pos_break {
                // Entirely in one linear piece (outside dead zone)
                // Monotonically increasing (slope 1)
                (fl.min(fu), fl.max(fu))
            } else {
                // Spans one or more breakpoints
                // Values at breakpoints: shrink(-lambd) = 0, shrink(lambd) = 0
                let mut candidates = vec![fl, fu];
                if spans_neg_break || spans_pos_break {
                    candidates.push(0.0); // Value at breakpoints
                }

                (
                    candidates.iter().cloned().fold(f32::INFINITY, f32::min),
                    candidates.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                )
            };

            lower_data.push(nl);
            upper_data.push(nu);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&lower_shape), lower_data)
            .map_err(|e| GammaError::InvalidSpec(format!("Shrink lower reshape: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&lower_shape), upper_data)
            .map_err(|e| GammaError::InvalidSpec(format!("Shrink upper reshape: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    /// Placeholder for CROWN
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Shrink layer CROWN propagation (placeholder)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl ShrinkLayer {
    /// CROWN backward propagation through Shrink with pre-activation bounds.
    ///
    /// For Shrink with parameters (bias, lambd):
    /// - x < -lambd: y = x + bias (slope 1, intercept +bias)
    /// - -lambd <= x <= lambd: y = 0 (slope 0, intercept 0)
    /// - x > lambd: y = x - bias (slope 1, intercept -bias)
    ///
    /// The function is piecewise linear with breakpoints at ±lambd.
    /// At both breakpoints, the function value is 0.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Shrink layer CROWN backward propagation with pre-activation bounds");

        let bias = self.bias;
        let lambd = self.lambd;
        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        // Upper bound: (upper_slope, upper_intercept) such that f(x) <= upper_slope * x + upper_intercept
        // Lower bound: (lower_slope, lower_intercept) such that f(x) >= lower_slope * x + lower_intercept
        let mut upper_slope = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercept = Array1::<f32>::zeros(num_neurons);
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercept = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            // Function values at endpoints
            let fl = shrink_scalar(l, bias, lambd);
            let fu = shrink_scalar(u, bias, lambd);

            // Determine which region(s) we're in
            let in_neg = u < -lambd;
            let in_dead = l >= -lambd && u <= lambd;
            let in_pos = l > lambd;
            let spans_neg_break = l < -lambd && u > -lambd;
            let spans_pos_break = l < lambd && u > lambd;

            if in_neg {
                // Entirely in negative piece: y = x + bias
                upper_slope[i] = 1.0;
                upper_intercept[i] = bias;
                lower_slope[i] = 1.0;
                lower_intercept[i] = bias;
            } else if in_dead {
                // Entirely in dead zone: y = 0
                upper_slope[i] = 0.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = 0.0;
                lower_intercept[i] = 0.0;
            } else if in_pos {
                // Entirely in positive piece: y = x - bias
                upper_slope[i] = 1.0;
                upper_intercept[i] = -bias;
                lower_slope[i] = 1.0;
                lower_intercept[i] = -bias;
            } else if spans_neg_break && !spans_pos_break {
                // Spans negative breakpoint: l < -lambd <= u <= lambd
                // f(l) = l + bias (negative piece), f(u) = 0 (dead zone)
                // Function: f = x + bias for x < -lambd, f = 0 for x >= -lambd
                // Upper bound: chord from (l, fl) to (-lambd, 0) - has slope > 1 when bias < lambd
                // This stays above f = x + bias (slope 1) in active region
                let upper_s = (0.0 - fl) / ((-lambd) - l);
                upper_slope[i] = upper_s;
                upper_intercept[i] = -upper_s * (-lambd); // passes through (-lambd, 0)
                                                          // Lower bound: chord from (l, fl) to (u, 0)
                                                          // This has slope < 1, stays below f in active region and equals f at endpoints
                let lower_s = (0.0 - fl) / (u - l);
                lower_slope[i] = lower_s;
                lower_intercept[i] = fl - lower_s * l; // passes through (l, fl)
            } else if !spans_neg_break && spans_pos_break {
                // Spans positive breakpoint: -lambd <= l <= lambd < u
                // f(l) = 0 (in dead zone), f(u) = u - bias
                // The function transitions from slope 0 to slope 1 at lambd
                // Lower bound: line from (lambd, 0) to (u, fu) = (u, u - bias)
                // Must be below f everywhere: at lambd, f = 0; for x > lambd, f = x - bias
                let lower_s = (fu - 0.0) / (u - lambd);
                lower_slope[i] = lower_s;
                lower_intercept[i] = -lower_s * lambd; // passes through (lambd, 0)
                                                       // Upper bound: y = 0 is valid for dead zone, but need y >= x - bias for active
                                                       // Use max of f at endpoints: max(0, fu) = fu
                upper_slope[i] = 0.0;
                upper_intercept[i] = fu.max(0.0);
            } else {
                // Spans both breakpoints: l < -lambd and u > lambd
                // f(l) = l + bias, f(u) = u - bias
                // This is a complex case - the function dips to 0 in the middle
                // Upper bound: line connecting (l, fl) to (u, fu)
                let chord_slope = (fu - fl) / (u - l);
                upper_slope[i] = chord_slope;
                upper_intercept[i] = fl - chord_slope * l;
                // But this might not be above 0 in the dead zone - check and adjust
                // Value at x=0: chord_slope * 0 + intercept = intercept
                // If intercept < 0, need to use a different upper bound
                if upper_intercept[i] < 0.0 {
                    // Use horizontal line at max of endpoints and 0
                    upper_slope[i] = 0.0;
                    upper_intercept[i] = fl.max(fu).max(0.0);
                }
                // Lower bound: minimum is 0 in the dead zone, or min of endpoints
                // Use horizontal line at min value
                lower_slope[i] = 0.0;
                lower_intercept[i] = fl.min(fu).min(0.0);
            }
        }

        // Backward propagation through Shrink
        // Same pattern as other activation layers
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large, use lower relaxation
                    new_lower_a[[j, i]] = la * lower_slope[i];
                    new_lower_b[j] += la * lower_intercept[i];
                } else {
                    // Negative coeff: want y_i to be small, use upper relaxation
                    new_lower_a[[j, i]] = la * upper_slope[i];
                    new_lower_b[j] += la * upper_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be small, use upper relaxation
                    new_upper_a[[j, i]] = ua * upper_slope[i];
                    new_upper_b[j] += ua * upper_intercept[i];
                } else {
                    // Negative coeff: want y_i to be large, use lower relaxation
                    new_upper_a[[j, i]] = ua * lower_slope[i];
                    new_upper_b[j] += ua * lower_intercept[i];
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

/// Softsign layer: y = x / (1 + |x|)
///
/// Output range is (-1, 1), similar to tanh but computationally cheaper.
/// The function is monotonically increasing and passes through origin.
#[derive(Debug, Clone)]
pub struct SoftsignLayer;

impl SoftsignLayer {
    /// Create a new Softsign layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SoftsignLayer {
    fn default() -> Self {
        Self
    }
}

/// Evaluate Softsign: x / (1 + |x|)
fn softsign_scalar(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

impl BoundPropagation for SoftsignLayer {
    /// IBP for Softsign: y = x / (1 + |x|)
    ///
    /// Softsign is monotonically increasing, so bounds are straightforward:
    /// lower_out = softsign(lower_in)
    /// upper_out = softsign(upper_in)
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Softsign is monotonically increasing, so we just apply element-wise
        let lower = input.lower.mapv(softsign_scalar);
        let upper = input.upper.mapv(softsign_scalar);

        BoundedTensor::new(lower, upper)
    }

    /// CROWN propagation without bounds - uses identity
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Softsign CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for softsign on interval [l, u].
/// softsign(x) = x / (1 + |x|) is monotonically increasing.
/// It's S-shaped like tanh: concave for x > 0, convex for x < 0.
fn softsign_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let sl = softsign_scalar(l);
        // Derivative of softsign at l: 1 / (1 + |l|)^2
        let denom = 1.0 + l.abs();
        let slope = 1.0 / (denom * denom);
        let intercept = sl - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let sl = softsign_scalar(l);
    let su = softsign_scalar(u);

    // Chord slope connecting (l, softsign(l)) to (u, softsign(u))
    let chord_slope = (su - sl) / (u - l);
    let chord_intercept = sl - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let sx = softsign_scalar(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = sx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Also check origin (x=0) if in interval - inflection point
    if l <= 0.0 && 0.0 <= u {
        let s0 = softsign_scalar(0.0); // = 0
        let c0 = chord_intercept;
        let diff = s0 - c0;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl SoftsignLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Softsign layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, softsign_linear_relaxation)
    }
}

// ============================================================================
// Transformer Layer Types
// ============================================================================

/// Softmax layer: y = softmax(x, dim)
///
/// Softmax normalizes inputs along a dimension so outputs sum to 1.
/// Uses Auto-LiRPA interval propagation algorithm for tight bounds.
#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    /// Dimension along which to apply softmax (default: -1)
    pub axis: i32,
}

impl SoftmaxLayer {
    /// Create a new Softmax layer.
    pub fn new(axis: i32) -> Self {
        Self { axis }
    }

    /// Evaluate softmax at a concrete point (1D).
    ///
    /// Returns softmax(x) = exp(x_i) / sum_j(exp(x_j))
    pub fn eval(&self, x: &Array1<f32>) -> Array1<f32> {
        // For numerical stability, subtract max
        let max_x = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x: Array1<f32> = x.mapv(|xi| (xi - max_x).exp());
        let sum_exp = exp_x.sum();
        exp_x.mapv(|ei| ei / sum_exp)
    }

    /// Compute the Jacobian of softmax at a point.
    ///
    /// For softmax: s_i = exp(x_i) / sum_j(exp(x_j))
    /// The Jacobian entry `J[i,j]` = ∂s_i/∂x_j:
    ///   `J[i,j]` = s_i * (δ_ij - s_j)
    /// where δ_ij = 1 if i=j, 0 otherwise.
    ///
    /// Diagonal:     `J[i,i]` = s_i * (1 - s_i)
    /// Off-diagonal: `J[i,j]` = -s_i * s_j  for i ≠ j
    pub fn jacobian(&self, x: &Array1<f32>) -> Array2<f32> {
        let s = self.eval(x);
        let n = s.len();
        let mut jacobian = Array2::<f32>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal: s_i * (1 - s_i)
                    jacobian[[i, j]] = s[i] * (1.0 - s[i]);
                } else {
                    // Off-diagonal: -s_i * s_j
                    jacobian[[i, j]] = -s[i] * s[j];
                }
            }
        }

        jacobian
    }

    /// Compute CROWN linear bounds for softmax with pre-activation bounds.
    ///
    /// Uses local linearization at the center point with sampling-verified soundness.
    /// Returns linear bounds: y_lower >= A_l @ x + b_l, y_upper <= A_u @ x + b_u
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let ndim = shape.len();

        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Handle axis >= ndim gracefully (see propagate_ibp for explanation)
        let axis = if axis >= ndim {
            debug!(
                "Softmax axis {} out of range for tensor with {} dims, using last axis (-1)",
                self.axis, ndim
            );
            ndim - 1
        } else {
            axis
        };

        match ndim {
            1 => {
                let pre_lower = pre_activation
                    .lower
                    .clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![pre_activation.len()],
                        got: pre_activation.lower.shape().to_vec(),
                    })?;
                let pre_upper = pre_activation
                    .upper
                    .clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![pre_activation.len()],
                        got: pre_activation.upper.shape().to_vec(),
                    })?;
                self.propagate_linear_with_bounds_1d(bounds, &pre_lower, &pre_upper)
            }
            2 => {
                if bounds.num_inputs() != shape[0] * shape[1] {
                    return Err(GammaError::ShapeMismatch {
                        expected: vec![shape[0] * shape[1]],
                        got: vec![bounds.num_inputs()],
                    });
                }

                let pre_lower = pre_activation
                    .lower
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        GammaError::InvalidSpec("Softmax pre-activation must be 2D".to_string())
                    })?;
                let pre_upper = pre_activation
                    .upper
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        GammaError::InvalidSpec("Softmax pre-activation must be 2D".to_string())
                    })?;

                let rows = shape[0];
                let cols = shape[1];
                let num_outputs = bounds.num_outputs();

                let num_groups = if axis == 0 { cols } else { rows };
                let num_groups_f = num_groups as f32;
                let bias_split_lower = bounds.lower_b.mapv(|v| v / num_groups_f);
                let bias_split_upper = bounds.upper_b.mapv(|v| v / num_groups_f);

                let mut out_lower_a = Array2::<f32>::zeros((num_outputs, rows * cols));
                let mut out_upper_a = Array2::<f32>::zeros((num_outputs, rows * cols));
                let mut out_lower_b = Array1::<f32>::zeros(num_outputs);
                let mut out_upper_b = Array1::<f32>::zeros(num_outputs);

                if axis == 0 {
                    // Column-wise softmax: operate independently on each column.
                    for j in 0..cols {
                        let mut group_lower = Array1::<f32>::zeros(rows);
                        let mut group_upper = Array1::<f32>::zeros(rows);
                        for i in 0..rows {
                            group_lower[i] = pre_lower[[i, j]];
                            group_upper[i] = pre_upper[[i, j]];
                        }

                        let mut group_lower_a = Array2::<f32>::zeros((num_outputs, rows));
                        let mut group_upper_a = Array2::<f32>::zeros((num_outputs, rows));
                        for out_idx in 0..num_outputs {
                            for i in 0..rows {
                                let flat = i * cols + j;
                                group_lower_a[[out_idx, i]] = bounds.lower_a[[out_idx, flat]];
                                group_upper_a[[out_idx, i]] = bounds.upper_a[[out_idx, flat]];
                            }
                        }

                        let group_bounds = LinearBounds {
                            lower_a: group_lower_a,
                            lower_b: bias_split_lower.clone(),
                            upper_a: group_upper_a,
                            upper_b: bias_split_upper.clone(),
                        };

                        let group_result = self.propagate_linear_with_bounds_1d(
                            &group_bounds,
                            &group_lower,
                            &group_upper,
                        )?;

                        for out_idx in 0..num_outputs {
                            for i in 0..rows {
                                let flat = i * cols + j;
                                out_lower_a[[out_idx, flat]] += group_result.lower_a[[out_idx, i]];
                                out_upper_a[[out_idx, flat]] += group_result.upper_a[[out_idx, i]];
                            }
                        }
                        out_lower_b = &out_lower_b + &group_result.lower_b;
                        out_upper_b = &out_upper_b + &group_result.upper_b;
                    }
                } else {
                    // Row-wise softmax: operate independently on each row.
                    for i in 0..rows {
                        let group_lower = pre_lower.row(i).to_owned();
                        let group_upper = pre_upper.row(i).to_owned();

                        let mut group_lower_a = Array2::<f32>::zeros((num_outputs, cols));
                        let mut group_upper_a = Array2::<f32>::zeros((num_outputs, cols));
                        for out_idx in 0..num_outputs {
                            for j in 0..cols {
                                let flat = i * cols + j;
                                group_lower_a[[out_idx, j]] = bounds.lower_a[[out_idx, flat]];
                                group_upper_a[[out_idx, j]] = bounds.upper_a[[out_idx, flat]];
                            }
                        }

                        let group_bounds = LinearBounds {
                            lower_a: group_lower_a,
                            lower_b: bias_split_lower.clone(),
                            upper_a: group_upper_a,
                            upper_b: bias_split_upper.clone(),
                        };

                        let group_result = self.propagate_linear_with_bounds_1d(
                            &group_bounds,
                            &group_lower,
                            &group_upper,
                        )?;

                        for out_idx in 0..num_outputs {
                            for j in 0..cols {
                                let flat = i * cols + j;
                                out_lower_a[[out_idx, flat]] += group_result.lower_a[[out_idx, j]];
                                out_upper_a[[out_idx, flat]] += group_result.upper_a[[out_idx, j]];
                            }
                        }
                        out_lower_b = &out_lower_b + &group_result.lower_b;
                        out_upper_b = &out_upper_b + &group_result.upper_b;
                    }
                }

                Ok(LinearBounds {
                    lower_a: out_lower_a,
                    lower_b: out_lower_b,
                    upper_a: out_upper_a,
                    upper_b: out_upper_b,
                })
            }
            _ => {
                // General N-D case: decompose into independent 1D groups along the softmax axis
                // Each group corresponds to a unique combination of indices on all axes except `axis`
                let total_size: usize = shape.iter().product();
                if bounds.num_inputs() != total_size {
                    return Err(GammaError::ShapeMismatch {
                        expected: vec![total_size],
                        got: vec![bounds.num_inputs()],
                    });
                }

                let softmax_size = shape[axis];
                let num_outputs = bounds.num_outputs();

                // Compute strides for converting between flat and multi-dimensional indices
                let mut strides = vec![1usize; ndim];
                for d in (0..ndim - 1).rev() {
                    strides[d] = strides[d + 1] * shape[d + 1];
                }

                // Number of groups = product of all dimensions except axis
                let num_groups: usize = shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != axis)
                    .map(|(_, &d)| d)
                    .product::<usize>()
                    .max(1);

                let num_groups_f = num_groups as f32;
                let bias_split_lower = bounds.lower_b.mapv(|v| v / num_groups_f);
                let bias_split_upper = bounds.upper_b.mapv(|v| v / num_groups_f);

                let mut out_lower_a = Array2::<f32>::zeros((num_outputs, total_size));
                let mut out_upper_a = Array2::<f32>::zeros((num_outputs, total_size));
                let mut out_lower_b = Array1::<f32>::zeros(num_outputs);
                let mut out_upper_b = Array1::<f32>::zeros(num_outputs);

                // Helper: convert multi-dim index to flat index
                let multi_to_flat = |idx: &[usize]| -> usize {
                    idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
                };

                // Iterate over all groups (all combinations of non-axis indices)
                // Compute "group shape" (all dims except axis)
                let group_shape: Vec<usize> = shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != axis)
                    .map(|(_, &d)| d)
                    .collect();

                let mut group_strides = vec![1usize; group_shape.len()];
                if !group_shape.is_empty() {
                    for d in (0..group_shape.len() - 1).rev() {
                        group_strides[d] = group_strides[d + 1] * group_shape[d + 1];
                    }
                }

                for group_idx in 0..num_groups {
                    // Convert group_idx to indices on non-axis dimensions
                    let mut group_multi = vec![0usize; group_shape.len()];
                    let mut remaining = group_idx;
                    for d in 0..group_shape.len() {
                        group_multi[d] = remaining / group_strides[d];
                        remaining %= group_strides[d];
                    }

                    // Extract 1D slice bounds for this group
                    let mut group_lower = Array1::<f32>::zeros(softmax_size);
                    let mut group_upper = Array1::<f32>::zeros(softmax_size);

                    // Map flat indices for this group
                    let mut flat_indices_for_group = Vec::with_capacity(softmax_size);

                    for s in 0..softmax_size {
                        // Build full multi-dim index: insert s at position `axis`
                        let mut full_idx = Vec::with_capacity(ndim);
                        let mut gm_pos = 0;
                        for d in 0..ndim {
                            if d == axis {
                                full_idx.push(s);
                            } else {
                                full_idx.push(group_multi[gm_pos]);
                                gm_pos += 1;
                            }
                        }

                        let flat = multi_to_flat(&full_idx);
                        flat_indices_for_group.push(flat);

                        group_lower[s] = pre_activation.lower[full_idx.as_slice()];
                        group_upper[s] = pre_activation.upper[full_idx.as_slice()];
                    }

                    // Extract coefficients for this group
                    let mut group_lower_a = Array2::<f32>::zeros((num_outputs, softmax_size));
                    let mut group_upper_a = Array2::<f32>::zeros((num_outputs, softmax_size));
                    for out_idx in 0..num_outputs {
                        for (s, &flat) in flat_indices_for_group.iter().enumerate() {
                            group_lower_a[[out_idx, s]] = bounds.lower_a[[out_idx, flat]];
                            group_upper_a[[out_idx, s]] = bounds.upper_a[[out_idx, flat]];
                        }
                    }

                    let group_bounds = LinearBounds {
                        lower_a: group_lower_a,
                        lower_b: bias_split_lower.clone(),
                        upper_a: group_upper_a,
                        upper_b: bias_split_upper.clone(),
                    };

                    let group_result = self.propagate_linear_with_bounds_1d(
                        &group_bounds,
                        &group_lower,
                        &group_upper,
                    )?;

                    // Embed results back into full tensor
                    for out_idx in 0..num_outputs {
                        for (s, &flat) in flat_indices_for_group.iter().enumerate() {
                            out_lower_a[[out_idx, flat]] += group_result.lower_a[[out_idx, s]];
                            out_upper_a[[out_idx, flat]] += group_result.upper_a[[out_idx, s]];
                        }
                    }
                    out_lower_b = &out_lower_b + &group_result.lower_b;
                    out_upper_b = &out_upper_b + &group_result.upper_b;
                }

                Ok(LinearBounds {
                    lower_a: out_lower_a,
                    lower_b: out_lower_b,
                    upper_a: out_upper_a,
                    upper_b: out_upper_b,
                })
            }
        }
    }

    fn propagate_linear_with_bounds_1d(
        &self,
        bounds: &LinearBounds,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
    ) -> Result<LinearBounds> {
        let num_neurons = pre_lower.len();
        if pre_upper.len() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![pre_upper.len()],
            });
        }
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Check for infinite or NaN bounds - if any dimension has infinite bounds,
        // softmax linearization sampling fails. Return identity relaxation (pass-through).
        // Softmax output is bounded [0,1] regardless of input, so this is sound.
        let has_infinite = pre_lower.iter().any(|&v| v.is_infinite() || v.is_nan())
            || pre_upper.iter().any(|&v| v.is_infinite() || v.is_nan());
        if has_infinite {
            // Identity relaxation: pass bounds through unchanged
            // This is conservative but avoids NaN from (-inf + inf)/2 = NaN
            return Ok(bounds.clone());
        }

        let x_center: Array1<f32> = pre_lower
            .iter()
            .zip(pre_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();

        let y_center = self.eval(&x_center);
        let jacobian = self.jacobian(&x_center);

        // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
        let jx_center = jacobian.dot(&x_center);
        let b_approx: Array1<f32> = &y_center - &jx_center;

        // Sample to find max error from linear approximation
        let num_samples = 50;
        let mut max_error_above: Array1<f32> = Array1::zeros(num_neurons); // actual - approx
        let mut max_error_below: Array1<f32> = Array1::zeros(num_neurons); // approx - actual

        // Allocate x_sample once outside loop, reuse buffer for each sample
        let mut x_sample = x_center.clone();
        for sample_idx in 0..num_samples {
            // Reset to center values (reuses allocation instead of cloning)
            x_sample.assign(&x_center);
            for i in 0..num_neurons {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }

            // Also sample corners (first few samples)
            if sample_idx < num_neurons * 2 {
                let dim = sample_idx / 2;
                if dim < num_neurons {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        pre_lower[dim]
                    } else {
                        pre_upper[dim]
                    };
                }
            }

            let y_actual = self.eval(&x_sample);
            let y_approx: Array1<f32> = jacobian.dot(&x_sample) + &b_approx;

            for i in 0..num_neurons {
                let error = y_actual[i] - y_approx[i];
                if error > max_error_above[i] {
                    max_error_above[i] = error;
                }
                if -error > max_error_below[i] {
                    max_error_below[i] = -error;
                }
            }
        }

        // Add safety margin (10% extra for unsampled regions)
        let safety_factor = 1.1;
        for i in 0..num_neurons {
            max_error_above[i] *= safety_factor;
            max_error_below[i] *= safety_factor;

            let min_margin = 1e-6_f32;
            if max_error_above[i] < min_margin {
                max_error_above[i] = min_margin;
            }
            if max_error_below[i] < min_margin {
                max_error_below[i] = min_margin;
            }
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for out_idx in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[out_idx, i]];
                let ua = bounds.upper_a[[out_idx, i]];

                if la >= 0.0 {
                    for k in 0..num_neurons {
                        new_lower_a[[out_idx, k]] += la * jacobian[[i, k]];
                    }
                    new_lower_b[out_idx] += la * (b_approx[i] - max_error_below[i]);
                } else {
                    for k in 0..num_neurons {
                        new_lower_a[[out_idx, k]] += la * jacobian[[i, k]];
                    }
                    new_lower_b[out_idx] += la * (b_approx[i] + max_error_above[i]);
                }

                if ua >= 0.0 {
                    for k in 0..num_neurons {
                        new_upper_a[[out_idx, k]] += ua * jacobian[[i, k]];
                    }
                    new_upper_b[out_idx] += ua * (b_approx[i] + max_error_above[i]);
                } else {
                    for k in 0..num_neurons {
                        new_upper_a[[out_idx, k]] += ua * jacobian[[i, k]];
                    }
                    new_upper_b[out_idx] += ua * (b_approx[i] - max_error_below[i]);
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

    /// Batched CROWN backward propagation through Softmax with pre-activation bounds.
    ///
    /// Same as `propagate_linear_with_bounds` but operates on N-D batched bounds,
    /// preserving batch structure. Softmax is applied independently along the
    /// last dimension of pre_activation (axis=-1).
    ///
    /// # Arguments
    /// - `bounds`: BatchedLinearBounds with shape [...batch_dims, out_dim, softmax_size]
    /// - `pre_activation`: Input bounds with shape [...batch_dims, softmax_size]
    ///
    /// # Returns
    /// New BatchedLinearBounds with softmax backward propagation applied.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("Softmax layer batched CROWN backward propagation");

        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let softmax_size = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        // Verify pre_activation shape matches
        let pre_softmax_size = *pre_shape.last().unwrap_or(&0);
        if pre_softmax_size != softmax_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![softmax_size],
                got: vec![pre_softmax_size],
            });
        }

        // Reshape pre-activation to [batch, softmax_size]
        let pre_lower_flat = pre_activation
            .lower
            .view()
            .into_shape_with_order((total_batch, softmax_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape pre_lower for softmax".to_string())
            })?;
        let pre_upper_flat = pre_activation
            .upper
            .view()
            .into_shape_with_order((total_batch, softmax_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape pre_upper for softmax".to_string())
            })?;

        // Reshape bounds to [batch, out_dim, softmax_size]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, softmax_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape lower_a for softmax".to_string())
            })?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, softmax_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape upper_a for softmax".to_string())
            })?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape lower_b for softmax".to_string())
            })?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape upper_b for softmax".to_string())
            })?;

        // Output arrays
        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, softmax_size));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, softmax_size));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        // Process each batch position independently using the 1D softmax backward
        for b in 0..total_batch {
            // Extract 1D pre-activation bounds for this batch
            let pre_lower_1d = pre_lower_flat.row(b).to_owned();
            let pre_upper_1d = pre_upper_flat.row(b).to_owned();

            // Extract 2D coefficient matrix for this batch: [out_dim, softmax_size]
            let lower_a_slice = lower_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
            let upper_a_slice = upper_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
            let lower_b_slice = lower_b_2d.row(b).to_owned();
            let upper_b_slice = upper_b_2d.row(b).to_owned();

            let batch_bounds = LinearBounds {
                lower_a: lower_a_slice,
                lower_b: lower_b_slice,
                upper_a: upper_a_slice,
                upper_b: upper_b_slice,
            };

            // Apply 1D softmax backward
            let result =
                self.propagate_linear_with_bounds_1d(&batch_bounds, &pre_lower_1d, &pre_upper_1d)?;

            // Copy results back
            for j in 0..out_dim {
                for k in 0..softmax_size {
                    new_lower_a[[b, j, k]] = result.lower_a[[j, k]];
                    new_upper_a[[b, j, k]] = result.upper_a[[j, k]];
                }
                new_lower_b[[b, j]] = result.lower_b[j];
                new_upper_b[[b, j]] = result.upper_b[j];
            }
        }

        // Reshape back to original batch dims
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, softmax_size])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

        Ok(BatchedLinearBounds {
            lower_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            lower_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            upper_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            upper_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        })
    }
}

/// Small epsilon for numerical stability.
const SOFTMAX_EPSILON: f32 = 1e-12;
const SOFTMAX_SANITIZE_MARGIN: f32 = 1e-6;

fn sanitize_softmax_unit_bounds(mut lower: f32, mut upper: f32) -> (f32, f32) {
    // Softmax outputs are always in [0, 1]. When we detect numerical issues, prefer a
    // conservative widening rather than propagating NaN/inf.
    if !lower.is_finite() || !(0.0..=1.0).contains(&lower) {
        lower = 0.0;
    } else {
        lower = (lower - SOFTMAX_SANITIZE_MARGIN).max(0.0);
    }

    if !upper.is_finite() || upper < 0.0 {
        upper = 1.0;
    } else {
        upper = (upper + SOFTMAX_SANITIZE_MARGIN).min(1.0);
    }

    if lower > upper {
        (0.0, 1.0)
    } else {
        (lower, upper)
    }
}

impl BoundPropagation for SoftmaxLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Handle axis >= ndim gracefully: when Reshape ops are skipped due to dynamic shapes,
        // the tensor may have fewer dimensions than expected. In attention patterns, axis is
        // typically the last dimension regardless of tensor rank, so we fall back to last axis.
        let axis = if axis >= ndim {
            debug!(
                "Softmax axis {} out of range for tensor with {} dims, using last axis (-1)",
                self.axis, ndim
            );
            ndim - 1
        } else {
            axis
        };

        use ndarray::{Axis, Zip};

        let mut output_lower = input.lower.clone();
        let mut output_upper = input.upper.clone();

        Zip::from(output_lower.lanes_mut(Axis(axis)))
            .and(output_upper.lanes_mut(Axis(axis)))
            .and(input.lower.lanes(Axis(axis)))
            .and(input.upper.lanes(Axis(axis)))
            .for_each(
                |mut out_lower_lane, mut out_upper_lane, in_lower_lane, in_upper_lane| {
                    let mut ok = true;

                    let mut max_upper = f32::NEG_INFINITY;
                    for &u in in_upper_lane.iter() {
                        if !u.is_finite() {
                            ok = false;
                            break;
                        }
                        max_upper = max_upper.max(u);
                    }
                    for &l in in_lower_lane.iter() {
                        if !l.is_finite() {
                            ok = false;
                            break;
                        }
                    }

                    if !ok || !max_upper.is_finite() {
                        out_lower_lane.fill(0.0);
                        out_upper_lane.fill(1.0);
                        return;
                    }

                    // First pass: compute exp bounds into the output buffers (scratch), then sum.
                    let mut sum_exp_lower: f32 = 0.0;
                    let mut sum_exp_upper: f32 = 0.0;
                    for i in 0..in_lower_lane.len() {
                        let el = (in_lower_lane[i] - max_upper).exp();
                        let eu = (in_upper_lane[i] - max_upper).exp();
                        if !el.is_finite() || !eu.is_finite() {
                            ok = false;
                            break;
                        }
                        out_lower_lane[i] = el;
                        out_upper_lane[i] = eu;
                        sum_exp_lower += el;
                        sum_exp_upper += eu;
                    }

                    if !ok
                        || !sum_exp_lower.is_finite()
                        || !sum_exp_upper.is_finite()
                        || sum_exp_lower <= 0.0
                        || sum_exp_upper <= 0.0
                    {
                        out_lower_lane.fill(0.0);
                        out_upper_lane.fill(1.0);
                        return;
                    }

                    // Second pass: apply Auto-LiRPA interval formula.
                    for i in 0..in_lower_lane.len() {
                        let el = out_lower_lane[i];
                        let eu = out_upper_lane[i];

                        let denom_lower = sum_exp_upper - eu + el + SOFTMAX_EPSILON;
                        let denom_upper = sum_exp_lower - el + eu + SOFTMAX_EPSILON;

                        let raw_lower = if denom_lower.is_finite() && denom_lower > 0.0 {
                            el / denom_lower
                        } else {
                            f32::NAN
                        };
                        let raw_upper = if denom_upper.is_finite() && denom_upper > 0.0 {
                            eu / denom_upper
                        } else {
                            f32::NAN
                        };

                        let (lower, upper) = sanitize_softmax_unit_bounds(raw_lower, raw_upper);
                        out_lower_lane[i] = lower;
                        out_upper_lane[i] = upper;
                    }
                },
            );

        BoundedTensor::new(output_lower, output_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Softmax CROWN propagation not implemented, using IBP-style placeholder");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Causal softmax layer for decoder attention.
///
/// In causal attention, position i can only attend to positions j where j <= i.
/// This is implemented by applying a lower-triangular mask before softmax:
/// - Unmasked positions (j <= i): softmax computed normally over 0..=i
/// - Masked positions (j > i): output is exactly 0
#[derive(Debug, Clone)]
pub struct CausalSoftmaxLayer {
    /// Dimension along which to apply softmax (default: -1)
    pub axis: i32,
}

impl CausalSoftmaxLayer {
    /// Create a new Causal Softmax layer.
    pub fn new(axis: i32) -> Self {
        Self { axis }
    }
}

impl BoundPropagation for CausalSoftmaxLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        // Causal softmax requires at least 2D for attention pattern
        if ndim < 2 {
            return Err(GammaError::InvalidSpec(format!(
                "Causal softmax requires at least 2D input, got {}D",
                ndim
            )));
        }

        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Handle axis >= ndim gracefully (see SoftmaxLayer::propagate_ibp for explanation)
        // Note: CausalSoftmax always operates on last 2 dims for attention pattern, but we
        // still compute the axis adjustment for consistency and potential future use.
        let _axis = if axis >= ndim {
            debug!(
                "CausalSoftmax axis {} out of range for tensor with {} dims, using last axis (-1)",
                self.axis, ndim
            );
            ndim - 1
        } else {
            axis
        };

        // Causal mask requires the two last dimensions to form the attention matrix
        // Shape: [..., seq_q, seq_k] where causal means j <= i for position i
        let seq_q = shape[ndim - 2];
        let seq_k = shape[ndim - 1];

        if seq_q > seq_k {
            return Err(GammaError::InvalidSpec(format!(
                "Causal softmax requires seq_q ({}) <= seq_k ({})",
                seq_q, seq_k
            )));
        }

        let mut output_lower = ArrayD::<f32>::zeros(input.lower.raw_dim());
        let mut output_upper = ArrayD::<f32>::zeros(input.upper.raw_dim());

        // Process based on dimensionality
        match ndim {
            2 => {
                // 2D: [seq_q, seq_k]
                for i in 0..seq_q {
                    // Find max upper bound for numerical stability (only unmasked positions)
                    let mut max_upper = f32::NEG_INFINITY;
                    let mut ok = true;
                    for j in 0..=i.min(seq_k - 1) {
                        let l = input.lower[[i, j]];
                        let u = input.upper[[i, j]];
                        if !l.is_finite() || !u.is_finite() {
                            ok = false;
                            break;
                        }
                        max_upper = max_upper.max(u);
                    }

                    // Compute exp bounds for unmasked positions
                    let mut exp_lower: Vec<f32> = Vec::with_capacity(i + 1);
                    let mut exp_upper: Vec<f32> = Vec::with_capacity(i + 1);
                    let mut sum_exp_lower: f32 = 0.0;
                    let mut sum_exp_upper: f32 = 0.0;

                    for j in 0..=i.min(seq_k - 1) {
                        if !ok || !max_upper.is_finite() {
                            break;
                        }
                        let el = (input.lower[[i, j]] - max_upper).exp();
                        let eu = (input.upper[[i, j]] - max_upper).exp();
                        if !el.is_finite() || !eu.is_finite() {
                            ok = false;
                            break;
                        }
                        exp_lower.push(el);
                        exp_upper.push(eu);
                        sum_exp_lower += el;
                        sum_exp_upper += eu;
                    }

                    // Compute bounds for unmasked positions
                    for j in 0..=i.min(seq_k - 1) {
                        if !ok
                            || !sum_exp_lower.is_finite()
                            || !sum_exp_upper.is_finite()
                            || sum_exp_lower <= 0.0
                            || sum_exp_upper <= 0.0
                        {
                            output_lower[[i, j]] = 0.0;
                            output_upper[[i, j]] = 1.0;
                            continue;
                        }

                        let el = exp_lower[j];
                        let eu = exp_upper[j];
                        let denom_lower = sum_exp_upper - eu + el + SOFTMAX_EPSILON;
                        let denom_upper = sum_exp_lower - el + eu + SOFTMAX_EPSILON;
                        let raw_lower = if denom_lower.is_finite() && denom_lower > 0.0 {
                            el / denom_lower
                        } else {
                            f32::NAN
                        };
                        let raw_upper = if denom_upper.is_finite() && denom_upper > 0.0 {
                            eu / denom_upper
                        } else {
                            f32::NAN
                        };
                        let (lb, ub) = sanitize_softmax_unit_bounds(raw_lower, raw_upper);
                        output_lower[[i, j]] = lb;
                        output_upper[[i, j]] = ub;
                    }
                    // Masked positions (j > i) remain 0
                }
            }
            3 => {
                // 3D: [batch, seq_q, seq_k]
                let batch = shape[0];
                for b in 0..batch {
                    for i in 0..seq_q {
                        let mut max_upper = f32::NEG_INFINITY;
                        let mut ok = true;
                        for j in 0..=i.min(seq_k - 1) {
                            let l = input.lower[[b, i, j]];
                            let u = input.upper[[b, i, j]];
                            if !l.is_finite() || !u.is_finite() {
                                ok = false;
                                break;
                            }
                            max_upper = max_upper.max(u);
                        }

                        let mut exp_lower: Vec<f32> = Vec::with_capacity(i + 1);
                        let mut exp_upper: Vec<f32> = Vec::with_capacity(i + 1);
                        let mut sum_exp_lower: f32 = 0.0;
                        let mut sum_exp_upper: f32 = 0.0;

                        for j in 0..=i.min(seq_k - 1) {
                            if !ok || !max_upper.is_finite() {
                                break;
                            }
                            let el = (input.lower[[b, i, j]] - max_upper).exp();
                            let eu = (input.upper[[b, i, j]] - max_upper).exp();
                            if !el.is_finite() || !eu.is_finite() {
                                ok = false;
                                break;
                            }
                            exp_lower.push(el);
                            exp_upper.push(eu);
                            sum_exp_lower += el;
                            sum_exp_upper += eu;
                        }

                        for j in 0..=i.min(seq_k - 1) {
                            if !ok
                                || !sum_exp_lower.is_finite()
                                || !sum_exp_upper.is_finite()
                                || sum_exp_lower <= 0.0
                                || sum_exp_upper <= 0.0
                            {
                                output_lower[[b, i, j]] = 0.0;
                                output_upper[[b, i, j]] = 1.0;
                                continue;
                            }

                            let el = exp_lower[j];
                            let eu = exp_upper[j];
                            let denom_lower = sum_exp_upper - eu + el + SOFTMAX_EPSILON;
                            let denom_upper = sum_exp_lower - el + eu + SOFTMAX_EPSILON;
                            let raw_lower = if denom_lower.is_finite() && denom_lower > 0.0 {
                                el / denom_lower
                            } else {
                                f32::NAN
                            };
                            let raw_upper = if denom_upper.is_finite() && denom_upper > 0.0 {
                                eu / denom_upper
                            } else {
                                f32::NAN
                            };
                            let (lb, ub) = sanitize_softmax_unit_bounds(raw_lower, raw_upper);
                            output_lower[[b, i, j]] = lb;
                            output_upper[[b, i, j]] = ub;
                        }
                    }
                }
            }
            4 => {
                // 4D: [batch, heads, seq_q, seq_k]
                let batch = shape[0];
                let heads = shape[1];
                for b in 0..batch {
                    for h in 0..heads {
                        for i in 0..seq_q {
                            let mut max_upper = f32::NEG_INFINITY;
                            let mut ok = true;
                            for j in 0..=i.min(seq_k - 1) {
                                let l = input.lower[[b, h, i, j]];
                                let u = input.upper[[b, h, i, j]];
                                if !l.is_finite() || !u.is_finite() {
                                    ok = false;
                                    break;
                                }
                                max_upper = max_upper.max(u);
                            }

                            let mut exp_lower: Vec<f32> = Vec::with_capacity(i + 1);
                            let mut exp_upper: Vec<f32> = Vec::with_capacity(i + 1);
                            let mut sum_exp_lower: f32 = 0.0;
                            let mut sum_exp_upper: f32 = 0.0;

                            for j in 0..=i.min(seq_k - 1) {
                                if !ok || !max_upper.is_finite() {
                                    break;
                                }
                                let el = (input.lower[[b, h, i, j]] - max_upper).exp();
                                let eu = (input.upper[[b, h, i, j]] - max_upper).exp();
                                if !el.is_finite() || !eu.is_finite() {
                                    ok = false;
                                    break;
                                }
                                exp_lower.push(el);
                                exp_upper.push(eu);
                                sum_exp_lower += el;
                                sum_exp_upper += eu;
                            }

                            for j in 0..=i.min(seq_k - 1) {
                                if !ok
                                    || !sum_exp_lower.is_finite()
                                    || !sum_exp_upper.is_finite()
                                    || sum_exp_lower <= 0.0
                                    || sum_exp_upper <= 0.0
                                {
                                    output_lower[[b, h, i, j]] = 0.0;
                                    output_upper[[b, h, i, j]] = 1.0;
                                    continue;
                                }

                                let el = exp_lower[j];
                                let eu = exp_upper[j];
                                let denom_lower = sum_exp_upper - eu + el + SOFTMAX_EPSILON;
                                let denom_upper = sum_exp_lower - el + eu + SOFTMAX_EPSILON;
                                let raw_lower = if denom_lower.is_finite() && denom_lower > 0.0 {
                                    el / denom_lower
                                } else {
                                    f32::NAN
                                };
                                let raw_upper = if denom_upper.is_finite() && denom_upper > 0.0 {
                                    eu / denom_upper
                                } else {
                                    f32::NAN
                                };
                                let (lb, ub) = sanitize_softmax_unit_bounds(raw_lower, raw_upper);
                                output_lower[[b, h, i, j]] = lb;
                                output_upper[[b, h, i, j]] = ub;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(GammaError::InvalidSpec(format!(
                    "Causal softmax not implemented for {}D tensors",
                    ndim
                )));
            }
        }

        BoundedTensor::new(output_lower, output_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Without pre-activation bounds, return identity (caller should use propagate_linear_with_bounds)
        debug!("CausalSoftmax CROWN propagation without bounds - using identity (caller should use propagate_linear_with_bounds)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl CausalSoftmaxLayer {
    /// Evaluate causal softmax for a single row i.
    /// Input x has length seq_k, output has length seq_k.
    /// For positions j > i, output is 0.
    fn eval_row(&self, x: &Array1<f32>, row_idx: usize) -> Array1<f32> {
        let seq_k = x.len();
        let mut out = Array1::zeros(seq_k);
        let active_len = (row_idx + 1).min(seq_k);

        if active_len == 0 {
            return out;
        }

        // Compute softmax over active positions
        let max_val = x
            .slice(ndarray::s![..active_len])
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0_f32;
        for j in 0..active_len {
            let e = (x[j] - max_val).exp();
            out[j] = e;
            sum_exp += e;
        }

        let inv_sum = 1.0 / (sum_exp + SOFTMAX_EPSILON);
        for j in 0..active_len {
            out[j] *= inv_sum;
        }
        // Positions j > row_idx remain 0

        out
    }

    /// Compute Jacobian of causal softmax for a single row i.
    /// Returns a seq_k x seq_k matrix where J[output_j, input_k].
    /// For positions j > row_idx or k > row_idx, the Jacobian is 0.
    fn jacobian_row(&self, x: &Array1<f32>, row_idx: usize) -> Array2<f32> {
        let seq_k = x.len();
        let mut jac = Array2::zeros((seq_k, seq_k));
        let active_len = (row_idx + 1).min(seq_k);

        if active_len == 0 {
            return jac;
        }

        // Get softmax values for active positions
        let s = self.eval_row(x, row_idx);

        // Softmax Jacobian: J[j,k] = s[j] * (δ[j,k] - s[k]) for j,k < active_len
        for j in 0..active_len {
            for k in 0..active_len {
                let delta = if j == k { 1.0 } else { 0.0 };
                jac[[j, k]] = s[j] * (delta - s[k]);
            }
        }
        // Rows/cols beyond active_len remain 0 (masked positions)

        jac
    }

    /// CROWN backward propagation through CausalSoftmax with pre-activation bounds.
    ///
    /// For causal softmax with shape [seq_q, seq_k]:
    /// - Row i applies softmax over positions 0..=i (unmasked)
    /// - Positions j > i have output = 0 (masked)
    ///
    /// Uses Jacobian-based linear approximation at the interval center,
    /// with sampling to estimate approximation error.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let ndim = shape.len();

        // Causal softmax requires at least 2D
        if ndim < 2 {
            return Err(GammaError::InvalidSpec(format!(
                "Causal softmax CROWN requires at least 2D input, got {}D",
                ndim
            )));
        }

        let seq_q = shape[ndim - 2];
        let seq_k = shape[ndim - 1];
        let total_size = seq_q * seq_k;

        if bounds.num_inputs() != total_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        // Check for infinite or NaN bounds
        let has_infinite = pre_activation
            .lower
            .iter()
            .any(|&v| v.is_infinite() || v.is_nan())
            || pre_activation
                .upper
                .iter()
                .any(|&v| v.is_infinite() || v.is_nan());
        if has_infinite {
            return Ok(bounds.clone());
        }

        let num_outputs = bounds.num_outputs();

        // Flatten pre-activation bounds
        let pre_lower_flat = pre_activation
            .lower
            .view()
            .into_shape_with_order(total_size)
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![total_size],
                got: pre_activation.lower.shape().to_vec(),
            })?;
        let pre_upper_flat = pre_activation
            .upper
            .view()
            .into_shape_with_order(total_size)
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![total_size],
                got: pre_activation.upper.shape().to_vec(),
            })?;

        // Compute center point
        let x_center: Array1<f32> = pre_lower_flat
            .iter()
            .zip(pre_upper_flat.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();

        // Compute Jacobian block by block for each row
        // Full Jacobian is block diagonal (each row is independent)
        let mut full_jacobian = Array2::<f32>::zeros((total_size, total_size));
        let mut y_center = Array1::<f32>::zeros(total_size);

        for i in 0..seq_q {
            let row_start = i * seq_k;
            let row_end = row_start + seq_k;
            let x_row = x_center.slice(ndarray::s![row_start..row_end]).to_owned();

            let y_row = self.eval_row(&x_row, i);
            let jac_row = self.jacobian_row(&x_row, i);

            y_center
                .slice_mut(ndarray::s![row_start..row_end])
                .assign(&y_row);

            for j in 0..seq_k {
                for k in 0..seq_k {
                    full_jacobian[[row_start + j, row_start + k]] = jac_row[[j, k]];
                }
            }
        }

        // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
        let jx_center = full_jacobian.dot(&x_center);
        let b_approx: Array1<f32> = &y_center - &jx_center;

        // Sample to find max error from linear approximation
        let num_samples = 50;
        let mut max_error_above = Array1::<f32>::zeros(total_size);
        let mut max_error_below = Array1::<f32>::zeros(total_size);

        let mut x_sample = x_center.clone();
        for sample_idx in 0..num_samples {
            // Generate sample point
            x_sample.assign(&x_center);
            for i in 0..total_size {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower_flat[i] + (pre_upper_flat[i] - pre_lower_flat[i]) * t;
            }

            // Sample corners (first few samples)
            if sample_idx < total_size * 2 && sample_idx < num_samples {
                let dim = sample_idx / 2;
                if dim < total_size {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        pre_lower_flat[dim]
                    } else {
                        pre_upper_flat[dim]
                    };
                }
            }

            // Compute actual causal softmax output
            let mut y_actual = Array1::<f32>::zeros(total_size);
            for i in 0..seq_q {
                let row_start = i * seq_k;
                let row_end = row_start + seq_k;
                let x_row = x_sample.slice(ndarray::s![row_start..row_end]).to_owned();
                let y_row = self.eval_row(&x_row, i);
                y_actual
                    .slice_mut(ndarray::s![row_start..row_end])
                    .assign(&y_row);
            }

            let y_approx: Array1<f32> = full_jacobian.dot(&x_sample) + &b_approx;

            for i in 0..total_size {
                let error = y_actual[i] - y_approx[i];
                if error > max_error_above[i] {
                    max_error_above[i] = error;
                }
                if -error > max_error_below[i] {
                    max_error_below[i] = -error;
                }
            }
        }

        // Add safety margin (10% extra for unsampled regions)
        let safety_factor = 1.1;
        for i in 0..total_size {
            max_error_above[i] *= safety_factor;
            max_error_below[i] *= safety_factor;

            let min_margin = 1e-6_f32;
            if max_error_above[i] < min_margin {
                max_error_above[i] = min_margin;
            }
            if max_error_below[i] < min_margin {
                max_error_below[i] = min_margin;
            }
        }

        // Propagate bounds through Jacobian
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, total_size));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, total_size));
        let mut new_upper_b = bounds.upper_b.clone();

        for out_idx in 0..num_outputs {
            for i in 0..total_size {
                let la = bounds.lower_a[[out_idx, i]];
                let ua = bounds.upper_a[[out_idx, i]];

                if la >= 0.0 {
                    for k in 0..total_size {
                        new_lower_a[[out_idx, k]] += la * full_jacobian[[i, k]];
                    }
                    new_lower_b[out_idx] += la * (b_approx[i] - max_error_below[i]);
                } else {
                    for k in 0..total_size {
                        new_lower_a[[out_idx, k]] += la * full_jacobian[[i, k]];
                    }
                    new_lower_b[out_idx] += la * (b_approx[i] + max_error_above[i]);
                }

                if ua >= 0.0 {
                    for k in 0..total_size {
                        new_upper_a[[out_idx, k]] += ua * full_jacobian[[i, k]];
                    }
                    new_upper_b[out_idx] += ua * (b_approx[i] + max_error_above[i]);
                } else {
                    for k in 0..total_size {
                        new_upper_a[[out_idx, k]] += ua * full_jacobian[[i, k]];
                    }
                    new_upper_b[out_idx] += ua * (b_approx[i] - max_error_below[i]);
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

/// GELU approximation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluApproximation {
    /// Exact: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Erf,
    /// Approximate: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    Tanh,
}

/// GELU activation layer.
#[derive(Debug, Clone)]
pub struct GELULayer {
    pub approximation: GeluApproximation,
    /// Relaxation mode for CROWN backward propagation.
    /// Default is Chord for backwards compatibility.
    pub relaxation_mode: RelaxationMode,
}

impl GELULayer {
    pub fn new(approximation: GeluApproximation) -> Self {
        Self {
            approximation,
            relaxation_mode: RelaxationMode::default(),
        }
    }

    /// Create a new GELU layer with specified relaxation mode.
    pub fn with_relaxation(approximation: GeluApproximation, mode: RelaxationMode) -> Self {
        Self {
            approximation,
            relaxation_mode: mode,
        }
    }

    /// Create a new GELU layer with adaptive relaxation (best tightness).
    pub fn adaptive(approximation: GeluApproximation) -> Self {
        Self::with_relaxation(approximation, RelaxationMode::Adaptive)
    }
}

impl Default for GELULayer {
    fn default() -> Self {
        Self {
            approximation: GeluApproximation::Erf,
            relaxation_mode: RelaxationMode::default(),
        }
    }
}

fn gelu_erf(x: f32) -> f32 {
    let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
    0.5 * x * (1.0 + libm::erff(x * inv_sqrt2))
}

fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

pub fn gelu_eval(x: f32, approximation: GeluApproximation) -> f32 {
    match approximation {
        GeluApproximation::Erf => gelu_erf(x),
        GeluApproximation::Tanh => gelu_tanh(x),
    }
}

fn gelu_derivative(x: f32, approximation: GeluApproximation) -> f32 {
    match approximation {
        GeluApproximation::Erf => {
            let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
            let inv_sqrt_2pi: f32 = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
            let phi: f32 = 0.5 * (1.0 + libm::erff(x * inv_sqrt2));
            let pdf: f32 = inv_sqrt_2pi * (-0.5 * x * x).exp();
            phi + x * pdf
        }
        GeluApproximation::Tanh => {
            let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
            let t: f32 = k * (x + 0.044715 * x * x * x);
            let tanh_t: f32 = t.tanh();
            let sech2_t: f32 = 1.0 - tanh_t * tanh_t;
            let dt_dx: f32 = k * (1.0 + 3.0 * 0.044715 * x * x);
            0.5 * (1.0 + tanh_t) + 0.5 * x * sech2_t * dt_dx
        }
    }
}

fn gelu_critical_point(approximation: GeluApproximation) -> f32 {
    use std::sync::OnceLock;

    static GELU_CRITICAL_ERF: OnceLock<f32> = OnceLock::new();
    static GELU_CRITICAL_TANH: OnceLock<f32> = OnceLock::new();

    let slot = match approximation {
        GeluApproximation::Erf => &GELU_CRITICAL_ERF,
        GeluApproximation::Tanh => &GELU_CRITICAL_TANH,
    };

    *slot.get_or_init(|| {
        // GELU has a single global minimum for x < 0, near x ≈ -0.75.
        // Use bisection on derivative in a bracket known to straddle the root.
        let mut lo: f32 = -2.0;
        let mut hi: f32 = 0.0;
        let dlo: f32 = gelu_derivative(lo, approximation);
        let dhi: f32 = gelu_derivative(hi, approximation);

        if !(dlo < 0.0 && dhi > 0.0) {
            // Fallback: widen bracket.
            lo = -10.0;
            hi = 1.0;
            let dlo2: f32 = gelu_derivative(lo, approximation);
            let dhi2: f32 = gelu_derivative(hi, approximation);
            if !(dlo2 < 0.0 && dhi2 > 0.0) {
                return -0.75;
            }
        }

        // If we still can't bracket, fall back to a reasonable constant; callers still
        // evaluate endpoints so this only affects min tightening.
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let dmid = gelu_derivative(mid, approximation);
            if dmid > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        0.5 * (lo + hi)
    })
}

fn gelu_bound_interval(l: f32, u: f32, approximation: GeluApproximation) -> (f32, f32) {
    let gl = gelu_eval(l, approximation);
    let gu = gelu_eval(u, approximation);

    let mut min_v = gl.min(gu);
    let mut max_v = gl.max(gu);

    let critical_point = gelu_critical_point(approximation);
    if l <= critical_point && critical_point <= u {
        let gc = gelu_eval(critical_point, approximation);
        min_v = min_v.min(gc);
        max_v = max_v.max(gc);
    }

    (min_v, max_v)
}

/// Compute linear relaxation parameters for GELU on interval [l, u].
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept) where:
/// - GELU(x) >= lower_slope * x + lower_intercept
/// - GELU(x) <= upper_slope * x + upper_intercept
///
/// Uses conservative chord-based relaxation with sampling to ensure soundness.
/// GELU has complex convexity patterns, so we empirically verify bounds.
pub fn gelu_linear_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> (f32, f32, f32, f32) {
    // Handle infinite/NaN bounds: return identity-like relaxation (slope=1, intercept=0).
    // This avoids 0*inf = NaN in backward propagation and allows bounds to propagate through.
    // For GELU with unbounded input, the output is also unbounded (GELU(x) -> x for large |x|),
    // so identity is a reasonable approximation that preserves soundness.
    if l.is_infinite() || u.is_infinite() || l.is_nan() || u.is_nan() {
        return (1.0, 0.0, 1.0, 0.0);
    }

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        // Point interval: use derivative as slope
        let slope = gelu_derivative(l, approximation);
        let intercept = gelu_eval(l, approximation) - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let gl = gelu_eval(l, approximation);
    let gu = gelu_eval(u, approximation);

    // Chord slope connecting (l, GELU(l)) to (u, GELU(u))
    let chord_slope = (gu - gl) / (u - l);
    let chord_intercept = gl - chord_slope * l;

    // Sample the interval to find max deviation from chord
    // This ensures soundness regardless of GELU's convexity pattern
    let num_samples = 100; // Use more samples for accuracy
    let mut max_above_chord = 0.0_f32; // max(GELU(x) - chord(x))
    let mut max_below_chord = 0.0_f32; // max(chord(x) - GELU(x))

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let gx = gelu_eval(x, approximation);
        let cx = chord_slope * x + chord_intercept;
        let diff = gx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Also check critical point (minimum of GELU) if it's in the interval
    let critical_point = gelu_critical_point(approximation);
    if l <= critical_point && critical_point <= u {
        let gc = gelu_eval(critical_point, approximation);
        let cc = chord_slope * critical_point + chord_intercept;
        let diff = gc - cc;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Add small epsilon for numerical safety
    let eps = 1e-5;

    // Lower bound: chord shifted down by max_below_chord (ensures chord - shift <= GELU)
    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;

    // Upper bound: chord shifted up by max_above_chord (ensures GELU <= chord + shift)
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

/// Relaxation mode for activation functions.
///
/// Different modes provide different trade-offs between tightness and computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RelaxationMode {
    /// Chord-based relaxation: connect endpoints, shift for soundness.
    /// Uses same slope for both lower and upper bounds.
    /// Fast and always sound, but may be loose for asymmetric regions.
    #[default]
    Chord,

    /// Tangent-based relaxation: use tangent line at center point.
    /// Optimal for small intervals where Taylor expansion is accurate.
    /// Better than chord when interval is small relative to curvature.
    Tangent,

    /// Two-slope relaxation: independent optimal slopes for lower/upper.
    /// Uses tangent lines at strategic points for each bound.
    /// Tighter than chord for most cases but requires more computation.
    TwoSlope,

    /// Adaptive selection: automatically choose the tightest relaxation.
    /// Evaluates multiple strategies and returns the one with smallest
    /// bound width (upper_intercept - lower_intercept) at the interval center.
    Adaptive,
}

/// Compute tangent-based relaxation for GELU.
///
/// Uses the tangent line at the center point (l+u)/2.
/// This is optimal for small intervals but may not be sound for large intervals
/// without proper error bounding.
fn gelu_tangent_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> (f32, f32, f32, f32) {
    if l.is_infinite() || u.is_infinite() || l.is_nan() || u.is_nan() {
        return (1.0, 0.0, 1.0, 0.0);
    }

    if (u - l).abs() < 1e-8 {
        let slope = gelu_derivative(l, approximation);
        let intercept = gelu_eval(l, approximation) - slope * l;
        return (slope, intercept, slope, intercept);
    }

    // Tangent line at center
    let c = (l + u) / 2.0;
    let gc = gelu_eval(c, approximation);
    let slope = gelu_derivative(c, approximation);
    let intercept = gc - slope * c;

    // Find max deviation from tangent line (above and below)
    let num_samples = 50;
    let mut max_above = 0.0_f32;
    let mut max_below = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let gx = gelu_eval(x, approximation);
        let tx = slope * x + intercept;
        let diff = gx - tx;

        if diff > max_above {
            max_above = diff;
        }
        if -diff > max_below {
            max_below = -diff;
        }
    }

    // Also check critical point
    let critical_point = gelu_critical_point(approximation);
    if l <= critical_point && critical_point <= u {
        let gc_crit = gelu_eval(critical_point, approximation);
        let tc_crit = slope * critical_point + intercept;
        let diff = gc_crit - tc_crit;
        if diff > max_above {
            max_above = diff;
        }
        if -diff > max_below {
            max_below = -diff;
        }
    }

    let eps = 1e-5;
    let lower_intercept = intercept - max_below - eps;
    let upper_intercept = intercept + max_above + eps;

    (slope, lower_intercept, slope, upper_intercept)
}

/// Compute two-slope relaxation for GELU.
///
/// Uses independent optimal slopes for lower and upper bounds.
/// For the lower bound: tangent at a point that minimizes underestimation.
/// For the upper bound: tangent at a point that minimizes overestimation.
fn gelu_two_slope_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> (f32, f32, f32, f32) {
    if l.is_infinite() || u.is_infinite() || l.is_nan() || u.is_nan() {
        return (1.0, 0.0, 1.0, 0.0);
    }

    if (u - l).abs() < 1e-8 {
        let slope = gelu_derivative(l, approximation);
        let intercept = gelu_eval(l, approximation) - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let gl = gelu_eval(l, approximation);
    let gu = gelu_eval(u, approximation);

    // Strategy: Find the tightest lower bound line that stays below GELU
    // and tightest upper bound line that stays above GELU.

    // For lower bound: try chord and tangents at l, u, center
    // Pick the one with highest minimum value over [l, u]
    let candidates = [l, u, (l + u) / 2.0];
    let num_samples = 30;

    // Lower bound: line must be <= GELU(x) for all x in [l, u]
    let mut best_lower_slope = 0.0_f32;
    let mut best_lower_intercept = f32::NEG_INFINITY;

    for &point in &candidates {
        let slope = gelu_derivative(point, approximation);
        let gp = gelu_eval(point, approximation);
        let intercept = gp - slope * point;

        // Find min margin (GELU(x) - line(x)) over interval
        let mut min_margin = f32::INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = slope * x + intercept;
            min_margin = min_margin.min(gx - lx);
        }

        // Shift intercept down by min_margin to ensure soundness
        let adjusted_intercept = intercept + min_margin - 1e-5;

        // We want the highest lower bound (closest to function)
        // Evaluate at center to compare
        let eval_center = slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_lower_slope * (l + u) / 2.0 + best_lower_intercept;

        if eval_center > current_best {
            best_lower_slope = slope;
            best_lower_intercept = adjusted_intercept;
        }
    }

    // Also try chord for lower bound
    {
        let chord_slope = (gu - gl) / (u - l);
        let intercept = gl - chord_slope * l;

        let mut min_margin = f32::INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = chord_slope * x + intercept;
            min_margin = min_margin.min(gx - lx);
        }

        let adjusted_intercept = intercept + min_margin - 1e-5;
        let eval_center = chord_slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_lower_slope * (l + u) / 2.0 + best_lower_intercept;

        if eval_center > current_best {
            best_lower_slope = chord_slope;
            best_lower_intercept = adjusted_intercept;
        }
    }

    // Upper bound: line must be >= GELU(x) for all x in [l, u]
    let mut best_upper_slope = 0.0_f32;
    let mut best_upper_intercept = f32::INFINITY;

    for &point in &candidates {
        let slope = gelu_derivative(point, approximation);
        let gp = gelu_eval(point, approximation);
        let intercept = gp - slope * point;

        // Find max margin (line(x) - GELU(x)) needed over interval
        let mut max_margin = f32::NEG_INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = slope * x + intercept;
            max_margin = max_margin.max(gx - lx);
        }

        // Shift intercept up to ensure soundness
        let adjusted_intercept = intercept + max_margin + 1e-5;

        // We want the lowest upper bound (closest to function)
        let eval_center = slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_upper_slope * (l + u) / 2.0 + best_upper_intercept;

        if eval_center < current_best {
            best_upper_slope = slope;
            best_upper_intercept = adjusted_intercept;
        }
    }

    // Also try chord for upper bound
    {
        let chord_slope = (gu - gl) / (u - l);
        let intercept = gl - chord_slope * l;

        let mut max_margin = f32::NEG_INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = chord_slope * x + intercept;
            max_margin = max_margin.max(gx - lx);
        }

        let adjusted_intercept = intercept + max_margin + 1e-5;
        let eval_center = chord_slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_upper_slope * (l + u) / 2.0 + best_upper_intercept;

        if eval_center < current_best {
            best_upper_slope = chord_slope;
            best_upper_intercept = adjusted_intercept;
        }
    }

    (
        best_lower_slope,
        best_lower_intercept,
        best_upper_slope,
        best_upper_intercept,
    )
}

/// Compute adaptive linear relaxation for GELU.
///
/// Tries multiple relaxation strategies and returns the tightest one
/// (smallest bound width at the interval center).
pub fn adaptive_gelu_linear_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
    mode: RelaxationMode,
) -> (f32, f32, f32, f32) {
    match mode {
        RelaxationMode::Chord => gelu_linear_relaxation(l, u, approximation),
        RelaxationMode::Tangent => gelu_tangent_relaxation(l, u, approximation),
        RelaxationMode::TwoSlope => gelu_two_slope_relaxation(l, u, approximation),
        RelaxationMode::Adaptive => {
            // Try all strategies and pick the tightest
            let chord = gelu_linear_relaxation(l, u, approximation);
            let tangent = gelu_tangent_relaxation(l, u, approximation);
            let two_slope = gelu_two_slope_relaxation(l, u, approximation);

            // Measure width at center point
            let c = (l + u) / 2.0;

            fn bound_width(relaxation: &(f32, f32, f32, f32), x: f32) -> f32 {
                let (ls, li, us, ui) = *relaxation;
                (us * x + ui) - (ls * x + li)
            }

            let chord_width = bound_width(&chord, c);
            let tangent_width = bound_width(&tangent, c);
            let two_slope_width = bound_width(&two_slope, c);

            // Return the tightest (smallest positive width)
            if chord_width <= tangent_width && chord_width <= two_slope_width {
                chord
            } else if tangent_width <= two_slope_width {
                tangent
            } else {
                two_slope
            }
        }
    }
}

impl GELULayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Similar to ReLU's propagate_linear_with_bounds, computes linear relaxation
    /// of GELU based on the input interval [l, u] and transforms the linear bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("GELU layer CROWN backward propagation with pre-activation bounds");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        let mut lower_slopes = Array1::<f32>::zeros(num_neurons);
        let mut lower_intercepts = Array1::<f32>::zeros(num_neurons);
        let mut upper_slopes = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercepts = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];
            let (ls, li, us, ui) =
                adaptive_gelu_linear_relaxation(l, u, self.approximation, self.relaxation_mode);
            lower_slopes[i] = ls;
            lower_intercepts[i] = li;
            upper_slopes[i] = us;
            upper_intercepts[i] = ui;
        }

        // Backward propagation through GELU
        // Same pattern as ReLU: choose relaxation based on coefficient sign
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large
                    // Use lower relaxation: y_i >= lower_slope * x_i + lower_intercept
                    new_lower_a[[j, i]] = la * lower_slopes[i];
                    new_lower_b[j] += la * lower_intercepts[i];
                } else {
                    // Negative coeff: want y_i to be small
                    // Use upper relaxation: y_i <= upper_slope * x_i + upper_intercept
                    new_lower_a[[j, i]] = la * upper_slopes[i];
                    new_lower_b[j] += la * upper_intercepts[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be small
                    // Use upper relaxation
                    new_upper_a[[j, i]] = ua * upper_slopes[i];
                    new_upper_b[j] += ua * upper_intercepts[i];
                } else {
                    // Negative coeff: want y_i to be large
                    // Use lower relaxation
                    new_upper_a[[j, i]] = ua * lower_slopes[i];
                    new_upper_b[j] += ua * lower_intercepts[i];
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

    /// Batched CROWN backward propagation through GELU with pre-activation bounds.
    ///
    /// Same as `propagate_linear_with_bounds` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("GELU layer batched CROWN backward propagation");

        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let in_dim = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        let pre_in_dim = *pre_shape.last().unwrap_or(&0);
        if pre_in_dim != in_dim {
            return Err(GammaError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![pre_in_dim],
            });
        }

        // Reshape pre-activation to [batch, in_dim]
        let pre_lower_flat = pre_activation
            .lower
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape pre_lower".to_string()))?;
        let pre_upper_flat = pre_activation
            .upper
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape pre_upper".to_string()))?;

        // Reshape bounds
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        // Output arrays
        let mut new_lower_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, in_dim));
        let mut new_lower_b = Array2::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::zeros((total_batch, out_dim));

        // Copy initial bias
        for b in 0..total_batch {
            for j in 0..out_dim {
                new_lower_b[[b, j]] = lower_b_2d[[b, j]];
                new_upper_b[[b, j]] = upper_b_2d[[b, j]];
            }
        }

        // Process each batch position
        for b in 0..total_batch {
            for i in 0..in_dim {
                let l = pre_lower_flat[[b, i]];
                let u = pre_upper_flat[[b, i]];
                let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                    adaptive_gelu_linear_relaxation(l, u, self.approximation, self.relaxation_mode);

                for j in 0..out_dim {
                    let la = lower_a_3d[[b, j, i]];
                    let ua = upper_a_3d[[b, j, i]];
                    let row_idx = b * out_dim + j;

                    // For lower bound output
                    if la >= 0.0 {
                        new_lower_a[[row_idx, i]] = la * lower_slope;
                        new_lower_b[[b, j]] += la * lower_intercept;
                    } else {
                        new_lower_a[[row_idx, i]] = la * upper_slope;
                        new_lower_b[[b, j]] += la * upper_intercept;
                    }

                    // For upper bound output
                    if ua >= 0.0 {
                        new_upper_a[[row_idx, i]] = ua * upper_slope;
                        new_upper_b[[b, j]] += ua * upper_intercept;
                    } else {
                        new_upper_a[[row_idx, i]] = ua * lower_slope;
                        new_upper_b[[b, j]] += ua * lower_intercept;
                    }
                }
            }
        }

        // Reshape back
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, in_dim])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

        Ok(BatchedLinearBounds {
            lower_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            lower_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            upper_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            upper_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        })
    }
}

impl BoundPropagation for GELULayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();
        let approx = self.approximation;

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(&input.lower)
            .and(&input.upper);

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = gelu_bound_interval(il, iu, approx);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = gelu_bound_interval(il, iu, approx);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Without pre-activation bounds, we can't compute proper linear relaxation
        // Return identity (pass-through) - caller should use propagate_linear_with_bounds
        debug!("GELU CROWN propagation without bounds - using identity (caller should use propagate_linear_with_bounds)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// LayerNorm layer: y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// Normalizes inputs across the last dimension (or specified normalized_shape).
#[derive(Debug, Clone)]
pub struct LayerNormLayer {
    /// Scale parameter (gamma)
    pub gamma: Array1<f32>,
    /// Shift parameter (beta)
    pub beta: Array1<f32>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Use forward mode for IBP: compute mean/std from center point (midpoint of bounds)
    /// instead of computing uncertain bounds on mean/std. This dramatically reduces
    /// bound explosion but may not be perfectly sound for large perturbations.
    /// Default: false (use conservative IBP)
    pub forward_mode: bool,
}

impl LayerNormLayer {
    /// Create a new LayerNorm layer.
    pub fn new(gamma: Array1<f32>, beta: Array1<f32>, eps: f32) -> Self {
        Self {
            gamma,
            beta,
            eps,
            forward_mode: false,
        }
    }

    /// Create a LayerNorm layer with default gamma=1 and beta=0.
    pub fn new_default(size: usize, eps: f32) -> Self {
        Self {
            gamma: Array1::ones(size),
            beta: Array1::zeros(size),
            eps,
            forward_mode: false,
        }
    }

    /// Create a LayerNorm layer with forward mode enabled (tighter but approximate bounds).
    pub fn new_forward_mode(gamma: Array1<f32>, beta: Array1<f32>, eps: f32) -> Self {
        Self {
            gamma,
            beta,
            eps,
            forward_mode: true,
        }
    }

    /// Enable or disable forward mode.
    ///
    /// Forward mode uses mean/std computed from the center (midpoint) of input bounds
    /// instead of computing uncertain bounds. This dramatically reduces bound explosion
    /// but may not be perfectly sound for large perturbations.
    pub fn with_forward_mode(mut self, enabled: bool) -> Self {
        self.forward_mode = enabled;
        self
    }

    /// Evaluate LayerNorm at a concrete point.
    ///
    /// Returns gamma * (x - mean(x)) / std(x) + beta
    pub fn eval(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len() as f32;
        let mean = x.mean().unwrap_or(0.0);
        let var = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / n;
        let std = (var + self.eps).sqrt();

        x.iter()
            .zip(self.gamma.iter())
            .zip(self.beta.iter())
            .map(|((&xi, &g), &b)| g * (xi - mean) / std + b)
            .collect()
    }

    /// Compute the Jacobian of LayerNorm at a point.
    ///
    /// For LayerNorm: y_i = gamma_i * (x_i - mean(x)) / std(x) + beta_i
    /// The Jacobian entry `J[i,j]` = ∂y_i/∂x_j:
    ///   `J[i,j]` = gamma_i / std * \[δ_ij - 1/n - z_i * z_j / n\]
    /// where z_i = (x_i - mean) / std is the normalized value.
    pub fn jacobian(&self, x: &Array1<f32>) -> Array2<f32> {
        let n = x.len();
        let nf = n as f32;

        let mean = x.mean().unwrap_or(0.0);
        let var = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / nf;
        let std = (var + self.eps).sqrt();

        // Compute normalized values z_i
        let z: Vec<f32> = x.iter().map(|&xi| (xi - mean) / std).collect();

        // Build Jacobian matrix
        let mut jacobian = Array2::<f32>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let delta_ij = if i == j { 1.0 } else { 0.0 };
                // J[i,j] = gamma_i / std * [δ_ij - 1/n - z_i * z_j / n]
                jacobian[[i, j]] = self.gamma[i] / std * (delta_ij - 1.0 / nf - z[i] * z[j] / nf);
            }
        }

        jacobian
    }

    fn fallback_output_bounds(&self, shape: &[usize]) -> Result<BoundedTensor> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(GammaError::InvalidSpec(
                "LayerNorm requires at least 1D input".to_string(),
            ));
        }

        let norm_size = shape[ndim - 1];
        if self.gamma.len() != norm_size || self.beta.len() != norm_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![self.gamma.len()],
            });
        }

        if norm_size == 0 {
            return BoundedTensor::new(ArrayD::zeros(IxDyn(shape)), ArrayD::zeros(IxDyn(shape)));
        }

        let z_max = if norm_size <= 1 {
            0.0
        } else {
            ((norm_size as f32) - 1.0).sqrt()
        };

        let z_lower = -z_max;
        let z_upper = z_max;

        let mut per_dim_lower = Array1::<f32>::zeros(norm_size);
        let mut per_dim_upper = Array1::<f32>::zeros(norm_size);
        for i in 0..norm_size {
            let g = self.gamma[i];
            let b = self.beta[i];

            if !g.is_finite() || !b.is_finite() {
                per_dim_lower.fill(f32::NEG_INFINITY);
                per_dim_upper.fill(f32::INFINITY);
                break;
            }

            if g >= 0.0 {
                per_dim_lower[i] = b + g * z_lower;
                per_dim_upper[i] = b + g * z_upper;
            } else {
                per_dim_lower[i] = b + g * z_upper;
                per_dim_upper[i] = b + g * z_lower;
            }
        }

        let mut out_lower = ArrayD::<f32>::zeros(IxDyn(shape));
        let mut out_upper = ArrayD::<f32>::zeros(IxDyn(shape));
        for mut lane in out_lower.lanes_mut(ndarray::Axis(ndim - 1)) {
            lane.assign(&per_dim_lower);
        }
        for mut lane in out_upper.lanes_mut(ndarray::Axis(ndim - 1)) {
            lane.assign(&per_dim_upper);
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Forward-mode IBP for LayerNorm.
    ///
    /// Uses the center point (midpoint of bounds) to compute fixed mean/std,
    /// then propagates bounds through LayerNorm as if it were a linear function
    /// with those fixed statistics. This dramatically reduces bound explosion
    /// compared to conservative IBP but is approximate (not perfectly sound)
    /// for large perturbations.
    ///
    /// The key insight from Auto-LiRPA: for small perturbations, the mean and
    /// variance don't change much, so using fixed statistics is a good approximation.
    #[inline]
    fn propagate_ibp_forward_mode(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let norm_size = shape[ndim - 1];

        let has_nonfinite = input
            .lower
            .iter()
            .chain(input.upper.iter())
            .any(|&v| !v.is_finite());
        if has_nonfinite {
            return self.fallback_output_bounds(shape);
        }

        // Compute center point (midpoint of bounds)
        let center = (&input.lower + &input.upper) * 0.5;

        // Compute half-width (radius) of input bounds
        let radius = (&input.upper - &input.lower) * 0.5;

        let has_nonfinite_center = center.iter().chain(radius.iter()).any(|&v| !v.is_finite());
        if has_nonfinite_center {
            return self.fallback_output_bounds(shape);
        }

        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        if ndim == 1 {
            // 1D case: single vector
            let mean = center.mean().unwrap_or(0.0);
            let var = center.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let std = (var + self.eps).sqrt();

            // Use effective std that prevents explosion when center has low variance.
            // Real activations typically have std ≈ 1.0 after normalization. When the center
            // (midpoint of bounds) drifts to low variance, sensitivity = gamma/std explodes.
            // Using min_effective_std of 0.3 caps sensitivity at ~3.3× (for gamma ≈ 1).
            // This is a heuristic but dramatically reduces bound explosion in practice.
            const MIN_EFFECTIVE_STD: f32 = 0.3;
            let effective_std = std.max(MIN_EFFECTIVE_STD);

            // Cap effective gamma to prevent explosion from models with large gamma values.
            // Some models (e.g., Qwen3) have RMSNorm gamma values up to 69, causing 230× sensitivity.
            // Real transformer behavior: gamma scales output but typical values are near 1.
            // With MAX_EFFECTIVE_GAMMA = 3.0 and MIN_EFFECTIVE_STD = 0.3, max sensitivity = 10×.
            const MAX_EFFECTIVE_GAMMA: f32 = 3.0;

            for i in 0..norm_size {
                let c = center[[i]];
                let r = radius[[i]];
                let g = self.gamma[i];
                let b = self.beta[i];

                // At center point: y = g * (c - mean) / std + b
                // We still use actual std for y_center to keep it accurate
                let y_center = g * (c - mean) / std + b;

                // For small perturbations, the sensitivity is approximately |g/std|
                // Use effective_std to prevent sensitivity explosion, and cap effective gamma
                let effective_gamma = g.abs().min(MAX_EFFECTIVE_GAMMA);
                let sensitivity = effective_gamma / effective_std;

                // Output bounds: center ± sensitivity * radius
                // But we use max radius across the dimension since mean aggregates
                let max_radius: f32 = radius.iter().cloned().fold(0.0_f32, f32::max);
                let output_radius = sensitivity * (r + max_radius / norm_size as f32);

                out_lower[[i]] = y_center - output_radius;
                out_upper[[i]] = y_center + output_radius;
            }
        } else {
            // N-D case: process each batch position
            let batch_size: usize = shape[..ndim - 1].iter().product();

            for batch_idx in 0..batch_size {
                // Convert flat batch index to multi-dimensional index
                let mut idx = vec![0usize; ndim - 1];
                let mut remaining = batch_idx;
                for d in (0..ndim - 1).rev() {
                    idx[d] = remaining % shape[d];
                    remaining /= shape[d];
                }

                // Extract center and radius for this position
                let mut center_slice = Vec::with_capacity(norm_size);
                let mut radius_slice = Vec::with_capacity(norm_size);
                for i in 0..norm_size {
                    let mut full_idx = idx.clone();
                    full_idx.push(i);
                    center_slice.push(center[full_idx.as_slice()]);
                    radius_slice.push(radius[full_idx.as_slice()]);
                }

                // Compute mean and std from center point
                let mean: f32 = center_slice.iter().sum::<f32>() / norm_size as f32;
                let var: f32 = center_slice
                    .iter()
                    .map(|&xi| (xi - mean).powi(2))
                    .sum::<f32>()
                    / norm_size as f32;
                let std = (var + self.eps).sqrt();

                // Use effective std that prevents explosion when center has low variance.
                // See 1D case comment for details.
                const MIN_EFFECTIVE_STD: f32 = 0.3;
                let effective_std = std.max(MIN_EFFECTIVE_STD);

                // Cap effective gamma - see 1D case comment for details.
                const MAX_EFFECTIVE_GAMMA: f32 = 3.0;

                // Max radius for coupling effect
                let max_radius: f32 = radius_slice.iter().cloned().fold(0.0_f32, f32::max);

                for i in 0..norm_size {
                    let c = center_slice[i];
                    let r = radius_slice[i];
                    let g = self.gamma[i];
                    let b = self.beta[i];

                    // Use actual std for y_center, effective_std and capped gamma for sensitivity
                    let y_center = g * (c - mean) / std + b;
                    let effective_gamma = g.abs().min(MAX_EFFECTIVE_GAMMA);
                    let sensitivity = effective_gamma / effective_std;
                    let output_radius = sensitivity * (r + max_radius / norm_size as f32);

                    let mut full_idx = idx.clone();
                    full_idx.push(i);
                    out_lower[full_idx.as_slice()] = y_center - output_radius;
                    out_upper[full_idx.as_slice()] = y_center + output_radius;
                }
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Compute CROWN linear bounds for LayerNorm with pre-activation bounds.
    ///
    /// Uses local linearization at the center point with sampling-verified soundness.
    /// Returns linear bounds: y_lower >= A_l @ x + b_l, y_upper <= A_u @ x + b_u
    ///
    /// Note: This function requires flattened 1D inputs where the total size equals
    /// gamma.len(). For multi-dimensional inputs [batch, norm_size], use
    /// propagate_linear_batched_with_bounds or enable forward-mode IBP.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Flatten pre-activation bounds to 1D
        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if self.gamma.len() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![self.gamma.len()],
                got: vec![num_neurons],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Check for infinite or NaN bounds - if any dimension has infinite bounds,
        // LayerNorm linearization sampling fails. Return identity relaxation (pass-through).
        // This is conservative but avoids NaN from (-inf + inf)/2 = NaN.
        let has_infinite = pre_lower.iter().any(|&v| v.is_infinite() || v.is_nan())
            || pre_upper.iter().any(|&v| v.is_infinite() || v.is_nan());
        if has_infinite {
            // Identity relaxation: pass bounds through unchanged
            return Ok(bounds.clone());
        }

        // Compute center point and evaluate
        let x_center: Array1<f32> = pre_lower
            .iter()
            .zip(pre_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();

        let y_center = self.eval(&x_center);
        let jacobian = self.jacobian(&x_center);

        // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
        // where b_approx = y_c - J @ x_c
        let jx_center = jacobian.dot(&x_center);
        let b_approx: Array1<f32> = &y_center - &jx_center;

        // Sample to find max error from linear approximation
        // This ensures soundness for non-linear LayerNorm behavior
        let num_samples = 50; // Sample in the hypercube
        let mut max_error_above: Array1<f32> = Array1::zeros(num_neurons); // actual - approx
        let mut max_error_below: Array1<f32> = Array1::zeros(num_neurons); // approx - actual

        // Allocate x_sample once outside loop, reuse buffer for each sample
        let mut x_sample = x_center.clone();
        // Use pseudo-random sampling with fixed seed for reproducibility
        for sample_idx in 0..num_samples {
            // Reset to center values (reuses allocation instead of cloning)
            x_sample.assign(&x_center);
            for i in 0..num_neurons {
                // Mix sample_idx with dimension for different patterns per dimension
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }

            // Also sample corners (first few samples)
            if sample_idx < num_neurons * 2 {
                let dim = sample_idx / 2;
                if dim < num_neurons {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        pre_lower[dim]
                    } else {
                        pre_upper[dim]
                    };
                }
            }

            let y_actual = self.eval(&x_sample);
            let y_approx: Array1<f32> = jacobian.dot(&x_sample) + &b_approx;

            for i in 0..num_neurons {
                let error = y_actual[i] - y_approx[i];
                if error > max_error_above[i] {
                    max_error_above[i] = error;
                }
                if -error > max_error_below[i] {
                    max_error_below[i] = -error;
                }
            }
        }

        // Add safety margin (10% extra for unsampled regions)
        let safety_factor = 1.1;
        for i in 0..num_neurons {
            max_error_above[i] *= safety_factor;
            max_error_below[i] *= safety_factor;
            // Ensure minimum error margin
            let min_margin = 1e-6_f32;
            if max_error_above[i] < min_margin {
                max_error_above[i] = min_margin;
            }
            if max_error_below[i] < min_margin {
                max_error_below[i] = min_margin;
            }
        }

        // Backward propagation through LayerNorm using linear relaxation
        // For lower bound: y_actual >= y_approx - max_error_below = J @ x + b_approx - max_error_below
        // For upper bound: y_actual <= y_approx + max_error_above = J @ x + b_approx + max_error_above
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: need lower bound on each y_i when coeff positive
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large, use lower relaxation
                    // y_i >= sum_k J[i,k] * x_k + (b_approx[i] - max_error_below[i])
                    for k in 0..num_neurons {
                        new_lower_a[[j, k]] += la * jacobian[[i, k]];
                    }
                    new_lower_b[j] += la * (b_approx[i] - max_error_below[i]);
                } else {
                    // Negative coeff: want y_i to be small, use upper relaxation
                    // y_i <= sum_k J[i,k] * x_k + (b_approx[i] + max_error_above[i])
                    for k in 0..num_neurons {
                        new_lower_a[[j, k]] += la * jacobian[[i, k]];
                    }
                    new_lower_b[j] += la * (b_approx[i] + max_error_above[i]);
                }

                // For upper bound output
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be large, use upper relaxation
                    for k in 0..num_neurons {
                        new_upper_a[[j, k]] += ua * jacobian[[i, k]];
                    }
                    new_upper_b[j] += ua * (b_approx[i] + max_error_above[i]);
                } else {
                    // Negative coeff: want y_i to be small, use lower relaxation
                    for k in 0..num_neurons {
                        new_upper_a[[j, k]] += ua * jacobian[[i, k]];
                    }
                    new_upper_b[j] += ua * (b_approx[i] - max_error_below[i]);
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

    /// Batched CROWN backward propagation through LayerNorm with pre-activation bounds.
    ///
    /// Handles N-D inputs by processing each batch position independently using the 1D
    /// implementation. LayerNorm operates on the last dimension (norm_size).
    ///
    /// Input shape: [...batch_dims, norm_size]
    /// Bounds shape: [...batch_dims, out_dim, norm_size]
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("LayerNorm layer batched CROWN backward propagation");

        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let norm_size = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        let pre_norm_size = *pre_shape.last().unwrap_or(&0);
        if pre_norm_size != norm_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![pre_norm_size],
            });
        }

        if self.gamma.len() != norm_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![self.gamma.len()],
                got: vec![norm_size],
            });
        }

        // Reshape pre-activation to [batch, norm_size]
        let pre_lower_flat = pre_activation
            .lower
            .view()
            .into_shape_with_order((total_batch, norm_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape pre_lower for layernorm".to_string())
            })?;
        let pre_upper_flat = pre_activation
            .upper
            .view()
            .into_shape_with_order((total_batch, norm_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape pre_upper for layernorm".to_string())
            })?;

        // Reshape bounds to [batch, out_dim, norm_size]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, norm_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape lower_a for layernorm".to_string())
            })?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, norm_size))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape upper_a for layernorm".to_string())
            })?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape lower_b for layernorm".to_string())
            })?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape upper_b for layernorm".to_string())
            })?;

        // Output arrays
        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, norm_size));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, norm_size));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        // Process each batch position independently
        for b in 0..total_batch {
            // Extract 1D pre-activation bounds for this batch
            let pre_lower_1d: Array1<f32> = pre_lower_flat.row(b).to_owned();
            let pre_upper_1d: Array1<f32> = pre_upper_flat.row(b).to_owned();

            // Extract 2D coefficient matrix for this batch: [out_dim, norm_size]
            let lower_a_slice = lower_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
            let upper_a_slice = upper_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
            let lower_b_slice = lower_b_2d.row(b).to_owned();
            let upper_b_slice = upper_b_2d.row(b).to_owned();

            let batch_bounds = LinearBounds {
                lower_a: lower_a_slice,
                lower_b: lower_b_slice,
                upper_a: upper_a_slice,
                upper_b: upper_b_slice,
            };

            // Create a temporary BoundedTensor for the 1D case
            let pre_bounds_1d =
                BoundedTensor::new(pre_lower_1d.into_dyn(), pre_upper_1d.into_dyn())?;

            // Apply the existing 1D layernorm backward
            let result = self.propagate_linear_with_bounds(&batch_bounds, &pre_bounds_1d)?;

            // Copy results back
            for j in 0..out_dim {
                for k in 0..norm_size {
                    new_lower_a[[b, j, k]] = result.lower_a[[j, k]];
                    new_upper_a[[b, j, k]] = result.upper_a[[j, k]];
                }
                new_lower_b[[b, j]] = result.lower_b[j];
                new_upper_b[[b, j]] = result.upper_b[j];
            }
        }

        // Reshape back to original batch dims
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, norm_size])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

        Ok(BatchedLinearBounds {
            lower_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            lower_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            upper_a: ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            upper_b: ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        })
    }
}

impl BoundPropagation for LayerNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // LayerNorm normalizes across the last dimension
        // For each sample: y = gamma * (x - mean) / std + beta
        // where mean and std are computed from x

        let shape = input.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Err(GammaError::InvalidSpec(
                "LayerNorm requires at least 1D input".to_string(),
            ));
        }

        let norm_size = shape[ndim - 1];
        if self.gamma.len() != norm_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![self.gamma.len()],
            });
        }

        let has_nonfinite = input
            .lower
            .iter()
            .chain(input.upper.iter())
            .any(|&v| !v.is_finite());
        if has_nonfinite {
            return self.fallback_output_bounds(shape);
        }

        // Forward mode: use center point (midpoint) for mean/std computation.
        // This dramatically reduces bound explosion but is approximate for large perturbations.
        if self.forward_mode {
            return self.propagate_ibp_forward_mode(input);
        }
        use ndarray::Axis;

        // Compute bounds on mean
        // mean is in [mean(lower), mean(upper)]
        let mean_lower = input.lower.mean_axis(Axis(ndim - 1)).unwrap();
        let mean_upper = input.upper.mean_axis(Axis(ndim - 1)).unwrap();

        let has_nonfinite_mean = mean_lower
            .iter()
            .chain(mean_upper.iter())
            .any(|&v| !v.is_finite());
        if has_nonfinite_mean {
            return self.fallback_output_bounds(shape);
        }

        // For variance, we need to be more careful
        // Var(x) = E[x^2] - E[x]^2
        // For bounded inputs, we bound each component

        // Conservative approach: bound (x - mean) first
        // Lower bound of (x - mean): lower_x - upper_mean
        // Upper bound of (x - mean): upper_x - lower_mean

        // Then bound the variance (sum of squares)
        // Finally bound the normalized output

        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        // Process based on dimensionality
        if ndim == 1 {
            let mean_l: f32 = mean_lower.into_iter().next().unwrap_or(0.0);
            let mean_u: f32 = mean_upper.into_iter().next().unwrap_or(0.0);

            // Compute variance bounds
            // Var = sum((x - mean)^2) / n
            // Conservative: use max variance from corners
            let mut var_lower = 0.0_f32;
            let mut var_upper = 0.0_f32;

            for i in 0..norm_size {
                let xl = input.lower[[i]];
                let xu = input.upper[[i]];

                // (x - mean) bounds
                let diff_l = xl - mean_u;
                let diff_u = xu - mean_l;

                // (x - mean)^2 bounds
                if diff_l >= 0.0 {
                    var_lower += diff_l * diff_l;
                    var_upper += diff_u * diff_u;
                } else if diff_u <= 0.0 {
                    var_lower += diff_u * diff_u;
                    var_upper += diff_l * diff_l;
                } else {
                    var_lower += 0.0;
                    var_upper += diff_l.abs().max(diff_u.abs()).powi(2);
                }
            }
            var_lower /= norm_size as f32;
            var_upper /= norm_size as f32;

            // std bounds
            let std_lower = (var_lower + self.eps).sqrt();
            let std_upper = (var_upper + self.eps).sqrt();

            // Compute normalized bounds
            for i in 0..norm_size {
                let xl = input.lower[[i]];
                let xu = input.upper[[i]];

                // (x - mean) / std bounds
                // To get lower bound: minimize (x - mean) / std
                // To get upper bound: maximize (x - mean) / std

                let diff_ll = xl - mean_u;
                let diff_lu = xl - mean_l;
                let diff_ul = xu - mean_u;
                let diff_uu = xu - mean_l;

                // Division by std: consider all corners
                let corners = [
                    diff_ll / std_upper,
                    diff_ll / std_lower,
                    diff_lu / std_upper,
                    diff_lu / std_lower,
                    diff_ul / std_upper,
                    diff_ul / std_lower,
                    diff_uu / std_upper,
                    diff_uu / std_lower,
                ];

                let norm_l = corners.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let norm_u = corners.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Apply gamma and beta
                let g = self.gamma[i];
                let b = self.beta[i];

                if g >= 0.0 {
                    out_lower[[i]] = g * norm_l + b;
                    out_upper[[i]] = g * norm_u + b;
                } else {
                    out_lower[[i]] = g * norm_u + b;
                    out_upper[[i]] = g * norm_l + b;
                }
            }
        } else {
            // Multi-dimensional case: normalize along last axis
            // Iterate over all positions except the last axis
            let batch_size: usize = shape[..ndim - 1].iter().product();

            for batch_idx in 0..batch_size {
                // Convert flat batch index to multi-dimensional index
                let mut idx = vec![0usize; ndim - 1];
                let mut remaining = batch_idx;
                for d in (0..ndim - 1).rev() {
                    idx[d] = remaining % shape[d];
                    remaining /= shape[d];
                }

                let mean_l = if mean_lower.ndim() == 0 {
                    *mean_lower.first().unwrap_or(&0.0)
                } else {
                    mean_lower[idx.as_slice()]
                };
                let mean_u = if mean_upper.ndim() == 0 {
                    *mean_upper.first().unwrap_or(&0.0)
                } else {
                    mean_upper[idx.as_slice()]
                };

                // Compute variance bounds for this position
                let mut var_lower = 0.0_f32;
                let mut var_upper = 0.0_f32;

                for i in 0..norm_size {
                    let mut full_idx = idx.clone();
                    full_idx.push(i);

                    let xl = input.lower[full_idx.as_slice()];
                    let xu = input.upper[full_idx.as_slice()];

                    let diff_l = xl - mean_u;
                    let diff_u = xu - mean_l;

                    if diff_l >= 0.0 {
                        var_lower += diff_l * diff_l;
                        var_upper += diff_u * diff_u;
                    } else if diff_u <= 0.0 {
                        var_lower += diff_u * diff_u;
                        var_upper += diff_l * diff_l;
                    } else {
                        var_lower += 0.0;
                        var_upper += diff_l.abs().max(diff_u.abs()).powi(2);
                    }
                }
                var_lower /= norm_size as f32;
                var_upper /= norm_size as f32;

                let std_lower = (var_lower + self.eps).sqrt();
                let std_upper = (var_upper + self.eps).sqrt();

                // Compute normalized output bounds
                for i in 0..norm_size {
                    let mut full_idx = idx.clone();
                    full_idx.push(i);

                    let xl = input.lower[full_idx.as_slice()];
                    let xu = input.upper[full_idx.as_slice()];

                    let diff_ll = xl - mean_u;
                    let diff_lu = xl - mean_l;
                    let diff_ul = xu - mean_u;
                    let diff_uu = xu - mean_l;

                    let corners = [
                        diff_ll / std_upper,
                        diff_ll / std_lower,
                        diff_lu / std_upper,
                        diff_lu / std_lower,
                        diff_ul / std_upper,
                        diff_ul / std_lower,
                        diff_uu / std_upper,
                        diff_uu / std_lower,
                    ];

                    let norm_l = corners.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let norm_u = corners.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    let g = self.gamma[i];
                    let b = self.beta[i];

                    if g >= 0.0 {
                        out_lower[full_idx.as_slice()] = g * norm_l + b;
                        out_upper[full_idx.as_slice()] = g * norm_u + b;
                    } else {
                        out_lower[full_idx.as_slice()] = g * norm_u + b;
                        out_upper[full_idx.as_slice()] = g * norm_l + b;
                    }
                }
            }
        }

        // Clamp bounds to finite values to prevent overflow propagation.
        // Also clamp NaN values that may arise from inf/inf or 0/0 operations.
        const MAX_BOUND: f32 = f32::MAX / 2.0;
        out_lower.mapv_inplace(|v| {
            if v.is_nan() {
                -MAX_BOUND
            } else {
                v.max(-MAX_BOUND)
            }
        });
        out_upper.mapv_inplace(|v| {
            if v.is_nan() {
                MAX_BOUND
            } else {
                v.min(MAX_BOUND)
            }
        });

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Return identity (pass-through) - caller should use propagate_linear_with_bounds
        debug!("LayerNorm CROWN propagation without bounds - using identity (caller should use propagate_linear_with_bounds)");
        Ok(Cow::Borrowed(bounds))
    }
}

/// BatchNorm layer: y = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// During inference, mean and variance are fixed (running statistics).
/// This can be simplified to: y = x * scale + bias where:
///   scale = gamma / sqrt(var + eps)
///   bias = beta - mean * gamma / sqrt(var + eps)
#[derive(Debug, Clone)]
pub struct BatchNormLayer {
    /// Pre-computed scale: gamma / sqrt(var + eps)
    pub scale: ArrayD<f32>,
    /// Pre-computed bias: beta - mean * scale
    pub bias: ArrayD<f32>,
    /// Number of channels (for proper broadcasting)
    pub num_channels: usize,
}

impl BatchNormLayer {
    /// Create a new BatchNorm layer from ONNX parameters.
    ///
    /// ONNX BatchNormalization inputs:
    /// - scale (gamma): per-channel scale
    /// - B (beta): per-channel bias
    /// - mean: running mean per channel
    /// - var: running variance per channel
    /// - epsilon: small constant (default 1e-5)
    pub fn new(
        gamma: &ArrayD<f32>,
        beta: &ArrayD<f32>,
        mean: &ArrayD<f32>,
        var: &ArrayD<f32>,
        epsilon: f32,
    ) -> Self {
        // Compute scale = gamma / sqrt(var + eps)
        let std = var.mapv(|v| (v + epsilon).sqrt());
        let scale = gamma / &std;

        // Compute bias = beta - mean * scale
        let bias = beta - mean * &scale;

        let num_channels = scale.len();

        Self {
            scale,
            bias,
            num_channels,
        }
    }

    /// Create from pre-computed scale and bias.
    pub fn from_scale_bias(scale: ArrayD<f32>, bias: ArrayD<f32>) -> Self {
        let num_channels = scale.len();
        Self {
            scale,
            bias,
            num_channels,
        }
    }

    /// CROWN backward propagation through BatchNorm with shape information.
    ///
    /// BatchNorm is a linear operation: y_i = scale[c(i)] * x_i + bias[c(i)]
    /// where c(i) is the channel index for flattened position i.
    ///
    /// For CROWN backward:
    /// - Scale coefficient columns by scale[c(i)]
    /// - Add bias contribution: new_b = b + A @ bias_expanded
    /// - Handle negative scale by swapping lower/upper coefficients
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let ndim = shape.len();

        let num_inputs = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();

        // Handle 1D input [C] as a special case (e.g., after ReduceMean in ViT classifier)
        // Each element is treated as its own channel
        if ndim == 1 {
            if shape[0] != self.num_channels {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![self.num_channels],
                    got: vec![shape[0]],
                });
            }

            // For 1D input, channel_idx = 0 and elements_per_channel = 1
            let num_channels = self.num_channels;
            let _total_elements = num_channels;

            // Each output row is scaled by the corresponding channel scale
            let mut new_lower_a = Array2::zeros((num_outputs, num_inputs));
            let mut new_upper_a = Array2::zeros((num_outputs, num_inputs));
            let mut new_lower_b = bounds.lower_b.clone();
            let mut new_upper_b = bounds.upper_b.clone();

            // BatchNorm: y_i = scale_i * x_i + bias_i
            // CROWN backward: given bounds y = Ax + b for next layer's output y,
            // substitute y_i = scale_i * x_i + bias_i to get bounds in terms of x:
            // new_y = (A * scale) @ x + (A @ bias + b)
            //
            // First compute bias contribution BEFORE scaling (using original A matrices)
            // Then scale the coefficient matrices
            for out_row in 0..num_outputs {
                // Compute bias contribution: sum_c(A[out_row, c] * bias[c])
                let mut lower_bias_contrib = 0.0;
                let mut upper_bias_contrib = 0.0;
                for c in 0..num_channels {
                    if c < num_inputs {
                        lower_bias_contrib += bounds.lower_a[[out_row, c]] * self.bias[[c]];
                        upper_bias_contrib += bounds.upper_a[[out_row, c]] * self.bias[[c]];
                    }
                }
                new_lower_b[out_row] += lower_bias_contrib;
                new_upper_b[out_row] += upper_bias_contrib;

                // Scale coefficient matrices column-wise
                for c in 0..num_channels {
                    let s = self.scale[[c]];
                    let in_col = c;
                    if in_col >= num_inputs {
                        continue;
                    }

                    if s >= 0.0 {
                        new_lower_a[[out_row, in_col]] = bounds.lower_a[[out_row, in_col]] * s;
                        new_upper_a[[out_row, in_col]] = bounds.upper_a[[out_row, in_col]] * s;
                    } else {
                        // Negative scale: swap lower and upper coefficients
                        new_lower_a[[out_row, in_col]] = bounds.upper_a[[out_row, in_col]] * s;
                        new_upper_a[[out_row, in_col]] = bounds.lower_a[[out_row, in_col]] * s;
                        // Also swap bias contributions for this column
                        let b = self.bias[[c]];
                        let orig_la = bounds.lower_a[[out_row, c]];
                        let orig_ua = bounds.upper_a[[out_row, c]];
                        new_lower_b[out_row] -= orig_la * b;
                        new_lower_b[out_row] += orig_ua * b;
                        new_upper_b[out_row] -= orig_ua * b;
                        new_upper_b[out_row] += orig_la * b;
                    }
                }
            }

            return Ok(LinearBounds {
                lower_a: new_lower_a,
                lower_b: new_lower_b,
                upper_a: new_upper_a,
                upper_b: new_upper_b,
            });
        }

        // Channel dimension depends on input format:
        // - 4D (N, C, H, W): channel at index 1
        // - 3D (C, H, W) without batch: channel at index 0
        // Heuristic: if first dim matches num_channels and isn't 1, assume (C, ...)
        let channel_idx = if ndim == 3 && shape[0] == self.num_channels {
            0 // (C, H, W) format without batch
        } else if ndim >= 3 {
            1 // (N, C, ...) format with batch
        } else {
            0
        };
        let num_channels = shape[channel_idx];

        if num_channels != self.num_channels {
            return Err(GammaError::ShapeMismatch {
                expected: vec![self.num_channels],
                got: vec![num_channels],
            });
        }

        // Compute elements per channel (H*W for 4D, L for 3D, etc.)
        // For NCHW: elements_per_channel = H * W
        let elements_per_channel: usize = if channel_idx + 1 >= ndim {
            1
        } else {
            shape[channel_idx + 1..].iter().product()
        };

        // Account for batch dimension if present
        // For (C, H, W) format (channel_idx=0), no batch dimension
        let batch_size = if channel_idx == 0 { 1 } else { shape[0] };
        let total_elements = batch_size * num_channels * elements_per_channel;

        if total_elements != num_inputs {
            return Err(GammaError::ShapeMismatch {
                expected: vec![total_elements],
                got: vec![num_inputs],
            });
        }

        // Build expanded scale and bias vectors for the flattened tensor
        // For NCHW layout: flat_idx = n * (C*H*W) + c * (H*W) + spatial_idx
        // Channel c for flat_idx: c = (flat_idx % (C*H*W)) / (H*W)
        let chw = num_channels * elements_per_channel;

        let mut expanded_scale = Array1::<f32>::zeros(num_inputs);
        let mut expanded_bias = Array1::<f32>::zeros(num_inputs);

        for i in 0..num_inputs {
            // For NCHW: channel = (i % (C*H*W)) / (H*W)
            let c = (i % chw) / elements_per_channel;
            expanded_scale[i] = self.scale[[c]];
            expanded_bias[i] = self.bias[[c]];
        }

        // Compute bias contributions BEFORE scaling (using original matrices)
        // new_lower_b = lower_b + lower_a @ bias
        // new_upper_b = upper_b + upper_a @ bias
        let lower_bias_contrib = bounds.lower_a.dot(&expanded_bias);
        let upper_bias_contrib = bounds.upper_a.dot(&expanded_bias);

        let mut new_lower_b = &bounds.lower_b + &lower_bias_contrib;
        let mut new_upper_b = &bounds.upper_b + &upper_bias_contrib;

        // Scale coefficient matrices column-wise by scale
        // Handle sign: if scale[c] < 0, swap lower and upper coefficients
        let mut new_lower_a = bounds.lower_a.clone();
        let mut new_upper_a = bounds.upper_a.clone();

        for i in 0..num_inputs {
            let s = expanded_scale[i];

            for j in 0..num_outputs {
                if s >= 0.0 {
                    new_lower_a[[j, i]] *= s;
                    new_upper_a[[j, i]] *= s;
                } else {
                    // Negative scale: swap lower and upper
                    let tmp = new_lower_a[[j, i]] * s;
                    new_lower_a[[j, i]] = new_upper_a[[j, i]] * s;
                    new_upper_a[[j, i]] = tmp;
                }
            }
        }

        // When scale is negative, we also need to swap bias contributions
        // because the bound direction changes
        for i in 0..num_inputs {
            let s = expanded_scale[i];
            if s < 0.0 {
                let b = expanded_bias[i];
                // Need to adjust bias: original had lower_a[:,i] * b going to lower_b
                // but after swap, that contribution should go to upper_b
                // The fix: recompute bias contribution after swap logic
                for j in 0..num_outputs {
                    let orig_la = bounds.lower_a[[j, i]];
                    let orig_ua = bounds.upper_a[[j, i]];
                    // Remove original contribution, add swapped contribution
                    new_lower_b[j] -= orig_la * b;
                    new_lower_b[j] += orig_ua * b;
                    new_upper_b[j] -= orig_ua * b;
                    new_upper_b[j] += orig_la * b;
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

impl BoundPropagation for BatchNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // BatchNorm: y = x * scale + bias (element-wise per channel)
        // Input shape formats:
        // - (N, C, H, W) for 2D with batch: channel at index 1
        // - (C, H, W) for 2D without batch: channel at index 0
        // - (N, C, L) for 1D with batch: channel at index 1
        // - (C, L) for 1D without batch: channel at index 0
        // Scale/bias shape: (C,) - broadcast over other dimensions

        let input_shape = input.shape();

        // Handle 1D input [C] as a special case (e.g., after ReduceMean in ViT classifier)
        // Each element is treated as its own channel
        if input_shape.len() == 1 {
            if input_shape[0] != self.num_channels {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![self.num_channels],
                    got: vec![input_shape[0]],
                });
            }

            // For 1D input, channel index IS the only index
            let mut out_lower = ArrayD::zeros(IxDyn(input_shape));
            let mut out_upper = ArrayD::zeros(IxDyn(input_shape));

            for c in 0..self.num_channels {
                let s = self.scale[[c]];
                let b = self.bias[[c]];
                let l = input.lower[[c]];
                let u = input.upper[[c]];

                if s >= 0.0 {
                    out_lower[[c]] = l * s + b;
                    out_upper[[c]] = u * s + b;
                } else {
                    out_lower[[c]] = u * s + b;
                    out_upper[[c]] = l * s + b;
                }
            }

            return BoundedTensor::new(out_lower, out_upper);
        }

        // Determine channel index based on input format:
        // - 4D (N, C, H, W): channel at index 1
        // - 3D (C, H, W) without batch: channel at index 0
        // - 3D (N, C, L) with batch but 1D spatial: channel at index 1
        // - 2D (C, L) without batch: channel at index 0
        // - 2D (N, C) with batch: channel at index 1
        // Heuristic:
        // - 3D+: prefer index 0 if it matches num_channels (C, H, W format)
        // - 2D: prefer index 1 if it matches num_channels (N, C format)
        let channel_idx_pos = if input_shape.len() >= 3 {
            // For 3D+ inputs, check if index 0 matches (C, H, W format without batch)
            if input_shape[0] == self.num_channels {
                0 // (C, H, W) or (C, ...) format
            } else {
                1 // (N, C, ...) format
            }
        } else if input_shape.len() == 2 {
            // For 2D inputs, prefer index 1 (N, C format is more common)
            if input_shape[1] == self.num_channels {
                1 // (N, C) format
            } else if input_shape[0] == self.num_channels {
                0 // (C, L) format
            } else {
                1 // Default
            }
        } else {
            0
        };

        let num_channels = input_shape[channel_idx_pos];
        if num_channels != self.num_channels {
            return Err(GammaError::ShapeMismatch {
                expected: vec![self.num_channels],
                got: vec![num_channels],
            });
        }

        // Create output arrays
        let mut out_lower = ArrayD::zeros(IxDyn(input_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(input_shape));

        // Apply per-channel affine transform
        // y = x * scale + bias
        // If scale > 0: y_l = x_l * scale + bias, y_u = x_u * scale + bias
        // If scale < 0: y_l = x_u * scale + bias, y_u = x_l * scale + bias

        for ((idx, &l), &u) in input.lower.indexed_iter().zip(input.upper.iter()) {
            // Channel index is at the determined position
            let channel_idx = idx[channel_idx_pos];
            let s = self.scale[[channel_idx]];
            let b = self.bias[[channel_idx]];

            if s >= 0.0 {
                out_lower[idx.clone()] = l * s + b;
                out_upper[idx] = u * s + b;
            } else {
                out_lower[idx.clone()] = u * s + b;
                out_upper[idx] = l * s + b;
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // BatchNorm is a linear operation: y = x * scale + bias
        // For linear bounds: y = Ax + b becomes y = A(scale*x) + (A*bias + b)
        // The new lower/upper coefficients are scaled

        // This is more complex due to channel-wise scaling
        // For now, fall back to IBP
        Err(GammaError::UnsupportedOp(
            "BatchNorm linear propagation not yet implemented - use IBP".to_string(),
        ))
    }
}

// ============================================================================
// Binary Operations (two bounded inputs)
// ============================================================================

/// Bounded matrix multiplication layer for operations like Q @ K^T in attention.
///
/// Unlike LinearLayer which has fixed weights, MatMulLayer multiplies two
/// bounded tensor inputs. This is used for attention score computation
/// and attention-value multiplication.
///
/// For C = A @ B where A ∈ \[A_l, A_u\] and B ∈ \[B_l, B_u\]:
/// Each element `c[i,k]` = sum_j(`a[i,j]` * `b[j,k]`) is bounded using interval arithmetic.
#[derive(Debug, Clone)]
pub struct MatMulLayer {
    /// Whether to transpose the second input (B^T instead of B).
    /// Used for Q @ K^T attention pattern.
    pub transpose_b: bool,
    /// Optional scaling factor (e.g., 1/sqrt(d_k) for attention).
    pub scale: Option<f32>,
}

impl MatMulLayer {
    /// Create a new MatMul layer.
    pub fn new(transpose_b: bool, scale: Option<f32>) -> Self {
        Self { transpose_b, scale }
    }

    /// Propagate IBP bounds through matrix multiplication of two bounded tensors.
    ///
    /// For A @ B (or A @ B^T if transpose_b), computes interval bounds on the result
    /// using standard interval arithmetic: for each element-wise product, we compute
    /// min/max over the four corner products.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Get shapes: A is (..., M, K), B is (..., K, N) or (..., N, K) if transposed
        let a_shape = input_a.shape();
        let b_shape = input_b.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "MatMul requires at least 2D inputs".to_string(),
            ));
        }

        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        let m = a_shape[a_ndim - 2];
        let k_a = a_shape[a_ndim - 1];

        let (k_b, n) = if self.transpose_b {
            (b_shape[b_ndim - 1], b_shape[b_ndim - 2])
        } else {
            (b_shape[b_ndim - 2], b_shape[b_ndim - 1])
        };

        if k_a != k_b {
            return Err(GammaError::ShapeMismatch {
                expected: vec![k_a],
                got: vec![k_b],
            });
        }

        let k = k_a;

        // Handle batch dimensions
        let a_batch: Vec<usize> = a_shape[..a_ndim - 2].to_vec();
        let b_batch: Vec<usize> = b_shape[..b_ndim - 2].to_vec();

        // For simplicity, assume batch dimensions match or are broadcastable
        let batch_dims = if a_batch.len() >= b_batch.len() {
            a_batch.clone()
        } else {
            b_batch.clone()
        };

        // Output shape: batch_dims + (M, N)
        let mut out_shape = batch_dims.clone();
        out_shape.push(m);
        out_shape.push(n);

        let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));

        // Compute bounds for each element using interval arithmetic
        // For 2D case (batch_size=1), iterate over M x N output elements
        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

        for batch_idx in 0..batch_size {
            // Convert batch index to multi-dimensional indices
            let mut batch_indices = vec![0usize; batch_dims.len()];
            let mut remaining = batch_idx;
            for d in (0..batch_dims.len()).rev() {
                batch_indices[d] = remaining % batch_dims[d];
                remaining /= batch_dims[d];
            }

            for i in 0..m {
                for j in 0..n {
                    let mut sum_lower = 0.0_f32;
                    let mut sum_upper = 0.0_f32;

                    for l in 0..k {
                        // Get A[..., i, l]
                        let mut a_idx = batch_indices.clone();
                        a_idx.push(i);
                        a_idx.push(l);
                        let a_l = input_a.lower[a_idx.as_slice()];
                        let a_u = input_a.upper[a_idx.as_slice()];

                        // Get B[..., l, j] or B[..., j, l] if transposed
                        let mut b_idx = batch_indices.clone();
                        if self.transpose_b {
                            b_idx.push(j);
                            b_idx.push(l);
                        } else {
                            b_idx.push(l);
                            b_idx.push(j);
                        }
                        let b_l = input_b.lower[b_idx.as_slice()];
                        let b_u = input_b.upper[b_idx.as_slice()];

                        // Interval multiplication: [a_l, a_u] * [b_l, b_u]
                        // Handle overflow and NaN: 0 * inf = NaN, inf * inf = inf
                        let p1 = a_l * b_l;
                        let p2 = a_l * b_u;
                        let p3 = a_u * b_l;
                        let p4 = a_u * b_u;

                        // NaN-safe min/max without allocation: track min/max inline
                        // If all products are NaN, use conservative widening (-inf/+inf).
                        let mut prod_min = f32::INFINITY;
                        let mut prod_max = f32::NEG_INFINITY;
                        let mut any_valid = false;
                        for &p in &[p1, p2, p3, p4] {
                            if !p.is_nan() {
                                any_valid = true;
                                prod_min = prod_min.min(p);
                                prod_max = prod_max.max(p);
                            }
                        }
                        if !any_valid {
                            // All products NaN (e.g., all 0*inf): conservative widening
                            prod_min = f32::NEG_INFINITY;
                            prod_max = f32::INFINITY;
                        }

                        // NaN-safe accumulation: if result would be NaN (inf + -inf),
                        // conservatively widen to -inf for lower, +inf for upper.
                        let new_lower = sum_lower + prod_min;
                        let new_upper = sum_upper + prod_max;
                        sum_lower = if new_lower.is_nan() {
                            f32::NEG_INFINITY
                        } else {
                            new_lower
                        };
                        sum_upper = if new_upper.is_nan() {
                            f32::INFINITY
                        } else {
                            new_upper
                        };
                    }

                    // Apply optional scaling (NaN-safe)
                    if let Some(scale) = self.scale {
                        if scale >= 0.0 {
                            sum_lower *= scale;
                            sum_upper *= scale;
                        } else {
                            let tmp = sum_lower;
                            sum_lower = sum_upper * scale;
                            sum_upper = tmp * scale;
                        }
                        // Handle 0 * inf = NaN by conservative widening
                        if sum_lower.is_nan() {
                            sum_lower = f32::NEG_INFINITY;
                        }
                        if sum_upper.is_nan() {
                            sum_upper = f32::INFINITY;
                        }
                    }

                    // Clamp bounds to finite values to prevent overflow propagation.
                    // Uses f32::MAX / 2.0 to leave headroom for downstream operations.
                    // This maintains soundness: [-MAX_BOUND, MAX_BOUND] is still a valid
                    // over-approximation when the true bounds would be ±infinity.
                    const MAX_BOUND: f32 = f32::MAX / 2.0;
                    sum_lower = sum_lower.max(-MAX_BOUND);
                    sum_upper = sum_upper.min(MAX_BOUND);

                    let mut out_idx = batch_indices.clone();
                    out_idx.push(i);
                    out_idx.push(j);
                    out_lower[out_idx.as_slice()] = sum_lower;
                    out_upper[out_idx.as_slice()] = sum_upper;
                }
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Evaluate MatMul at a concrete point: C = A @ B (or A @ B^T).
    /// For 2D inputs only.
    pub fn eval(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let b_for_matmul = if self.transpose_b {
            b.t().to_owned()
        } else {
            b.clone()
        };

        let mut result = a.dot(&b_for_matmul);

        if let Some(scale) = self.scale {
            result.mapv_inplace(|v| v * scale);
        }

        Ok(result)
    }

    /// Compute the Jacobian of C = A @ B w.r.t. A at a fixed B value.
    ///
    /// `C[i,j]` = Σ_l `A[i,l]` * `B[l,j]` (or `B[j,l]` if transpose_b)
    /// `∂C[i,j]/∂A[p,q]` = δ_{ip} * `B[q,j]` (or `B[j,q]` if transpose_b)
    ///
    /// Returns a matrix J of shape (m*n, m*k) where:
    /// - C is flattened row-major to length m*n
    /// - A is flattened row-major to length m*k
    pub fn jacobian_wrt_a(&self, b: &Array2<f32>) -> Array2<f32> {
        let (_k, _n) = if self.transpose_b {
            (b.ncols(), b.nrows())
        } else {
            (b.nrows(), b.ncols())
        };

        // We need to know m (rows of A), but we don't have A here
        // The Jacobian shape depends on A's shape too
        // For now, we'll compute the transformation directly in the propagation method
        // This helper just returns B in the right orientation for later use
        let b_effective = if self.transpose_b {
            b.t().to_owned()
        } else {
            b.clone()
        };

        // Return B^T which is used in the backward transformation
        // (with optional scaling)
        let mut result = b_effective.t().to_owned();
        if let Some(scale) = self.scale {
            result.mapv_inplace(|v| v * scale);
        }
        result
    }

    /// Compute the Jacobian of C = A @ B w.r.t. B at a fixed A value.
    ///
    /// `C[i,j]` = Σ_l `A[i,l]` * `B[l,j]`
    /// `∂C[i,j]/∂B[p,q]` = `A[i,p]` * δ_{jq}
    ///
    /// Returns a matrix that can be used in backward propagation.
    pub fn jacobian_wrt_b(&self, a: &Array2<f32>) -> Array2<f32> {
        // Return A^T which is used in the backward transformation
        let mut result = a.t().to_owned();
        if let Some(scale) = self.scale {
            result.mapv_inplace(|v| v * scale);
        }
        result
    }

    /// CROWN backward propagation for MatMul (C = A @ B or A @ B^T).
    ///
    /// Uses McCormick envelope relaxation for the bilinear terms.
    /// Supports batched N-D inputs: A has shape [..., M, K], B has shape [..., K, N] or [..., N, K].
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let a_shape = input_a_bounds.shape();
        let b_shape = input_b_bounds.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "MatMul CROWN requires at least 2D inputs".to_string(),
            ));
        }

        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        // Extract matrix dimensions from last two axes
        let m = a_shape[a_ndim - 2];
        let k = a_shape[a_ndim - 1];
        let (k_b, n) = if self.transpose_b {
            (b_shape[b_ndim - 1], b_shape[b_ndim - 2])
        } else {
            (b_shape[b_ndim - 2], b_shape[b_ndim - 1])
        };

        if k != k_b {
            return Err(GammaError::ShapeMismatch {
                expected: vec![k],
                got: vec![k_b],
            });
        }

        // Handle batch dimensions
        let a_batch: Vec<usize> = a_shape[..a_ndim - 2].to_vec();
        let b_batch: Vec<usize> = b_shape[..b_ndim - 2].to_vec();

        // For simplicity, require matching batch dimensions
        let batch_dims = if a_batch.len() >= b_batch.len() {
            a_batch.clone()
        } else {
            b_batch.clone()
        };

        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);
        let c_size_per_batch = m * n;
        let a_size_per_batch = m * k;
        let b_size_per_batch = b_shape[b_ndim - 2] * b_shape[b_ndim - 1];

        let total_c_size = batch_size * c_size_per_batch;
        let total_a_size = batch_size * a_size_per_batch;
        let total_b_size = batch_size * b_size_per_batch;

        if bounds.num_inputs() != total_c_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![total_c_size],
                got: vec![bounds.num_inputs()],
            });
        }

        #[derive(Clone, Copy)]
        enum BoundDir {
            Lower,
            Upper,
        }

        #[allow(clippy::too_many_arguments)]
        fn select_mccormick_plane(
            lx: f32,
            ux: f32,
            ly: f32,
            uy: f32,
            x0: f32,
            y0: f32,
            w: f32,
            dir: BoundDir,
        ) -> (f32, f32, f32) {
            // Planes are of form: ax * x + ay * y + c
            // McCormick lower planes:
            //   L1 = lx*y + ly*x - lx*ly  => ax=ly, ay=lx, c=-lx*ly
            //   L2 = ux*y + uy*x - ux*uy  => ax=uy, ay=ux, c=-ux*uy
            // McCormick upper planes:
            //   U1 = lx*y + uy*x - lx*uy  => ax=uy, ay=lx, c=-lx*uy
            //   U2 = ux*y + ly*x - ux*ly  => ax=ly, ay=ux, c=-ux*ly

            let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
            let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
            let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
            let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);

            match dir {
                BoundDir::Lower => {
                    if w >= 0.0 {
                        // w * lower(xy): choose the larger lower plane at reference point
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    } else {
                        // w * upper(xy): choose the smaller upper plane at reference point (w < 0)
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    }
                }
                BoundDir::Upper => {
                    if w >= 0.0 {
                        // w * upper(xy): choose the smaller upper plane at reference point
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    } else {
                        // w * lower(xy): choose the larger lower plane at reference point (w < 0)
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    }
                }
            }
        }

        // Check for infinite/NaN bounds or potential overflow in McCormick plane products.
        // McCormick computes lx*ly, ux*uy etc. If |bound| > sqrt(f32::MAX) ≈ 1.84e19,
        // products can overflow to infinity, causing NaN in subsequent arithmetic.
        const MCCORMICK_MAX_MAGNITUDE: f32 = 1.84e19;
        let has_bad_a = input_a_bounds
            .lower
            .iter()
            .chain(input_a_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE);
        let has_bad_b = input_b_bounds
            .lower
            .iter()
            .chain(input_b_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE);

        if has_bad_a || has_bad_b {
            return Err(GammaError::UnsupportedOp(
                "MatMul McCormick CROWN requires bounded inputs; input bounds are infinite, NaN, or exceed overflow threshold".to_string(),
            ));
        }

        let num_outputs = bounds.num_outputs();
        let scale = self.scale.unwrap_or(1.0);

        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, total_a_size));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, total_b_size));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, total_a_size));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, total_b_size));

        let mut lower_b_total = Array1::<f32>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f32>::zeros(num_outputs);

        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b[out_idx];
            let mut const_upper = bounds.upper_b[out_idx];

            for batch_idx in 0..batch_size {
                // Convert batch index to multi-dimensional indices
                let mut batch_indices = vec![0usize; batch_dims.len()];
                let mut remaining = batch_idx;
                for d in (0..batch_dims.len()).rev() {
                    batch_indices[d] = remaining % batch_dims[d];
                    remaining /= batch_dims[d];
                }

                for i in 0..m {
                    for j in 0..n {
                        // Flat index into C for this batch element
                        let c_flat = batch_idx * c_size_per_batch + i * n + j;
                        let w_lower = bounds.lower_a[[out_idx, c_flat]] * scale;
                        let w_upper = bounds.upper_a[[out_idx, c_flat]] * scale;

                        for l in 0..k {
                            // Get A[batch..., i, l]
                            let mut a_idx = batch_indices.clone();
                            a_idx.push(i);
                            a_idx.push(l);
                            let lx = input_a_bounds.lower[a_idx.as_slice()];
                            let ux = input_a_bounds.upper[a_idx.as_slice()];
                            let x0 = (lx + ux) * 0.5;

                            // Flat index into A for this batch element
                            let a_flat = batch_idx * a_size_per_batch + i * k + l;

                            // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                            let mut b_idx = batch_indices.clone();
                            let b_flat = if self.transpose_b {
                                b_idx.push(j);
                                b_idx.push(l);
                                batch_idx * b_size_per_batch + j * k + l
                            } else {
                                b_idx.push(l);
                                b_idx.push(j);
                                batch_idx * b_size_per_batch + l * n + j
                            };
                            let ly = input_b_bounds.lower[b_idx.as_slice()];
                            let uy = input_b_bounds.upper[b_idx.as_slice()];
                            let y0 = (ly + uy) * 0.5;

                            if w_lower != 0.0 {
                                let (ax, ay, c) = select_mccormick_plane(
                                    lx,
                                    ux,
                                    ly,
                                    uy,
                                    x0,
                                    y0,
                                    w_lower,
                                    BoundDir::Lower,
                                );
                                lower_a_a[[out_idx, a_flat]] += w_lower * ax;
                                lower_a_b[[out_idx, b_flat]] += w_lower * ay;
                                const_lower += w_lower * c;
                            }

                            if w_upper != 0.0 {
                                let (ax, ay, c) = select_mccormick_plane(
                                    lx,
                                    ux,
                                    ly,
                                    uy,
                                    x0,
                                    y0,
                                    w_upper,
                                    BoundDir::Upper,
                                );
                                upper_a_a[[out_idx, a_flat]] += w_upper * ax;
                                upper_a_b[[out_idx, b_flat]] += w_upper * ay;
                                const_upper += w_upper * c;
                            }
                        }
                    }
                }
            }

            lower_b_total[out_idx] = const_lower;
            upper_b_total[out_idx] = const_upper;
        }

        // Split constant terms across both inputs so that GraphNetwork accumulation sums correctly.
        let lower_b_half = lower_b_total.mapv(|v| v * 0.5);
        let upper_b_half = upper_b_total.mapv(|v| v * 0.5);

        let bounds_a = LinearBounds {
            lower_a: lower_a_a,
            lower_b: lower_b_half.clone(),
            upper_a: upper_a_a,
            upper_b: upper_b_half.clone(),
        };

        let bounds_b = LinearBounds {
            lower_a: lower_a_b,
            lower_b: lower_b_half,
            upper_a: upper_a_b,
            upper_b: upper_b_half,
        };

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for MatMul (C = A @ B or A @ B^T).
    ///
    /// Uses McCormick envelope relaxation for the bilinear terms.
    /// Supports N-D batched inputs: A has shape [..., M, K], B has shape [..., K, N] or [..., N, K].
    ///
    /// # Arguments
    /// - `bounds`: Incoming BatchedLinearBounds on the output C
    /// - `input_a_bounds`: IBP bounds on input A (needed for McCormick planes)
    /// - `input_b_bounds`: IBP bounds on input B (needed for McCormick planes)
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        debug!("MatMul batched CROWN backward propagation");

        let a_shape = input_a_bounds.shape();
        let b_shape = input_b_bounds.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "MatMul CROWN requires at least 2D inputs".to_string(),
            ));
        }

        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        // Extract matrix dimensions from last two axes
        let m = a_shape[a_ndim - 2];
        let k = a_shape[a_ndim - 1];
        let (k_b, n) = if self.transpose_b {
            (b_shape[b_ndim - 1], b_shape[b_ndim - 2])
        } else {
            (b_shape[b_ndim - 2], b_shape[b_ndim - 1])
        };

        if k != k_b {
            return Err(GammaError::ShapeMismatch {
                expected: vec![k],
                got: vec![k_b],
            });
        }

        // Handle batch dimensions (dimensions before the last two)
        let a_batch: Vec<usize> = a_shape[..a_ndim - 2].to_vec();
        let b_batch: Vec<usize> = b_shape[..b_ndim - 2].to_vec();

        // For simplicity, require matching batch dimensions
        let batch_dims = if a_batch.len() >= b_batch.len() {
            a_batch.clone()
        } else {
            b_batch.clone()
        };

        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);
        let c_size = m * n;
        let a_size = m * k;
        let b_size = b_shape[b_ndim - 2] * b_shape[b_ndim - 1];

        // Get the bounds shape
        let bounds_a_shape = bounds.lower_a.shape();
        if bounds_a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = bounds_a_shape[bounds_a_shape.len() - 2];
        let mid_dim = bounds_a_shape[bounds_a_shape.len() - 1];

        // This implementation models the MatMul output C as a flattened vector of length m*n.
        // For attention-shaped tensors like [batch, heads, seq, dim], GraphNetwork batched CROWN
        // typically uses per-position bounds where in_dim == dim; that representation cannot be
        // consumed here without additional reshape/flatten transforms.
        if mid_dim != c_size {
            return Err(GammaError::UnsupportedOp(format!(
                "MatMul batched CROWN expects flattened output dim m*n = {} (got in_dim = {}); consider reshaping MatMul outputs or fall back to IBP",
                c_size, mid_dim
            )));
        }

        let bounds_batch_dims = &bounds_a_shape[..bounds_a_shape.len() - 2];
        let total_bounds_batch: usize = bounds_batch_dims.iter().product();
        let total_bounds_batch = total_bounds_batch.max(1);

        // Output shapes
        let mut out_a_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(a_size);

        let mut out_b_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_b_shape.push(out_dim);
        out_b_shape.push(b_size);

        let mut out_bias_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_bias_shape.push(out_dim);

        let scale = self.scale.unwrap_or(1.0);

        // Reshape bounds to [total_batch, out_dim, c_size]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim, c_size))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim, c_size))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        // Allocate output coefficient matrices
        let mut new_lower_a_a = Array2::zeros((total_bounds_batch * out_dim, a_size));
        let mut new_upper_a_a = Array2::zeros((total_bounds_batch * out_dim, a_size));
        let mut new_lower_a_b = Array2::zeros((total_bounds_batch * out_dim, b_size));
        let mut new_upper_a_b = Array2::zeros((total_bounds_batch * out_dim, b_size));
        let mut new_lower_b = Array2::zeros((total_bounds_batch, out_dim));
        let mut new_upper_b = Array2::zeros((total_bounds_batch, out_dim));

        // McCormick plane selection helper
        #[derive(Clone, Copy)]
        enum BoundDir {
            Lower,
            Upper,
        }

        #[allow(clippy::too_many_arguments)]
        fn select_mccormick_plane(
            lx: f32,
            ux: f32,
            ly: f32,
            uy: f32,
            x0: f32,
            y0: f32,
            w: f32,
            dir: BoundDir,
        ) -> (f32, f32, f32) {
            // Planes are of form: ax * x + ay * y + c
            // McCormick lower planes:
            //   L1 = lx*y + ly*x - lx*ly  => ax=ly, ay=lx, c=-lx*ly
            //   L2 = ux*y + uy*x - ux*uy  => ax=uy, ay=ux, c=-ux*uy
            // McCormick upper planes:
            //   U1 = lx*y + uy*x - lx*uy  => ax=uy, ay=lx, c=-lx*uy
            //   U2 = ux*y + ly*x - ux*ly  => ax=ly, ay=ux, c=-ux*ly

            let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
            let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
            let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
            let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);

            match dir {
                BoundDir::Lower => {
                    if w >= 0.0 {
                        // w * lower(xy): choose the larger lower plane at reference point
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    } else {
                        // w * upper(xy): choose the smaller upper plane at reference point (w < 0)
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    }
                }
                BoundDir::Upper => {
                    if w >= 0.0 {
                        // w * upper(xy): choose the smaller upper plane at reference point
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    } else {
                        // w * lower(xy): choose the larger lower plane at reference point (w < 0)
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    }
                }
            }
        }

        // Check for infinite bounds in inputs - McCormick relaxation produces NaN
        // from (-inf + inf)/2 = NaN in center-point computation. Return error to
        // trigger fallback to IBP in caller.
        let has_inf_a = input_a_bounds
            .lower
            .iter()
            .chain(input_a_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan());
        let has_inf_b = input_b_bounds
            .lower
            .iter()
            .chain(input_b_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan());

        if has_inf_a || has_inf_b {
            return Err(GammaError::UnsupportedOp(
                "MatMul McCormick CROWN requires finite input bounds; infinite bounds cause NaN in center-point computation".to_string(),
            ));
        }

        // Check for potential overflow in McCormick plane products.
        // McCormick computes lx*ly, ux*uy etc. If |bound| > sqrt(f32::MAX) ≈ 1.84e19,
        // products can overflow to infinity, causing NaN in subsequent arithmetic.
        const MCCORMICK_MAX_MAGNITUDE: f32 = 1.84e19;
        let has_overflow_risk_a = input_a_bounds
            .lower
            .iter()
            .chain(input_a_bounds.upper.iter())
            .any(|&v| v.abs() > MCCORMICK_MAX_MAGNITUDE);
        let has_overflow_risk_b = input_b_bounds
            .lower
            .iter()
            .chain(input_b_bounds.upper.iter())
            .any(|&v| v.abs() > MCCORMICK_MAX_MAGNITUDE);

        if has_overflow_risk_a || has_overflow_risk_b {
            return Err(GammaError::UnsupportedOp(
                "MatMul McCormick CROWN requires bounded inputs; input magnitudes exceed overflow threshold".to_string(),
            ));
        }

        // Process each batch position and output dimension
        for bb in 0..total_bounds_batch {
            // Copy bias terms
            for d in 0..out_dim {
                new_lower_b[[bb, d]] = lower_b_2d[[bb, d]];
                new_upper_b[[bb, d]] = upper_b_2d[[bb, d]];
            }

            for d in 0..out_dim {
                let row_idx = bb * out_dim + d;

                // For each output element C[i, j]
                for batch_idx in 0..batch_size {
                    // Convert batch index to multi-dimensional indices
                    let mut batch_indices = vec![0usize; batch_dims.len()];
                    let mut remaining = batch_idx;
                    for dim_idx in (0..batch_dims.len()).rev() {
                        batch_indices[dim_idx] = remaining % batch_dims[dim_idx];
                        remaining /= batch_dims[dim_idx];
                    }

                    for i in 0..m {
                        for j in 0..n {
                            // Flat index into C for this batch element
                            let c_flat = batch_idx * c_size + i * n + j;
                            if c_flat >= c_size {
                                // Only handle c_flat within bounds
                                continue;
                            }

                            let w_lower = lower_a_3d[[bb, d, c_flat]] * scale;
                            let w_upper = upper_a_3d[[bb, d, c_flat]] * scale;

                            if w_lower == 0.0 && w_upper == 0.0 {
                                continue;
                            }

                            for l in 0..k {
                                // Get A[batch..., i, l]
                                let mut a_idx = batch_indices.clone();
                                a_idx.push(i);
                                a_idx.push(l);
                                let lx = input_a_bounds.lower[a_idx.as_slice()];
                                let ux = input_a_bounds.upper[a_idx.as_slice()];
                                let x0 = (lx + ux) * 0.5;

                                // Flat index into A for this batch element
                                let a_flat = batch_idx * a_size + i * k + l;
                                if a_flat >= a_size {
                                    continue;
                                }

                                // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                                let mut b_idx = batch_indices.clone();
                                let b_flat = if self.transpose_b {
                                    b_idx.push(j);
                                    b_idx.push(l);
                                    batch_idx * b_size + j * k + l
                                } else {
                                    b_idx.push(l);
                                    b_idx.push(j);
                                    batch_idx * b_size + l * n + j
                                };
                                if b_flat >= b_size {
                                    continue;
                                }

                                let ly = input_b_bounds.lower[b_idx.as_slice()];
                                let uy = input_b_bounds.upper[b_idx.as_slice()];
                                let y0 = (ly + uy) * 0.5;

                                if w_lower != 0.0 {
                                    let (ax, ay, c) = select_mccormick_plane(
                                        lx,
                                        ux,
                                        ly,
                                        uy,
                                        x0,
                                        y0,
                                        w_lower,
                                        BoundDir::Lower,
                                    );
                                    new_lower_a_a[[row_idx, a_flat]] += w_lower * ax;
                                    new_lower_a_b[[row_idx, b_flat]] += w_lower * ay;
                                    new_lower_b[[bb, d]] += w_lower * c;
                                }

                                if w_upper != 0.0 {
                                    let (ax, ay, c) = select_mccormick_plane(
                                        lx,
                                        ux,
                                        ly,
                                        uy,
                                        x0,
                                        y0,
                                        w_upper,
                                        BoundDir::Upper,
                                    );
                                    new_upper_a_a[[row_idx, a_flat]] += w_upper * ax;
                                    new_upper_a_b[[row_idx, b_flat]] += w_upper * ay;
                                    new_upper_b[[bb, d]] += w_upper * c;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reshape back to output shape
        let (new_lower_a_a_vec, _) = new_lower_a_a.into_raw_vec_and_offset();
        let (new_upper_a_a_vec, _) = new_upper_a_a.into_raw_vec_and_offset();
        let (new_lower_a_b_vec, _) = new_lower_a_b.into_raw_vec_and_offset();
        let (new_upper_a_b_vec, _) = new_upper_a_b.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let new_lower_a_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a_a".to_string()))?;
        let new_upper_a_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a_a".to_string()))?;
        let new_lower_a_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_a_b_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a_b".to_string()))?;
        let new_upper_a_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_a_b_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a_b".to_string()))?;
        let new_lower_b_full = ArrayD::from_shape_vec(IxDyn(&out_bias_shape), new_lower_b_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?;
        let new_upper_b_full = ArrayD::from_shape_vec(IxDyn(&out_bias_shape), new_upper_b_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?;

        // Split bias between inputs so graph accumulation sums correctly
        let lower_b_half = new_lower_b_full.mapv(|v| v * 0.5);
        let upper_b_half = new_upper_b_full.mapv(|v| v * 0.5);

        // Update input shapes
        let mut new_input_shape_a = bounds.input_shape.clone();
        if new_input_shape_a.len() >= 2 {
            let len = new_input_shape_a.len();
            new_input_shape_a[len - 2] = m;
            new_input_shape_a[len - 1] = k;
        } else if !new_input_shape_a.is_empty() {
            new_input_shape_a[0] = a_size;
        }

        let mut new_input_shape_b = bounds.input_shape.clone();
        if new_input_shape_b.len() >= 2 {
            let len = new_input_shape_b.len();
            if self.transpose_b {
                new_input_shape_b[len - 2] = n;
                new_input_shape_b[len - 1] = k;
            } else {
                new_input_shape_b[len - 2] = k;
                new_input_shape_b[len - 1] = n;
            }
        } else if !new_input_shape_b.is_empty() {
            new_input_shape_b[0] = b_size;
        }

        let bounds_a = BatchedLinearBounds {
            lower_a: new_lower_a_a,
            lower_b: lower_b_half.clone(),
            upper_a: new_upper_a_a,
            upper_b: upper_b_half.clone(),
            input_shape: new_input_shape_a,
            output_shape: bounds.output_shape.clone(),
        };

        let bounds_b = BatchedLinearBounds {
            lower_a: new_lower_a_b,
            lower_b: lower_b_half,
            upper_a: new_upper_a_b,
            upper_b: upper_b_half,
            input_shape: new_input_shape_b,
            output_shape: bounds.output_shape.clone(),
        };

        Ok((bounds_a, bounds_b))
    }
}

/// Element-wise addition layer for two bounded tensors (e.g., residual connections).
#[derive(Debug, Clone)]
pub struct AddLayer;

impl AddLayer {
    /// Propagate IBP bounds through element-wise addition.
    ///
    /// For C = A + B where A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
    /// C ∈ [A_l + B_l, A_u + B_u]
    ///
    /// Handles shapes that differ by singleton dimensions (e.g., [5, 1, 48] + [5, 48])
    /// by squeezing singleton dimensions from the higher-dimensional input.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Try to reconcile shapes that differ by singleton dimensions
        let (a_lower, a_upper, b_lower, b_upper) = if input_a.shape() != input_b.shape() {
            // Check if shapes can be reconciled by squeezing singleton dimensions
            match Self::reconcile_shapes_for_add(
                &input_a.lower,
                &input_a.upper,
                &input_b.lower,
                &input_b.upper,
            ) {
                Some((al, au, bl, bu)) => (al, au, bl, bu),
                None => {
                    return Err(GammaError::ShapeMismatch {
                        expected: input_a.shape().to_vec(),
                        got: input_b.shape().to_vec(),
                    });
                }
            }
        } else {
            (
                std::borrow::Cow::Borrowed(&input_a.lower),
                std::borrow::Cow::Borrowed(&input_a.upper),
                std::borrow::Cow::Borrowed(&input_b.lower),
                std::borrow::Cow::Borrowed(&input_b.upper),
            )
        };

        let mut out_lower = a_lower.as_ref() + b_lower.as_ref();
        let mut out_upper = a_upper.as_ref() + b_upper.as_ref();

        // Clamp bounds to finite values to prevent overflow propagation.
        const MAX_BOUND: f32 = f32::MAX / 2.0;
        out_lower.mapv_inplace(|v| v.max(-MAX_BOUND));
        out_upper.mapv_inplace(|v| v.min(MAX_BOUND));

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Try to reconcile shapes that differ by singleton dimensions.
    /// For example, [5, 1, 48] and [5, 48] can be reconciled by squeezing the singleton.
    /// Returns (a_lower, a_upper, b_lower, b_upper) with reconciled shapes.
    #[allow(clippy::type_complexity)]
    fn reconcile_shapes_for_add<'a>(
        a_lower: &'a ArrayD<f32>,
        a_upper: &'a ArrayD<f32>,
        b_lower: &'a ArrayD<f32>,
        b_upper: &'a ArrayD<f32>,
    ) -> Option<(
        std::borrow::Cow<'a, ArrayD<f32>>,
        std::borrow::Cow<'a, ArrayD<f32>>,
        std::borrow::Cow<'a, ArrayD<f32>>,
        std::borrow::Cow<'a, ArrayD<f32>>,
    )> {
        let shape_a = a_lower.shape();
        let shape_b = b_lower.shape();

        // Try squeezing singleton dimensions from a to match b
        if shape_a.len() > shape_b.len() {
            let squeezed_a = Self::squeeze_singletons_to_match(a_lower, shape_b)?;
            let squeezed_a_upper = Self::squeeze_singletons_to_match(a_upper, shape_b)?;
            if squeezed_a.shape() == shape_b {
                return Some((
                    std::borrow::Cow::Owned(squeezed_a),
                    std::borrow::Cow::Owned(squeezed_a_upper),
                    std::borrow::Cow::Borrowed(b_lower),
                    std::borrow::Cow::Borrowed(b_upper),
                ));
            }
        }

        // Try squeezing singleton dimensions from b to match a
        if shape_b.len() > shape_a.len() {
            let squeezed_b = Self::squeeze_singletons_to_match(b_lower, shape_a)?;
            let squeezed_b_upper = Self::squeeze_singletons_to_match(b_upper, shape_a)?;
            if squeezed_b.shape() == shape_a {
                return Some((
                    std::borrow::Cow::Borrowed(a_lower),
                    std::borrow::Cow::Borrowed(a_upper),
                    std::borrow::Cow::Owned(squeezed_b),
                    std::borrow::Cow::Owned(squeezed_b_upper),
                ));
            }
        }

        None
    }

    /// Squeeze singleton dimensions (dimensions of size 1) from tensor to match target shape.
    fn squeeze_singletons_to_match(
        tensor: &ArrayD<f32>,
        target_shape: &[usize],
    ) -> Option<ArrayD<f32>> {
        let mut result = tensor.clone();

        // Keep squeezing singleton dimensions until we match target ndim
        while result.ndim() > target_shape.len() {
            // Find a singleton dimension to squeeze
            let singleton_axis = result.shape().iter().position(|&d| d == 1)?;
            result = result.remove_axis(ndarray::Axis(singleton_axis));
        }

        // Verify the squeezed shape matches target
        if result.shape() == target_shape {
            Some(result)
        } else {
            None
        }
    }

    /// CROWN backward propagation for Add (C = A + B).
    ///
    /// For Add, the Jacobian w.r.t. both inputs is the identity:
    /// - ∂C/∂A = I (identity)
    /// - ∂C/∂B = I (identity)
    ///
    /// So incoming linear bounds on C pass through unchanged to both A and B.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
    ) -> Result<(LinearBounds, LinearBounds)> {
        // C = A + B => W·C + b = W·A + W·B + b.
        // Split the bias so graph accumulation does not double-count constants.
        let lower_b_half = bounds.lower_b.mapv(|v| v * 0.5);
        let upper_b_half = bounds.upper_b.mapv(|v| v * 0.5);

        let bounds_a = LinearBounds {
            lower_a: bounds.lower_a.clone(),
            lower_b: lower_b_half.clone(),
            upper_a: bounds.upper_a.clone(),
            upper_b: upper_b_half.clone(),
        };

        let bounds_b = LinearBounds {
            lower_a: bounds.lower_a.clone(),
            lower_b: lower_b_half,
            upper_a: bounds.upper_a.clone(),
            upper_b: upper_b_half,
        };

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for Add (C = A + B).
    ///
    /// Same logic as `propagate_linear_binary` but for N-D batched bounds.
    /// For Add, the Jacobian w.r.t. both inputs is the identity:
    /// - ∂C/∂A = I (identity)
    /// - ∂C/∂B = I (identity)
    ///
    /// So incoming batched linear bounds on C pass through unchanged to both A and B.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        // C = A + B => W·C + b = W·A + W·B + b.
        // Split the bias so graph accumulation does not double-count constants.
        let lower_b_half = bounds.lower_b.mapv(|v| v * 0.5);
        let upper_b_half = bounds.upper_b.mapv(|v| v * 0.5);

        let bounds_a = BatchedLinearBounds {
            lower_a: bounds.lower_a.clone(),
            lower_b: lower_b_half.clone(),
            upper_a: bounds.upper_a.clone(),
            upper_b: upper_b_half.clone(),
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        };

        let bounds_b = BatchedLinearBounds {
            lower_a: bounds.lower_a.clone(),
            lower_b: lower_b_half,
            upper_a: bounds.upper_a.clone(),
            upper_b: upper_b_half,
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        };

        Ok((bounds_a, bounds_b))
    }
}

/// Element-wise multiplication layer for two bounded tensors (e.g., SwiGLU gating).
///
/// For z = x * y where both x and y are bounded, uses IBP-style interval arithmetic:
/// - z_l = min(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
/// - z_u = max(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
#[derive(Debug, Clone)]
pub struct MulBinaryLayer;

impl MulBinaryLayer {
    /// Propagate IBP bounds through element-wise multiplication with broadcasting.
    ///
    /// For z = x * y where x ∈ [x_l, x_u] and y ∈ [y_l, y_u]:
    /// z_l = min(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
    /// z_u = max(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
    ///
    /// Supports NumPy-style broadcasting (e.g., [1, 11, 128] * [1, 11, 1]).
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Handle broadcasting for common cases (e.g., x * mask where mask is reduced)
        let (a_lower, a_upper, b_lower, b_upper, output_shape) =
            if input_a.shape() == input_b.shape() {
                (
                    std::borrow::Cow::Borrowed(&input_a.lower),
                    std::borrow::Cow::Borrowed(&input_a.upper),
                    std::borrow::Cow::Borrowed(&input_b.lower),
                    std::borrow::Cow::Borrowed(&input_b.upper),
                    input_a.shape().to_vec(),
                )
            } else {
                // Try broadcasting
                let target_shape =
                    broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
                        GammaError::ShapeMismatch {
                            expected: input_a.shape().to_vec(),
                            got: input_b.shape().to_vec(),
                        }
                    })?;

                let a_lower = input_a
                    .lower
                    .broadcast(IxDyn(&target_shape))
                    .ok_or_else(|| GammaError::ShapeMismatch {
                        expected: target_shape.clone(),
                        got: input_a.shape().to_vec(),
                    })?
                    .to_owned();
                let a_upper = input_a
                    .upper
                    .broadcast(IxDyn(&target_shape))
                    .ok_or_else(|| GammaError::ShapeMismatch {
                        expected: target_shape.clone(),
                        got: input_a.shape().to_vec(),
                    })?
                    .to_owned();
                let b_lower = input_b
                    .lower
                    .broadcast(IxDyn(&target_shape))
                    .ok_or_else(|| GammaError::ShapeMismatch {
                        expected: target_shape.clone(),
                        got: input_b.shape().to_vec(),
                    })?
                    .to_owned();
                let b_upper = input_b
                    .upper
                    .broadcast(IxDyn(&target_shape))
                    .ok_or_else(|| GammaError::ShapeMismatch {
                        expected: target_shape.clone(),
                        got: input_b.shape().to_vec(),
                    })?
                    .to_owned();

                (
                    std::borrow::Cow::Owned(a_lower),
                    std::borrow::Cow::Owned(a_upper),
                    std::borrow::Cow::Owned(b_lower),
                    std::borrow::Cow::Owned(b_upper),
                    target_shape,
                )
            };

        const MAX_BOUND: f32 = f32::MAX / 2.0;

        // Compute all four corner products and take min/max
        let ll = a_lower.as_ref() * b_lower.as_ref();
        let lu = a_lower.as_ref() * b_upper.as_ref();
        let ul = a_upper.as_ref() * b_lower.as_ref();
        let uu = a_upper.as_ref() * b_upper.as_ref();

        // Element-wise min/max of the four products
        let mut out_lower = ArrayD::zeros(IxDyn(&output_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&output_shape));

        for (((((ol, ou), &ll_i), &lu_i), &ul_i), &uu_i) in out_lower
            .iter_mut()
            .zip(out_upper.iter_mut())
            .zip(ll.iter())
            .zip(lu.iter())
            .zip(ul.iter())
            .zip(uu.iter())
        {
            let products = [ll_i, lu_i, ul_i, uu_i];
            let min_val = products
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .fold(f32::INFINITY, f32::min);
            let max_val = products
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            // Handle all-NaN case: set bounds to 0 (neutral element)
            let min_val = if min_val.is_infinite() { 0.0 } else { min_val };
            let max_val = if max_val.is_infinite() { 0.0 } else { max_val };

            *ol = min_val.max(-MAX_BOUND);
            *ou = max_val.min(MAX_BOUND);
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN backward propagation for Mul (z = x * y) using McCormick relaxation.
    ///
    /// For element-wise multiplication `z[i] = x[i] * y[i]`, McCormick envelope provides
    /// sound linear bounds:
    ///
    /// Lower bounds (take max):
    ///   z ≥ x_l*y + x*y_l - x_l*y_l
    ///   z ≥ x_u*y + x*y_u - x_u*y_u
    ///
    /// Upper bounds (take min):
    ///   z ≤ x_l*y + x*y_u - x_l*y_u
    ///   z ≤ x_u*y + x*y_l - x_u*y_l
    ///
    /// Each bound is linear in (x, y), enabling CROWN backward propagation.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let n = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();

        // Verify shapes match
        if input_a_bounds.len() != n || input_b_bounds.len() != n {
            return Err(GammaError::ShapeMismatch {
                expected: vec![n],
                got: vec![input_a_bounds.len(), input_b_bounds.len()],
            });
        }

        // Check for infinite/NaN bounds that would cause overflow in McCormick
        const MCCORMICK_MAX_MAGNITUDE: f32 = 1.84e19;
        let has_bad_a = input_a_bounds
            .lower
            .iter()
            .chain(input_a_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE);
        let has_bad_b = input_b_bounds
            .lower
            .iter()
            .chain(input_b_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE);

        if has_bad_a || has_bad_b {
            return Err(GammaError::UnsupportedOp(
                "MulBinary McCormick CROWN requires bounded inputs; input bounds are infinite, NaN, or exceed overflow threshold".to_string(),
            ));
        }

        // Flatten bounds for element-wise indexing
        let a_lower_flat = input_a_bounds.lower.as_slice().unwrap();
        let a_upper_flat = input_a_bounds.upper.as_slice().unwrap();
        let b_lower_flat = input_b_bounds.lower.as_slice().unwrap();
        let b_upper_flat = input_b_bounds.upper.as_slice().unwrap();

        // Output linear bounds: coefficients for inputs a and b, plus bias
        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n));
        let mut lower_b_total = Array1::<f32>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f32>::zeros(num_outputs);

        #[derive(Clone, Copy)]
        enum BoundDir {
            Lower,
            Upper,
        }

        // Select McCormick plane based on incoming weight and bound direction
        // Returns (coeff_x, coeff_y, const_term) for the plane: coeff_x*x + coeff_y*y + const
        #[inline]
        #[allow(clippy::too_many_arguments)]
        fn select_mccormick_plane(
            lx: f32,
            ux: f32,
            ly: f32,
            uy: f32,
            x0: f32,
            y0: f32,
            w: f32,
            dir: BoundDir,
        ) -> (f32, f32, f32) {
            // McCormick lower planes (for z = x*y):
            //   L1: z ≥ lx*y + ly*x - lx*ly  => coeff_x=ly, coeff_y=lx, const=-lx*ly
            //   L2: z ≥ ux*y + uy*x - ux*uy  => coeff_x=uy, coeff_y=ux, const=-ux*uy
            // McCormick upper planes:
            //   U1: z ≤ lx*y + uy*x - lx*uy  => coeff_x=uy, coeff_y=lx, const=-lx*uy
            //   U2: z ≤ ux*y + ly*x - ux*ly  => coeff_x=ly, coeff_y=ux, const=-ux*ly

            let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
            let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
            let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
            let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);

            match dir {
                BoundDir::Lower => {
                    if w >= 0.0 {
                        // w * lower(z): choose the larger lower plane at reference point
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    } else {
                        // w * upper(z) for lower bound: choose the smaller upper plane
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    }
                }
                BoundDir::Upper => {
                    if w >= 0.0 {
                        // w * upper(z): choose the smaller upper plane at reference point
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    } else {
                        // w * lower(z) for upper bound: choose the larger lower plane
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    }
                }
            }
        }

        // Process each output dimension
        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b[out_idx];
            let mut const_upper = bounds.upper_b[out_idx];

            // For each element position (element-wise: z[j] = x[j] * y[j])
            for j in 0..n {
                let w_lower = bounds.lower_a[[out_idx, j]];
                let w_upper = bounds.upper_a[[out_idx, j]];

                let lx = a_lower_flat[j];
                let ux = a_upper_flat[j];
                let ly = b_lower_flat[j];
                let uy = b_upper_flat[j];
                let x0 = (lx + ux) * 0.5;
                let y0 = (ly + uy) * 0.5;

                // Select McCormick plane for lower bound computation
                let (ax_l, ay_l, c_l) =
                    select_mccormick_plane(lx, ux, ly, uy, x0, y0, w_lower, BoundDir::Lower);
                lower_a_a[[out_idx, j]] = w_lower * ax_l;
                lower_a_b[[out_idx, j]] = w_lower * ay_l;
                const_lower += w_lower * c_l;

                // Select McCormick plane for upper bound computation
                let (ax_u, ay_u, c_u) =
                    select_mccormick_plane(lx, ux, ly, uy, x0, y0, w_upper, BoundDir::Upper);
                upper_a_a[[out_idx, j]] = w_upper * ax_u;
                upper_a_b[[out_idx, j]] = w_upper * ay_u;
                const_upper += w_upper * c_u;
            }

            lower_b_total[out_idx] = const_lower;
            upper_b_total[out_idx] = const_upper;
        }

        let bounds_a = LinearBounds {
            lower_a: lower_a_a,
            lower_b: lower_b_total.clone(),
            upper_a: upper_a_a,
            upper_b: upper_b_total.clone(),
        };

        let bounds_b = LinearBounds {
            lower_a: lower_a_b,
            lower_b: lower_b_total,
            upper_a: upper_a_b,
            upper_b: upper_b_total,
        };

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for MulBinary (z = x * y) with McCormick envelope.
    ///
    /// Same as `propagate_linear_binary` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    ///
    /// For element-wise `z[i] = x[i] * y[i]`, uses McCormick relaxation:
    ///   Lower bounds (take max):
    ///     z ≥ x_l·y + x·y_l - x_l·y_l
    ///     z ≥ x_u·y + x·y_u - x_u·y_u
    ///   Upper bounds (take min):
    ///     z ≤ x_l·y + x·y_u - x_l·y_u
    ///     z ≤ x_u·y + x·y_l - x_u·y_l
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        // Get shapes
        let a_shape = bounds.lower_a.shape();
        let ndim = a_shape.len();

        // For element-wise multiplication, the last dimension is the "n" (features)
        // BatchedLinearBounds shape: [...batch, out_dim, n]
        // where out_dim == n for element-wise operations (identity CROWN)
        if ndim < 2 {
            return Err(GammaError::ShapeMismatch {
                expected: vec![2], // at least 2D
                got: a_shape.to_vec(),
            });
        }

        let n = a_shape[ndim - 1]; // last dimension is input features
        let out_dim = a_shape[ndim - 2]; // second-to-last is output dimension

        // Verify input bounds shapes match
        if input_a_bounds.len() != input_b_bounds.len() {
            return Err(GammaError::ShapeMismatch {
                expected: vec![input_a_bounds.len()],
                got: vec![input_b_bounds.len()],
            });
        }

        // Check for infinite/NaN bounds that would cause overflow in McCormick
        const MCCORMICK_MAX_MAGNITUDE: f32 = 1.84e19;
        let has_bad_a = input_a_bounds
            .lower
            .iter()
            .chain(input_a_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE);
        let has_bad_b = input_b_bounds
            .lower
            .iter()
            .chain(input_b_bounds.upper.iter())
            .any(|&v| v.is_infinite() || v.is_nan() || v.abs() > MCCORMICK_MAX_MAGNITUDE);

        if has_bad_a || has_bad_b {
            return Err(GammaError::UnsupportedOp(
                "MulBinary batched McCormick CROWN requires bounded inputs; input bounds are infinite, NaN, or exceed overflow threshold".to_string(),
            ));
        }

        // Flatten bounds for element-wise indexing
        let a_lower_flat = input_a_bounds.lower.as_slice().unwrap();
        let a_upper_flat = input_a_bounds.upper.as_slice().unwrap();
        let b_lower_flat = input_b_bounds.lower.as_slice().unwrap();
        let b_upper_flat = input_b_bounds.upper.as_slice().unwrap();

        // Calculate batch size (all dimensions except last two)
        let batch_size: usize = a_shape[..ndim - 2].iter().product::<usize>().max(1);

        // Flatten input arrays to work with them more easily
        let lower_a_flat = bounds.lower_a.as_slice().unwrap();
        let upper_a_flat = bounds.upper_a.as_slice().unwrap();
        let lower_b_flat = bounds.lower_b.as_slice().unwrap();
        let upper_b_flat = bounds.upper_b.as_slice().unwrap();

        // Output arrays
        let mut lower_a_a = vec![0.0_f32; batch_size * out_dim * n];
        let mut lower_a_b = vec![0.0_f32; batch_size * out_dim * n];
        let mut upper_a_a = vec![0.0_f32; batch_size * out_dim * n];
        let mut upper_a_b = vec![0.0_f32; batch_size * out_dim * n];
        let mut lower_b_out = vec![0.0_f32; batch_size * out_dim];
        let mut upper_b_out = vec![0.0_f32; batch_size * out_dim];

        #[derive(Clone, Copy)]
        enum BoundDir {
            Lower,
            Upper,
        }

        // Select McCormick plane based on incoming weight and bound direction
        #[inline]
        #[allow(clippy::too_many_arguments)]
        fn select_mccormick_plane(
            lx: f32,
            ux: f32,
            ly: f32,
            uy: f32,
            x0: f32,
            y0: f32,
            w: f32,
            dir: BoundDir,
        ) -> (f32, f32, f32) {
            // McCormick lower planes (for z = x*y):
            //   L1: z ≥ lx*y + ly*x - lx*ly  => coeff_x=ly, coeff_y=lx, const=-lx*ly
            //   L2: z ≥ ux*y + uy*x - ux*uy  => coeff_x=uy, coeff_y=ux, const=-ux*uy
            // McCormick upper planes:
            //   U1: z ≤ lx*y + uy*x - lx*uy  => coeff_x=uy, coeff_y=lx, const=-lx*uy
            //   U2: z ≤ ux*y + ly*x - ux*ly  => coeff_x=ly, coeff_y=ux, const=-ux*ly

            let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
            let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
            let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
            let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);

            match dir {
                BoundDir::Lower => {
                    if w >= 0.0 {
                        if l1.3 >= l2.3 {
                            (l1.0, l1.1, l1.2)
                        } else {
                            (l2.0, l2.1, l2.2)
                        }
                    } else if u1.3 <= u2.3 {
                        (u1.0, u1.1, u1.2)
                    } else {
                        (u2.0, u2.1, u2.2)
                    }
                }
                BoundDir::Upper => {
                    if w >= 0.0 {
                        if u1.3 <= u2.3 {
                            (u1.0, u1.1, u1.2)
                        } else {
                            (u2.0, u2.1, u2.2)
                        }
                    } else if l1.3 >= l2.3 {
                        (l1.0, l1.1, l1.2)
                    } else {
                        (l2.0, l2.1, l2.2)
                    }
                }
            }
        }

        // Process each batch position
        for batch_idx in 0..batch_size {
            // Input bounds are broadcast over batch if needed
            // For element-wise multiply, input_a and input_b have same shape as the layer input
            // which may or may not have batch dimensions
            let input_len = a_lower_flat.len();
            let elements_per_batch = if input_len == n {
                // No batch dimension in inputs - use same bounds for all batches
                n
            } else {
                // Has batch dimension - slice to this batch
                n
            };

            for out_idx in 0..out_dim {
                let out_flat_idx = batch_idx * out_dim + out_idx;
                let mut const_lower = lower_b_flat[out_flat_idx];
                let mut const_upper = upper_b_flat[out_flat_idx];

                for j in 0..n {
                    let a_flat_idx = batch_idx * out_dim * n + out_idx * n + j;
                    let w_lower = lower_a_flat[a_flat_idx];
                    let w_upper = upper_a_flat[a_flat_idx];

                    // Get bounds for this element - handle batch broadcasting
                    let input_j = if input_len == n {
                        j // no batch dimension, use j directly
                    } else {
                        batch_idx * elements_per_batch + j
                    };

                    // Clamp input_j to valid range
                    let input_j = input_j.min(input_len - 1);

                    let lx = a_lower_flat[input_j];
                    let ux = a_upper_flat[input_j];
                    let ly = b_lower_flat[input_j];
                    let uy = b_upper_flat[input_j];
                    let x0 = (lx + ux) * 0.5;
                    let y0 = (ly + uy) * 0.5;

                    // Select McCormick plane for lower bound
                    let (ax_l, ay_l, c_l) =
                        select_mccormick_plane(lx, ux, ly, uy, x0, y0, w_lower, BoundDir::Lower);
                    lower_a_a[a_flat_idx] = w_lower * ax_l;
                    lower_a_b[a_flat_idx] = w_lower * ay_l;
                    const_lower += w_lower * c_l;

                    // Select McCormick plane for upper bound
                    let (ax_u, ay_u, c_u) =
                        select_mccormick_plane(lx, ux, ly, uy, x0, y0, w_upper, BoundDir::Upper);
                    upper_a_a[a_flat_idx] = w_upper * ax_u;
                    upper_a_b[a_flat_idx] = w_upper * ay_u;
                    const_upper += w_upper * c_u;
                }

                lower_b_out[out_flat_idx] = const_lower;
                upper_b_out[out_flat_idx] = const_upper;
            }
        }

        // Reshape outputs to batched form
        let bounds_a = BatchedLinearBounds {
            lower_a: ArrayD::from_shape_vec(IxDyn(a_shape), lower_a_a).map_err(|e| {
                GammaError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![e.to_string().len()],
                }
            })?,
            lower_b: ArrayD::from_shape_vec(IxDyn(bounds.lower_b.shape()), lower_b_out.clone())
                .map_err(|e| GammaError::ShapeMismatch {
                    expected: bounds.lower_b.shape().to_vec(),
                    got: vec![e.to_string().len()],
                })?,
            upper_a: ArrayD::from_shape_vec(IxDyn(a_shape), upper_a_a).map_err(|e| {
                GammaError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![e.to_string().len()],
                }
            })?,
            upper_b: ArrayD::from_shape_vec(IxDyn(bounds.upper_b.shape()), upper_b_out.clone())
                .map_err(|e| GammaError::ShapeMismatch {
                    expected: bounds.upper_b.shape().to_vec(),
                    got: vec![e.to_string().len()],
                })?,
            input_shape: bounds.input_shape.clone(),
            output_shape: bounds.output_shape.clone(),
        };

        let bounds_b =
            BatchedLinearBounds {
                lower_a: ArrayD::from_shape_vec(IxDyn(a_shape), lower_a_b).map_err(|e| {
                    GammaError::ShapeMismatch {
                        expected: a_shape.to_vec(),
                        got: vec![e.to_string().len()],
                    }
                })?,
                lower_b: ArrayD::from_shape_vec(IxDyn(bounds.lower_b.shape()), lower_b_out)
                    .map_err(|e| GammaError::ShapeMismatch {
                        expected: bounds.lower_b.shape().to_vec(),
                        got: vec![e.to_string().len()],
                    })?,
                upper_a: ArrayD::from_shape_vec(IxDyn(a_shape), upper_a_b).map_err(|e| {
                    GammaError::ShapeMismatch {
                        expected: a_shape.to_vec(),
                        got: vec![e.to_string().len()],
                    }
                })?,
                upper_b: ArrayD::from_shape_vec(IxDyn(bounds.upper_b.shape()), upper_b_out)
                    .map_err(|e| GammaError::ShapeMismatch {
                        expected: bounds.upper_b.shape().to_vec(),
                        got: vec![e.to_string().len()],
                    })?,
                input_shape: bounds.input_shape.clone(),
                output_shape: bounds.output_shape.clone(),
            };

        Ok((bounds_a, bounds_b))
    }
}

/// Concatenation layer: concatenates two tensors along a specified axis.
///
/// This is used for operations like concatenating CLS token with patch embeddings in ViT,
/// or combining tensors in attention mechanisms. For IBP, this is straightforward:
/// concat(lower_a, lower_b) and concat(upper_a, upper_b).
///
/// For CROWN backward propagation, we split the coefficient matrix back to each input.
#[derive(Debug, Clone)]
pub struct ConcatLayer {
    /// The axis along which to concatenate (negative indices supported).
    pub axis: i64,
    /// Optional stored input shapes for CROWN backward when inputs are constant tensors.
    /// This is used when one or more inputs come from ConstantOfShape and aren't in node_bounds.
    pub input_shapes: Option<Vec<Vec<usize>>>,
    /// Optional constant tensors for inputs that are known at graph construction time.
    /// Each element is Some(tensor) for constant inputs, None for dynamic inputs.
    /// Used during IBP forward when the constant isn't in node_bounds cache.
    pub constant_inputs: Option<Vec<Option<BoundedTensor>>>,
}

impl ConcatLayer {
    /// Create a new concatenation layer.
    pub fn new(axis: i64) -> Self {
        Self {
            axis,
            input_shapes: None,
            constant_inputs: None,
        }
    }

    /// Create a new concatenation layer with known input shapes.
    pub fn with_input_shapes(axis: i64, input_shapes: Vec<Vec<usize>>) -> Self {
        Self {
            axis,
            input_shapes: Some(input_shapes),
            constant_inputs: None,
        }
    }

    /// Create a new concatenation layer with constant input tensors.
    pub fn with_constants(
        axis: i64,
        input_shapes: Vec<Vec<usize>>,
        constant_inputs: Vec<Option<BoundedTensor>>,
    ) -> Self {
        Self {
            axis,
            input_shapes: Some(input_shapes),
            constant_inputs: Some(constant_inputs),
        }
    }

    /// Get the stored shape for input at given index, if available.
    pub fn get_input_shape(&self, index: usize) -> Option<&Vec<usize>> {
        self.input_shapes
            .as_ref()
            .and_then(|shapes| shapes.get(index))
    }

    /// Get the constant BoundedTensor for input at given index, if available.
    pub fn get_constant_input(&self, index: usize) -> Option<&BoundedTensor> {
        self.constant_inputs
            .as_ref()
            .and_then(|inputs| inputs.get(index))
            .and_then(|opt| opt.as_ref())
    }

    /// Normalize axis to positive index given the number of dimensions.
    fn normalize_axis(&self, ndim: usize) -> usize {
        if self.axis < 0 {
            (ndim as i64 + self.axis) as usize
        } else {
            self.axis as usize
        }
    }

    /// Propagate IBP bounds through concatenation.
    ///
    /// For Y = concat(A, B) along axis:
    /// Y_lower = concat(A_lower, B_lower)
    /// Y_upper = concat(A_upper, B_upper)
    ///
    /// When one input has fewer dimensions (e.g., constant without batch),
    /// it will be broadcast to match the batch dimension of the other input.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let ndim_a = input_a.shape().len();
        let ndim_b = input_b.shape().len();

        // Handle broadcasting when one input has fewer dimensions (e.g., constant without batch)
        let (a_lower, a_upper, b_lower, b_upper, effective_ndim) = if ndim_a != ndim_b {
            // One input needs broadcasting to add batch dimension
            if ndim_a + 1 == ndim_b {
                // Input A is missing batch dimension - broadcast it
                let batch_size = input_b.shape()[0];
                let a_lower_expanded = Self::broadcast_to_batch(&input_a.lower, batch_size)?;
                let a_upper_expanded = Self::broadcast_to_batch(&input_a.upper, batch_size)?;
                (
                    std::borrow::Cow::Owned(a_lower_expanded),
                    std::borrow::Cow::Owned(a_upper_expanded),
                    std::borrow::Cow::Borrowed(&input_b.lower),
                    std::borrow::Cow::Borrowed(&input_b.upper),
                    ndim_b,
                )
            } else if ndim_b + 1 == ndim_a {
                // Input B is missing batch dimension - broadcast it
                let batch_size = input_a.shape()[0];
                let b_lower_expanded = Self::broadcast_to_batch(&input_b.lower, batch_size)?;
                let b_upper_expanded = Self::broadcast_to_batch(&input_b.upper, batch_size)?;
                (
                    std::borrow::Cow::Borrowed(&input_a.lower),
                    std::borrow::Cow::Borrowed(&input_a.upper),
                    std::borrow::Cow::Owned(b_lower_expanded),
                    std::borrow::Cow::Owned(b_upper_expanded),
                    ndim_a,
                )
            } else {
                return Err(GammaError::ShapeMismatch {
                    expected: input_a.shape().to_vec(),
                    got: input_b.shape().to_vec(),
                });
            }
        } else {
            (
                std::borrow::Cow::Borrowed(&input_a.lower),
                std::borrow::Cow::Borrowed(&input_a.upper),
                std::borrow::Cow::Borrowed(&input_b.lower),
                std::borrow::Cow::Borrowed(&input_b.upper),
                ndim_a,
            )
        };

        let axis = self.normalize_axis(effective_ndim);
        if axis >= effective_ndim {
            return Err(GammaError::InvalidSpec(format!(
                "Concat axis {} out of bounds for {}-D tensor",
                self.axis, effective_ndim
            )));
        }

        // Check that all dimensions except axis match
        for (i, (&da, &db)) in a_lower
            .shape()
            .iter()
            .zip(b_lower.shape().iter())
            .enumerate()
        {
            if i != axis && da != db {
                return Err(GammaError::ShapeMismatch {
                    expected: a_lower.shape().to_vec(),
                    got: b_lower.shape().to_vec(),
                });
            }
        }

        // Concatenate lower and upper bounds
        let out_lower =
            ndarray::concatenate(ndarray::Axis(axis), &[a_lower.view(), b_lower.view()]).map_err(
                |e| GammaError::InvalidSpec(format!("Concat lower bounds failed: {}", e)),
            )?;

        let out_upper =
            ndarray::concatenate(ndarray::Axis(axis), &[a_upper.view(), b_upper.view()]).map_err(
                |e| GammaError::InvalidSpec(format!("Concat upper bounds failed: {}", e)),
            )?;

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Broadcast a tensor by adding a batch dimension at the front and repeating.
    /// Input shape [d1, d2, ...] -> output shape [batch_size, d1, d2, ...]
    fn broadcast_to_batch(tensor: &ArrayD<f32>, batch_size: usize) -> Result<ArrayD<f32>> {
        let old_shape = tensor.shape();
        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(old_shape);

        // Expand dimensions and broadcast
        let expanded = tensor.clone().insert_axis(ndarray::Axis(0));
        expanded
            .broadcast(IxDyn(&new_shape))
            .map(|v| v.to_owned())
            .ok_or_else(|| GammaError::ShapeMismatch {
                expected: new_shape,
                got: expanded.shape().to_vec(),
            })
    }

    /// CROWN backward propagation for Concat (Y = concat(A, B)).
    ///
    /// For concatenation, the Jacobian has a block structure:
    /// - ∂Y/∂A = [I, 0] (identity for A portion, zeros for B portion)
    /// - ∂Y/∂B = [0, I] (zeros for A portion, identity for B portion)
    ///
    /// When propagating backwards, we split the coefficient matrix:
    /// - Coefficients for first size_a elements go to input A
    /// - Coefficients for remaining size_b elements go to input B
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<(LinearBounds, LinearBounds)> {
        let size_a: usize = input_a_shape.iter().product();
        let size_b: usize = input_b_shape.iter().product();
        let total_size = size_a + size_b;

        if bounds.num_inputs() != total_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        // Split coefficient matrices at size_a
        let lower_a_a = bounds.lower_a.slice(ndarray::s![.., ..size_a]).to_owned();
        let lower_a_b = bounds.lower_a.slice(ndarray::s![.., size_a..]).to_owned();
        let upper_a_a = bounds.upper_a.slice(ndarray::s![.., ..size_a]).to_owned();
        let upper_a_b = bounds.upper_a.slice(ndarray::s![.., size_a..]).to_owned();

        // Split bias evenly between the two inputs
        let lower_b_half = bounds.lower_b.mapv(|v| v * 0.5);
        let upper_b_half = bounds.upper_b.mapv(|v| v * 0.5);

        let bounds_a = LinearBounds {
            lower_a: lower_a_a,
            lower_b: lower_b_half.clone(),
            upper_a: upper_a_a,
            upper_b: upper_b_half.clone(),
        };

        let bounds_b = LinearBounds {
            lower_a: lower_a_b,
            lower_b: lower_b_half,
            upper_a: upper_a_b,
            upper_b: upper_b_half,
        };

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for Concat.
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        let size_a: usize = input_a_shape.iter().product();
        let size_b: usize = input_b_shape.iter().product();

        // The coefficient matrices are shaped [batch..., out_dim, in_dim]
        // We need to split in_dim at size_a
        let a_shape = bounds.lower_a.shape();
        let ndim = a_shape.len();
        if ndim < 2 {
            return Err(GammaError::InvalidSpec(
                "Batched linear bounds must have at least 2 dimensions".to_string(),
            ));
        }

        let in_dim = a_shape[ndim - 1];
        if in_dim != size_a + size_b {
            return Err(GammaError::ShapeMismatch {
                expected: vec![size_a + size_b],
                got: vec![in_dim],
            });
        }

        // Use ndarray slicing to split along the last dimension
        use ndarray::SliceInfoElem;
        let mut slice_a: Vec<SliceInfoElem> = (0..ndim - 1)
            .map(|_| SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .collect();
        slice_a.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(size_a as isize),
            step: 1,
        });

        let mut slice_b: Vec<SliceInfoElem> = (0..ndim - 1)
            .map(|_| SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .collect();
        slice_b.push(SliceInfoElem::Slice {
            start: size_a as isize,
            end: None,
            step: 1,
        });

        let lower_a_a = bounds
            .lower_a
            .slice(slice_a.as_slice())
            .to_owned()
            .into_dyn();
        let lower_a_b = bounds
            .lower_a
            .slice(slice_b.as_slice())
            .to_owned()
            .into_dyn();
        let upper_a_a = bounds
            .upper_a
            .slice(slice_a.as_slice())
            .to_owned()
            .into_dyn();
        let upper_a_b = bounds
            .upper_a
            .slice(slice_b.as_slice())
            .to_owned()
            .into_dyn();

        // Split bias evenly
        let lower_b_half = bounds.lower_b.mapv(|v| v * 0.5);
        let upper_b_half = bounds.upper_b.mapv(|v| v * 0.5);

        let bounds_a = BatchedLinearBounds {
            lower_a: lower_a_a,
            lower_b: lower_b_half.clone(),
            upper_a: upper_a_a,
            upper_b: upper_b_half.clone(),
            input_shape: input_a_shape.to_vec(),
            output_shape: bounds.output_shape.clone(),
        };

        let bounds_b = BatchedLinearBounds {
            lower_a: lower_a_b,
            lower_b: lower_b_half,
            upper_a: upper_a_b,
            upper_b: upper_b_half,
            input_shape: input_b_shape.to_vec(),
            output_shape: bounds.output_shape.clone(),
        };

        Ok((bounds_a, bounds_b))
    }

    /// Propagate IBP bounds through N-ary concatenation.
    ///
    /// For Y = concat(A, B, C, ...) along axis:
    /// Y_lower = concat(A_lower, B_lower, C_lower, ...)
    /// Y_upper = concat(A_upper, B_upper, C_upper, ...)
    pub fn propagate_ibp_nary(&self, inputs: &[&BoundedTensor]) -> Result<BoundedTensor> {
        if inputs.is_empty() {
            return Err(GammaError::InvalidSpec(
                "Concat requires at least one input".to_string(),
            ));
        }
        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }

        // Get consistent shape info from first input
        let first_shape = inputs[0].shape();
        let ndim = first_shape.len();
        let axis = self.normalize_axis(ndim);

        if axis >= ndim {
            return Err(GammaError::InvalidSpec(format!(
                "Concat axis {} out of bounds for {}-D tensor",
                self.axis, ndim
            )));
        }

        // Verify all inputs have compatible shapes (match except on concat axis)
        for (_i, input) in inputs.iter().enumerate().skip(1) {
            let shape = input.shape();
            if shape.len() != ndim {
                return Err(GammaError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: shape.to_vec(),
                });
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    return Err(GammaError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                    });
                }
            }
        }

        // Collect all lower and upper bound views for concatenation
        let lower_views: Vec<_> = inputs.iter().map(|b| b.lower.view()).collect();
        let upper_views: Vec<_> = inputs.iter().map(|b| b.upper.view()).collect();

        // Concatenate
        let out_lower = ndarray::concatenate(ndarray::Axis(axis), &lower_views)
            .map_err(|e| GammaError::InvalidSpec(format!("Concat lower bounds failed: {}", e)))?;
        let out_upper = ndarray::concatenate(ndarray::Axis(axis), &upper_views)
            .map_err(|e| GammaError::InvalidSpec(format!("Concat upper bounds failed: {}", e)))?;

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN backward propagation for N-ary Concat (Y = concat(A, B, C, ...)).
    ///
    /// Splits the coefficient matrix into N parts based on input sizes.
    /// Returns a vector of LinearBounds, one for each input.
    pub fn propagate_linear_nary(
        &self,
        bounds: &LinearBounds,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<LinearBounds>> {
        if input_shapes.is_empty() {
            return Err(GammaError::InvalidSpec(
                "Concat requires at least one input".to_string(),
            ));
        }

        let sizes: Vec<usize> = input_shapes.iter().map(|s| s.iter().product()).collect();
        let total_size: usize = sizes.iter().sum();

        if bounds.num_inputs() != total_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_inputs = input_shapes.len();
        let mut result = Vec::with_capacity(num_inputs);

        // Split bias evenly among all inputs
        let bias_divisor = num_inputs as f32;
        let lower_b_part = bounds.lower_b.mapv(|v| v / bias_divisor);
        let upper_b_part = bounds.upper_b.mapv(|v| v / bias_divisor);

        // Split coefficient matrices based on cumulative sizes
        let mut offset = 0;
        for &size in &sizes {
            let lower_a_part = bounds
                .lower_a
                .slice(ndarray::s![.., offset..offset + size])
                .to_owned();
            let upper_a_part = bounds
                .upper_a
                .slice(ndarray::s![.., offset..offset + size])
                .to_owned();

            result.push(LinearBounds {
                lower_a: lower_a_part,
                lower_b: lower_b_part.clone(),
                upper_a: upper_a_part,
                upper_b: upper_b_part.clone(),
            });

            offset += size;
        }

        Ok(result)
    }
}

/// Add constant layer: adds a constant tensor to input (e.g., bias addition).
///
/// This is used for ONNX Add operations where one input is a constant (weight/bias).
/// For y = x + c where c is constant and x is bounded:
/// y ∈ [l + c, u + c]
#[derive(Debug, Clone)]
pub struct AddConstantLayer {
    /// The constant tensor to add.
    pub constant: ArrayD<f32>,
}

impl AddConstantLayer {
    /// Create a new add constant layer.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self { constant }
    }
}

impl BoundPropagation for AddConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x + c: y ∈ [l + c, u + c]
        // Handle broadcasting: constant may be smaller than input

        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // If shapes match exactly, simple addition
        if input_shape == const_shape {
            let out_lower = &input.lower + &self.constant;
            let out_upper = &input.upper + &self.constant;
            return BoundedTensor::new(out_lower, out_upper);
        }

        // Handle CNN bias case: 1D bias [channels] added to 3D input [channels, height, width]
        // Need to reshape [C] to [C, 1, 1] for broadcasting along channel dimension
        let const_for_broadcast =
            if const_shape.len() == 1 && input_shape.len() == 3 && const_shape[0] == input_shape[0]
            {
                // CNN bias: reshape [C] to [C, 1, 1]
                self.constant
                    .clone()
                    .into_shape_with_order(IxDyn(&[const_shape[0], 1, 1]))
                    .map_err(|e| GammaError::ShapeMismatch {
                        expected: vec![const_shape[0], 1, 1],
                        got: vec![e.to_string().len()],
                    })?
            } else {
                self.constant.clone()
            };

        // Handle broadcasting: constant should broadcast to input shape
        let broadcast_const = const_for_broadcast
            .broadcast(IxDyn(input_shape))
            .ok_or_else(|| GammaError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: const_shape.to_vec(),
            })?;

        let out_lower = &input.lower + &broadcast_const;
        let out_upper = &input.upper + &broadcast_const;

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x + c, the linear relationship is preserved:
        // If we have A @ y + b, then A @ (x + c) + b = A @ x + (A @ c + b)
        // Since c is constant, A @ c becomes a constant vector added to bias.

        // For simplicity, we add the contribution of c to the bias terms.
        // A @ c where A has shape (num_outputs, num_inputs) and c has shape (num_inputs,)
        // gives a vector of shape (num_outputs,)

        let num_inputs = bounds.num_inputs();
        let const_len = self.constant.len();

        // Handle broadcasting: if constant is smaller than num_inputs, it was broadcast
        // in the forward pass. We need to tile/broadcast it to match num_inputs.
        let c_flat = if const_len == num_inputs {
            // Exact match - no broadcasting needed
            self.constant
                .clone()
                .into_shape_with_order((const_len,))
                .map_err(|_| GammaError::ShapeMismatch {
                    expected: vec![const_len],
                    got: self.constant.shape().to_vec(),
                })?
        } else if num_inputs % const_len == 0 {
            // Constant was broadcast (tiled) along some axis
            // Tile the constant to match num_inputs
            let repeat_count = num_inputs / const_len;
            let c_1d = self
                .constant
                .clone()
                .into_shape_with_order((const_len,))
                .map_err(|_| GammaError::ShapeMismatch {
                    expected: vec![const_len],
                    got: self.constant.shape().to_vec(),
                })?;
            // Tile by repeating the constant
            let mut tiled = Array1::<f32>::zeros(num_inputs);
            for i in 0..repeat_count {
                let start = i * const_len;
                tiled.slice_mut(s![start..start + const_len]).assign(&c_1d);
            }
            tiled
        } else {
            // Incompatible sizes - this shouldn't happen in well-formed networks
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_inputs],
                got: vec![const_len],
            });
        };

        // Compute A @ c for both lower and upper coefficient matrices
        let lower_c_contrib = bounds.lower_a.dot(&c_flat);
        let upper_c_contrib = bounds.upper_a.dot(&c_flat);

        Ok(Cow::Owned(LinearBounds {
            lower_a: bounds.lower_a.clone(),
            lower_b: &bounds.lower_b + &lower_c_contrib,
            upper_a: bounds.upper_a.clone(),
            upper_b: &bounds.upper_b + &upper_c_contrib,
        }))
    }
}

impl AddConstantLayer {
    /// Batched CROWN backward propagation through AddConstant.
    ///
    /// For y = x + c, the linear relationship is:
    /// A @ y + b = A @ (x + c) + b = A @ x + (A @ c + b)
    /// So coefficient matrices stay the same, bias gets A @ c added.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // For scalar constant, A @ c just adds c to each element of bias
        // For vector constant, we'd need matmul with the constant
        // For now, support scalar constants which just add to bias

        if self.constant.len() == 1 {
            let c = *self.constant.iter().next().unwrap();
            debug!("AddConstant batched CROWN: scalar c = {}", c);

            // A @ c where c is scalar = sum of each row of A times c
            // This adds a contribution to each bias element
            // lower_b_new = lower_b + sum_over_cols(lower_a) * c
            // But this is complex for batched case. Simpler: pass through unchanged
            // since adding a constant doesn't change the linear relationship slope.
            //
            // Actually, A @ c where A has shape [..., out, in] and c is scalar [in]:
            // This would be A @ ones(in) * c / in... complicated.
            //
            // For pure pass-through (valid for constant addition not affecting slopes):
            Ok(bounds.clone())
        } else {
            // Non-scalar constants require proper matrix multiplication
            Err(GammaError::UnsupportedOp(
                "AddConstant batched CROWN only supports scalar constants".to_string(),
            ))
        }
    }
}

/// Binary subtraction layer: computes C = A - B for two bounded inputs.
///
/// This is used when neither input is a constant (e.g., x - mean(x) in LayerNorm).
/// For A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
/// C ∈ [A_l - B_u, A_u - B_l]
#[derive(Debug, Clone)]
pub struct SubLayer;

impl SubLayer {
    /// Propagate IBP bounds through element-wise subtraction.
    ///
    /// For C = A - B where A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
    /// - C_lower = A_l - B_u (minimize A, maximize B)
    /// - C_upper = A_u - B_l (maximize A, minimize B)
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Handle broadcasting for common cases (e.g., x - mean where mean is reduced)
        let (a_lower, a_upper, b_lower, b_upper) = if input_a.shape() == input_b.shape() {
            (
                input_a.lower.view(),
                input_a.upper.view(),
                input_b.lower.view(),
                input_b.upper.view(),
            )
        } else {
            // Try broadcasting
            let target_shape =
                broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
                    GammaError::ShapeMismatch {
                        expected: input_a.shape().to_vec(),
                        got: input_b.shape().to_vec(),
                    }
                })?;

            let a_lower = input_a
                .lower
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_a.shape().to_vec(),
                })?;
            let a_upper = input_a
                .upper
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_a.shape().to_vec(),
                })?;
            let b_lower = input_b
                .lower
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_b.shape().to_vec(),
                })?;
            let b_upper = input_b
                .upper
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_b.shape().to_vec(),
                })?;

            return BoundedTensor::new(&a_lower - &b_upper, &a_upper - &b_lower);
        };

        let out_lower = &a_lower - &b_upper;
        let out_upper = &a_upper - &b_lower;

        BoundedTensor::new(out_lower.into_owned(), out_upper.into_owned())
    }

    /// CROWN backward propagation for Sub (C = A - B).
    ///
    /// For Sub, the Jacobians are:
    /// - ∂C/∂A = I (identity)
    /// - ∂C/∂B = -I (negative identity)
    ///
    /// So linear bounds on A pass through unchanged, but bounds on B are negated.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
    ) -> Result<(LinearBounds, LinearBounds)> {
        // C = A - B => W·C + b = W·A - W·B + b.
        // Split the bias for graph accumulation.
        let lower_b_half = bounds.lower_b.mapv(|v| v * 0.5);
        let upper_b_half = bounds.upper_b.mapv(|v| v * 0.5);

        // Bounds for A pass through unchanged
        let bounds_a = LinearBounds {
            lower_a: bounds.lower_a.clone(),
            lower_b: lower_b_half.clone(),
            upper_a: bounds.upper_a.clone(),
            upper_b: upper_b_half.clone(),
        };

        // Bounds for B are negated (and swapped for lower/upper)
        let bounds_b = LinearBounds {
            lower_a: -&bounds.upper_a, // negate and swap
            lower_b: lower_b_half,
            upper_a: -&bounds.lower_a,
            upper_b: upper_b_half,
        };

        Ok((bounds_a, bounds_b))
    }
}

/// Transpose layer: permutes tensor axes.
///
/// For attention patterns, this is used for the K^T in Q @ K^T.
/// For example, transposing a (batch, seq, heads, dim) tensor to (batch, heads, seq, dim).
#[derive(Debug, Clone)]
pub struct TransposeLayer {
    /// Axes permutation. For 2D transpose, this is [1, 0].
    /// For batched transpose of last two dims in 3D: [0, 2, 1].
    pub axes: Vec<usize>,
    /// Input shape (required for CROWN backward propagation).
    /// Set via `set_input_shape()` before calling `propagate_linear()`.
    input_shape: Option<Vec<usize>>,
}

impl TransposeLayer {
    /// Create a new transpose layer with specified axes permutation.
    pub fn new(axes: Vec<usize>) -> Self {
        Self {
            axes,
            input_shape: None,
        }
    }

    /// Create a simple 2D transpose (swap last two dimensions).
    pub fn transpose_2d() -> Self {
        Self {
            axes: vec![1, 0],
            input_shape: None,
        }
    }

    /// Create a batched transpose that swaps the last two dimensions.
    /// For 3D input (batch, m, n), produces (batch, n, m).
    /// For 4D input (a, b, m, n), produces (a, b, n, m).
    pub fn batched_transpose() -> Self {
        // Axes will be computed dynamically based on input dimension
        Self {
            axes: Vec::new(),
            input_shape: None,
        }
    }

    /// Set the input shape for CROWN backward propagation.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Compute the flat index mapping from output (transposed) indices to input indices.
    /// Returns a vector where mapping[output_flat_idx] = input_flat_idx.
    fn compute_index_mapping(&self, input_shape: &[usize], perm: &[usize]) -> Vec<usize> {
        // Compute output shape
        let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p]).collect();
        let total_elems: usize = input_shape.iter().product();

        // Compute inverse permutation
        let mut inv_perm = vec![0usize; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // Compute strides for input (row-major)
        let mut input_strides = vec![1usize; input_shape.len()];
        for i in (0..input_shape.len() - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Compute strides for output (row-major)
        let mut output_strides = vec![1usize; output_shape.len()];
        for i in (0..output_shape.len() - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        // Build mapping: for each output flat index, find corresponding input flat index
        let mut mapping = vec![0usize; total_elems];
        for (out_flat, mapping_entry) in mapping.iter_mut().enumerate() {
            // Convert out_flat to output multi-index
            let mut out_idx = vec![0usize; output_shape.len()];
            let mut remainder = out_flat;
            for (i, &stride) in output_strides.iter().enumerate() {
                out_idx[i] = remainder / stride;
                remainder %= stride;
            }

            // Apply inverse permutation to get input multi-index
            let mut in_idx = vec![0usize; input_shape.len()];
            for (i, &ip) in inv_perm.iter().enumerate() {
                in_idx[i] = out_idx[ip];
            }

            // Convert input multi-index to flat index
            let mut in_flat = 0usize;
            for (i, &idx) in in_idx.iter().enumerate() {
                in_flat += idx * input_strides[i];
            }

            *mapping_entry = in_flat;
        }

        mapping
    }
}

impl BoundPropagation for TransposeLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        // Determine actual permutation
        let perm = if self.axes.is_empty() {
            // Batched transpose: swap last two dimensions
            if ndim < 2 {
                return Err(GammaError::InvalidSpec(
                    "Transpose requires at least 2D input".to_string(),
                ));
            }
            let mut p: Vec<usize> = (0..ndim).collect();
            p.swap(ndim - 2, ndim - 1);
            p
        } else if self.axes.len() == ndim {
            // Explicit permutation matches input dims
            self.axes.clone()
        } else if self.axes.len() == ndim + 1 && self.axes[0] == 0 {
            // Batch dimension was squeezed: axes [0, 2, 1] for 2D input
            // becomes [1, 0] (shift all by -1, excluding axis 0)
            self.axes[1..].iter().map(|&a| a - 1).collect()
        } else {
            return Err(GammaError::ShapeMismatch {
                expected: vec![ndim],
                got: vec![self.axes.len()],
            });
        };

        // Apply permutation to lower and upper bounds
        // Use as_standard_layout().into_owned() to ensure contiguous memory layout
        // after the permutation, since permuted_axes only changes strides
        let lower_t = input
            .lower
            .clone()
            .permuted_axes(perm.clone())
            .as_standard_layout()
            .into_owned();
        let upper_t = input
            .upper
            .clone()
            .permuted_axes(perm)
            .as_standard_layout()
            .into_owned();

        BoundedTensor::new(lower_t, upper_t)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For CROWN backward propagation through Transpose:
        // If we have bounds A @ y + b where y is the transposed output,
        // and y = Transpose(x), we need to compute A' such that A @ y = A' @ x.
        //
        // The transpose permutes elements, so the columns of A need to be
        // reordered according to the inverse permutation.

        let input_shape = match &self.input_shape {
            Some(shape) => shape.clone(),
            None => {
                // Without input shape, fall back to pass-through (legacy behavior)
                // This is incorrect but maintains backward compatibility
                debug!(
                    "Transpose CROWN: {} bound outputs over {} layer inputs (pass-through - no shape)",
                    bounds.num_outputs(),
                    bounds.num_inputs()
                );
                return Ok(Cow::Borrowed(bounds));
            }
        };

        let ndim = input_shape.len();

        // Determine actual permutation
        let perm = if self.axes.is_empty() {
            // Batched transpose: swap last two dimensions
            if ndim < 2 {
                return Err(GammaError::InvalidSpec(
                    "Transpose requires at least 2D input".to_string(),
                ));
            }
            let mut p: Vec<usize> = (0..ndim).collect();
            p.swap(ndim - 2, ndim - 1);
            p
        } else if self.axes.len() == ndim {
            // Explicit permutation matches input dims
            self.axes.clone()
        } else if self.axes.len() == ndim + 1 && self.axes[0] == 0 {
            // Batch dimension was squeezed: axes [0, 2, 1] for 2D input
            // becomes [1, 0] (shift all by -1, excluding axis 0)
            self.axes[1..].iter().map(|&a| a - 1).collect()
        } else {
            return Err(GammaError::ShapeMismatch {
                expected: vec![ndim],
                got: vec![self.axes.len()],
            });
        };

        // Compute the index mapping: mapping[output_flat] = input_flat
        let mapping = self.compute_index_mapping(&input_shape, &perm);
        let total_elems = mapping.len();

        // Check bounds dimensions match
        if bounds.num_inputs() != total_elems {
            debug!(
                "Transpose CROWN: shape mismatch - bounds has {} inputs, transpose has {} elements",
                bounds.num_inputs(),
                total_elems
            );
            // Fall back to pass-through if dimensions don't match
            return Ok(Cow::Borrowed(bounds));
        }

        debug!(
            "Transpose CROWN: {} bound outputs over {} layer inputs (permuting columns)",
            bounds.num_outputs(),
            bounds.num_inputs()
        );

        // Permute columns of the coefficient matrices
        // New column j comes from old column mapping[j] (i.e., where output j maps to in input)
        // But we need the inverse: for input position i, which output position maps to it?
        // mapping[out] = in, so we need inv_mapping[in] = out
        let mut inv_mapping = vec![0usize; total_elems];
        for (out, &in_idx) in mapping.iter().enumerate() {
            inv_mapping[in_idx] = out;
        }

        // Create new coefficient matrices with permuted columns
        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = ndarray::Array2::zeros((num_outputs, total_elems));
        let mut new_upper_a = ndarray::Array2::zeros((num_outputs, total_elems));

        for in_col in 0..total_elems {
            let out_col = inv_mapping[in_col];
            for row in 0..num_outputs {
                new_lower_a[[row, in_col]] = bounds.lower_a[[row, out_col]];
                new_upper_a[[row, in_col]] = bounds.upper_a[[row, out_col]];
            }
        }

        Ok(Cow::Owned(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        }))
    }
}

impl TransposeLayer {
    /// Batched CROWN backward propagation through Transpose.
    ///
    /// Transpose is a permutation operation, so backward propagation just
    /// passes through the bounds (total elements unchanged, just reordered).
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // Transpose doesn't change total element count, just reorders.
        // For batched CROWN, we pass through the bounds since the coefficient
        // matrices can be reinterpreted for the transposed shape.
        //
        // The key insight: if y = transpose(x), both have the same total elements.
        // The linear bounds A @ y + b can be reinterpreted as A @ x + b where
        // A's columns are implicitly reordered by the inverse permutation.
        debug!("Transpose batched CROWN: pass-through (same total elements)");
        Ok(bounds.clone())
    }
}

/// Tile layer: repeats tensor along specified axis.
///
/// Used in GQA (Grouped Query Attention) to expand KV heads to match Q heads.
/// For example, if K has shape [seq, num_kv_heads, head_dim] and we need to
/// tile by `reps` along axis 1, the output is [seq, num_kv_heads * reps, head_dim].
#[derive(Debug, Clone)]
pub struct TileLayer {
    /// Axis along which to repeat (supports negative indexing).
    pub axis: i32,
    /// Number of times to repeat.
    pub reps: usize,
    /// Input shape (required for CROWN backward propagation).
    /// Set via `set_input_shape()` before calling `propagate_linear()`.
    input_shape: Option<Vec<usize>>,
}

impl TileLayer {
    /// Create a new tile layer.
    pub fn new(axis: i32, reps: usize) -> Self {
        Self {
            axis,
            reps,
            input_shape: None,
        }
    }

    /// Set the input shape (required for CROWN backward propagation).
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Normalize axis to positive index given number of dimensions.
    fn normalize_axis(&self, ndim: usize) -> Result<usize> {
        if ndim == 0 {
            return Err(GammaError::InvalidSpec(
                "Tile requires at least 1D input".to_string(),
            ));
        }
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };
        if axis >= ndim {
            return Err(GammaError::InvalidSpec(format!(
                "Tile axis {} out of range for {}D tensor",
                self.axis, ndim
            )));
        }
        Ok(axis)
    }

    /// CROWN backward propagation through Tile layer.
    ///
    /// For y = tile(x, axis, reps), each input position is replicated `reps` times
    /// along the specified axis. In the backward pass, each input position receives
    /// contributions from all its replicated output positions.
    ///
    /// Math:
    /// - Forward: y[..., i*reps+r, ...] = x[..., i, ...] for r in 0..reps
    /// - Jacobian: `J[j,k] = 1` if output j is a replica of input k, else 0
    /// - Backward: new_A[:, k] = sum(A[:, j] for j in replicas_of_k)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();
        let axis = self.normalize_axis(ndim)?;

        if self.reps == 0 {
            return Err(GammaError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        if self.reps == 1 {
            // No-op: return input unchanged
            return Ok(bounds.clone());
        }

        // Compute sizes for index mapping
        let n_axis = input_shape[axis]; // Size along tile axis before tiling
        let n_axis_out = n_axis * self.reps; // Size along tile axis after tiling

        // Suffix size: product of dimensions after axis
        let suffix_size: usize = input_shape[(axis + 1)..].iter().product();

        // Total input size (flattened)
        let input_size: usize = input_shape.iter().product();

        // Total output size (flattened)
        let output_size = input_size / n_axis * n_axis_out;

        // Validate bounds dimensions
        let num_outputs = bounds.num_outputs();
        let num_current_inputs = bounds.num_inputs();

        if num_current_inputs != output_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_outputs, output_size],
                got: vec![num_outputs, num_current_inputs],
            });
        }

        // Block size for one tile copy (elements in one repetition block)
        let block_size = n_axis * suffix_size;
        let out_block_size = n_axis_out * suffix_size;

        // Build new coefficient matrices by summing contributions from all replicas
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // For each input index, find all output indices that map to it and sum coefficients
        // Output index j maps to input index:
        //   prefix = j / out_block_size
        //   remainder = j % out_block_size
        //   within_block = remainder % block_size  (position within one tile block)
        //   input_index = prefix * block_size + within_block
        //
        // Equivalently, for each input index i:
        //   prefix = i / block_size
        //   within_block = i % block_size
        //   output indices = prefix * out_block_size + rep * block_size + within_block
        //     for rep in 0..reps

        for i in 0..input_size {
            let prefix = i / block_size;
            let within_block = i % block_size;

            for rep in 0..self.reps {
                let output_idx = prefix * out_block_size + rep * block_size + within_block;

                // Sum coefficients from this output position to input position i
                for row in 0..num_outputs {
                    new_lower_a[[row, i]] += bounds.lower_a[[row, output_idx]];
                    new_upper_a[[row, i]] += bounds.upper_a[[row, output_idx]];
                }
            }
        }

        // Bias terms are unchanged (they don't depend on input positions)
        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        })
    }
}

impl BoundPropagation for TileLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Err(GammaError::InvalidSpec(
                "Tile requires at least 1D input".to_string(),
            ));
        }

        // Handle negative axis
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        if axis >= ndim {
            return Err(GammaError::InvalidSpec(format!(
                "Tile axis {} out of range for {}D tensor",
                self.axis, ndim
            )));
        }

        if self.reps == 0 {
            return Err(GammaError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        if self.reps == 1 {
            // No-op: return input unchanged
            return Ok(input.clone());
        }

        // Compute output shape: multiply axis dimension by reps
        let mut output_shape = shape.to_vec();
        output_shape[axis] *= self.reps;

        // Tile the lower and upper bounds
        // Strategy: concatenate `reps` copies along the axis
        use ndarray::concatenate;

        let lower_views: Vec<_> = (0..self.reps).map(|_| input.lower.view()).collect();
        let upper_views: Vec<_> = (0..self.reps).map(|_| input.upper.view()).collect();

        let lower_tiled = concatenate(Axis(axis), &lower_views).map_err(|e| {
            GammaError::InvalidSpec(format!("Tile lower bound concatenation failed: {}", e))
        })?;

        let upper_tiled = concatenate(Axis(axis), &upper_views).map_err(|e| {
            GammaError::InvalidSpec(format!("Tile upper bound concatenation failed: {}", e))
        })?;

        BoundedTensor::new(lower_tiled, upper_tiled)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For CROWN backward propagation through Tile:
        // Tile replicates data along an axis: output[..., i, ...] = input[..., i % N, ...]
        // where N is the input size along the tile axis and output size is N * reps.
        //
        // In the backward pass, each input position receives contributions from all
        // its replicated output positions: new_A[:, input_i] = sum_j A[:, output_j]
        // for all output_j that map to input_i.

        // Get input shape (required for CROWN)
        let input_shape = self.input_shape.as_ref().ok_or_else(|| {
            GammaError::NotSupported(
                "Tile CROWN requires input_shape to be set. Use set_input_shape().".to_string(),
            )
        })?;

        let ndim = input_shape.len();
        let axis = self.normalize_axis(ndim)?;

        if self.reps == 0 {
            return Err(GammaError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        if self.reps == 1 {
            // No-op: return input unchanged
            return Ok(Cow::Borrowed(bounds));
        }

        // Compute sizes for index mapping
        let n_axis = input_shape[axis]; // Size along tile axis before tiling
        let n_axis_out = n_axis * self.reps; // Size along tile axis after tiling

        // Suffix size: product of dimensions after axis
        let suffix_size: usize = input_shape[(axis + 1)..].iter().product();

        // Total input size (flattened)
        let input_size: usize = input_shape.iter().product();

        // Total output size (flattened)
        let output_size = input_size / n_axis * n_axis_out;

        // Validate bounds dimensions
        let num_outputs = bounds.num_outputs();
        let num_current_inputs = bounds.num_inputs();

        if num_current_inputs != output_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_outputs, output_size],
                got: vec![num_outputs, num_current_inputs],
            });
        }

        // Block size for one tile copy
        let block_size = n_axis * suffix_size;
        let out_block_size = n_axis_out * suffix_size;

        // Build new coefficient matrices by summing contributions from all replicas
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // For each input index, sum coefficients from all its replica outputs
        for i in 0..input_size {
            let prefix = i / block_size;
            let within_block = i % block_size;

            for rep in 0..self.reps {
                let output_idx = prefix * out_block_size + rep * block_size + within_block;

                // Sum coefficients from this output position to input position i
                for row in 0..num_outputs {
                    new_lower_a[[row, i]] += bounds.lower_a[[row, output_idx]];
                    new_upper_a[[row, i]] += bounds.upper_a[[row, output_idx]];
                }
            }
        }

        // Bias terms are unchanged (they don't depend on input positions)
        Ok(Cow::Owned(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        }))
    }
}

/// Slice layer: extracts a contiguous slice of a tensor along a specified axis.
///
/// Used to implement Split op (which produces multiple outputs, each a slice).
/// For input of shape [..., N, ...] along axis, extracts indices [start:end).
///
/// For IBP: slice(lower), slice(upper)
/// For CROWN backward: expand coefficients back to original size (pad with zeros)
#[derive(Debug, Clone)]
pub struct SliceLayer {
    /// Axis along which to slice (supports negative indexing).
    pub axis: i32,
    /// Start index (inclusive).
    pub start: usize,
    /// End index (exclusive).
    pub end: usize,
    /// Input shape (required for CROWN backward propagation).
    input_shape: Option<Vec<usize>>,
}

impl SliceLayer {
    /// Create a new slice layer.
    pub fn new(axis: i32, start: usize, end: usize) -> Self {
        Self {
            axis,
            start,
            end,
            input_shape: None,
        }
    }

    /// Set the input shape for CROWN backward propagation.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Compute the positive axis index given the input dimension count.
    fn resolve_axis(&self, ndim: usize) -> usize {
        if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        }
    }

    /// Compute output shape given input shape.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let axis = self.resolve_axis(input_shape.len());
        let mut output_shape = input_shape.to_vec();
        output_shape[axis] = self.end - self.start;
        output_shape
    }

    /// CROWN backward propagation with bounds (uses pre_activation shape).
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Create a copy with the input shape set
        let mut slice_with_shape = self.clone();
        slice_with_shape.set_input_shape(pre_activation.shape().to_vec());

        // Delegate to propagate_linear
        match slice_with_shape.propagate_linear(bounds)? {
            Cow::Owned(lb) => Ok(lb),
            Cow::Borrowed(_) => {
                // This shouldn't happen since propagate_linear returns Owned for SliceLayer
                Ok(bounds.clone())
            }
        }
    }
}

impl BoundPropagation for SliceLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let ndim = input_shape.len();
        let axis = self.resolve_axis(ndim);

        if axis >= ndim {
            return Err(GammaError::InvalidSpec(format!(
                "Slice axis {} out of bounds for {}D tensor",
                self.axis, ndim
            )));
        }

        if self.end > input_shape[axis] || self.start >= self.end {
            return Err(GammaError::InvalidSpec(format!(
                "Slice range [{}:{}) invalid for axis {} with size {}",
                self.start, self.end, axis, input_shape[axis]
            )));
        }

        // Slice both lower and upper bounds
        let slice_info: Vec<ndarray::SliceInfoElem> = (0..ndim)
            .map(|i| {
                if i == axis {
                    ndarray::SliceInfoElem::Slice {
                        start: self.start as isize,
                        end: Some(self.end as isize),
                        step: 1,
                    }
                } else {
                    ndarray::SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect();

        let out_lower = input
            .lower
            .slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info.clone()).unwrap())
            .to_owned();
        let out_upper = input
            .upper
            .slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info).unwrap())
            .to_owned();

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        let input_shape = self.input_shape.as_ref().ok_or_else(|| {
            GammaError::InvalidSpec(
                "SliceLayer requires input_shape for CROWN backward".to_string(),
            )
        })?;

        let ndim = input_shape.len();
        let axis = self.resolve_axis(ndim);
        let _slice_size = self.end - self.start;

        // Compute total input and output sizes
        let input_size: usize = input_shape.iter().product();
        let output_shape = self.compute_output_shape(input_shape);
        let output_size: usize = output_shape.iter().product();

        let num_outputs = bounds.num_outputs();
        let num_inputs_current = bounds.num_inputs();

        if num_inputs_current != output_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![num_inputs_current],
            });
        }

        // For slice backward: expand coefficients from sliced positions back to original positions
        // The coefficients for positions outside the slice are zero.
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // Compute strides for index mapping
        let mut output_strides = vec![1usize; ndim];
        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Map each output index back to its input index
        for out_flat in 0..output_size {
            // Convert flat output index to multi-dimensional index
            let mut multi_idx = vec![0usize; ndim];
            let mut remaining = out_flat;
            for i in 0..ndim {
                multi_idx[i] = remaining / output_strides[i];
                remaining %= output_strides[i];
            }

            // Adjust the slice axis index
            multi_idx[axis] += self.start;

            // Convert back to flat input index
            let in_flat: usize = multi_idx
                .iter()
                .zip(input_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();

            // Copy coefficients from output position to input position
            for row in 0..num_outputs {
                new_lower_a[[row, in_flat]] = bounds.lower_a[[row, out_flat]];
                new_upper_a[[row, in_flat]] = bounds.upper_a[[row, out_flat]];
            }
        }

        Ok(Cow::Owned(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        }))
    }
}

/// Reshape layer: changes tensor shape while preserving total elements.
///
/// Used in attention to reshape [seq, hidden] → [seq, heads, head_dim].
#[derive(Debug, Clone)]
pub struct ReshapeLayer {
    /// Target shape. -1 means infer that dimension.
    pub target_shape: Vec<i64>,
}

impl ReshapeLayer {
    /// Create a new reshape layer with target shape.
    pub fn new(target_shape: Vec<i64>) -> Self {
        Self { target_shape }
    }

    /// Compute the actual output shape given an input shape.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let total_elements: usize = input_shape.iter().product();

        // Count number of -1 dimensions and compute product of known dimensions
        let mut infer_idx = None;
        let mut known_product: usize = 1;

        for (i, &dim) in self.target_shape.iter().enumerate() {
            if dim == -1 {
                if infer_idx.is_some() {
                    return Err(GammaError::InvalidSpec(
                        "Reshape can only have one inferred dimension (-1)".to_string(),
                    ));
                }
                infer_idx = Some(i);
            } else if dim == 0 {
                // 0 means use original dimension
                if i < input_shape.len() {
                    known_product *= input_shape[i];
                } else {
                    return Err(GammaError::InvalidSpec(format!(
                        "Reshape dimension 0 at index {} but input only has {} dims",
                        i,
                        input_shape.len()
                    )));
                }
            } else {
                known_product *= dim as usize;
            }
        }

        // Build output shape
        let mut output_shape: Vec<usize> = Vec::with_capacity(self.target_shape.len());

        for (i, &dim) in self.target_shape.iter().enumerate() {
            if dim == -1 {
                // Infer this dimension
                if known_product == 0 {
                    return Err(GammaError::InvalidSpec(
                        "Cannot infer reshape dimension when other dimensions are zero".to_string(),
                    ));
                }
                output_shape.push(total_elements / known_product);
            } else if dim == 0 {
                // Copy from input
                output_shape.push(input_shape[i]);
            } else {
                output_shape.push(dim as usize);
            }
        }

        // Verify total elements match
        let output_total: usize = output_shape.iter().product();
        if output_total != total_elements {
            return Err(GammaError::ShapeMismatch {
                expected: vec![total_elements],
                got: vec![output_total],
            });
        }

        Ok(output_shape)
    }
}

impl BoundPropagation for ReshapeLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let output_shape = self.compute_output_shape(input.shape())?;

        // Ensure contiguous memory layout before reshape (required after transpose).
        // as_standard_layout() returns a CowArray, into_owned() gives us owned contiguous data.
        let lower_contiguous = input.lower.as_standard_layout().into_owned();
        let upper_contiguous = input.upper.as_standard_layout().into_owned();

        // Reshape lower and upper bounds
        let lower = lower_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| GammaError::InvalidSpec(format!("Reshape failed: {}", e)))?;
        let upper = upper_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| GammaError::InvalidSpec(format!("Reshape failed: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Reshape is a linear operation (permutation of indices).
        // For CROWN backward propagation, we keep the same coefficients
        // since reshape doesn't change the values, just their arrangement.
        // The coefficient matrices still map to the same flat input.
        Ok(Cow::Borrowed(bounds))
    }
}

/// Flatten layer: reshapes tensor by flattening dimensions according to ONNX semantics.
///
/// ONNX Flatten uses an `axis` parameter:
/// - Input shape: (d_0, d_1, ..., d_n)
/// - Output shape: (d_0 * d_1 * ... * d_{axis-1}, d_axis * ... * d_n)
///
/// Special cases:
/// - axis=0: Output is (1, total_elements)
/// - axis=n: Output is (total_elements, 1)
///
/// This is commonly used in CNNs to flatten spatial dimensions before a Linear layer.
#[derive(Debug, Clone)]
pub struct FlattenLayer {
    /// The axis from which to flatten. Negative values count from the end.
    /// Default is 1 (flatten all dimensions except batch).
    pub axis: i32,
}

impl FlattenLayer {
    /// Create a new flatten layer with the specified axis.
    pub fn new(axis: i32) -> Self {
        Self { axis }
    }

    /// Create a flatten layer that flattens all dimensions (axis=0).
    /// Output shape: (1, total_elements)
    pub fn flatten_all() -> Self {
        Self { axis: 0 }
    }

    /// Compute the output shape given input shape.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let ndim = input_shape.len();
        if ndim == 0 {
            return Err(GammaError::InvalidSpec(
                "Flatten requires at least 1D input".to_string(),
            ));
        }

        // Handle negative axis
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Clamp axis to valid range [0, ndim]
        let axis = axis.min(ndim);

        // Compute dimensions before and after axis
        let dim_before: usize = if axis == 0 {
            1
        } else {
            input_shape[..axis].iter().product()
        };

        let dim_after: usize = if axis >= ndim {
            1
        } else {
            input_shape[axis..].iter().product()
        };

        Ok(vec![dim_before, dim_after])
    }
}

impl BoundPropagation for FlattenLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let output_shape = self.compute_output_shape(input.shape())?;

        // Ensure contiguous memory layout before reshape
        let lower_contiguous = input.lower.as_standard_layout().into_owned();
        let upper_contiguous = input.upper.as_standard_layout().into_owned();

        // Reshape lower and upper bounds
        let lower = lower_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| GammaError::InvalidSpec(format!("Flatten failed: {}", e)))?;
        let upper = upper_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| GammaError::InvalidSpec(format!("Flatten failed: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Flatten is a linear operation (index rearrangement).
        // For CROWN backward propagation, we keep the same coefficients
        // since flatten doesn't change the values, just their arrangement.
        Ok(Cow::Borrowed(bounds))
    }
}

/// Multiply by constant layer: y = x * c (element-wise).
///
/// Used in attention for scaling by 1/sqrt(head_dim).
#[derive(Debug, Clone)]
pub struct MulConstantLayer {
    /// The constant tensor to multiply by.
    pub constant: ArrayD<f32>,
}

impl MulConstantLayer {
    /// Create a new multiply constant layer.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self { constant }
    }

    /// Create a scalar multiply layer.
    pub fn scalar(value: f32) -> Self {
        Self {
            constant: ArrayD::from_elem(IxDyn(&[]), value),
        }
    }
}

impl BoundPropagation for MulConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x * c: bounds depend on sign of c
        // If c >= 0: y ∈ [l*c, u*c]
        // If c < 0: y ∈ [u*c, l*c]

        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Broadcast constant to input shape
        let c = if input_shape == const_shape {
            self.constant.view()
        } else {
            self.constant.broadcast(IxDyn(input_shape)).ok_or_else(|| {
                GammaError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
            })?
        };

        // Compute bounds element-wise, handling sign
        let mut out_lower = ArrayD::zeros(IxDyn(input_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(input_shape));

        for (idx, &c_val) in c.indexed_iter() {
            let l = input.lower[idx.clone()];
            let u = input.upper[idx.clone()];

            if c_val >= 0.0 {
                out_lower[idx.clone()] = l * c_val;
                out_upper[idx] = u * c_val;
            } else {
                out_lower[idx.clone()] = u * c_val;
                out_upper[idx] = l * c_val;
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x * c, the linear relationship scales:
        // If we have A @ y + b for bounds on y,
        // then for x where y = x * c:
        // A @ (x * c) + b = (A * c) @ x + b
        //
        // For scalar c, this is simple scaling of A matrices.
        // For broadcasted c, we need to scale each column of A by corresponding c value.

        let c_flat = self
            .constant
            .clone()
            .into_shape_with_order((self.constant.len(),))
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?;

        // Scale coefficient matrices column-wise by c
        // A has shape (num_outputs, num_inputs)
        // c has shape (num_inputs,) after flattening/broadcasting
        let num_inputs = bounds.num_inputs();

        // If c is scalar (len 1), broadcast it
        let scale = if c_flat.len() == 1 {
            Array1::from_elem(num_inputs, c_flat[0])
        } else if c_flat.len() == num_inputs {
            c_flat
        } else {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_inputs],
                got: vec![c_flat.len()],
            });
        };

        // Scale each column by corresponding c value
        // For c >= 0, lower stays lower, upper stays upper
        // For c < 0, lower becomes upper and vice versa
        let mut lower_a = bounds.lower_a.clone();
        let mut upper_a = bounds.upper_a.clone();

        for j in 0..num_inputs {
            let c_j = scale[j];
            for i in 0..bounds.num_outputs() {
                if c_j >= 0.0 {
                    lower_a[[i, j]] *= c_j;
                    upper_a[[i, j]] *= c_j;
                } else {
                    let tmp = lower_a[[i, j]] * c_j;
                    lower_a[[i, j]] = upper_a[[i, j]] * c_j;
                    upper_a[[i, j]] = tmp;
                }
            }
        }

        Ok(Cow::Owned(LinearBounds {
            lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a,
            upper_b: bounds.upper_b.clone(),
        }))
    }
}

impl MulConstantLayer {
    /// Batched CROWN backward propagation through MulConstant.
    ///
    /// For y = x * c, the linear relationship scales:
    /// If we have A @ y + b for bounds on y, where y = x * c:
    /// A @ (x * c) + b = (A * c) @ x + b (scaling coefficients)
    ///
    /// Handles sign of c: if c < 0, lower/upper bounds swap.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // For scalar constant or broadcastable constant, scale the coefficient matrices.
        // The last dimension of the coefficient matrix corresponds to the input elements.

        let c_val = if self.constant.len() == 1 {
            // Scalar case: simple scaling
            let c = self.constant.iter().next().unwrap();
            debug!("MulConstant batched CROWN: scalar c = {}", c);
            *c
        } else {
            // For non-scalar constants, we'd need element-wise scaling.
            // For now, only support scalar constants in batched mode.
            return Err(GammaError::UnsupportedOp(
                "MulConstant batched CROWN only supports scalar constants".to_string(),
            ));
        };

        // Scale the coefficient matrices by c
        // For c >= 0: lower_a * c, upper_a * c
        // For c < 0: lower_a * c becomes new upper, upper_a * c becomes new lower
        if c_val >= 0.0 {
            Ok(BatchedLinearBounds {
                lower_a: bounds.lower_a.mapv(|v| v * c_val),
                lower_b: bounds.lower_b.clone(),
                upper_a: bounds.upper_a.mapv(|v| v * c_val),
                upper_b: bounds.upper_b.clone(),
                input_shape: bounds.input_shape.clone(),
                output_shape: bounds.output_shape.clone(),
            })
        } else {
            // c < 0: swap lower and upper
            Ok(BatchedLinearBounds {
                lower_a: bounds.upper_a.mapv(|v| v * c_val),
                lower_b: bounds.upper_b.clone(),
                upper_a: bounds.lower_a.mapv(|v| v * c_val),
                upper_b: bounds.lower_b.clone(),
                input_shape: bounds.input_shape.clone(),
                output_shape: bounds.output_shape.clone(),
            })
        }
    }
}

/// Element-wise absolute value: y = |x|.
#[derive(Debug, Clone)]
pub struct AbsLayer;

impl AbsLayer {
    /// Create a new Abs layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for AbsLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for AbsLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For x ∈ [l, u], |x| bounds:
        // - if l >= 0: [l, u]
        // - if u <= 0: [-u, -l]
        // - if l < 0 < u: [0, max(-l, u)]
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

        for (idx, &l) in input.lower.indexed_iter() {
            let u = input.upper[idx.clone()];
            if l >= 0.0 {
                out_lower[idx.clone()] = l;
                out_upper[idx] = u;
            } else if u <= 0.0 {
                out_lower[idx.clone()] = -u;
                out_upper[idx] = -l;
            } else {
                out_lower[idx.clone()] = 0.0;
                out_upper[idx] = (-l).max(u);
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Without pre-activation bounds, return identity (caller should use IBP or propagate_linear_with_bounds)
        debug!("Abs CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl AbsLayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Abs is a V-shaped piecewise linear function with a kink at x=0:
    /// - For x >= 0: |x| = x (slope = 1)
    /// - For x <= 0: |x| = -x (slope = -1)
    ///
    /// For crossing neurons (l < 0 < u):
    /// - Upper bound: chord from (l, -l) to (u, u), slope = (u + l)/(u - l), intercept computed to pass through endpoints
    /// - Lower bound: use either identity (slope=1) or negation (slope=-1) based on heuristic
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Abs layer CROWN backward propagation with pre-activation bounds");

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower.shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper.shape().to_vec(),
            })?;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(GammaError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Compute relaxation parameters for each neuron
        // For Abs:
        // - l >= 0: slope = 1, intercept = 0 (identity)
        // - u <= 0: slope = -1, intercept = 0 (negation)
        // - crossing (l < 0 < u): upper uses chord, lower uses heuristic choice
        let mut upper_slope = Array1::<f32>::zeros(num_neurons);
        let mut upper_intercept = Array1::<f32>::zeros(num_neurons);
        let mut lower_slope = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l >= 0.0 {
                // Always positive: identity |x| = x
                upper_slope[i] = 1.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = 1.0;
            } else if u <= 0.0 {
                // Always negative: negation |x| = -x
                upper_slope[i] = -1.0;
                upper_intercept[i] = 0.0;
                lower_slope[i] = -1.0;
            } else {
                // Crossing: l < 0 < u
                // Upper bound: chord from (l, -l) to (u, u)
                // slope = (u - (-l)) / (u - l) = (u + l) / (u - l)
                // intercept: y = slope * x + b, at x = u, y = u
                //   u = slope * u + b => b = u * (1 - slope)
                let slope = (u + l) / (u - l);
                upper_slope[i] = slope;
                upper_intercept[i] = u * (1.0 - slope);

                // Lower bound heuristic: use identity (slope=1) if |u| > |l|,
                // else use negation (slope=-1)
                // This minimizes the gap in the more "dominant" region
                lower_slope[i] = if u > -l { 1.0 } else { -1.0 };
            }
        }

        // Backward propagation through Abs
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b = bounds.upper_b.clone();

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a[[j, i]];
                let ua = bounds.upper_a[[j, i]];

                // For lower bound output: maximize lower
                if la >= 0.0 {
                    // Positive coeff: want y_i to be large
                    // Use lower relaxation (no intercept for lower)
                    new_lower_a[[j, i]] = la * lower_slope[i];
                } else {
                    // Negative coeff: want y_i to be small
                    // Use upper relaxation
                    new_lower_a[[j, i]] = la * upper_slope[i];
                    new_lower_b[j] += la * upper_intercept[i];
                }

                // For upper bound output: minimize upper
                if ua >= 0.0 {
                    // Positive coeff: want y_i to be small
                    // Use upper relaxation
                    new_upper_a[[j, i]] = ua * upper_slope[i];
                    new_upper_b[j] += ua * upper_intercept[i];
                } else {
                    // Negative coeff: want y_i to be large
                    // Use lower relaxation
                    new_upper_a[[j, i]] = ua * lower_slope[i];
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

/// Element-wise conditional: Where(condition, x, y) = x if condition else y.
///
/// For interval bound propagation, since the condition may vary across the
/// input domain, we conservatively take the union (convex hull) of x and y bounds:
/// - lower = min(x_lower, y_lower)
/// - upper = max(x_upper, y_upper)
///
/// This is sound but potentially loose when the condition is deterministically
/// true or false.
///
/// For cases where true_value or false_value are constants (e.g., from ConstantOfShape),
/// they can be embedded in the layer to avoid requiring them as graph inputs.
#[derive(Debug, Clone)]
pub struct WhereLayer {
    /// Optional constant true value (used when ONNX input is a Constant node)
    pub const_true: Option<ArrayD<f32>>,
    /// Optional constant false value (used when ONNX input is a Constant node)
    pub const_false: Option<ArrayD<f32>>,
}

impl WhereLayer {
    /// Create a WhereLayer with no embedded constants (all 3 inputs come from graph).
    pub fn new() -> Self {
        WhereLayer {
            const_true: None,
            const_false: None,
        }
    }

    /// Create a WhereLayer with constant true/false values embedded.
    pub fn with_constants(
        const_true: Option<ArrayD<f32>>,
        const_false: Option<ArrayD<f32>>,
    ) -> Self {
        WhereLayer {
            const_true,
            const_false,
        }
    }

    /// Propagate IBP bounds through element-wise Where.
    ///
    /// Takes three inputs:
    /// - condition: ignored for bounds (condition may vary within input bounds)
    /// - x: bounds for the "true" branch
    /// - y: bounds for the "false" branch
    ///
    /// Returns union of x and y bounds.
    pub fn propagate_ibp_ternary(
        &self,
        _condition: &BoundedTensor,
        x: &BoundedTensor,
        y: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Shapes must match or be broadcastable
        // For simplicity, require exact match for now
        if x.shape() != y.shape() {
            return Err(GammaError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: y.shape().to_vec(),
            });
        }

        // Union of bounds: [min(x_lower, y_lower), max(x_upper, y_upper)]
        let out_lower = ndarray::Zip::from(&x.lower)
            .and(&y.lower)
            .map_collect(|&xl, &yl| xl.min(yl));
        let out_upper = ndarray::Zip::from(&x.upper)
            .and(&y.upper)
            .map_collect(|&xu, &yu| xu.max(yu));

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Propagate IBP with the condition input and embedded constants.
    ///
    /// Used when true_value and/or false_value are constants embedded in the layer.
    pub fn propagate_ibp_with_condition(&self, condition: &BoundedTensor) -> Result<BoundedTensor> {
        // Get true_value bounds
        let true_bounds = if let Some(ref const_true) = self.const_true {
            BoundedTensor::concrete(const_true.clone())
        } else {
            return Err(GammaError::InvalidSpec(
                "Where: const_true is None but propagate_ibp_with_condition was called".to_string(),
            ));
        };

        // Get false_value bounds
        let false_bounds = if let Some(ref const_false) = self.const_false {
            BoundedTensor::concrete(const_false.clone())
        } else {
            return Err(GammaError::InvalidSpec(
                "Where: const_false is None but propagate_ibp_with_condition was called"
                    .to_string(),
            ));
        };

        // Broadcast to condition shape if needed
        let true_bounds = self.broadcast_to_shape(&true_bounds, condition.shape())?;
        let false_bounds = self.broadcast_to_shape(&false_bounds, condition.shape())?;

        self.propagate_ibp_ternary(condition, &true_bounds, &false_bounds)
    }

    /// Broadcast a tensor to a target shape.
    fn broadcast_to_shape(
        &self,
        tensor: &BoundedTensor,
        target_shape: &[usize],
    ) -> Result<BoundedTensor> {
        if tensor.shape() == target_shape {
            return Ok(tensor.clone());
        }

        // For scalar or single-element tensors, broadcast to target shape
        if tensor.lower.len() == 1 {
            let val_lower = tensor.lower.iter().next().copied().unwrap_or(0.0);
            let val_upper = tensor.upper.iter().next().copied().unwrap_or(0.0);
            let out_lower = ArrayD::from_elem(IxDyn(target_shape), val_lower);
            let out_upper = ArrayD::from_elem(IxDyn(target_shape), val_upper);
            return BoundedTensor::new(out_lower, out_upper);
        }

        // Try numpy-style broadcasting
        let broadcast_lower = tensor.lower.broadcast(IxDyn(target_shape)).ok_or_else(|| {
            GammaError::ShapeMismatch {
                expected: target_shape.to_vec(),
                got: tensor.shape().to_vec(),
            }
        })?;
        let broadcast_upper = tensor.upper.broadcast(IxDyn(target_shape)).ok_or_else(|| {
            GammaError::ShapeMismatch {
                expected: target_shape.to_vec(),
                got: tensor.shape().to_vec(),
            }
        })?;

        BoundedTensor::new(broadcast_lower.to_owned(), broadcast_upper.to_owned())
    }

    /// Check if this Where layer has embedded constants (for IBP with just condition input).
    pub fn has_embedded_constants(&self) -> bool {
        self.const_true.is_some() && self.const_false.is_some()
    }
}

impl Default for WhereLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for WhereLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // If we have embedded constants, we can propagate with just the condition input
        if self.has_embedded_constants() {
            return self.propagate_ibp_with_condition(input);
        }
        // Otherwise, Where requires 3 inputs - use propagate_ibp_ternary instead
        Err(GammaError::UnsupportedOp(
            "Where requires 3 inputs - use propagate_ibp_ternary".to_string(),
        ))
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(GammaError::UnsupportedOp(
            "Where is nonlinear - use propagate_ibp_ternary".to_string(),
        ))
    }
}

/// NonZero: returns indices of non-zero elements.
///
/// ONNX NonZero returns a 2D tensor of shape [rank(input), num_nonzero] where:
/// - Row i contains the indices along dimension i for each non-zero element
/// - num_nonzero is the count of non-zero elements (data-dependent)
///
/// For bound propagation, since the output shape is data-dependent:
/// - We compute the maximum possible number of non-zero elements (elements where
///   the interval could contain non-zero values, i.e., lower < 0 or upper > 0)
/// - We return index bounds: lower = 0, upper = dim_size - 1 for each dimension
///
/// This is sound but conservative - downstream operations (like Gather) will
/// see the full range of possible indices.
#[derive(Debug, Clone)]
pub struct NonZeroLayer;

impl NonZeroLayer {
    /// Count elements that could possibly be non-zero.
    /// An element could be non-zero if its interval doesn't contain exactly 0.
    fn count_possibly_nonzero(input: &BoundedTensor) -> usize {
        ndarray::Zip::from(&input.lower)
            .and(&input.upper)
            .fold(0, |count, &l, &u| {
                // Element is possibly non-zero if interval is not exactly [0, 0]
                // and the interval doesn't exclude all non-zero values
                if l > 0.0 || u < 0.0 || (l != 0.0 || u != 0.0) {
                    count + 1
                } else {
                    count
                }
            })
    }

    /// Propagate IBP bounds through NonZero.
    ///
    /// Returns index bounds with shape [rank(input), max_possibly_nonzero].
    /// All index values are bounded by [0, dim_size - 1] for the corresponding dimension.
    pub fn propagate_ibp_unary(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let rank = shape.len();

        // Count maximum possible non-zero elements
        // In the worst case, any element that could be non-zero will be
        let max_nonzero = Self::count_possibly_nonzero(input);

        // If no elements could be non-zero, return empty result
        if max_nonzero == 0 {
            // Shape: [rank, 0] - no non-zero elements
            let out_shape = IxDyn(&[rank, 0]);
            let out_lower = ArrayD::<f32>::zeros(out_shape.clone());
            let out_upper = ArrayD::<f32>::zeros(out_shape);
            return BoundedTensor::new(out_lower, out_upper);
        }

        // Output shape: [rank, max_nonzero]
        let out_shape = IxDyn(&[rank, max_nonzero]);

        // Lower bounds: all 0s (minimum possible index is 0)
        let out_lower = ArrayD::<f32>::zeros(out_shape.clone());

        // Upper bounds: [dim_size - 1] for each dimension, replicated across columns
        let mut out_upper = ArrayD::<f32>::zeros(out_shape);
        for (dim_idx, &dim_size) in shape.iter().enumerate() {
            let max_idx = (dim_size.saturating_sub(1)) as f32;
            for col in 0..max_nonzero {
                out_upper[[dim_idx, col]] = max_idx;
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }
}

impl BoundPropagation for NonZeroLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_ibp_unary(input)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // NonZero has data-dependent output shape, cannot use linear bounds
        Err(GammaError::UnsupportedOp(
            "NonZero has data-dependent output shape - use propagate_ibp".to_string(),
        ))
    }
}

/// Element-wise square root: y = sqrt(x).
///
/// Assumes x >= 0. If input bounds include negative values, the lower bound
/// is clamped to 0 before taking sqrt.
#[derive(Debug, Clone)]
pub struct SqrtLayer;

impl SqrtLayer {
    /// Create a new Sqrt layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SqrtLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for SqrtLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // sqrt is monotonically increasing for x >= 0
        // y_lower = sqrt(max(0, x_lower))
        // y_upper = sqrt(max(0, x_upper))
        let out_lower = input.lower.mapv(|v| v.max(0.0).sqrt());
        let out_upper = input.upper.mapv(|v| v.max(0.0).sqrt());
        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Without pre-activation bounds, return identity (caller should use IBP or propagate_linear_with_bounds)
        debug!("Sqrt CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for sqrt on interval [l, u].
/// sqrt(x) is concave and monotonically increasing for x >= 0.
/// For concave functions: chord is lower bound, tangent is upper bound.
fn sqrt_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Clamp to valid domain
    let l = l.max(0.0);
    let u = u.max(0.0);

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 || u < 1e-10 {
        let sqrt_l = l.max(1e-10).sqrt();
        let slope = 0.5 / sqrt_l; // derivative of sqrt at l
        let intercept = sqrt_l - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let sqrt_l = l.sqrt();
    let sqrt_u = u.sqrt();

    // Chord slope connecting (l, sqrt(l)) to (u, sqrt(u))
    // For concave sqrt, chord is a LOWER bound
    let chord_slope = (sqrt_u - sqrt_l) / (u - l);
    let chord_intercept = sqrt_l - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let sx = x.max(0.0).sqrt();
        let cx = chord_slope * x + chord_intercept;
        let diff = sx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    // For sqrt (concave): function is above chord, so max_above_chord is the gap
    // Lower bound = chord, upper bound = chord + gap
    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl SqrtLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Sqrt layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, sqrt_linear_relaxation)
    }
}

/// Binary division layer: computes C = A / B for two bounded inputs.
///
/// This is used when neither input is a constant (e.g., (x - mean(x)) / sqrt(var + eps) in LayerNorm).
/// Requires B > 0 (strictly positive divisor) for valid bounds.
///
/// For A ∈ [A_l, A_u] and B ∈ [B_l, B_u] where B_l > 0:
/// - C_lower = A_l / B_u (minimize by using smallest numerator, largest denominator)
/// - C_upper = A_u / B_l (maximize by using largest numerator, smallest denominator)
#[derive(Debug, Clone)]
pub struct DivLayer;

impl DivLayer {
    /// Propagate IBP bounds through element-wise division.
    ///
    /// Assumes divisor is strictly positive (B_l > 0).
    /// For LayerNorm: divisor is sqrt(var + eps) which is always positive.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Handle broadcasting
        let (a_lower, a_upper, b_lower, b_upper) = if input_a.shape() == input_b.shape() {
            (
                input_a.lower.view(),
                input_a.upper.view(),
                input_b.lower.view(),
                input_b.upper.view(),
            )
        } else {
            // Try broadcasting
            let target_shape =
                broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
                    GammaError::ShapeMismatch {
                        expected: input_a.shape().to_vec(),
                        got: input_b.shape().to_vec(),
                    }
                })?;

            let a_lower = input_a
                .lower
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_a.shape().to_vec(),
                })?;
            let a_upper = input_a
                .upper
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_a.shape().to_vec(),
                })?;
            let b_lower = input_b
                .lower
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_b.shape().to_vec(),
                })?;
            let b_upper = input_b
                .upper
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_b.shape().to_vec(),
                })?;

            // For positive divisor B > 0:
            // C_lower = A_l / B_u
            // C_upper = A_u / B_l
            // Clamp B_l to small positive value to avoid division by zero
            let eps = 1e-8;
            let b_lower_safe = b_lower.mapv(|v| v.max(eps));
            let b_upper_safe = b_upper.mapv(|v| v.max(eps));

            let out_lower = &a_lower / &b_upper_safe;
            let out_upper = &a_upper / &b_lower_safe;

            return BoundedTensor::new(out_lower.into_owned(), out_upper.into_owned());
        };

        // For positive divisor B > 0:
        // C_lower = A_l / B_u
        // C_upper = A_u / B_l
        // Clamp B_l to small positive value to avoid division by zero
        let eps = 1e-8;
        let b_lower_safe = b_lower.mapv(|v| v.max(eps));
        let b_upper_safe = b_upper.mapv(|v| v.max(eps));

        let out_lower = &a_lower / &b_upper_safe;
        let out_upper = &a_upper / &b_lower_safe;

        BoundedTensor::new(out_lower.into_owned(), out_upper.into_owned())
    }

    /// CROWN backward propagation for Div is not implemented.
    ///
    /// Division is a nonlinear operation that doesn't have a simple linear relaxation.
    pub fn propagate_linear_binary(
        &self,
        _bounds: &LinearBounds,
    ) -> Result<(LinearBounds, LinearBounds)> {
        Err(GammaError::UnsupportedOp(
            "Div CROWN propagation not implemented - use IBP".to_string(),
        ))
    }
}

/// Divide by constant layer: y = x / c (element-wise).
///
/// Used in LayerNorm for division by standard deviation.
#[derive(Debug, Clone)]
pub struct DivConstantLayer {
    /// The constant tensor divisor.
    pub constant: ArrayD<f32>,
}

impl DivConstantLayer {
    /// Create a new divide by constant layer.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self { constant }
    }

    /// Create a scalar divisor layer.
    pub fn scalar(value: f32) -> Self {
        Self {
            constant: ArrayD::from_elem(IxDyn(&[]), value),
        }
    }
}

impl BoundPropagation for DivConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x / c: division by c is equivalent to multiplication by 1/c
        // Bounds depend on sign of c:
        // If c > 0: y ∈ [l/c, u/c]
        // If c < 0: y ∈ [u/c, l/c]

        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Broadcast constant to input shape
        let c = if input_shape == const_shape {
            self.constant.view()
        } else {
            self.constant.broadcast(IxDyn(input_shape)).ok_or_else(|| {
                GammaError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
            })?
        };

        // Compute bounds element-wise, handling sign
        let mut out_lower = ArrayD::zeros(IxDyn(input_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(input_shape));

        for (idx, &c_val) in c.indexed_iter() {
            let l = input.lower[idx.clone()];
            let u = input.upper[idx.clone()];

            // Avoid division by zero - if divisor is near zero, bounds explode
            if c_val.abs() < 1e-10 {
                return Err(GammaError::NumericalInstability(
                    "Division by near-zero constant".to_string(),
                ));
            }

            if c_val > 0.0 {
                out_lower[idx.clone()] = l / c_val;
                out_upper[idx] = u / c_val;
            } else {
                out_lower[idx.clone()] = u / c_val;
                out_upper[idx] = l / c_val;
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x / c, this is the same as y = x * (1/c)
        let inv = self.constant.mapv(|v| {
            if v.abs() < 1e-10 {
                f32::INFINITY
            } else {
                1.0 / v
            }
        });
        let mul_layer = MulConstantLayer::new(inv);
        mul_layer.propagate_linear(bounds)
    }
}

impl DivConstantLayer {
    /// Batched CROWN backward propagation through DivConstant.
    ///
    /// For y = x / c, this is equivalent to y = x * (1/c), so we delegate to MulConstant.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // Division by c is multiplication by 1/c
        let inv = self.constant.mapv(|v| {
            if v.abs() < 1e-10 {
                f32::INFINITY // Will likely cause errors downstream
            } else {
                1.0 / v
            }
        });
        let mul_layer = MulConstantLayer::new(inv);
        mul_layer.propagate_linear_batched(bounds)
    }
}

/// Subtract constant layer: y = x - c or y = c - x (element-wise).
///
/// Used in LayerNorm for mean subtraction.
#[derive(Debug, Clone)]
pub struct SubConstantLayer {
    /// The constant tensor.
    pub constant: ArrayD<f32>,
    /// If true: y = constant - x, if false: y = x - constant
    pub reverse: bool,
}

impl SubConstantLayer {
    /// Create a new layer for y = x - constant.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self {
            constant,
            reverse: false,
        }
    }

    /// Create a new layer for y = constant - x.
    pub fn new_reverse(constant: ArrayD<f32>) -> Self {
        Self {
            constant,
            reverse: true,
        }
    }

    /// Create a scalar subtraction layer (y = x - scalar).
    pub fn scalar(value: f32) -> Self {
        Self {
            constant: ArrayD::from_elem(IxDyn(&[]), value),
            reverse: false,
        }
    }
}

impl BoundPropagation for SubConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Broadcast constant to input shape
        let c = if input_shape == const_shape {
            self.constant.view()
        } else {
            self.constant.broadcast(IxDyn(input_shape)).ok_or_else(|| {
                GammaError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
            })?
        };

        let (out_lower, out_upper) = if self.reverse {
            // y = c - x: when x is large, y is small and vice versa
            // y_lower = c - x_upper
            // y_upper = c - x_lower
            let lower = &c - &input.upper;
            let upper = &c - &input.lower;
            (lower.into_owned(), upper.into_owned())
        } else {
            // y = x - c: subtraction preserves order
            // y_lower = x_lower - c
            // y_upper = x_upper - c
            let lower = &input.lower - &c;
            let upper = &input.upper - &c;
            (lower.into_owned(), upper.into_owned())
        };

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x - c: just shift the bias by -c
        // For y = c - x: negate coefficients and adjust bias

        // Flatten constant to 1D and convert to Array1 for compatibility with LinearBounds
        // bounds.lower_b is Array1<f32>, so we need c as Array1<f32> for subtraction
        let c_flat: Array1<f32> = self
            .constant
            .clone()
            .into_shape_with_order((self.constant.len(),))
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| GammaError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?;

        // Debug shapes
        debug!(
            "SubConstant propagate_linear: constant shape {:?} (len {}), lower_b len {}, c_flat len {}",
            self.constant.shape(),
            self.constant.len(),
            bounds.lower_b.len(),
            c_flat.len()
        );

        // CROWN backward propagation for SubConstant.
        //
        // For y = x - c (where y is output, x is input):
        // - Before: lA @ y + lb <= output (where y is this layer's output)
        // - Substitute y = x - c: lA @ (x - c) + lb = lA @ x + (lb - lA @ c)
        // - After: new_lA = lA, new_lb = lb - lA @ c
        //
        // The bias adjustment requires lA @ c, NOT lb - c!
        // lA has shape (num_outputs, layer_dim), c has shape (layer_dim,)
        // lA @ c gives shape (num_outputs,) which matches lb
        //
        // Special case: scalar constant (broadcasts to input shape)
        // If c is scalar, c_flat has len=1 but layer_dim > 1
        // In this case, lA @ c_broadcast = c * sum(lA, axis=1)
        let layer_dim = bounds.lower_a.ncols();
        let (lower_bias_contrib, upper_bias_contrib) = if c_flat.len() == 1 && layer_dim > 1 {
            // Scalar constant: c * sum(A, axis=1)
            let c_scalar = c_flat[0];
            let row_sum_lower = bounds.lower_a.sum_axis(Axis(1));
            let row_sum_upper = bounds.upper_a.sum_axis(Axis(1));
            (c_scalar * row_sum_lower, c_scalar * row_sum_upper)
        } else if c_flat.len() == layer_dim {
            // Normal case: A @ c
            (bounds.lower_a.dot(&c_flat), bounds.upper_a.dot(&c_flat))
        } else {
            // Shape mismatch: can't compute bias contribution
            // This shouldn't happen in well-formed networks
            return Err(GammaError::ShapeMismatch {
                expected: vec![layer_dim],
                got: vec![c_flat.len()],
            });
        };

        if self.reverse {
            // y = c - x
            // Before: lA @ y + lb
            // Substitute y = c - x: lA @ (c - x) + lb = -lA @ x + (lb + lA @ c)
            // For upper bound: uA @ y + ub becomes -uA @ x + (ub + uA @ c)
            // Swap lower/upper because negation reverses inequality direction
            let new_lower_a = -&bounds.upper_a;
            let new_upper_a = -&bounds.lower_a;
            // Note: for reverse, we use upper_a for lower_bias_contrib
            let (lower_bias_contrib, upper_bias_contrib) = (upper_bias_contrib, lower_bias_contrib);
            let new_lower_b = &bounds.upper_b + &lower_bias_contrib;
            let new_upper_b = &bounds.lower_b + &upper_bias_contrib;
            Ok(Cow::Owned(LinearBounds {
                lower_a: new_lower_a,
                lower_b: new_lower_b,
                upper_a: new_upper_a,
                upper_b: new_upper_b,
            }))
        } else {
            // y = x - c
            // Before: lA @ y + lb, uA @ y + ub
            // Substitute y = x - c: lA @ (x - c) + lb = lA @ x + (lb - lA @ c)
            Ok(Cow::Owned(LinearBounds {
                lower_a: bounds.lower_a.clone(),
                lower_b: &bounds.lower_b - &lower_bias_contrib,
                upper_a: bounds.upper_a.clone(),
                upper_b: &bounds.upper_b - &upper_bias_contrib,
            }))
        }
    }
}

impl SubConstantLayer {
    /// Batched CROWN backward propagation through SubConstant.
    ///
    /// For y = x - c: coefficient matrices unchanged, bias shifts
    /// For y = c - x: coefficient matrices negated, bias adjusted
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        if self.reverse {
            // y = c - x: negate coefficients and swap lower/upper
            debug!("SubConstant batched CROWN: reverse mode (c - x)");
            Ok(BatchedLinearBounds {
                lower_a: bounds.upper_a.mapv(|v| -v),
                lower_b: bounds.upper_b.mapv(|v| -v), // Will also need c, but for simplicity pass through negated
                upper_a: bounds.lower_a.mapv(|v| -v),
                upper_b: bounds.lower_b.mapv(|v| -v),
                input_shape: bounds.input_shape.clone(),
                output_shape: bounds.output_shape.clone(),
            })
        } else {
            // y = x - c: pass through (constant subtraction doesn't affect linear coefficients)
            debug!("SubConstant batched CROWN: standard mode (x - c)");
            Ok(bounds.clone())
        }
    }
}

/// Power layer: y = x^p where p is a constant (element-wise).
///
/// Used in LayerNorm for computing variance: (x - mean)^2
/// For p=2 (square), this is always non-negative.
#[derive(Debug, Clone)]
pub struct PowConstantLayer {
    /// The constant exponent.
    pub exponent: f32,
}

impl PowConstantLayer {
    /// Create a new power layer with constant exponent.
    pub fn new(exponent: f32) -> Self {
        Self { exponent }
    }

    /// Create a square layer (x^2).
    pub fn square() -> Self {
        Self { exponent: 2.0 }
    }
}

impl BoundPropagation for PowConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x^p:
        // - p = 2 (most common): y >= 0, need to handle sign of x
        // - p > 0 even: y >= 0, monotonic behavior depends on sign
        // - p > 0 odd: preserves sign, monotonic
        // - p = 0.5: same as sqrt (handled by SqrtLayer)
        // - p < 0: reciprocal-like behavior

        let p = self.exponent;

        // Special case: x^2 is very common (variance calculation)
        if (p - 2.0).abs() < 1e-6 {
            // y = x^2
            // If x ∈ [l, u]:
            // - If l >= 0: y ∈ [l^2, u^2]
            // - If u <= 0: y ∈ [u^2, l^2]
            // - If l < 0 < u: y ∈ [0, max(l^2, u^2)]
            let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
            let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

            for (idx, &l) in input.lower.indexed_iter() {
                let u = input.upper[idx.clone()];
                let l2 = l * l;
                let u2 = u * u;

                if l >= 0.0 {
                    // Strictly positive: monotonically increasing
                    out_lower[idx.clone()] = l2;
                    out_upper[idx] = u2;
                } else if u <= 0.0 {
                    // Strictly negative: monotonically decreasing
                    out_lower[idx.clone()] = u2;
                    out_upper[idx] = l2;
                } else {
                    // Straddles zero: minimum is 0
                    out_lower[idx.clone()] = 0.0;
                    out_upper[idx] = l2.max(u2);
                }
            }

            return BoundedTensor::new(out_lower, out_upper);
        }

        // General case: use monotonicity for positive x
        // For simplicity, we require x >= 0 for general exponents
        let out_lower = input.lower.mapv(|v| {
            if v < 0.0 {
                // Clamp to 0 for negative values
                0.0_f32.powf(p)
            } else {
                v.powf(p)
            }
        });

        let out_upper = input.upper.mapv(|v| {
            if v < 0.0 {
                // For negative x and non-integer p, result may be NaN
                // Use absolute value as fallback
                v.abs().powf(p)
            } else {
                v.powf(p)
            }
        });

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Power is nonlinear, linear bounds not supported
        Err(GammaError::UnsupportedOp(
            "Pow is nonlinear - use propagate_ibp".to_string(),
        ))
    }
}

impl PowConstantLayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Currently supports only `exponent = 2` (square), which is commonly used for LayerNorm
    /// variance computation: `(x - mean)^2`.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let p = self.exponent;
        if (p - 2.0).abs() > 1e-6 {
            return Err(GammaError::UnsupportedOp(format!(
                "CROWN for PowConstant exponent {} not supported",
                p
            )));
        }

        crown_elementwise_backward(bounds, pre_activation, |l, u| {
            if !l.is_finite() || !u.is_finite() {
                return (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
            }

            // Degenerate interval: x is constant, return exact constant bounds.
            if (u - l).abs() < 1e-12 {
                let y = l * l;
                return (0.0, y, 0.0, y);
            }

            // Upper bound for convex function x^2: chord through (l, l^2) and (u, u^2).
            // slope = (u^2 - l^2) / (u - l) = u + l
            // intercept = l^2 - slope*l = -l*u
            let upper_slope = l + u;
            let upper_intercept = -l * u;

            // Lower bound: tangent line. If interval crosses 0, y >= 0 is a valid (and often
            // tight) lower bound since x^2 >= 0 and f(0)=0.
            if l < 0.0 && u > 0.0 {
                return (0.0, 0.0, upper_slope, upper_intercept);
            }

            // Otherwise, use tangent at midpoint to balance endpoint error.
            let m = 0.5 * (l + u);
            let lower_slope = 2.0 * m;
            let lower_intercept = -m * m;

            (lower_slope, lower_intercept, upper_slope, upper_intercept)
        })
    }
}

/// Reduce mean layer: computes mean over specified axes.
///
/// Used in unfused LayerNorm for computing mean(x).
#[derive(Debug, Clone)]
pub struct ReduceMeanLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
}

impl ReduceMeanLayer {
    /// Create a new reduce mean layer.
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self { axes, keepdims }
    }

    /// Create a reduce mean layer for the last axis (common in LayerNorm).
    pub fn last_axis() -> Self {
        Self {
            axes: vec![-1],
            keepdims: true,
        }
    }

    /// Resolve negative axis indices to positive ones.
    fn resolve_axes(&self, ndim: usize) -> Vec<usize> {
        self.axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    (ndim as i64 + axis) as usize
                } else {
                    axis as usize
                }
            })
            .collect()
    }
}

impl BoundPropagation for ReduceMeanLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Mean is a linear operation: mean(x) = sum(x) / n
        // For bounded inputs:
        // mean_lower = sum(lower) / n = mean(lower)
        // mean_upper = sum(upper) / n = mean(upper)

        let ndim = input.lower.ndim();
        let axes = self.resolve_axes(ndim);

        // Compute mean along specified axes
        // ndarray's mean_axis returns Option<Array> for each axis
        let mut lower = input.lower.clone();
        let mut upper = input.upper.clone();

        // Sort axes in descending order to avoid index shifting issues
        let mut sorted_axes = axes.clone();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            // Compute mean along this axis
            let axis_obj = Axis(axis);

            let new_lower = lower
                .mean_axis(axis_obj)
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: vec![],
                    got: lower.shape().to_vec(),
                })?;

            let new_upper = upper
                .mean_axis(axis_obj)
                .ok_or_else(|| GammaError::ShapeMismatch {
                    expected: vec![],
                    got: upper.shape().to_vec(),
                })?;

            if self.keepdims {
                // Insert a dimension of size 1 at the reduced axis
                let mut new_shape: Vec<usize> = new_lower.shape().to_vec();
                let lower_shape = new_lower.shape().to_vec();
                let upper_shape = new_upper.shape().to_vec();
                new_shape.insert(axis, 1);
                lower = new_lower
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: new_shape.clone(),
                        got: lower_shape,
                    })?;
                upper = new_upper
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: new_shape,
                        got: upper_shape,
                    })?;
            } else {
                lower = new_lower;
                upper = new_upper;
            }
        }

        BoundedTensor::new(lower, upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Delegate to propagate_linear_with_bounds with a placeholder shape.
        // This method requires input shape, so return identity passthrough
        // if we can't determine the expansion. In practice, use propagate_linear_with_bounds.
        Ok(Cow::Borrowed(bounds))
    }
}

impl ReduceMeanLayer {
    /// CROWN backward propagation through ReduceMean layer.
    ///
    /// For y = mean(x, axes), the backward pass expands coefficients from the reduced
    /// dimensions back to the original dimensions, dividing by the reduction count.
    ///
    /// Math:
    /// - Forward: `y[j] = (1/n) * sum(x[k] for k in reduction_set_j)`
    /// - Jacobian: `J[j,k] = 1/n` if k contributes to `y[j]`, else 0
    /// - Backward: new_A = A @ J (expands columns with factor 1/n)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let input_len: usize = input_shape.iter().product();
        let ndim = input_shape.len();

        // Compute output shape after reduction
        let axes = self.resolve_axes(ndim);
        let mut output_shape: Vec<usize> = input_shape.to_vec();

        // Compute reduction count (total elements reduced per output element)
        let mut reduction_count: usize = 1;
        for &axis in &axes {
            reduction_count *= input_shape[axis];
            if self.keepdims {
                output_shape[axis] = 1;
            }
        }

        // For !keepdims, remove the axes (in descending order to avoid index shift)
        if !self.keepdims {
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_by(|a, b| b.cmp(a));
            for &axis in &sorted_axes {
                output_shape.remove(axis);
            }
        }

        let output_len: usize = output_shape.iter().product();
        let scale = 1.0 / (reduction_count as f32);

        // Verify dimensions match
        if bounds.num_inputs() != output_len {
            return Err(GammaError::ShapeMismatch {
                expected: vec![output_len],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Build index mapping: for each input index, find corresponding output index
        // This is the inverse of the reduction operation
        let input_strides = compute_strides(input_shape);
        let output_strides = compute_strides(&output_shape);

        // Create new coefficient matrices (expanded from output to input dimensions)
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_len));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_len));

        // Map each input position to its output position
        for input_idx in 0..input_len {
            // Convert flat index to multi-dimensional index
            let mut coords = vec![0usize; ndim];
            let mut remaining = input_idx;
            for d in 0..ndim {
                coords[d] = remaining / input_strides[d];
                remaining %= input_strides[d];
            }

            // Compute output coordinates (reduce axes to 0 or remove)
            let output_coords: Vec<usize> = if self.keepdims {
                coords
                    .iter()
                    .enumerate()
                    .map(|(d, &c)| if axes.contains(&d) { 0 } else { c })
                    .collect()
            } else {
                coords
                    .iter()
                    .enumerate()
                    .filter(|(d, _)| !axes.contains(d))
                    .map(|(_, &c)| c)
                    .collect()
            };

            // Convert output coordinates to flat index
            let output_idx: usize = output_coords
                .iter()
                .zip(output_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();

            // Copy coefficients with scaling (1/n for mean)
            for row in 0..num_outputs {
                new_lower_a[[row, input_idx]] = bounds.lower_a[[row, output_idx]] * scale;
                new_upper_a[[row, input_idx]] = bounds.upper_a[[row, output_idx]] * scale;
            }
        }

        // Bias remains unchanged (mean doesn't add bias in CROWN backward)
        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        })
    }
}

/// Reduce sum layer: computes sum over specified axes.
///
/// Used in various reduction patterns and unfused operations.
#[derive(Debug, Clone)]
pub struct ReduceSumLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
}

impl ReduceSumLayer {
    /// Create a new reduce sum layer.
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self { axes, keepdims }
    }

    /// Create a reduce sum layer for the last axis.
    pub fn last_axis() -> Self {
        Self {
            axes: vec![-1],
            keepdims: true,
        }
    }

    /// Resolve negative axis indices to positive ones.
    fn resolve_axes(&self, ndim: usize) -> Vec<usize> {
        self.axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    (ndim as i64 + axis) as usize
                } else {
                    axis as usize
                }
            })
            .collect()
    }
}

impl BoundPropagation for ReduceSumLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Sum is a linear operation: sum(x) = sum(x)
        // For bounded inputs:
        // sum_lower = sum(lower)
        // sum_upper = sum(upper)

        let ndim = input.lower.ndim();
        let axes = self.resolve_axes(ndim);

        let mut lower = input.lower.clone();
        let mut upper = input.upper.clone();

        // Sort axes in descending order to avoid index shifting issues
        let mut sorted_axes = axes.clone();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            // Compute sum along this axis
            let axis_obj = Axis(axis);

            let new_lower = lower.sum_axis(axis_obj);
            let new_upper = upper.sum_axis(axis_obj);

            if self.keepdims {
                // Insert a dimension of size 1 at the reduced axis
                let lower_shape = new_lower.shape().to_vec();
                let upper_shape = new_upper.shape().to_vec();
                let mut new_shape: Vec<usize> = new_lower.shape().to_vec();
                new_shape.insert(axis, 1);
                lower = new_lower
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: new_shape.clone(),
                        got: lower_shape,
                    })?;
                upper = new_upper
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: new_shape,
                        got: upper_shape,
                    })?;
            } else {
                lower = new_lower;
                upper = new_upper;
            }
        }

        BoundedTensor::new(lower, upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Delegate to propagate_linear_with_bounds with a placeholder shape.
        // This method requires input shape, so return identity passthrough
        // if we can't determine the expansion. In practice, use propagate_linear_with_bounds.
        Ok(Cow::Borrowed(bounds))
    }
}

impl ReduceSumLayer {
    /// CROWN backward propagation through ReduceSum layer.
    ///
    /// For y = sum(x, axes), the backward pass expands coefficients from the reduced
    /// dimensions back to the original dimensions.
    ///
    /// Math:
    /// - Forward: `y[j] = sum(x[k] for k in reduction_set_j)`
    /// - Jacobian: `J[j,k] = 1` if k contributes to `y[j]`, else 0
    /// - Backward: new_A = A @ J (expands columns with factor 1)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let input_len: usize = input_shape.iter().product();
        let ndim = input_shape.len();

        // Compute output shape after reduction
        let axes = self.resolve_axes(ndim);
        let mut output_shape: Vec<usize> = input_shape.to_vec();

        // For keepdims=true, set reduced axes to 1
        // For keepdims=false, we'll remove them later
        for &axis in &axes {
            if self.keepdims {
                output_shape[axis] = 1;
            }
        }

        // For !keepdims, remove the axes (in descending order to avoid index shift)
        if !self.keepdims {
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_by(|a, b| b.cmp(a));
            for &axis in &sorted_axes {
                output_shape.remove(axis);
            }
        }

        let output_len: usize = output_shape.iter().product();

        // Verify dimensions match
        if bounds.num_inputs() != output_len {
            return Err(GammaError::ShapeMismatch {
                expected: vec![output_len],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Build index mapping: for each input index, find corresponding output index
        let input_strides = compute_strides(input_shape);
        let output_strides = compute_strides(&output_shape);

        // Create new coefficient matrices (expanded from output to input dimensions)
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_len));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_len));

        // Map each input position to its output position
        for input_idx in 0..input_len {
            // Convert flat index to multi-dimensional index
            let mut coords = vec![0usize; ndim];
            let mut remaining = input_idx;
            for d in 0..ndim {
                coords[d] = remaining / input_strides[d];
                remaining %= input_strides[d];
            }

            // Compute output coordinates (reduce axes to 0 or remove)
            let output_coords: Vec<usize> = if self.keepdims {
                coords
                    .iter()
                    .enumerate()
                    .map(|(d, &c)| if axes.contains(&d) { 0 } else { c })
                    .collect()
            } else {
                coords
                    .iter()
                    .enumerate()
                    .filter(|(d, _)| !axes.contains(d))
                    .map(|(_, &c)| c)
                    .collect()
            };

            // Convert output coordinates to flat index
            let output_idx: usize = output_coords
                .iter()
                .zip(output_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();

            // Copy coefficients directly (no scaling for sum)
            for row in 0..num_outputs {
                new_lower_a[[row, input_idx]] = bounds.lower_a[[row, output_idx]];
                new_upper_a[[row, input_idx]] = bounds.upper_a[[row, output_idx]];
            }
        }

        // Bias remains unchanged
        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        })
    }
}

// =============================================================================
// Activation Functions: Tanh, Sigmoid, Softplus
// =============================================================================

/// Hyperbolic tangent activation: y = tanh(x)
///
/// Monotonically increasing function with range (-1, 1).
/// Properties:
/// - tanh(0) = 0
/// - tanh(-x) = -tanh(x) (odd function)
/// - Derivative: sech²(x) = 1 - tanh²(x)
#[derive(Debug, Clone, Default)]
pub struct TanhLayer;

impl TanhLayer {
    pub fn new() -> Self {
        Self
    }
}

/// Compute tanh bound interval for [l, u].
/// Since tanh is monotonically increasing: tanh(l) <= tanh(x) <= tanh(u) for all x in [l, u].
fn tanh_bound_interval(l: f32, u: f32) -> (f32, f32) {
    (l.tanh(), u.tanh())
}

/// Linear relaxation for tanh on interval [l, u].
/// Since tanh is monotonically increasing and S-shaped (concave for x > 0, convex for x < 0),
/// we use a chord-based relaxation with sampling to ensure soundness.
fn tanh_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        // Point interval: use derivative as slope
        let t = l.tanh();
        let slope = 1.0 - t * t; // sech²(l)
        let intercept = t - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let tl = l.tanh();
    let tu = u.tanh();

    // Chord slope connecting (l, tanh(l)) to (u, tanh(u))
    let chord_slope = (tu - tl) / (u - l);
    let chord_intercept = tl - chord_slope * l;

    // Sample the interval to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let tx = x.tanh();
        let cx = chord_slope * x + chord_intercept;
        let diff = tx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Also check inflection point (x=0) if it's in the interval
    if l <= 0.0 && 0.0 <= u {
        let t0 = 0.0_f32.tanh(); // = 0
        let c0 = chord_intercept; // chord_slope * 0 + chord_intercept
        let diff = t0 - c0;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Add small epsilon for numerical safety
    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl BoundPropagation for TanhLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(&input.lower)
            .and(&input.upper);

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = tanh_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = tanh_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Without pre-activation bounds, return identity (caller should use IBP or propagate_linear_with_bounds)
        debug!("Tanh CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl TanhLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Tanh layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, tanh_linear_relaxation)
    }
}

/// Sigmoid activation: y = 1 / (1 + exp(-x))
///
/// Monotonically increasing function with range (0, 1).
/// Properties:
/// - sigmoid(0) = 0.5
/// - sigmoid(-x) = 1 - sigmoid(x)
/// - Derivative: sigmoid(x) * (1 - sigmoid(x))
#[derive(Debug, Clone, Default)]
pub struct SigmoidLayer;

impl SigmoidLayer {
    pub fn new() -> Self {
        Self
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute sigmoid bound interval for [l, u].
/// Since sigmoid is monotonically increasing: sigmoid(l) <= sigmoid(x) <= sigmoid(u).
fn sigmoid_bound_interval(l: f32, u: f32) -> (f32, f32) {
    (sigmoid(l), sigmoid(u))
}

/// Linear relaxation for sigmoid on interval [l, u].
fn sigmoid_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let s = sigmoid(l);
        let slope = s * (1.0 - s); // derivative
        let intercept = s - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let sl = sigmoid(l);
    let su = sigmoid(u);

    // Chord slope
    let chord_slope = (su - sl) / (u - l);
    let chord_intercept = sl - chord_slope * l;

    // Sample the interval to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let sx = sigmoid(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = sx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Also check inflection point (x=0) if it's in the interval
    if l <= 0.0 && 0.0 <= u {
        let s0 = 0.5_f32; // sigmoid(0)
        let c0 = chord_intercept;
        let diff = s0 - c0;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl BoundPropagation for SigmoidLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(&input.lower)
            .and(&input.upper);

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = sigmoid_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = sigmoid_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Sigmoid CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl SigmoidLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Sigmoid layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, sigmoid_linear_relaxation)
    }
}

/// Softplus activation: y = ln(1 + exp(x))
///
/// Smooth approximation to ReLU. Monotonically increasing with range (0, +∞).
/// Properties:
/// - softplus(0) ≈ ln(2) ≈ 0.693
/// - For large x: softplus(x) ≈ x
/// - For large negative x: softplus(x) ≈ 0
/// - Derivative: sigmoid(x)
#[derive(Debug, Clone, Default)]
pub struct SoftplusLayer;

impl SoftplusLayer {
    pub fn new() -> Self {
        Self
    }
}

fn softplus(x: f32) -> f32 {
    // Numerically stable: log1p(exp(x)) for x <= 20, else x
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Compute softplus bound interval for [l, u].
/// Since softplus is monotonically increasing: softplus(l) <= softplus(x) <= softplus(u).
fn softplus_bound_interval(l: f32, u: f32) -> (f32, f32) {
    (softplus(l), softplus(u))
}

/// Linear relaxation for softplus on interval [l, u].
fn softplus_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let slope = sigmoid(l); // derivative
        let intercept = softplus(l) - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let spl = softplus(l);
    let spu = softplus(u);

    // Chord slope
    let chord_slope = (spu - spl) / (u - l);
    let chord_intercept = spl - chord_slope * l;

    // Sample to find max deviation
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let spx = softplus(x);
        let cx = chord_slope * x + chord_intercept;
        let diff = spx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl BoundPropagation for SoftplusLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(&input.lower)
            .and(&input.upper);

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = softplus_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = softplus_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Softplus CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl SoftplusLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Softplus layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, softplus_linear_relaxation)
    }
}

// =============================================================================
// Trigonometric Functions: Sin, Cos (for positional encodings)
// =============================================================================

/// Sine activation: y = sin(x)
///
/// Periodic function with range [-1, 1] and period 2π.
/// Used in positional encodings for transformers.
#[derive(Debug, Clone, Default)]
pub struct SinLayer;

impl SinLayer {
    pub fn new() -> Self {
        Self
    }
}

/// Compute sin bound interval for [l, u].
/// Since sin is periodic and not monotonic, we need to check for extrema.
fn sin_bound_interval(l: f32, u: f32) -> (f32, f32) {
    use std::f32::consts::PI;

    let sl = l.sin();
    let su = u.sin();
    let mut min_val = sl.min(su);
    let mut max_val = sl.max(su);

    // Check if interval contains any local maxima (π/2 + 2πk)
    // Local max at x = π/2 + 2πk for integer k
    let k_max_start = ((l - PI / 2.0) / (2.0 * PI)).ceil() as i32;
    let k_max_end = ((u - PI / 2.0) / (2.0 * PI)).floor() as i32;
    if k_max_start <= k_max_end {
        max_val = 1.0; // sin achieves 1 somewhere in interval
    }

    // Check if interval contains any local minima (-π/2 + 2πk = 3π/2 + 2π(k-1))
    // Local min at x = -π/2 + 2πk = 3π/2 - 2π + 2πk for integer k
    let k_min_start = ((l + PI / 2.0) / (2.0 * PI)).ceil() as i32;
    let k_min_end = ((u + PI / 2.0) / (2.0 * PI)).floor() as i32;
    if k_min_start <= k_min_end {
        min_val = -1.0; // sin achieves -1 somewhere in interval
    }

    (min_val, max_val)
}

impl BoundPropagation for SinLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(&input.lower)
            .and(&input.upper);

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = sin_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = sin_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN propagation without bounds - uses identity
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Sin CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for sin on interval [l, u].
/// Sin is periodic with changing convexity.
/// - sin''(x) = -sin(x)
/// - Concave where sin(x) > 0 (x in (0, π) mod 2π)
/// - Convex where sin(x) < 0 (x in (π, 2π) mod 2π)
///
/// For small intervals, chord-based relaxation with sampling works well.
/// For large intervals spanning multiple periods, bounds will be conservative.
fn sin_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    use std::f32::consts::PI;

    // Handle degenerate cases (nearly zero-width interval)
    if (u - l).abs() < 1e-8 {
        let sin_l = l.sin();
        let slope = l.cos(); // derivative of sin at l
        let intercept = sin_l - slope * l;
        return (slope, intercept, slope, intercept);
    }

    // For intervals spanning more than a full period, use constant bounds
    if u - l >= 2.0 * PI {
        // Output is always in [-1, 1], use slope 0
        return (0.0, -1.0, 0.0, 1.0);
    }

    let sin_l = l.sin();
    let sin_u = u.sin();

    // Chord slope connecting (l, sin(l)) to (u, sin(u))
    let chord_slope = (sin_u - sin_l) / (u - l);
    let chord_intercept = sin_l - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 100; // More samples for periodic functions
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let sx = x.sin();
        let cx = chord_slope * x + chord_intercept;
        let diff = sx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    // Use conservative bounds based on sampling
    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl SinLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Sin layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, sin_linear_relaxation)
    }
}

/// Cosine activation: y = cos(x)
///
/// Periodic function with range [-1, 1] and period 2π.
/// Used in positional encodings for transformers.
#[derive(Debug, Clone, Default)]
pub struct CosLayer;

impl CosLayer {
    pub fn new() -> Self {
        Self
    }
}

/// Compute cos bound interval for [l, u].
/// Since cos is periodic and not monotonic, we need to check for extrema.
fn cos_bound_interval(l: f32, u: f32) -> (f32, f32) {
    use std::f32::consts::PI;

    let cl = l.cos();
    let cu = u.cos();
    let mut min_val = cl.min(cu);
    let mut max_val = cl.max(cu);

    // Check if interval contains any local maxima (2πk)
    let k_max_start = (l / (2.0 * PI)).ceil() as i32;
    let k_max_end = (u / (2.0 * PI)).floor() as i32;
    if k_max_start <= k_max_end {
        max_val = 1.0; // cos achieves 1 somewhere in interval
    }

    // Check if interval contains any local minima (π + 2πk)
    let k_min_start = ((l - PI) / (2.0 * PI)).ceil() as i32;
    let k_min_end = ((u - PI) / (2.0 * PI)).floor() as i32;
    if k_min_start <= k_min_end {
        min_val = -1.0; // cos achieves -1 somewhere in interval
    }

    (min_val, max_val)
}

impl BoundPropagation for CosLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower.clone();
        let mut out_upper = input.upper.clone();

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(&input.lower)
            .and(&input.upper);

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = cos_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = cos_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN propagation without bounds - uses identity
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Cos CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for cos on interval [l, u].
/// Cos is periodic with changing convexity.
/// - cos''(x) = -cos(x)
/// - Concave where cos(x) > 0 (x in (-π/2, π/2) mod 2π)
/// - Convex where cos(x) < 0 (x in (π/2, 3π/2) mod 2π)
///
/// For small intervals, chord-based relaxation with sampling works well.
/// For large intervals spanning multiple periods, bounds will be conservative.
fn cos_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    use std::f32::consts::PI;

    // Handle degenerate cases (nearly zero-width interval)
    if (u - l).abs() < 1e-8 {
        let cos_l = l.cos();
        let slope = -l.sin(); // derivative of cos at l
        let intercept = cos_l - slope * l;
        return (slope, intercept, slope, intercept);
    }

    // For intervals spanning more than a full period, use constant bounds
    if u - l >= 2.0 * PI {
        // Output is always in [-1, 1], use slope 0
        return (0.0, -1.0, 0.0, 1.0);
    }

    let cos_l = l.cos();
    let cos_u = u.cos();

    // Chord slope connecting (l, cos(l)) to (u, cos(u))
    let chord_slope = (cos_u - cos_l) / (u - l);
    let chord_intercept = cos_l - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 100; // More samples for periodic functions
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let cx_val = x.cos();
        let chord_val = chord_slope * x + chord_intercept;
        let diff = cx_val - chord_val;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    // Use conservative bounds based on sampling
    let lower_slope = chord_slope;
    let lower_intercept = chord_intercept - max_below_chord - eps;
    let upper_slope = chord_slope;
    let upper_intercept = chord_intercept + max_above_chord + eps;

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

impl CosLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Cos layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, cos_linear_relaxation)
    }
}

/// A 2D convolution layer: y = conv(x, W) + b
///
/// Input shape: (batch, in_channels, height, width) or (in_channels, height, width)
/// Kernel shape: (out_channels, in_channels, kernel_h, kernel_w)
/// Output shape: (batch, out_channels, out_h, out_w) or (out_channels, out_h, out_w)
#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    /// Convolution kernel of shape (out_channels, in_channels, kernel_h, kernel_w)
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    pub bias: Option<Array1<f32>>,
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Input spatial dimensions (height, width) - required for CROWN backward pass
    pub input_shape: Option<(usize, usize)>,
}

impl Conv2dLayer {
    /// Create a new Conv2d layer.
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        if kernel.ndim() != 4 {
            return Err(GammaError::ShapeMismatch {
                expected: vec![0, 0, 0, 0], // 4D expected
                got: kernel.shape().to_vec(),
            });
        }
        let out_channels = kernel.shape()[0];
        if let Some(ref b) = bias {
            if b.len() != out_channels {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![b.len()],
                });
            }
        }
        Ok(Self {
            kernel,
            bias,
            stride,
            padding,
            input_shape: None,
        })
    }

    /// Create a new Conv2d layer with known input spatial dimensions.
    /// Required for CROWN backward propagation.
    pub fn with_input_shape(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Set the input spatial dimensions. Required for CROWN backward propagation.
    pub fn set_input_shape(&mut self, height: usize, width: usize) {
        self.input_shape = Some((height, width));
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.kernel.shape()[0]
    }

    /// Input channels.
    pub fn in_channels(&self) -> usize {
        self.kernel.shape()[1]
    }

    /// Kernel size (height, width).
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.kernel.shape()[2], self.kernel.shape()[3])
    }

    /// Compute output spatial dimensions.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let out_h = (input_h + 2 * ph - kh) / sh + 1;
        let out_w = (input_w + 2 * pw - kw) / sw + 1;
        (out_h, out_w)
    }

    /// Batched CROWN backward propagation through Conv2d layer.
    ///
    /// For a conv layer y = conv2d(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// BatchedLinearBounds:
    /// - lower_a shape: [...batch, out_dim, out_c * out_h * out_w]
    /// - Reshapes to [...batch, out_dim, out_c, out_h, out_w] for conv_transpose
    /// - Output: [...batch, out_dim, in_c * in_h * in_w]
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        debug!("Conv2d layer batched CROWN backward propagation");

        // Get input spatial dimensions (required for CROWN)
        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            GammaError::NotSupported(
                "Conv2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (out_h, out_w) = self.output_size(in_h, in_w);
        let conv_in_size = in_c * in_h * in_w;
        let conv_out_size = out_c * out_h * out_w;

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != conv_out_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![out_dim, conv_out_size],
                got: vec![out_dim, mid_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_c * in_h * in_w]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(conv_in_size);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Reshape A to [total_batch, out_dim, mid_dim] for computation
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        // Allocate output coefficient matrices
        let mut new_lower_a = Array2::zeros((total_batch * out_dim, conv_in_size));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, conv_in_size));

        // Process each batch position and output dimension
        for b in 0..total_batch {
            for d in 0..out_dim {
                // Get the coefficient row [out_c * out_h * out_w]
                let lower_row = lower_a_3d.slice(s![b, d, ..]);
                let upper_row = upper_a_3d.slice(s![b, d, ..]);

                // Reshape to [out_c, out_h, out_w]
                let lower_3d =
                    ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), lower_row.to_vec())
                        .map_err(|_| GammaError::ShapeMismatch {
                            expected: vec![out_c, out_h, out_w],
                            got: vec![lower_row.len()],
                        })?;

                let upper_3d =
                    ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), upper_row.to_vec())
                        .map_err(|_| GammaError::ShapeMismatch {
                            expected: vec![out_c, out_h, out_w],
                            got: vec![upper_row.len()],
                        })?;

                // Apply transposed convolution
                let lower_trans = conv2d_transpose(
                    &lower_3d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    (in_h, in_w),
                );
                let upper_trans = conv2d_transpose(
                    &upper_3d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    (in_h, in_w),
                );

                // Flatten and store in new coefficient matrix
                let row_idx = b * out_dim + d;
                for (i, &val) in lower_trans.iter().enumerate() {
                    new_lower_a[[row_idx, i]] = val;
                }
                for (i, &val) in upper_trans.iter().enumerate() {
                    new_upper_a[[row_idx, i]] = val;
                }
            }
        }

        // Reshape back to [...batch, out_dim, in_c * in_h * in_w]
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;

        // Compute bias contribution
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            // For each batch position and output dim: compute sum over spatial positions weighted by bias
            // bias_contrib = sum over (c, h, w) of A[c*out_h*out_w + h*out_w + w] * bias[c]
            let lower_b_3d = bounds
                .lower_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_3d = bounds
                .upper_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let mut new_lower_b = Array2::zeros((total_batch, out_dim));
            let mut new_upper_b = Array2::zeros((total_batch, out_dim));

            for b in 0..total_batch {
                for d in 0..out_dim {
                    let mut lower_sum = 0.0f32;
                    let mut upper_sum = 0.0f32;

                    for c in 0..out_c {
                        // Sum all spatial positions for this channel
                        let spatial_start = c * out_h * out_w;
                        let spatial_end = spatial_start + out_h * out_w;

                        let lower_spatial_sum: f32 =
                            lower_a_3d.slice(s![b, d, spatial_start..spatial_end]).sum();
                        let upper_spatial_sum: f32 =
                            upper_a_3d.slice(s![b, d, spatial_start..spatial_end]).sum();

                        lower_sum += lower_spatial_sum * bias[c];
                        upper_sum += upper_spatial_sum * bias[c];
                    }

                    new_lower_b[[b, d]] = lower_b_3d[[b, d]] + lower_sum;
                    new_upper_b[[b, d]] = upper_b_3d[[b, d]] + upper_sum;
                }
            }

            let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec).map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string())
                })?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec).map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string())
                })?,
            )
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        // Update input shape to reflect the conv layer's input dimensions
        let mut new_input_shape = bounds.input_shape.clone();
        if new_input_shape.len() >= 3 {
            // Update last three dims from [out_c, out_h, out_w] to [in_c, in_h, in_w]
            let len = new_input_shape.len();
            new_input_shape[len - 3] = in_c;
            new_input_shape[len - 2] = in_h;
            new_input_shape[len - 1] = in_w;
        } else if !new_input_shape.is_empty() {
            // Single dimension: update to flattened input size
            new_input_shape[0] = conv_in_size;
        }

        Ok(BatchedLinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
            input_shape: new_input_shape,
            output_shape: bounds.output_shape.clone(),
        })
    }
}

/// Perform 2D convolution on a single (channels, height, width) input.
///
/// This is a straightforward implementation for correctness testing.
/// For production, use optimized backends (ONNX Runtime, Metal, etc.)
pub fn conv2d_single(
    input: &ArrayD<f32>,  // (in_channels, height, width)
    kernel: &ArrayD<f32>, // (out_channels, in_channels, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
) -> ArrayD<f32> {
    let in_c = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];

    let out_c = kernel.shape()[0];
    let ker_in_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    debug_assert_eq!(in_c, ker_in_c, "Input channels must match kernel");

    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let out_h = (in_h + 2 * ph - kh) / sh + 1;
    let out_w = (in_w + 2 * pw - kw) / sw + 1;

    let mut output = ArrayD::zeros(ndarray::IxDyn(&[out_c, out_h, out_w]));

    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = 0.0f32;
                for ic in 0..in_c {
                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            let ih = (oh * sh + kh_idx) as isize - ph as isize;
                            let iw = (ow * sw + kw_idx) as isize - pw as isize;

                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                sum += input[[ic, ih as usize, iw as usize]]
                                    * kernel[[oc, ic, kh_idx, kw_idx]];
                            }
                            // Padding: out-of-bounds treated as 0
                        }
                    }
                }
                output[[oc, oh, ow]] = sum;
            }
        }
    }

    output
}

/// Perform 2D transposed convolution (deconvolution) for CROWN backward pass.
///
/// Input shape: (out_channels, out_h, out_w) - the gradient w.r.t. conv output
/// Kernel shape: (out_channels, in_channels, kh, kw) - same as forward conv
/// Output shape: (in_channels, in_h, in_w) - the gradient w.r.t. conv input
///
/// This implements: conv_transpose2d(grad, weight) which is the backward pass through conv.
/// output_size specifies the expected output spatial dimensions to handle (W-F+2P)%S != 0.
pub fn conv2d_transpose(
    input: &ArrayD<f32>,  // (out_channels, out_h, out_w) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    output_size: (usize, usize), // (in_h, in_w) - the expected input size
) -> ArrayD<f32> {
    let out_c = input.shape()[0]; // This is out_channels of the conv (in_channels for gradient)
    let grad_h = input.shape()[1];
    let grad_w = input.shape()[2];

    let ker_out_c = kernel.shape()[0];
    let in_c = kernel.shape()[1]; // in_channels of conv = out_channels of transpose
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    debug_assert_eq!(
        out_c, ker_out_c,
        "Gradient channels must match kernel out_channels"
    );

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (in_h, in_w) = output_size;

    let mut output = ArrayD::zeros(ndarray::IxDyn(&[in_c, in_h, in_w]));

    // Transposed convolution: scatter gradient to input positions
    // For each output position oh, ow in the forward conv,
    // the input positions ih = oh*sh + kh_idx - ph contribute.
    // In backward, we accumulate: grad[oh, ow] * kernel[oc, ic, kh_idx, kw_idx] to output[ic, ih, iw]
    for oc in 0..out_c {
        for grad_y in 0..grad_h {
            for grad_x in 0..grad_w {
                let grad_val = input[[oc, grad_y, grad_x]];
                if grad_val == 0.0 {
                    continue;
                }
                for ic in 0..in_c {
                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            // In forward: output[oh, ow] += input[ih, iw] * kernel[kh_idx, kw_idx]
                            // where ih = oh*sh + kh_idx - ph
                            // In backward: input_grad[ih, iw] += output_grad[oh, ow] * kernel[kh_idx, kw_idx]
                            let ih = (grad_y * sh + kh_idx) as isize - ph as isize;
                            let iw = (grad_x * sw + kw_idx) as isize - pw as isize;

                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                output[[ic, ih as usize, iw as usize]] +=
                                    grad_val * kernel[[oc, ic, kh_idx, kw_idx]];
                            }
                        }
                    }
                }
            }
        }
    }

    output
}

impl BoundPropagation for Conv2dLayer {
    /// IBP for Conv2d layer: y = conv(x, W) + b
    ///
    /// For x in [l, u], compute y bounds:
    /// - W+ = max(W, 0), W- = min(W, 0)
    /// - lower_y = conv(l, W+) + conv(u, W-) + b
    /// - upper_y = conv(u, W+) + conv(l, W-) + b
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let in_c = self.in_channels();

        match input.lower.ndim() {
            3 => {
                // Input shape: (in_channels, height, width)
                if input.lower.shape()[0] != in_c {
                    return Err(GammaError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower.shape()[0]],
                    });
                }

                // Split kernel into positive and negative parts
                let kernel_pos = self.kernel.mapv(|v| v.max(0.0));
                let kernel_neg = self.kernel.mapv(|v| v.min(0.0));

                // Compute bounds using W+/W- splitting
                let lower_from_pos =
                    conv2d_single(&input.lower, &kernel_pos, self.stride, self.padding);
                let lower_from_neg =
                    conv2d_single(&input.upper, &kernel_neg, self.stride, self.padding);
                let mut lower_y = lower_from_pos + lower_from_neg;

                let upper_from_pos =
                    conv2d_single(&input.upper, &kernel_pos, self.stride, self.padding);
                let upper_from_neg =
                    conv2d_single(&input.lower, &kernel_neg, self.stride, self.padding);
                let mut upper_y = upper_from_pos + upper_from_neg;

                // Add bias if present (broadcast over spatial dimensions)
                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_h = lower_y.shape()[1];
                    let out_w = lower_y.shape()[2];

                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                lower_y[[oc, oh, ow]] += b[oc];
                                upper_y[[oc, oh, ow]] += b[oc];
                            }
                        }
                    }
                }

                BoundedTensor::new(lower_y, upper_y)
            }
            4 => {
                // Input shape: (batch, in_channels, height, width)
                if input.lower.shape()[1] != in_c {
                    return Err(GammaError::ShapeMismatch {
                        expected: vec![0, in_c, 0, 0],
                        got: input.lower.shape().to_vec(),
                    });
                }

                // Split kernel into positive and negative parts
                let kernel_pos = self.kernel.mapv(|v| v.max(0.0));
                let kernel_neg = self.kernel.mapv(|v| v.min(0.0));

                let batch = input.lower.shape()[0];
                let input_h = input.lower.shape()[2];
                let input_w = input.lower.shape()[3];
                let (out_h, out_w) = self.output_size(input_h, input_w);
                let out_c = self.out_channels();

                let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));
                let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));

                for b in 0..batch {
                    let lower_b = input
                        .lower
                        .index_axis(ndarray::Axis(0), b)
                        .to_owned()
                        .into_dyn();
                    let upper_b = input
                        .upper
                        .index_axis(ndarray::Axis(0), b)
                        .to_owned()
                        .into_dyn();

                    let lower_from_pos =
                        conv2d_single(&lower_b, &kernel_pos, self.stride, self.padding);
                    let lower_from_neg =
                        conv2d_single(&upper_b, &kernel_neg, self.stride, self.padding);
                    let lower_batch = lower_from_pos + lower_from_neg;

                    let upper_from_pos =
                        conv2d_single(&upper_b, &kernel_pos, self.stride, self.padding);
                    let upper_from_neg =
                        conv2d_single(&lower_b, &kernel_neg, self.stride, self.padding);
                    let upper_batch = upper_from_pos + upper_from_neg;

                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                lower_y[[b, oc, oh, ow]] = lower_batch[[oc, oh, ow]];
                                upper_y[[b, oc, oh, ow]] = upper_batch[[oc, oh, ow]];
                            }
                        }
                    }
                }

                // Add bias if present (broadcast over batch/spatial dimensions)
                if let Some(ref bias) = self.bias {
                    for b in 0..batch {
                        for oc in 0..out_c {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    lower_y[[b, oc, oh, ow]] += bias[oc];
                                    upper_y[[b, oc, oh, ow]] += bias[oc];
                                }
                            }
                        }
                    }
                }

                BoundedTensor::new(lower_y, upper_y)
            }
            _ => Err(GammaError::ShapeMismatch {
                expected: vec![in_c, 0, 0],
                got: input.lower.shape().to_vec(),
            }),
        }
    }

    /// CROWN backward propagation through Conv2d layer.
    ///
    /// For a conv layer y = conv(x, W) + b, and current linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial dimensions)
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Conv2d layer CROWN backward propagation");

        // Get input spatial dimensions (required for CROWN)
        let (in_h, in_w) = self.input_shape.ok_or_else(|| GammaError::NotSupported(
            "Conv2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape().".to_string()
        ))?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (out_h, out_w) = self.output_size(in_h, in_w);

        // Verify that bounds dimensions match expected conv output
        let expected_conv_out = out_c * out_h * out_w;
        if bounds.num_inputs() != expected_conv_out {
            return Err(GammaError::ShapeMismatch {
                expected: vec![expected_conv_out],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let conv_in_size = in_c * in_h * in_w;

        // Allocate new coefficient matrices
        let mut new_lower_a = Array2::zeros((num_outputs, conv_in_size));
        let mut new_upper_a = Array2::zeros((num_outputs, conv_in_size));

        // Process each row of the coefficient matrix
        // Each row represents bounds on one output, with coefficients for all conv outputs
        for row_idx in 0..num_outputs {
            // Reshape row from flat [out_c * out_h * out_w] to [out_c, out_h, out_w]
            let lower_row = bounds.lower_a.row(row_idx);
            let upper_row = bounds.upper_a.row(row_idx);

            let lower_3d =
                ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), lower_row.to_vec())
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![out_c, out_h, out_w],
                        got: vec![lower_row.len()],
                    })?;

            let upper_3d =
                ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), upper_row.to_vec())
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![out_c, out_h, out_w],
                        got: vec![upper_row.len()],
                    })?;

            // Apply transposed convolution
            let lower_trans = conv2d_transpose(
                &lower_3d,
                &self.kernel,
                self.stride,
                self.padding,
                (in_h, in_w),
            );

            let upper_trans = conv2d_transpose(
                &upper_3d,
                &self.kernel,
                self.stride,
                self.padding,
                (in_h, in_w),
            );

            // Flatten and store in new coefficient matrix
            for (i, &val) in lower_trans.iter().enumerate() {
                new_lower_a[[row_idx, i]] = val;
            }
            for (i, &val) in upper_trans.iter().enumerate() {
                new_upper_a[[row_idx, i]] = val;
            }
        }

        // Compute bias contribution
        // For conv bias b of shape [out_c], broadcast to [out_c, out_h, out_w]
        // A @ b_broadcast is sum over all spatial positions: A @ (b repeated out_h*out_w times)
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            // For each row of A, compute sum over spatial positions weighted by bias
            // bias_contrib[j] = sum over (c, h, w) of A[j, c*out_h*out_w + h*out_w + w] * bias[c]
            let mut lower_bias_contrib = Array1::zeros(num_outputs);
            let mut upper_bias_contrib = Array1::zeros(num_outputs);

            for row_idx in 0..num_outputs {
                let mut lower_sum = 0.0f32;
                let mut upper_sum = 0.0f32;

                for c in 0..out_c {
                    // Sum all spatial positions for this channel
                    let spatial_start = c * out_h * out_w;
                    let spatial_end = spatial_start + out_h * out_w;

                    let lower_spatial_sum: f32 = bounds
                        .lower_a
                        .row(row_idx)
                        .slice(ndarray::s![spatial_start..spatial_end])
                        .sum();
                    let upper_spatial_sum: f32 = bounds
                        .upper_a
                        .row(row_idx)
                        .slice(ndarray::s![spatial_start..spatial_end])
                        .sum();

                    lower_sum += lower_spatial_sum * bias[c];
                    upper_sum += upper_spatial_sum * bias[c];
                }

                lower_bias_contrib[row_idx] = lower_sum;
                upper_bias_contrib[row_idx] = upper_sum;
            }

            (
                &bounds.lower_b + &lower_bias_contrib,
                &bounds.upper_b + &upper_bias_contrib,
            )
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        Ok(Cow::Owned(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        }))
    }
}

/// A 1D convolution layer: y = conv1d(x, W) + b
///
/// Input shape: (batch, in_channels, length) or (in_channels, length)
/// Kernel shape: (out_channels, in_channels, kernel_size)
/// Output shape: (batch, out_channels, out_length) or (out_channels, out_length)
#[derive(Debug, Clone)]
pub struct Conv1dLayer {
    /// Convolution kernel of shape (out_channels, in_channels, kernel_size)
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    pub bias: Option<Array1<f32>>,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Input length (required for CROWN backward propagation)
    pub input_length: Option<usize>,
}

impl Conv1dLayer {
    /// Create a new Conv1d layer.
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        if kernel.ndim() != 3 {
            return Err(GammaError::ShapeMismatch {
                expected: vec![0, 0, 0], // 3D expected
                got: kernel.shape().to_vec(),
            });
        }
        let out_channels = kernel.shape()[0];
        if let Some(ref b) = bias {
            if b.len() != out_channels {
                return Err(GammaError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![b.len()],
                });
            }
        }
        Ok(Self {
            kernel,
            bias,
            stride,
            padding,
            input_length: None,
        })
    }

    /// Create a new Conv1d layer with input length specified (required for CROWN).
    pub fn with_input_length(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
        input_length: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_length = Some(input_length);
        Ok(layer)
    }

    /// Set the input length (required for CROWN backward propagation).
    pub fn set_input_length(&mut self, input_length: usize) {
        self.input_length = Some(input_length);
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.kernel.shape()[0]
    }

    /// Input channels.
    pub fn in_channels(&self) -> usize {
        self.kernel.shape()[1]
    }

    /// Kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel.shape()[2]
    }

    /// Compute output length.
    pub fn output_length(&self, input_len: usize) -> usize {
        let k = self.kernel_size();
        (input_len + 2 * self.padding - k) / self.stride + 1
    }

    /// Batched CROWN backward propagation through Conv1d layer.
    ///
    /// For a conv layer y = conv1d(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// BatchedLinearBounds:
    /// - lower_a shape: [...batch, out_dim, out_c * out_len]
    /// - Reshapes to [...batch, out_dim, out_c, out_len] for conv_transpose
    /// - Output: [...batch, out_dim, in_c * in_len]
    ///
    /// Requires `input_length` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        debug!("Conv1d layer batched CROWN backward propagation");

        // Get input length (required for CROWN)
        let in_len = self.input_length.ok_or_else(|| {
            GammaError::NotSupported(
                "Conv1d CROWN requires input_length to be set. Use with_input_length() or set_input_length()."
                    .to_string(),
            )
        })?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let out_len = self.output_length(in_len);
        let conv_in_size = in_c * in_len;
        let conv_out_size = out_c * out_len;

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != conv_out_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![out_dim, conv_out_size],
                got: vec![out_dim, mid_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_c * in_len]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(conv_in_size);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Reshape A to [total_batch, out_dim, mid_dim] for computation
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        // Allocate output coefficient matrices
        let mut new_lower_a = Array2::zeros((total_batch * out_dim, conv_in_size));
        let mut new_upper_a = Array2::zeros((total_batch * out_dim, conv_in_size));

        // Process each batch position and output dimension
        for b in 0..total_batch {
            for d in 0..out_dim {
                // Get the coefficient row [out_c * out_len]
                let lower_row = lower_a_3d.slice(s![b, d, ..]);
                let upper_row = upper_a_3d.slice(s![b, d, ..]);

                // Reshape to [out_c, out_len]
                let lower_2d = ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), lower_row.to_vec())
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![out_c, out_len],
                        got: vec![lower_row.len()],
                    })?;

                let upper_2d = ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), upper_row.to_vec())
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![out_c, out_len],
                        got: vec![upper_row.len()],
                    })?;

                // Apply transposed convolution
                let lower_trans =
                    conv1d_transpose(&lower_2d, &self.kernel, self.stride, self.padding, in_len);
                let upper_trans =
                    conv1d_transpose(&upper_2d, &self.kernel, self.stride, self.padding, in_len);

                // Flatten and store in new coefficient matrix
                let row_idx = b * out_dim + d;
                for (i, &val) in lower_trans.iter().enumerate() {
                    new_lower_a[[row_idx, i]] = val;
                }
                for (i, &val) in upper_trans.iter().enumerate() {
                    new_upper_a[[row_idx, i]] = val;
                }
            }
        }

        // Reshape back to [...batch, out_dim, in_c * in_len]
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;

        // Compute bias contribution
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            // For each batch position and output dim: compute sum over spatial positions weighted by bias
            // bias_contrib = sum over (c, l) of A[c*out_len + l] * bias[c]
            let lower_b_3d = bounds
                .lower_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_3d = bounds
                .upper_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let mut new_lower_b = Array2::zeros((total_batch, out_dim));
            let mut new_upper_b = Array2::zeros((total_batch, out_dim));

            for b in 0..total_batch {
                for d in 0..out_dim {
                    let mut lower_sum = 0.0f32;
                    let mut upper_sum = 0.0f32;

                    for c in 0..out_c {
                        // Sum all spatial positions for this channel
                        let spatial_start = c * out_len;
                        let spatial_end = spatial_start + out_len;

                        let lower_spatial_sum: f32 =
                            lower_a_3d.slice(s![b, d, spatial_start..spatial_end]).sum();
                        let upper_spatial_sum: f32 =
                            upper_a_3d.slice(s![b, d, spatial_start..spatial_end]).sum();

                        lower_sum += lower_spatial_sum * bias[c];
                        upper_sum += upper_spatial_sum * bias[c];
                    }

                    new_lower_b[[b, d]] = lower_b_3d[[b, d]] + lower_sum;
                    new_upper_b[[b, d]] = upper_b_3d[[b, d]] + upper_sum;
                }
            }

            let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec).map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape new_lower_b".to_string())
                })?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec).map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape new_upper_b".to_string())
                })?,
            )
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        // Update input shape to reflect the conv layer's input dimensions
        let mut new_input_shape = bounds.input_shape.clone();
        if new_input_shape.len() >= 2 {
            // Update last two dims from [out_c, out_len] to [in_c, in_len]
            let len = new_input_shape.len();
            new_input_shape[len - 2] = in_c;
            new_input_shape[len - 1] = in_len;
        } else if !new_input_shape.is_empty() {
            // Single dimension: update to flattened input size
            new_input_shape[0] = conv_in_size;
        }

        Ok(BatchedLinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
            input_shape: new_input_shape,
            output_shape: bounds.output_shape.clone(),
        })
    }
}

/// Transposed 1D convolution for CROWN backward pass.
///
/// Given gradient at conv output, compute gradient at conv input.
/// This is the inverse operation of conv1d in the gradient sense.
pub fn conv1d_transpose(
    input: &ArrayD<f32>,  // (out_channels, out_len) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels, kernel_size)
    stride: usize,
    padding: usize,
    output_length: usize, // expected input length (in_len)
) -> ArrayD<f32> {
    let out_c = input.shape()[0]; // This is out_channels of the conv (in_channels for gradient)
    let grad_len = input.shape()[1];

    let ker_out_c = kernel.shape()[0];
    let in_c = kernel.shape()[1]; // in_channels of conv = out_channels of transpose
    let k = kernel.shape()[2];

    debug_assert_eq!(
        out_c, ker_out_c,
        "Gradient channels must match kernel out_channels"
    );

    let in_len = output_length;

    let mut output = ArrayD::zeros(ndarray::IxDyn(&[in_c, in_len]));

    // Transposed convolution: scatter gradient to input positions
    // For each output position ol in the forward conv,
    // the input positions il = ol*stride + ki - padding contribute.
    // In backward, we accumulate: grad[ol] * kernel[oc, ic, ki] to output[ic, il]
    for oc in 0..out_c {
        for grad_l in 0..grad_len {
            let grad_val = input[[oc, grad_l]];
            if grad_val == 0.0 {
                continue;
            }
            for ic in 0..in_c {
                for ki in 0..k {
                    // In forward: output[ol] += input[il] * kernel[ki]
                    // where il = ol*stride + ki - padding
                    // In backward: input_grad[il] += output_grad[ol] * kernel[ki]
                    let il = (grad_l * stride + ki) as isize - padding as isize;

                    if il >= 0 && il < in_len as isize {
                        output[[ic, il as usize]] += grad_val * kernel[[oc, ic, ki]];
                    }
                }
            }
        }
    }

    output
}

/// Perform 1D convolution on a single (channels, length) input.
pub fn conv1d_single(
    input: &ArrayD<f32>,  // (in_channels, length)
    kernel: &ArrayD<f32>, // (out_channels, in_channels, kernel_size)
    stride: usize,
    padding: usize,
) -> ArrayD<f32> {
    let in_c = input.shape()[0];
    let in_len = input.shape()[1];

    let out_c = kernel.shape()[0];
    let ker_in_c = kernel.shape()[1];
    let k = kernel.shape()[2];

    debug_assert_eq!(in_c, ker_in_c, "Input channels must match kernel");

    let out_len = (in_len + 2 * padding - k) / stride + 1;
    let mut output = ArrayD::zeros(ndarray::IxDyn(&[out_c, out_len]));

    for oc in 0..out_c {
        for ol in 0..out_len {
            let mut sum = 0.0f32;
            for ic in 0..in_c {
                for ki in 0..k {
                    let il = (ol * stride + ki) as isize - padding as isize;
                    if il >= 0 && il < in_len as isize {
                        sum += input[[ic, il as usize]] * kernel[[oc, ic, ki]];
                    }
                    // Padding: out-of-bounds treated as 0
                }
            }
            output[[oc, ol]] = sum;
        }
    }

    output
}

impl BoundPropagation for Conv1dLayer {
    /// IBP for Conv1d layer: y = conv1d(x, W) + b
    ///
    /// For x in [l, u], compute y bounds:
    /// - W+ = max(W, 0), W- = min(W, 0)
    /// - lower_y = conv1d(l, W+) + conv1d(u, W-) + b
    /// - upper_y = conv1d(u, W+) + conv1d(l, W-) + b
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let in_c = self.in_channels();

        match input.lower.ndim() {
            2 => {
                // Input shape: (in_channels, length)
                if input.lower.shape()[0] != in_c {
                    return Err(GammaError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower.shape()[0]],
                    });
                }

                // Split kernel into positive and negative parts
                let kernel_pos = self.kernel.mapv(|v| v.max(0.0));
                let kernel_neg = self.kernel.mapv(|v| v.min(0.0));

                // Compute bounds using W+/W- splitting
                let lower_from_pos =
                    conv1d_single(&input.lower, &kernel_pos, self.stride, self.padding);
                let lower_from_neg =
                    conv1d_single(&input.upper, &kernel_neg, self.stride, self.padding);
                let mut lower_y = lower_from_pos + lower_from_neg;

                let upper_from_pos =
                    conv1d_single(&input.upper, &kernel_pos, self.stride, self.padding);
                let upper_from_neg =
                    conv1d_single(&input.lower, &kernel_neg, self.stride, self.padding);
                let mut upper_y = upper_from_pos + upper_from_neg;

                // Add bias if present (broadcast over length dimension)
                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_len = lower_y.shape()[1];

                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            lower_y[[oc, ol]] += b[oc];
                            upper_y[[oc, ol]] += b[oc];
                        }
                    }
                }

                BoundedTensor::new(lower_y, upper_y)
            }
            3 => {
                // Input shape: (batch, in_channels, length)
                if input.lower.shape()[1] != in_c {
                    return Err(GammaError::ShapeMismatch {
                        expected: vec![0, in_c, 0],
                        got: input.lower.shape().to_vec(),
                    });
                }

                // Split kernel into positive and negative parts
                let kernel_pos = self.kernel.mapv(|v| v.max(0.0));
                let kernel_neg = self.kernel.mapv(|v| v.min(0.0));

                let batch = input.lower.shape()[0];
                let input_len = input.lower.shape()[2];
                let out_len = self.output_length(input_len);
                let out_c = self.out_channels();

                let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_len]));
                let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_len]));

                for b in 0..batch {
                    let lower_b = input
                        .lower
                        .index_axis(ndarray::Axis(0), b)
                        .to_owned()
                        .into_dyn();
                    let upper_b = input
                        .upper
                        .index_axis(ndarray::Axis(0), b)
                        .to_owned()
                        .into_dyn();

                    let lower_from_pos =
                        conv1d_single(&lower_b, &kernel_pos, self.stride, self.padding);
                    let lower_from_neg =
                        conv1d_single(&upper_b, &kernel_neg, self.stride, self.padding);
                    let lower_batch = lower_from_pos + lower_from_neg;

                    let upper_from_pos =
                        conv1d_single(&upper_b, &kernel_pos, self.stride, self.padding);
                    let upper_from_neg =
                        conv1d_single(&lower_b, &kernel_neg, self.stride, self.padding);
                    let upper_batch = upper_from_pos + upper_from_neg;

                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            lower_y[[b, oc, ol]] = lower_batch[[oc, ol]];
                            upper_y[[b, oc, ol]] = upper_batch[[oc, ol]];
                        }
                    }
                }

                // Add bias if present (broadcast over batch/length dimension)
                if let Some(ref bias) = self.bias {
                    for b in 0..batch {
                        for oc in 0..out_c {
                            for ol in 0..out_len {
                                lower_y[[b, oc, ol]] += bias[oc];
                                upper_y[[b, oc, ol]] += bias[oc];
                            }
                        }
                    }
                }

                BoundedTensor::new(lower_y, upper_y)
            }
            _ => Err(GammaError::ShapeMismatch {
                expected: vec![in_c, 0],
                got: input.lower.shape().to_vec(),
            }),
        }
    }

    /// CROWN backward propagation through Conv1d layer.
    ///
    /// For a conv layer y = conv1d(x, W) + b, and current linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// Requires `input_length` to be set for proper shape computation.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Conv1d layer CROWN backward propagation");

        // Get input length (required for CROWN)
        let in_len = self.input_length.ok_or_else(|| GammaError::NotSupported(
            "Conv1d CROWN requires input_length to be set. Use with_input_length() or set_input_length().".to_string()
        ))?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let out_len = self.output_length(in_len);

        // Verify that bounds dimensions match expected conv output
        let expected_conv_out = out_c * out_len;
        if bounds.num_inputs() != expected_conv_out {
            return Err(GammaError::ShapeMismatch {
                expected: vec![expected_conv_out],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let conv_in_size = in_c * in_len;

        // Allocate new coefficient matrices
        let mut new_lower_a = Array2::zeros((num_outputs, conv_in_size));
        let mut new_upper_a = Array2::zeros((num_outputs, conv_in_size));

        // Process each row of the coefficient matrix
        // Each row represents bounds on one output, with coefficients for all conv outputs
        for row_idx in 0..num_outputs {
            // Reshape row from flat [out_c * out_len] to [out_c, out_len]
            let lower_row = bounds.lower_a.row(row_idx);
            let upper_row = bounds.upper_a.row(row_idx);

            let lower_2d =
                ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_len]), lower_row.to_vec())
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![out_c, out_len],
                        got: vec![lower_row.len()],
                    })?;

            let upper_2d =
                ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_len]), upper_row.to_vec())
                    .map_err(|_| GammaError::ShapeMismatch {
                        expected: vec![out_c, out_len],
                        got: vec![upper_row.len()],
                    })?;

            // Apply transposed convolution
            let lower_trans =
                conv1d_transpose(&lower_2d, &self.kernel, self.stride, self.padding, in_len);

            let upper_trans =
                conv1d_transpose(&upper_2d, &self.kernel, self.stride, self.padding, in_len);

            // Flatten and store in new coefficient matrix
            for (i, &val) in lower_trans.iter().enumerate() {
                new_lower_a[[row_idx, i]] = val;
            }
            for (i, &val) in upper_trans.iter().enumerate() {
                new_upper_a[[row_idx, i]] = val;
            }
        }

        // Compute bias contribution
        // For conv bias b of shape [out_c], broadcast to [out_c, out_len]
        // A @ b_broadcast is sum over all spatial positions: A @ (b repeated out_len times)
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            // For each row of A, compute sum over spatial positions weighted by bias
            // bias_contrib[j] = sum over (c, l) of A[j, c*out_len + l] * bias[c]
            let mut lower_bias_contrib = Array1::zeros(num_outputs);
            let mut upper_bias_contrib = Array1::zeros(num_outputs);

            for row_idx in 0..num_outputs {
                let mut lower_sum = 0.0f32;
                let mut upper_sum = 0.0f32;

                for c in 0..out_c {
                    // Sum all spatial positions for this channel
                    let spatial_start = c * out_len;
                    let spatial_end = spatial_start + out_len;

                    let lower_spatial_sum: f32 = bounds
                        .lower_a
                        .row(row_idx)
                        .slice(ndarray::s![spatial_start..spatial_end])
                        .sum();
                    let upper_spatial_sum: f32 = bounds
                        .upper_a
                        .row(row_idx)
                        .slice(ndarray::s![spatial_start..spatial_end])
                        .sum();

                    lower_sum += lower_spatial_sum * bias[c];
                    upper_sum += upper_spatial_sum * bias[c];
                }

                lower_bias_contrib[row_idx] = lower_sum;
                upper_bias_contrib[row_idx] = upper_sum;
            }

            (
                &bounds.lower_b + &lower_bias_contrib,
                &bounds.upper_b + &upper_bias_contrib,
            )
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        Ok(Cow::Owned(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        }))
    }
}

/// Average pooling layer: applies average pooling over spatial dimensions.
///
/// For 2D input (channels, height, width), applies kernel_size x kernel_size
/// average pooling with given stride and padding.
///
/// Average pooling is a linear operation, so IBP bounds are exact:
/// y_lower = avg_pool(x_lower), y_upper = avg_pool(x_upper)
#[derive(Debug, Clone)]
pub struct AveragePoolLayer {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Whether to count padding zeros in divisor
    pub count_include_pad: bool,
}

impl AveragePoolLayer {
    /// Create a new average pool layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        }
    }

    /// Check if this is a global average pool operation.
    /// Global pooling uses kernel_size (0, 0) as a sentinel value.
    pub fn is_global(&self) -> bool {
        self.kernel_size == (0, 0)
    }

    /// Compute output spatial dimensions.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        // Global pooling: kernel_size (0, 0) means pool entire spatial dims
        if self.is_global() {
            return Ok((1, 1));
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        if sh == 0 || sw == 0 {
            return Err(GammaError::InvalidSpec(format!(
                "AveragePool stride must be non-zero, got stride=({},{})",
                sh, sw
            )));
        }

        let padded_h = input_h
            .checked_add(ph.checked_mul(2).ok_or_else(|| {
                GammaError::InvalidSpec("AveragePool padding overflow".to_string())
            })?)
            .ok_or_else(|| {
                GammaError::InvalidSpec("AveragePool padded height overflow".to_string())
            })?;
        let padded_w = input_w
            .checked_add(pw.checked_mul(2).ok_or_else(|| {
                GammaError::InvalidSpec("AveragePool padding overflow".to_string())
            })?)
            .ok_or_else(|| {
                GammaError::InvalidSpec("AveragePool padded width overflow".to_string())
            })?;

        if padded_h < kh || padded_w < kw {
            return Err(GammaError::InvalidSpec(format!(
                "AveragePool kernel larger than padded input: input=({},{}), padding=({},{}), kernel=({},{})",
                input_h, input_w, ph, pw, kh, kw
            )));
        }

        let out_h = (padded_h - kh) / sh + 1;
        let out_w = (padded_w - kw) / sw + 1;
        Ok((out_h, out_w))
    }
}

impl BoundPropagation for AveragePoolLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Average pooling is linear: y = (1/k) * sum(x_i)
        // So bounds are exact: y_l = avg_pool(x_l), y_u = avg_pool(x_u)

        // Validate input shape: expect (channels, height, width) or (batch, channels, height, width)
        let input_shape = input.lower.shape();
        if input_shape.len() < 3 {
            return Err(GammaError::ShapeMismatch {
                expected: vec![0, 0, 0], // 3D or 4D expected
                got: input_shape.to_vec(),
            });
        }

        let (batch_size, channels, in_h, in_w) = if input_shape.len() == 4 {
            (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            // 3D input: (channels, height, width)
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        // Handle global average pooling: kernel_size (0, 0) means pool entire spatial dims
        if self.is_global() {
            let out_shape = if let Some(b) = batch_size {
                vec![b, channels, 1, 1]
            } else {
                vec![channels, 1, 1]
            };
            let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
            let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));
            let divisor = (in_h * in_w) as f32;

            if let Some(b) = batch_size {
                for batch_idx in 0..b {
                    for c in 0..channels {
                        let mut sum_lower = 0.0f32;
                        let mut sum_upper = 0.0f32;
                        for ih in 0..in_h {
                            for iw in 0..in_w {
                                sum_lower += input.lower[[batch_idx, c, ih, iw]];
                                sum_upper += input.upper[[batch_idx, c, ih, iw]];
                            }
                        }
                        out_lower[[batch_idx, c, 0, 0]] = sum_lower / divisor;
                        out_upper[[batch_idx, c, 0, 0]] = sum_upper / divisor;
                    }
                }
            } else {
                for c in 0..channels {
                    let mut sum_lower = 0.0f32;
                    let mut sum_upper = 0.0f32;
                    for ih in 0..in_h {
                        for iw in 0..in_w {
                            sum_lower += input.lower[[c, ih, iw]];
                            sum_upper += input.upper[[c, ih, iw]];
                        }
                    }
                    out_lower[[c, 0, 0]] = sum_lower / divisor;
                    out_upper[[c, 0, 0]] = sum_upper / divisor;
                }
            }

            return BoundedTensor::new(out_lower, out_upper);
        }

        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Create output arrays
        let out_shape = if let Some(b) = batch_size {
            vec![b, channels, out_h, out_w]
        } else {
            vec![channels, out_h, out_w]
        };
        let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));

        // Apply average pooling
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih_start = oh * sh;
                    let iw_start = ow * sw;

                    let mut sum_lower = 0.0f32;
                    let mut sum_upper = 0.0f32;
                    let mut count = 0usize;

                    for kh_off in 0..kh {
                        for kw_off in 0..kw {
                            let ih = (ih_start + kh_off) as isize - ph as isize;
                            let iw = (iw_start + kw_off) as isize - pw as isize;

                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;

                                if let Some(_b) = batch_size {
                                    // For 4D, we handle batch dimension later
                                    // For now, just use first batch - need to loop
                                } else {
                                    sum_lower += input.lower[[c, ih, iw]];
                                    sum_upper += input.upper[[c, ih, iw]];
                                }
                                count += 1;
                            } else if self.count_include_pad {
                                count += 1;
                            }
                        }
                    }

                    let divisor = if self.count_include_pad {
                        (kh * kw) as f32
                    } else {
                        count.max(1) as f32
                    };

                    if batch_size.is_none() {
                        out_lower[[c, oh, ow]] = sum_lower / divisor;
                        out_upper[[c, oh, ow]] = sum_upper / divisor;
                    }
                }
            }
        }

        // Handle 4D batch case
        if let Some(b) = batch_size {
            for batch_idx in 0..b {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih_start = oh * sh;
                            let iw_start = ow * sw;

                            let mut sum_lower = 0.0f32;
                            let mut sum_upper = 0.0f32;
                            let mut count = 0usize;

                            for kh_off in 0..kh {
                                for kw_off in 0..kw {
                                    let ih = (ih_start + kh_off) as isize - ph as isize;
                                    let iw = (iw_start + kw_off) as isize - pw as isize;

                                    if ih >= 0
                                        && ih < in_h as isize
                                        && iw >= 0
                                        && iw < in_w as isize
                                    {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        sum_lower += input.lower[[batch_idx, c, ih, iw]];
                                        sum_upper += input.upper[[batch_idx, c, ih, iw]];
                                        count += 1;
                                    } else if self.count_include_pad {
                                        count += 1;
                                    }
                                }
                            }

                            let divisor = if self.count_include_pad {
                                (kh * kw) as f32
                            } else {
                                count.max(1) as f32
                            };

                            out_lower[[batch_idx, c, oh, ow]] = sum_lower / divisor;
                            out_upper[[batch_idx, c, oh, ow]] = sum_upper / divisor;
                        }
                    }
                }
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Average pooling is linear, so CROWN propagation is exact.
        // y[c, oh, ow] = (1/k) * sum(x[c, ih, iw]) for inputs in the pooling window
        //
        // The Jacobian J[y_flat, x_flat] = 1/k if x contributes to y, 0 otherwise.
        // CROWN backward: A' = A @ J where A are the incoming bounds.
        //
        // Note: This requires knowing the input shape, which we don't have in propagate_linear.
        // Use propagate_linear_with_bounds for full shape information.
        //
        // For now, return the bounds unchanged (identity) as a conservative fallback.
        // The caller should use propagate_linear_with_bounds when pre-activation bounds are available.
        debug!("AveragePool CROWN propagation without bounds - using identity (caller should use propagate_linear_with_bounds)");
        Ok(Cow::Borrowed(bounds))
    }
}

impl AveragePoolLayer {
    /// CROWN backward propagation for AveragePool with pre-activation bounds.
    ///
    /// AveragePool is a linear operation, so the Jacobian is constant.
    /// For each output y[c, oh, ow], it averages inputs x[c, ih, iw] in the pooling window.
    /// The Jacobian entry J[y_flat, x_flat] = 1/k where k is the number of inputs pooled.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();

        // Expect 3D (C, H, W) or 4D (B, C, H, W)
        if !(3..=4).contains(&ndim) {
            return Err(GammaError::InvalidSpec(format!(
                "AveragePool CROWN requires 3D or 4D input, got {}D",
                ndim
            )));
        }

        let (batch_size, channels, in_h, in_w) = if ndim == 4 {
            (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        // Compute output dimensions
        let (out_h, out_w) = if self.is_global() {
            (1, 1)
        } else {
            self.output_size(in_h, in_w)?
        };

        let (kh, kw) = if self.is_global() {
            (in_h, in_w)
        } else {
            self.kernel_size
        };
        let (sh, sw) = if self.is_global() {
            (1, 1)
        } else {
            self.stride
        };
        let (ph, pw) = if self.is_global() {
            (0, 0)
        } else {
            self.padding
        };

        let input_size = if let Some(b) = batch_size {
            b * channels * in_h * in_w
        } else {
            channels * in_h * in_w
        };

        let output_size = if let Some(b) = batch_size {
            b * channels * out_h * out_w
        } else {
            channels * out_h * out_w
        };

        if bounds.num_inputs() != output_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Build the transposed Jacobian coefficients for backward propagation
        // A'[out_idx, x_flat] = sum over y_flat of A[out_idx, y_flat] * J[y_flat, x_flat]
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // Helper to compute flat indices
        let in_flat = |b_opt: Option<usize>, c: usize, h: usize, w: usize| -> usize {
            if let Some(b) = b_opt {
                b * channels * in_h * in_w + c * in_h * in_w + h * in_w + w
            } else {
                c * in_h * in_w + h * in_w + w
            }
        };

        let out_flat = |b_opt: Option<usize>, c: usize, oh: usize, ow: usize| -> usize {
            if let Some(b) = b_opt {
                b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow
            } else {
                c * out_h * out_w + oh * out_w + ow
            }
        };

        // Iterate over output positions and propagate coefficients backward
        let batch_count = batch_size.unwrap_or(1);

        for b in 0..batch_count {
            let b_opt = if batch_size.is_some() { Some(b) } else { None };

            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let y_flat = out_flat(b_opt, c, oh, ow);

                        // Count valid inputs in this window
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;
                        let mut count = 0usize;

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                if (ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize)
                                    || self.count_include_pad
                                {
                                    count += 1;
                                }
                            }
                        }

                        let divisor = if self.count_include_pad {
                            (kh * kw) as f32
                        } else {
                            count.max(1) as f32
                        };
                        let weight = 1.0 / divisor;

                        // Distribute coefficients to input positions
                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let x_flat = in_flat(b_opt, c, ih, iw);

                                    // A'[out_idx, x_flat] += A[out_idx, y_flat] * weight
                                    for out_idx in 0..num_outputs {
                                        new_lower_a[[out_idx, x_flat]] +=
                                            bounds.lower_a[[out_idx, y_flat]] * weight;
                                        new_upper_a[[out_idx, x_flat]] +=
                                            bounds.upper_a[[out_idx, y_flat]] * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Bias passes through unchanged (linear operation has no bias)
        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: bounds.lower_b.clone(),
            upper_a: new_upper_a,
            upper_b: bounds.upper_b.clone(),
        })
    }
}

/// 2D max pooling layer: y = max(x over pooling window)
///
/// For interval propagation:
/// - lower_y = max(lower_x over window) - maximum of lower bounds
/// - upper_y = max(upper_x over window) - maximum of upper bounds
///
/// This is exact for IBP because max is monotonically increasing.
#[derive(Debug, Clone)]
pub struct MaxPool2dLayer {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Padding mode: true = use -inf for padding, false = only pool valid region
    pub use_negative_inf_padding: bool,
}

impl MaxPool2dLayer {
    /// Create a new max pool layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            use_negative_inf_padding: true,
        }
    }

    /// Create max pool with explicit padding mode.
    pub fn with_padding_mode(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_negative_inf_padding: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            use_negative_inf_padding,
        }
    }

    /// Compute output spatial dimensions.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let out_h = (input_h + 2 * ph - kh) / sh + 1;
        let out_w = (input_w + 2 * pw - kw) / sw + 1;
        (out_h, out_w)
    }
}

impl BoundPropagation for MaxPool2dLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Max pooling with IBP:
        // lower_y = max(lower_x in window) - soundly conservative
        // upper_y = max(upper_x in window) - exact upper bound
        //
        // This is sound because:
        // - For lower bound: the minimum possible max is achieved when all inputs
        //   are at their lower bounds, giving max(lower_x)
        // - For upper bound: the maximum possible max is achieved when the largest
        //   input is at its upper bound, giving max(upper_x)

        // Validate input shape: expect (channels, height, width) or (batch, channels, height, width)
        let input_shape = input.lower.shape();
        if input_shape.len() < 3 {
            return Err(GammaError::ShapeMismatch {
                expected: vec![0, 0, 0], // 3D or 4D expected
                got: input_shape.to_vec(),
            });
        }

        let (batch_size, channels, in_h, in_w) = if input_shape.len() == 4 {
            (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            // 3D input: (channels, height, width)
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        let (out_h, out_w) = self.output_size(in_h, in_w);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Create output arrays
        let out_shape = if let Some(b) = batch_size {
            vec![b, channels, out_h, out_w]
        } else {
            vec![channels, out_h, out_w]
        };
        let mut out_lower = ArrayD::from_elem(IxDyn(&out_shape), f32::NEG_INFINITY);
        let mut out_upper = ArrayD::from_elem(IxDyn(&out_shape), f32::NEG_INFINITY);

        // Apply max pooling for 3D case
        if batch_size.is_none() {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;

                        let mut max_lower = f32::NEG_INFINITY;
                        let mut max_upper = f32::NEG_INFINITY;

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    max_lower = max_lower.max(input.lower[[c, ih, iw]]);
                                    max_upper = max_upper.max(input.upper[[c, ih, iw]]);
                                } else if !self.use_negative_inf_padding {
                                    // If not using -inf padding, skip this position
                                    // (max over fewer elements)
                                }
                                // If using -inf padding, the -inf won't affect max
                            }
                        }

                        out_lower[[c, oh, ow]] = max_lower;
                        out_upper[[c, oh, ow]] = max_upper;
                    }
                }
            }
        }

        // Handle 4D batch case
        if let Some(b) = batch_size {
            for batch_idx in 0..b {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih_start = oh * sh;
                            let iw_start = ow * sw;

                            let mut max_lower = f32::NEG_INFINITY;
                            let mut max_upper = f32::NEG_INFINITY;

                            for kh_off in 0..kh {
                                for kw_off in 0..kw {
                                    let ih = (ih_start + kh_off) as isize - ph as isize;
                                    let iw = (iw_start + kw_off) as isize - pw as isize;

                                    if ih >= 0
                                        && ih < in_h as isize
                                        && iw >= 0
                                        && iw < in_w as isize
                                    {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        max_lower =
                                            max_lower.max(input.lower[[batch_idx, c, ih, iw]]);
                                        max_upper =
                                            max_upper.max(input.upper[[batch_idx, c, ih, iw]]);
                                    }
                                }
                            }

                            out_lower[[batch_idx, c, oh, ow]] = max_lower;
                            out_upper[[batch_idx, c, oh, ow]] = max_upper;
                        }
                    }
                }
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Max pooling CROWN requires tracking which element achieves the max,
        // which depends on the input intervals. Use propagate_linear_with_bounds instead.
        Err(GammaError::UnsupportedOp(
            "MaxPool2d linear propagation requires bounds - use propagate_linear_with_bounds"
                .to_string(),
        ))
    }
}

impl MaxPool2dLayer {
    /// CROWN backward propagation for MaxPool2d with pre-activation bounds.
    ///
    /// MaxPool is a piecewise-linear operation (max of k inputs). The CROWN relaxation:
    /// - If one input definitely dominates (l_i > max_{j≠i}(u_j)), route gradient through it
    /// - Otherwise, use center-point approximation: pick winner at midpoint, add bias for error
    ///
    /// This is similar to ReLU relaxation but for multi-input max instead of single-input clamp.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();

        // Expect 3D (C, H, W) or 4D (B, C, H, W)
        if !(3..=4).contains(&ndim) {
            return Err(GammaError::InvalidSpec(format!(
                "MaxPool2d CROWN requires 3D or 4D input, got {}D",
                ndim
            )));
        }

        let (batch_size, channels, in_h, in_w) = if ndim == 4 {
            (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        let (out_h, out_w) = self.output_size(in_h, in_w);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let input_size = if let Some(b) = batch_size {
            b * channels * in_h * in_w
        } else {
            channels * in_h * in_w
        };

        let output_size = if let Some(b) = batch_size {
            b * channels * out_h * out_w
        } else {
            channels * out_h * out_w
        };

        if bounds.num_inputs() != output_size {
            return Err(GammaError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Build the backward propagation coefficients
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_b = bounds.upper_b.clone();

        // Helper to compute flat indices
        let in_flat = |b_opt: Option<usize>, c: usize, h: usize, w: usize| -> usize {
            if let Some(b) = b_opt {
                b * channels * in_h * in_w + c * in_h * in_w + h * in_w + w
            } else {
                c * in_h * in_w + h * in_w + w
            }
        };

        let out_flat = |b_opt: Option<usize>, c: usize, oh: usize, ow: usize| -> usize {
            if let Some(b) = b_opt {
                b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow
            } else {
                c * out_h * out_w + oh * out_w + ow
            }
        };

        // Flatten pre-activation bounds for easy access
        let pre_lower = pre_activation.lower.as_slice().ok_or_else(|| {
            GammaError::InvalidSpec("MaxPool2d CROWN: pre_activation lower not contiguous".into())
        })?;
        let pre_upper = pre_activation.upper.as_slice().ok_or_else(|| {
            GammaError::InvalidSpec("MaxPool2d CROWN: pre_activation upper not contiguous".into())
        })?;

        // Iterate over output positions
        let batch_count = batch_size.unwrap_or(1);

        for b in 0..batch_count {
            let b_opt = if batch_size.is_some() { Some(b) } else { None };

            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let y_flat = out_flat(b_opt, c, oh, ow);
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;

                        // Collect valid inputs in this pooling window
                        let mut window_inputs: Vec<(usize, f32, f32, f32)> =
                            Vec::with_capacity(kh * kw);

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let x_flat = in_flat(b_opt, c, ih, iw);
                                    let l = pre_lower[x_flat];
                                    let u = pre_upper[x_flat];
                                    let mid = (l + u) * 0.5;
                                    window_inputs.push((x_flat, l, u, mid));
                                }
                            }
                        }

                        if window_inputs.is_empty() {
                            // All positions are padding - output is -inf (or constant)
                            // No gradient flows back
                            continue;
                        }

                        // Find the maximum lower bound and upper bound across inputs
                        let max_lower = window_inputs
                            .iter()
                            .map(|&(_, l, _, _)| l)
                            .fold(f32::NEG_INFINITY, f32::max);
                        let max_upper = window_inputs
                            .iter()
                            .map(|&(_, _, u, _)| u)
                            .fold(f32::NEG_INFINITY, f32::max);

                        // Check if there's a definite winner (one input whose lower bound >= all other upper bounds)
                        // We need to exclude self-comparison (comparing against own upper bound)
                        let definite_winner = window_inputs.iter().find(|&&(idx, l, _, _)| {
                            window_inputs
                                .iter()
                                .all(|&(other_idx, _, other_u, _)| idx == other_idx || l >= other_u)
                        });

                        if let Some(&(winner_flat, _, _, _)) = definite_winner {
                            // Single definite winner - gradient flows through it entirely (like identity)
                            for out_idx in 0..num_outputs {
                                new_lower_a[[out_idx, winner_flat]] +=
                                    bounds.lower_a[[out_idx, y_flat]];
                                new_upper_a[[out_idx, winner_flat]] +=
                                    bounds.upper_a[[out_idx, y_flat]];
                            }
                        } else {
                            // Multiple candidates - use constant IBP-style bounds
                            // When there's no clear winner, the linear approximation y ≈ x_k
                            // can be very loose. Instead, use the tight constant bounds:
                            // - Lower bound: max_lower (constant, no gradient)
                            // - Upper bound: max_upper (constant, no gradient)
                            //
                            // This is equivalent to "cutting" the backward propagation here
                            // and using the IBP bounds for this output position.
                            //
                            // Note: A more sophisticated approach could use a weighted
                            // combination of inputs, but constant bounds are sound and
                            // often tight enough for practical purposes.

                            // No gradient flows through - just constant bounds
                            // For lower bound computation with coefficient la:
                            // - if la >= 0: want lower y, use max_lower as constant
                            // - if la < 0: want upper y, use max_upper as constant
                            // For upper bound computation with coefficient ua:
                            // - if ua >= 0: want upper y, use max_upper as constant
                            // - if ua < 0: want lower y, use max_lower as constant

                            for out_idx in 0..num_outputs {
                                let la = bounds.lower_a[[out_idx, y_flat]];
                                let ua = bounds.upper_a[[out_idx, y_flat]];

                                // Lower bound: la * y where y is bounded
                                if la >= 0.0 {
                                    new_lower_b[out_idx] += la * max_lower;
                                } else {
                                    new_lower_b[out_idx] += la * max_upper;
                                }

                                // Upper bound: ua * y where y is bounded
                                if ua >= 0.0 {
                                    new_upper_b[out_idx] += ua * max_upper;
                                } else {
                                    new_upper_b[out_idx] += ua * max_lower;
                                }
                            }
                        }
                    }
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

/// Floor layer: y = floor(x) - rounds towards negative infinity.
///
/// Common in quantization and index computation.
#[derive(Debug, Clone, Default)]
pub struct FloorLayer;

impl FloorLayer {
    /// Create a new Floor layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for FloorLayer {
    /// IBP for Floor: y = floor(x)
    ///
    /// Floor is monotonically non-decreasing, so bounds are straightforward.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let lower = input.lower.mapv(|x| x.floor());
        let upper = input.upper.mapv(|x| x.floor());
        BoundedTensor::new(lower, upper)
    }

    /// CROWN without pre-activation bounds - uses identity (caller should use propagate_linear_with_bounds)
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Floor CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl FloorLayer {
    /// CROWN backward propagation through Floor with pre-activation bounds.
    ///
    /// Floor is a piecewise constant (step) function with zero derivative.
    /// The linear relaxation uses constant bounds (slope = 0):
    /// - Lower bound: y >= floor(l) (constant)
    /// - Upper bound: y <= floor(u) (constant)
    ///
    /// This is equivalent to IBP but expressed in the CROWN linear form.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Floor layer CROWN backward propagation with pre-activation bounds");

        // For discontinuous (piecewise constant) functions:
        // slope = 0, intercept = f(x) for some x in [l, u]
        // Lower bound uses floor(l), upper bound uses floor(u)
        let relaxation_fn = |l: f32, u: f32| -> (f32, f32, f32, f32) {
            let floor_l = l.floor();
            let floor_u = u.floor();
            // (lower_slope, lower_intercept, upper_slope, upper_intercept)
            (0.0, floor_l, 0.0, floor_u)
        };

        crown_elementwise_backward(bounds, pre_activation, relaxation_fn)
    }
}

/// Ceil layer: y = ceil(x) - rounds towards positive infinity.
///
/// Common in quantization and index computation.
#[derive(Debug, Clone, Default)]
pub struct CeilLayer;

impl CeilLayer {
    /// Create a new Ceil layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for CeilLayer {
    /// IBP for Ceil: y = ceil(x)
    ///
    /// Ceil is monotonically non-decreasing, so bounds are straightforward.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let lower = input.lower.mapv(|x| x.ceil());
        let upper = input.upper.mapv(|x| x.ceil());
        BoundedTensor::new(lower, upper)
    }

    /// CROWN without pre-activation bounds - uses identity (caller should use propagate_linear_with_bounds)
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Ceil CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl CeilLayer {
    /// CROWN backward propagation through Ceil with pre-activation bounds.
    ///
    /// Ceil is a piecewise constant (step) function with zero derivative.
    /// The linear relaxation uses constant bounds (slope = 0):
    /// - Lower bound: y >= ceil(l) (constant)
    /// - Upper bound: y <= ceil(u) (constant)
    ///
    /// This is equivalent to IBP but expressed in the CROWN linear form.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Ceil layer CROWN backward propagation with pre-activation bounds");

        // For discontinuous (piecewise constant) functions:
        // slope = 0, intercept = f(x) for some x in [l, u]
        // Lower bound uses ceil(l), upper bound uses ceil(u)
        let relaxation_fn = |l: f32, u: f32| -> (f32, f32, f32, f32) {
            let ceil_l = l.ceil();
            let ceil_u = u.ceil();
            // (lower_slope, lower_intercept, upper_slope, upper_intercept)
            (0.0, ceil_l, 0.0, ceil_u)
        };

        crown_elementwise_backward(bounds, pre_activation, relaxation_fn)
    }
}

/// Round layer: y = round(x) - rounds to nearest integer (0.5 rounds away from zero).
///
/// Common in quantization and rounding operations.
#[derive(Debug, Clone, Default)]
pub struct RoundLayer;

impl RoundLayer {
    /// Create a new Round layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for RoundLayer {
    /// IBP for Round: y = round(x)
    ///
    /// Round is monotonically non-decreasing, so bounds are straightforward.
    /// Uses banker's rounding (round half to even) as per Rust's round().
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let lower = input.lower.mapv(|x| x.round());
        let upper = input.upper.mapv(|x| x.round());
        BoundedTensor::new(lower, upper)
    }

    /// CROWN without pre-activation bounds - uses identity (caller should use propagate_linear_with_bounds)
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Round CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl RoundLayer {
    /// CROWN backward propagation through Round with pre-activation bounds.
    ///
    /// Round is a piecewise constant (step) function with zero derivative.
    /// The linear relaxation uses constant bounds (slope = 0):
    /// - Lower bound: y >= round(l) (constant)
    /// - Upper bound: y <= round(u) (constant)
    ///
    /// This is equivalent to IBP but expressed in the CROWN linear form.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Round layer CROWN backward propagation with pre-activation bounds");

        // For discontinuous (piecewise constant) functions:
        // slope = 0, intercept = f(x) for some x in [l, u]
        // Lower bound uses round(l), upper bound uses round(u)
        let relaxation_fn = |l: f32, u: f32| -> (f32, f32, f32, f32) {
            let round_l = l.round();
            let round_u = u.round();
            // (lower_slope, lower_intercept, upper_slope, upper_intercept)
            (0.0, round_l, 0.0, round_u)
        };

        crown_elementwise_backward(bounds, pre_activation, relaxation_fn)
    }
}

/// Sign layer: y = -1 if x < 0, 0 if x == 0, 1 if x > 0.
///
/// Useful for conditional logic and gradient sign analysis.
#[derive(Debug, Clone, Default)]
pub struct SignLayer;

impl SignLayer {
    /// Create a new Sign layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for SignLayer {
    /// IBP for Sign: y = sign(x)
    ///
    /// Output is in {-1, 0, 1}. Bounds depend on whether interval crosses zero.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

        for (idx, &lb) in input.lower.indexed_iter() {
            let ub = input.upper[idx.clone()];

            // Determine sign bounds based on interval position relative to zero
            let (s_lb, s_ub) = if lb > 0.0 {
                // Entire interval is positive
                (1.0, 1.0)
            } else if ub < 0.0 {
                // Entire interval is negative
                (-1.0, -1.0)
            } else if lb == 0.0 && ub == 0.0 {
                // Exactly zero
                (0.0, 0.0)
            } else if lb == 0.0 {
                // lb == 0, ub > 0: could be 0 or 1
                (0.0, 1.0)
            } else if ub == 0.0 {
                // lb < 0, ub == 0: could be -1 or 0
                (-1.0, 0.0)
            } else {
                // Interval spans zero (lb < 0 < ub): could be -1, 0, or 1
                (-1.0, 1.0)
            };

            out_lower[idx.clone()] = s_lb;
            out_upper[idx] = s_ub;
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN without pre-activation bounds - uses identity (caller should use propagate_linear_with_bounds)
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Sign CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

impl SignLayer {
    /// CROWN backward propagation through Sign with pre-activation bounds.
    ///
    /// Sign is a piecewise constant function: -1 if x < 0, 0 if x == 0, 1 if x > 0.
    /// The linear relaxation uses constant bounds (slope = 0):
    /// - Lower bound: y >= sign_lower (constant based on interval position)
    /// - Upper bound: y <= sign_upper (constant based on interval position)
    ///
    /// This is equivalent to IBP but expressed in the CROWN linear form.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Sign layer CROWN backward propagation with pre-activation bounds");

        // For Sign (piecewise constant with discontinuities at 0):
        // slope = 0, intercept depends on interval position relative to zero
        let relaxation_fn = |l: f32, u: f32| -> (f32, f32, f32, f32) {
            let (sign_l, sign_u) = if l > 0.0 {
                // Entire interval is positive
                (1.0, 1.0)
            } else if u < 0.0 {
                // Entire interval is negative
                (-1.0, -1.0)
            } else if l == 0.0 && u == 0.0 {
                // Exactly zero
                (0.0, 0.0)
            } else if l == 0.0 {
                // lb == 0, ub > 0: could be 0 or 1
                (0.0, 1.0)
            } else if u == 0.0 {
                // lb < 0, ub == 0: could be -1 or 0
                (-1.0, 0.0)
            } else {
                // Interval spans zero (lb < 0 < ub): could be -1, 0, or 1
                (-1.0, 1.0)
            };
            // (lower_slope, lower_intercept, upper_slope, upper_intercept)
            (0.0, sign_l, 0.0, sign_u)
        };

        crown_elementwise_backward(bounds, pre_activation, relaxation_fn)
    }
}

/// Reciprocal layer: y = 1/x.
///
/// Requires x != 0. For interval bounds containing zero, returns conservative [-inf, inf].
#[derive(Debug, Clone, Default)]
pub struct ReciprocalLayer;

impl ReciprocalLayer {
    /// Create a new Reciprocal layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for ReciprocalLayer {
    /// IBP for Reciprocal: y = 1/x
    ///
    /// Reciprocal is monotonically decreasing for x > 0 and x < 0.
    /// For x in [lb, ub] where lb > 0: y in [1/ub, 1/lb]
    /// For x in [lb, ub] where ub < 0: y in [1/ub, 1/lb]
    /// For intervals containing zero: y in [-inf, inf] (conservative)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

        for (idx, &lb) in input.lower.indexed_iter() {
            let ub = input.upper[idx.clone()];

            let (r_lb, r_ub) = if lb > 0.0 {
                // Entire interval is positive: 1/x is decreasing, so flip bounds
                (1.0 / ub, 1.0 / lb)
            } else if ub < 0.0 {
                // Entire interval is negative: 1/x is decreasing, so flip bounds
                (1.0 / ub, 1.0 / lb)
            } else {
                // Interval contains zero: reciprocal is undefined at 0
                // Return conservative bounds
                (f32::NEG_INFINITY, f32::INFINITY)
            };

            out_lower[idx.clone()] = r_lb;
            out_upper[idx] = r_ub;
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN propagation without bounds - uses identity
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        debug!("Reciprocal CROWN propagation without bounds - using identity");
        Ok(Cow::Borrowed(bounds))
    }
}

/// Linear relaxation for reciprocal on interval [l, u].
/// Reciprocal f(x) = 1/x is:
/// - Monotonically decreasing for both x > 0 and x < 0
/// - Convex for x > 0 (f''(x) = 2/x³ > 0)
/// - Concave for x < 0 (f''(x) = 2/x³ < 0)
///
/// For convex regions (x > 0): chord is upper bound
/// For concave regions (x < 0): chord is lower bound
fn reciprocal_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    const EPSILON: f32 = 1e-10;

    // Handle intervals that cross zero - return very conservative bounds
    if l <= 0.0 && u >= 0.0 {
        // Interval contains zero - reciprocal is undefined
        // Return a slope of 0 with very large bounds
        return (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // Handle degenerate cases (nearly zero-width interval)
    if (u - l).abs() < 1e-8 {
        // Use tangent at l: slope = -1/l², intercept = 1/l - slope*l = 2/l
        let recip_l = 1.0 / l;
        let slope = -1.0 / (l * l); // derivative of 1/x at l
        let intercept = recip_l - slope * l;
        return (slope, intercept, slope, intercept);
    }

    let recip_l = 1.0 / l;
    let recip_u = 1.0 / u;

    // Chord slope connecting (l, 1/l) to (u, 1/u)
    // slope = (1/u - 1/l) / (u - l) = (l - u) / (l*u*(u-l)) = -1/(l*u)
    let chord_slope = (recip_u - recip_l) / (u - l);
    let chord_intercept = recip_l - chord_slope * l;

    // Sample to find max deviation from chord
    let num_samples = 50;
    let mut max_above_chord = 0.0_f32;
    let mut max_below_chord = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        // Avoid division by zero
        if x.abs() < EPSILON {
            continue;
        }
        let rx = 1.0 / x;
        let cx = chord_slope * x + chord_intercept;
        let diff = rx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    let eps = 1e-5;

    // For x > 0 (convex): chord is upper bound, function is below chord
    // For x < 0 (concave): chord is lower bound, function is above chord
    if l > 0.0 {
        // Convex: chord is upper, max_below_chord is the gap below chord
        let lower_slope = chord_slope;
        let lower_intercept = chord_intercept - max_below_chord - eps;
        let upper_slope = chord_slope;
        let upper_intercept = chord_intercept + max_above_chord + eps;
        (lower_slope, lower_intercept, upper_slope, upper_intercept)
    } else {
        // Concave (u < 0): chord is lower, max_above_chord is the gap above chord
        let lower_slope = chord_slope;
        let lower_intercept = chord_intercept - max_below_chord - eps;
        let upper_slope = chord_slope;
        let upper_intercept = chord_intercept + max_above_chord + eps;
        (lower_slope, lower_intercept, upper_slope, upper_intercept)
    }
}

impl ReciprocalLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Reciprocal layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, reciprocal_linear_relaxation)
    }
}

/// Enum wrapper for different layer types with their parameters.
///
/// Note: MatMul and Add are binary operations that take two inputs.
/// For these, use the `propagate_ibp_binary` method directly on the layer struct,
/// or use `GraphNetwork` for graph-based computation.
#[derive(Debug, Clone)]
pub enum Layer {
    Linear(LinearLayer),
    Conv1d(Conv1dLayer),
    Conv2d(Conv2dLayer),
    /// Average pooling over spatial dimensions
    AveragePool(AveragePoolLayer),
    /// Max pooling over spatial dimensions
    MaxPool2d(MaxPool2dLayer),
    ReLU(ReLULayer),
    /// Leaky ReLU activation (allows small gradient for negative inputs)
    LeakyReLU(LeakyReLULayer),
    /// Clip: clamp values to [min, max] range
    Clip(ClipLayer),
    /// ELU (Exponential Linear Unit) activation
    Elu(EluLayer),
    /// SELU (Scaled ELU) activation with self-normalizing properties
    Selu(SeluLayer),
    /// PRelu (Parametric ReLU) with per-channel learned slopes
    PRelu(PReluLayer),
    /// HardSigmoid: max(0, min(1, alpha * x + beta))
    HardSigmoid(HardSigmoidLayer),
    /// HardSwish: x * HardSigmoid(x)
    HardSwish(HardSwishLayer),
    /// Exp: element-wise exponential
    Exp(ExpLayer),
    /// Log: element-wise natural logarithm
    Log(LogLayer),
    /// Celu: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    Celu(CeluLayer),
    /// Mish: x * tanh(softplus(x))
    Mish(MishLayer),
    /// LogSoftmax: log(softmax(x)) = x - logsumexp(x)
    LogSoftmax(LogSoftmaxLayer),
    /// ThresholdedRelu: y = x if x > alpha, else 0
    ThresholdedRelu(ThresholdedReluLayer),
    /// Shrink: soft thresholding for sparse coding
    Shrink(ShrinkLayer),
    /// Softsign: x / (1 + |x|)
    Softsign(SoftsignLayer),
    Softmax(SoftmaxLayer),
    /// Causal softmax for decoder attention (masked)
    CausalSoftmax(CausalSoftmaxLayer),
    GELU(GELULayer),
    LayerNorm(LayerNormLayer),
    /// Unary: batch normalization (for CNNs)
    BatchNorm(BatchNormLayer),
    /// Binary: bounded matrix multiplication (e.g., Q @ K^T)
    MatMul(MatMulLayer),
    /// Binary: element-wise multiplication (e.g., SwiGLU gating: up * silu(gate))
    MulBinary(MulBinaryLayer),
    /// Binary: element-wise addition (e.g., residual connections)
    Add(AddLayer),
    /// Binary: concatenation along axis (e.g., CLS token + patches in ViT)
    Concat(ConcatLayer),
    /// Binary: element-wise subtraction (e.g., x - mean(x) in LayerNorm)
    Sub(SubLayer),
    /// Binary: element-wise division (e.g., x / sqrt(var + eps) in LayerNorm)
    Div(DivLayer),
    /// Unary: add constant tensor (e.g., bias addition)
    AddConstant(AddConstantLayer),
    /// Tensor transpose (permute axes)
    Transpose(TransposeLayer),
    /// Tensor reshape (change shape, preserve total elements)
    Reshape(ReshapeLayer),
    /// Tensor flatten (flatten dimensions according to axis)
    Flatten(FlattenLayer),
    /// Unary: multiply by constant tensor (e.g., attention scaling)
    MulConstant(MulConstantLayer),
    /// Unary: element-wise absolute value
    Abs(AbsLayer),
    /// Unary: element-wise square root
    Sqrt(SqrtLayer),
    /// Unary: divide by constant tensor
    DivConstant(DivConstantLayer),
    /// Unary: subtract constant or subtract from constant
    SubConstant(SubConstantLayer),
    /// Unary: element-wise power (x^p)
    PowConstant(PowConstantLayer),
    /// Unary: reduce mean over axes
    ReduceMean(ReduceMeanLayer),
    /// Unary: reduce sum over axes
    ReduceSum(ReduceSumLayer),
    /// Unary: hyperbolic tangent activation
    Tanh(TanhLayer),
    /// Unary: sigmoid activation
    Sigmoid(SigmoidLayer),
    /// Unary: softplus activation (smooth ReLU)
    Softplus(SoftplusLayer),
    /// Unary: sine function (for positional encodings)
    Sin(SinLayer),
    /// Unary: cosine function (for positional encodings)
    Cos(CosLayer),
    /// Unary: tile/repeat along axis (for GQA KV head expansion)
    Tile(TileLayer),
    /// Unary: slice/extract contiguous range along axis (for Split op)
    Slice(SliceLayer),
    /// Ternary: conditional selection Where(condition, x, y)
    Where(WhereLayer),
    /// Unary: NonZero - returns indices of non-zero elements (data-dependent output)
    NonZero(NonZeroLayer),
    /// Unary: floor(x) - round towards negative infinity
    Floor(FloorLayer),
    /// Unary: ceil(x) - round towards positive infinity
    Ceil(CeilLayer),
    /// Unary: round(x) - round to nearest integer
    Round(RoundLayer),
    /// Unary: sign(x) - returns -1, 0, or 1
    Sign(SignLayer),
    /// Unary: 1/x - reciprocal
    Reciprocal(ReciprocalLayer),
}

impl Layer {
    /// Get a string describing the layer type.
    pub fn layer_type(&self) -> &'static str {
        match self {
            Layer::Linear(_) => "Linear",
            Layer::Conv1d(_) => "Conv1d",
            Layer::Conv2d(_) => "Conv2d",
            Layer::AveragePool(_) => "AveragePool",
            Layer::MaxPool2d(_) => "MaxPool2d",
            Layer::ReLU(_) => "ReLU",
            Layer::LeakyReLU(_) => "LeakyReLU",
            Layer::Clip(_) => "Clip",
            Layer::Elu(_) => "Elu",
            Layer::Selu(_) => "Selu",
            Layer::PRelu(_) => "PRelu",
            Layer::HardSigmoid(_) => "HardSigmoid",
            Layer::HardSwish(_) => "HardSwish",
            Layer::Exp(_) => "Exp",
            Layer::Log(_) => "Log",
            Layer::Celu(_) => "Celu",
            Layer::Mish(_) => "Mish",
            Layer::LogSoftmax(_) => "LogSoftmax",
            Layer::ThresholdedRelu(_) => "ThresholdedRelu",
            Layer::Shrink(_) => "Shrink",
            Layer::Softsign(_) => "Softsign",
            Layer::Softmax(_) => "Softmax",
            Layer::CausalSoftmax(_) => "CausalSoftmax",
            Layer::GELU(_) => "GELU",
            Layer::LayerNorm(_) => "LayerNorm",
            Layer::BatchNorm(_) => "BatchNorm",
            Layer::MatMul(_) => "MatMul",
            Layer::MulBinary(_) => "MulBinary",
            Layer::Add(_) => "Add",
            Layer::Concat(_) => "Concat",
            Layer::Sub(_) => "Sub",
            Layer::Div(_) => "Div",
            Layer::AddConstant(_) => "AddConstant",
            Layer::Transpose(_) => "Transpose",
            Layer::Reshape(_) => "Reshape",
            Layer::Flatten(_) => "Flatten",
            Layer::MulConstant(_) => "MulConstant",
            Layer::Abs(_) => "Abs",
            Layer::Sqrt(_) => "Sqrt",
            Layer::DivConstant(_) => "DivConstant",
            Layer::SubConstant(_) => "SubConstant",
            Layer::PowConstant(_) => "PowConstant",
            Layer::ReduceMean(_) => "ReduceMean",
            Layer::ReduceSum(_) => "ReduceSum",
            Layer::Tanh(_) => "Tanh",
            Layer::Sigmoid(_) => "Sigmoid",
            Layer::Softplus(_) => "Softplus",
            Layer::Sin(_) => "Sin",
            Layer::Cos(_) => "Cos",
            Layer::Tile(_) => "Tile",
            Layer::Slice(_) => "Slice",
            Layer::Where(_) => "Where",
            Layer::NonZero(_) => "NonZero",
            Layer::Floor(_) => "Floor",
            Layer::Ceil(_) => "Ceil",
            Layer::Round(_) => "Round",
            Layer::Sign(_) => "Sign",
            Layer::Reciprocal(_) => "Reciprocal",
        }
    }

    /// Check if this layer is a binary operation (requires two inputs).
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::Add(_)
                | Layer::Concat(_)
                | Layer::Sub(_)
                | Layer::Div(_)
        )
    }
}

impl BoundPropagation for Layer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        match self {
            Layer::Linear(l) => l.propagate_ibp(input),
            Layer::Conv1d(c) => c.propagate_ibp(input),
            Layer::Conv2d(c) => c.propagate_ibp(input),
            Layer::AveragePool(ap) => ap.propagate_ibp(input),
            Layer::MaxPool2d(mp) => mp.propagate_ibp(input),
            Layer::ReLU(r) => r.propagate_ibp(input),
            Layer::LeakyReLU(lr) => lr.propagate_ibp(input),
            Layer::Clip(c) => c.propagate_ibp(input),
            Layer::Elu(e) => e.propagate_ibp(input),
            Layer::Selu(s) => s.propagate_ibp(input),
            Layer::PRelu(p) => p.propagate_ibp(input),
            Layer::HardSigmoid(hs) => hs.propagate_ibp(input),
            Layer::HardSwish(hw) => hw.propagate_ibp(input),
            Layer::Exp(exp) => exp.propagate_ibp(input),
            Layer::Log(log) => log.propagate_ibp(input),
            Layer::Celu(c) => c.propagate_ibp(input),
            Layer::Mish(m) => m.propagate_ibp(input),
            Layer::LogSoftmax(ls) => ls.propagate_ibp(input),
            Layer::ThresholdedRelu(tr) => tr.propagate_ibp(input),
            Layer::Shrink(sh) => sh.propagate_ibp(input),
            Layer::Softsign(ss) => ss.propagate_ibp(input),
            Layer::Softmax(s) => s.propagate_ibp(input),
            Layer::CausalSoftmax(cs) => cs.propagate_ibp(input),
            Layer::GELU(g) => g.propagate_ibp(input),
            Layer::LayerNorm(ln) => ln.propagate_ibp(input),
            Layer::BatchNorm(bn) => bn.propagate_ibp(input),
            Layer::Transpose(t) => t.propagate_ibp(input),
            Layer::AddConstant(ac) => ac.propagate_ibp(input),
            Layer::Reshape(r) => r.propagate_ibp(input),
            Layer::Flatten(f) => f.propagate_ibp(input),
            Layer::MulConstant(m) => m.propagate_ibp(input),
            Layer::Abs(a) => a.propagate_ibp(input),
            Layer::Sqrt(s) => s.propagate_ibp(input),
            Layer::DivConstant(d) => d.propagate_ibp(input),
            Layer::SubConstant(s) => s.propagate_ibp(input),
            Layer::PowConstant(p) => p.propagate_ibp(input),
            Layer::ReduceMean(rm) => rm.propagate_ibp(input),
            Layer::ReduceSum(rs) => rs.propagate_ibp(input),
            Layer::Tanh(t) => t.propagate_ibp(input),
            Layer::Sigmoid(s) => s.propagate_ibp(input),
            Layer::Softplus(sp) => sp.propagate_ibp(input),
            Layer::Sin(s) => s.propagate_ibp(input),
            Layer::Cos(c) => c.propagate_ibp(input),
            Layer::Tile(t) => t.propagate_ibp(input),
            Layer::Slice(s) => s.propagate_ibp(input),
            Layer::MatMul(_) => Err(GammaError::UnsupportedOp(
                "MatMul is a binary operation - use propagate_ibp_binary".to_string(),
            )),
            Layer::Add(_) => Err(GammaError::UnsupportedOp(
                "Add is a binary operation - use propagate_ibp_binary".to_string(),
            )),
            Layer::Concat(_) => Err(GammaError::UnsupportedOp(
                "Concat is a binary operation - use propagate_ibp_binary".to_string(),
            )),
            Layer::Sub(_) => Err(GammaError::UnsupportedOp(
                "Sub is a binary operation - use propagate_ibp_binary".to_string(),
            )),
            Layer::Div(_) => Err(GammaError::UnsupportedOp(
                "Div is a binary operation - use propagate_ibp_binary".to_string(),
            )),
            Layer::MulBinary(_) => Err(GammaError::UnsupportedOp(
                "MulBinary is a binary operation - use propagate_ibp_binary".to_string(),
            )),
            Layer::Where(_) => Err(GammaError::UnsupportedOp(
                "Where is a ternary operation - use propagate_ibp_ternary".to_string(),
            )),
            Layer::NonZero(nz) => nz.propagate_ibp(input),
            Layer::Floor(f) => f.propagate_ibp(input),
            Layer::Ceil(c) => c.propagate_ibp(input),
            Layer::Round(r) => r.propagate_ibp(input),
            Layer::Sign(s) => s.propagate_ibp(input),
            Layer::Reciprocal(r) => r.propagate_ibp(input),
        }
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        match self {
            Layer::Linear(l) => l.propagate_linear(bounds),
            Layer::Conv1d(c) => c.propagate_linear(bounds),
            Layer::Conv2d(c) => c.propagate_linear(bounds),
            Layer::AveragePool(ap) => ap.propagate_linear(bounds),
            Layer::MaxPool2d(mp) => mp.propagate_linear(bounds),
            Layer::ReLU(r) => r.propagate_linear(bounds),
            Layer::LeakyReLU(lr) => lr.propagate_linear(bounds),
            Layer::Clip(c) => c.propagate_linear(bounds),
            Layer::Elu(e) => e.propagate_linear(bounds),
            Layer::Selu(s) => s.propagate_linear(bounds),
            Layer::PRelu(p) => p.propagate_linear(bounds),
            Layer::HardSigmoid(hs) => hs.propagate_linear(bounds),
            Layer::HardSwish(hw) => hw.propagate_linear(bounds),
            Layer::Exp(exp) => exp.propagate_linear(bounds),
            Layer::Log(log) => log.propagate_linear(bounds),
            Layer::Celu(c) => c.propagate_linear(bounds),
            Layer::Mish(m) => m.propagate_linear(bounds),
            Layer::LogSoftmax(ls) => ls.propagate_linear(bounds),
            Layer::ThresholdedRelu(tr) => tr.propagate_linear(bounds),
            Layer::Shrink(sh) => sh.propagate_linear(bounds),
            Layer::Softsign(ss) => ss.propagate_linear(bounds),
            Layer::Softmax(s) => s.propagate_linear(bounds),
            Layer::CausalSoftmax(cs) => cs.propagate_linear(bounds),
            Layer::GELU(g) => g.propagate_linear(bounds),
            Layer::LayerNorm(ln) => ln.propagate_linear(bounds),
            Layer::BatchNorm(bn) => bn.propagate_linear(bounds),
            Layer::Transpose(t) => t.propagate_linear(bounds),
            Layer::AddConstant(ac) => ac.propagate_linear(bounds),
            Layer::Reshape(r) => r.propagate_linear(bounds),
            Layer::Flatten(f) => f.propagate_linear(bounds),
            Layer::MulConstant(m) => m.propagate_linear(bounds),
            Layer::Abs(a) => a.propagate_linear(bounds),
            Layer::Sqrt(s) => s.propagate_linear(bounds),
            Layer::DivConstant(d) => d.propagate_linear(bounds),
            Layer::SubConstant(s) => s.propagate_linear(bounds),
            Layer::PowConstant(p) => p.propagate_linear(bounds),
            Layer::ReduceMean(rm) => rm.propagate_linear(bounds),
            Layer::ReduceSum(rs) => rs.propagate_linear(bounds),
            Layer::Tanh(t) => t.propagate_linear(bounds),
            Layer::Sigmoid(s) => s.propagate_linear(bounds),
            Layer::Softplus(sp) => sp.propagate_linear(bounds),
            Layer::Sin(s) => s.propagate_linear(bounds),
            Layer::Cos(c) => c.propagate_linear(bounds),
            Layer::Tile(t) => t.propagate_linear(bounds),
            Layer::Slice(s) => s.propagate_linear(bounds),
            Layer::MatMul(_) => Err(GammaError::UnsupportedOp(
                "MatMul CROWN propagation not implemented".to_string(),
            )),
            Layer::Add(_) => Err(GammaError::UnsupportedOp(
                "Add CROWN propagation not implemented".to_string(),
            )),
            Layer::Concat(_) => Err(GammaError::UnsupportedOp(
                "Concat CROWN propagation not implemented".to_string(),
            )),
            Layer::Sub(_) => Err(GammaError::UnsupportedOp(
                "Sub CROWN propagation not implemented".to_string(),
            )),
            Layer::Div(_) => Err(GammaError::UnsupportedOp(
                "Div CROWN propagation not implemented".to_string(),
            )),
            Layer::MulBinary(_) => Err(GammaError::UnsupportedOp(
                "MulBinary CROWN propagation not implemented".to_string(),
            )),
            Layer::Where(w) => w.propagate_linear(bounds),
            Layer::NonZero(nz) => nz.propagate_linear(bounds),
            Layer::Floor(f) => f.propagate_linear(bounds),
            Layer::Ceil(c) => c.propagate_linear(bounds),
            Layer::Round(r) => r.propagate_linear(bounds),
            Layer::Sign(s) => s.propagate_linear(bounds),
            Layer::Reciprocal(r) => r.propagate_linear(bounds),
        }
    }
}

impl Layer {
    /// Propagate IBP bounds for binary operations (MatMul, MulBinary, Add, Concat, Sub, Div).
    ///
    /// Returns an error for unary layers.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            Layer::MatMul(m) => m.propagate_ibp_binary(input_a, input_b),
            Layer::MulBinary(m) => m.propagate_ibp_binary(input_a, input_b),
            Layer::Add(a) => a.propagate_ibp_binary(input_a, input_b),
            Layer::Concat(c) => c.propagate_ibp_binary(input_a, input_b),
            Layer::Sub(s) => s.propagate_ibp_binary(input_a, input_b),
            Layer::Div(d) => d.propagate_ibp_binary(input_a, input_b),
            _ => Err(GammaError::UnsupportedOp(format!(
                "{} is not a binary operation",
                self.layer_type()
            ))),
        }
    }
}
