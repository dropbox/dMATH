//! Linear bounds representation for CROWN-style bound propagation.
//!
//! This module contains:
//! - `LinearBounds`: 2D linear bounds (flattened) for basic CROWN
//! - `BatchedLinearBounds`: N-D batched bounds for transformer verification
//! - `AlphaCrownConfig`: Configuration for α-CROWN optimization
//! - `AlphaState`: Learnable α parameters for unstable ReLU neurons

use gamma_core::{GammaError, Result};
use gamma_tensor::BoundedTensor;
use ndarray::{s, Array1, Array2, Array3, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Linear bounds representation for CROWN-style propagation.
///
/// Represents bounds of the form: lA @ x + lb <= y <= uA @ x + ub
/// where x is the network input and y is some intermediate/output value.
///
/// Shape conventions:
/// - For N outputs and M inputs: lower_a/upper_a are (N, M), lower_b/upper_b are (N,)
#[derive(Debug, Clone)]
pub struct LinearBounds {
    /// Lower bound coefficient matrix: shape (num_outputs, num_inputs)
    pub lower_a: Array2<f32>,
    /// Lower bound bias: shape (num_outputs,)
    pub lower_b: Array1<f32>,
    /// Upper bound coefficient matrix: shape (num_outputs, num_inputs)
    pub upper_a: Array2<f32>,
    /// Upper bound bias: shape (num_outputs,)
    pub upper_b: Array1<f32>,
}

impl LinearBounds {
    /// Create identity linear bounds (output = input).
    ///
    /// Returns bounds where A = I (identity) and b = 0.
    pub fn identity(dim: usize) -> Self {
        Self {
            lower_a: Array2::eye(dim),
            lower_b: Array1::zeros(dim),
            upper_a: Array2::eye(dim),
            upper_b: Array1::zeros(dim),
        }
    }

    /// Create linear bounds from a specification matrix for spec-guided CROWN.
    ///
    /// This initializes the CROWN backward pass with a specification matrix `C`
    /// instead of identity. The backward pass will then compute bounds on `C @ y`
    /// directly, preserving correlation information between outputs.
    ///
    /// # Arguments
    /// * `c` - Specification matrix of shape [num_specs, output_dim]
    ///
    /// # Returns
    /// LinearBounds where A = C and b = 0
    ///
    /// # Example
    /// For verification property "output_0 > output_1", use C = [[1, -1, 0, ...]]
    /// to compute bounds on output_0 - output_1 directly.
    pub fn from_spec_matrix(c: Array2<f32>) -> Self {
        let num_specs = c.nrows();
        Self {
            lower_a: c.clone(),
            lower_b: Array1::zeros(num_specs),
            upper_a: c,
            upper_b: Array1::zeros(num_specs),
        }
    }

    /// Number of outputs (rows in coefficient matrix).
    pub fn num_outputs(&self) -> usize {
        self.lower_a.nrows()
    }

    /// Number of inputs (columns in coefficient matrix).
    pub fn num_inputs(&self) -> usize {
        self.lower_a.ncols()
    }

    /// Concretize linear bounds given input bounds.
    ///
    /// For linear bounds lA @ x + lb, with x in [l, u]:
    /// - Lower bound: sum of min(lA_i * l_i, lA_i * u_i) + lb
    /// - Upper bound: sum of max(uA_i * l_i, uA_i * u_i) + ub
    ///
    /// Handles 0 * inf = 0 correctly to prevent NaN propagation with saturated bounds.
    pub fn concretize(&self, input_bounds: &BoundedTensor) -> BoundedTensor {
        let input_flat = input_bounds.flatten();
        let input_lower = input_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .expect("Input should be flattenable to 1D");
        let input_upper = input_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .expect("Input should be flattenable to 1D");

        // For each output i, compute:
        // lower[i] = sum_j(min(lower_a[i,j] * input_lower[j], lower_a[i,j] * input_upper[j])) + lower_b[i]
        // This is equivalent to: lower_a_pos @ input_lower + lower_a_neg @ input_upper + lower_b

        let lower_a_pos = self.lower_a.mapv(|v| v.max(0.0));
        let lower_a_neg = self.lower_a.mapv(|v| v.min(0.0));
        let upper_a_pos = self.upper_a.mapv(|v| v.max(0.0));
        let upper_a_neg = self.upper_a.mapv(|v| v.min(0.0));

        // Check if inputs contain infinite values (need safe multiplication)
        let has_inf = input_lower.iter().any(|&v| v.is_infinite())
            || input_upper.iter().any(|&v| v.is_infinite());

        let (concrete_lower, concrete_upper) = if has_inf {
            // Use safe multiplication to handle 0 * inf = 0
            let m = self.lower_a.nrows();
            let n = self.lower_a.ncols();
            let mut lower = Array1::zeros(m);
            let mut upper = Array1::zeros(m);

            for i in 0..m {
                let mut sum_l = self.lower_b[i];
                let mut sum_u = self.upper_b[i];
                for j in 0..n {
                    // Safe multiplication: 0 * inf = 0
                    let a_pos_l = lower_a_pos[[i, j]];
                    let a_neg_l = lower_a_neg[[i, j]];
                    let a_pos_u = upper_a_pos[[i, j]];
                    let a_neg_u = upper_a_neg[[i, j]];
                    let in_l = input_lower[j];
                    let in_u = input_upper[j];

                    if a_pos_l != 0.0 {
                        sum_l += a_pos_l * in_l;
                    }
                    if a_neg_l != 0.0 {
                        sum_l += a_neg_l * in_u;
                    }
                    if a_pos_u != 0.0 {
                        sum_u += a_pos_u * in_u;
                    }
                    if a_neg_u != 0.0 {
                        sum_u += a_neg_u * in_l;
                    }
                }
                lower[i] = sum_l;
                upper[i] = sum_u;
            }
            (lower, upper)
        } else {
            // Fast path: use standard dot product when no infinities
            let concrete_lower =
                lower_a_pos.dot(&input_lower) + lower_a_neg.dot(&input_upper) + &self.lower_b;
            let concrete_upper =
                upper_a_pos.dot(&input_upper) + upper_a_neg.dot(&input_lower) + &self.upper_b;
            (concrete_lower, concrete_upper)
        };

        BoundedTensor {
            lower: concrete_lower.into_dyn(),
            upper: concrete_upper.into_dyn(),
        }
    }

    /// Concretize linear bounds over an ℓ2 ball input set.
    ///
    /// For bounds of the form:
    /// - Lower: y >= a_L^T x + b_L
    /// - Upper: y <= a_U^T x + b_U
    ///
    /// and input constraint `||x - x_hat||_2 <= rho`, the extrema of a linear function
    /// occur in the direction of the coefficient vector:
    /// - min_x a^T x = a^T x_hat - rho * ||a||_2
    /// - max_x a^T x = a^T x_hat + rho * ||a||_2
    pub fn concretize_l2_ball(&self, x_hat: &Array1<f32>, rho: f32) -> Result<BoundedTensor> {
        if rho < 0.0 {
            return Err(GammaError::InvalidSpec(format!(
                "rho must be >= 0 (got {rho})"
            )));
        }
        if self.num_inputs() != x_hat.len() {
            return Err(GammaError::shape_mismatch(
                vec![self.num_inputs()],
                vec![x_hat.len()],
            ));
        }

        let m = self.num_outputs();
        let n = self.num_inputs();
        let rho_f64 = rho as f64;

        let mut lower = Array1::<f32>::zeros(m);
        let mut upper = Array1::<f32>::zeros(m);

        for i in 0..m {
            let mut dot_l = self.lower_b[i] as f64;
            let mut dot_u = self.upper_b[i] as f64;
            let mut norm_l2_l = 0.0f64;
            let mut norm_l2_u = 0.0f64;
            for j in 0..n {
                let xj = x_hat[j] as f64;
                let al = self.lower_a[[i, j]] as f64;
                let au = self.upper_a[[i, j]] as f64;
                dot_l += al * xj;
                dot_u += au * xj;
                norm_l2_l += al * al;
                norm_l2_u += au * au;
            }
            let norm_l2_l = norm_l2_l.sqrt();
            let norm_l2_u = norm_l2_u.sqrt();
            lower[i] = (dot_l - rho_f64 * norm_l2_l) as f32;
            upper[i] = (dot_u + rho_f64 * norm_l2_u) as f32;
        }

        Ok(BoundedTensor {
            lower: lower.into_dyn(),
            upper: upper.into_dyn(),
        })
    }
}

/// N-D batched linear bounds for transformer verification.
///
/// Unlike `LinearBounds` which flattens everything to 2D, this maintains
/// the batch structure (e.g., [batch, seq, hidden]) and operates on the
/// last dimension only, following Auto-LiRPA's approach.
///
/// For input shape [...batch_dims, in_dim] and output shape [...batch_dims, out_dim]:
/// - lower_a: [...batch_dims, out_dim, in_dim] - coefficient matrix per position
/// - lower_b: [...batch_dims, out_dim] - bias per position
///
/// The backward pass broadcasts correctly: for y = Wx + b,
/// new_A = A @ W broadcasts over batch dimensions.
#[derive(Debug, Clone)]
pub struct BatchedLinearBounds {
    /// Lower bound coefficient matrix: shape [...batch_dims, out_dim, in_dim]
    /// Represents: lower(y) >= A_L @ x + b_L
    pub lower_a: ArrayD<f32>,
    /// Lower bound bias: shape [...batch_dims, out_dim]
    pub lower_b: ArrayD<f32>,
    /// Upper bound coefficient matrix: shape [...batch_dims, out_dim, in_dim]
    /// Represents: upper(y) <= A_U @ x + b_U
    pub upper_a: ArrayD<f32>,
    /// Upper bound bias: shape [...batch_dims, out_dim]
    pub upper_b: ArrayD<f32>,
    /// Input shape (what x represents): e.g., [batch, seq, hidden]
    pub input_shape: Vec<usize>,
    /// Output shape (what y represents): e.g., [batch, seq, hidden]
    pub output_shape: Vec<usize>,
}

impl BatchedLinearBounds {
    /// Create identity linear bounds (output = input) for given shape.
    ///
    /// For shape [..., dim], creates bounds where A = I (identity) and b = 0.
    /// This represents y >= x and y <= x, i.e., y = x.
    pub fn identity(shape: &[usize]) -> Self {
        if shape.is_empty() || shape.iter().product::<usize>() == 0 {
            return Self {
                lower_a: ArrayD::ones(IxDyn(&[1, 1])),
                lower_b: ArrayD::zeros(IxDyn(&[1])),
                upper_a: ArrayD::ones(IxDyn(&[1, 1])),
                upper_b: ArrayD::zeros(IxDyn(&[1])),
                input_shape: vec![1],
                output_shape: vec![1],
            };
        }

        let dim = *shape.last().unwrap();
        let batch_dims: Vec<usize> = shape[..shape.len() - 1].to_vec();

        // Coefficient shape: [batch..., out_dim, in_dim]
        let mut a_shape = batch_dims.clone();
        a_shape.push(dim);
        a_shape.push(dim);

        // Bias shape: [batch..., out_dim]
        let mut b_shape = batch_dims.clone();
        b_shape.push(dim);

        // Create eye matrix [dim, dim]
        let eye = Array2::eye(dim);

        // Broadcast to [batch..., dim, dim]
        let total_batch: usize = batch_dims.iter().product();
        let total_batch = total_batch.max(1);

        // Stack identity matrices
        let mut lower_a_data = Vec::with_capacity(total_batch * dim * dim);
        for _ in 0..total_batch {
            lower_a_data.extend(eye.iter());
        }

        let lower_a = ArrayD::from_shape_vec(IxDyn(&a_shape), lower_a_data.clone()).unwrap();
        let upper_a = ArrayD::from_shape_vec(IxDyn(&a_shape), lower_a_data).unwrap();

        Self {
            lower_a,
            lower_b: ArrayD::zeros(IxDyn(&b_shape)),
            upper_a,
            upper_b: ArrayD::zeros(IxDyn(&b_shape)),
            input_shape: shape.to_vec(),
            output_shape: shape.to_vec(),
        }
    }

    /// Concretize batched linear bounds given input bounds.
    ///
    /// For linear bounds A @ x + b, with x in [l, u]:
    /// - For each batch position, compute concrete bounds
    /// - Lower bound: A_pos @ l + A_neg @ u + b (per position)
    /// - Upper bound: A_pos @ u + A_neg @ l + b (per position)
    pub fn concretize(&self, input_bounds: &BoundedTensor) -> BoundedTensor {
        // input_bounds shape: [...batch, in_dim]
        // self.lower_a shape: [...batch, out_dim, in_dim]
        // self.lower_b shape: [...batch, out_dim]
        // output shape: [...batch, out_dim]

        let in_lower = &input_bounds.lower;
        let in_upper = &input_bounds.upper;

        // Split coefficients into positive and negative parts
        let lower_a_pos = self.lower_a.mapv(|v| v.max(0.0));
        let lower_a_neg = self.lower_a.mapv(|v| v.min(0.0));
        let upper_a_pos = self.upper_a.mapv(|v| v.max(0.0));
        let upper_a_neg = self.upper_a.mapv(|v| v.min(0.0));

        // Compute concrete bounds using batched matrix-vector multiplication
        // For each position: result[i] = sum_j(A[i,j] * x[j])
        // With positive/negative split for interval arithmetic

        // We need to do batched matmul: [...batch, out, in] @ [...batch, in] -> [...batch, out]
        // Use safe addition to handle inf + (-inf) = conservative bound
        let lower_pos_term = batched_matvec(&lower_a_pos, in_lower);
        let lower_neg_term = batched_matvec(&lower_a_neg, in_upper);
        let lower_sum = safe_array_add(&lower_pos_term, &lower_neg_term, true);
        let concrete_lower = safe_array_add(&lower_sum, &self.lower_b, true);

        let upper_pos_term = batched_matvec(&upper_a_pos, in_upper);
        let upper_neg_term = batched_matvec(&upper_a_neg, in_lower);
        let upper_sum = safe_array_add(&upper_pos_term, &upper_neg_term, false);
        let concrete_upper = safe_array_add(&upper_sum, &self.upper_b, false);

        BoundedTensor {
            lower: concrete_lower,
            upper: concrete_upper,
        }
    }

    /// Create identity bounds for attention-shaped output.
    ///
    /// For attention output with shape [batch, heads, seq, seq], creates bounds with
    /// the last two dimensions flattened to enable McCormick relaxation for Q@K^T.
    ///
    /// The resulting bounds have shape [batch, heads, seq*seq, seq*seq] for the A matrix,
    /// which matches the flattened c_size = seq * seq expected by McCormick CROWN.
    ///
    /// Returns None if the shape doesn't match attention pattern (4D with last two dims equal)
    /// or if the flattened size would be too large (> 16M elements per identity matrix).
    pub fn identity_for_attention(shape: &[usize]) -> Option<Self> {
        // Attention output shape: [batch, heads, seq, seq] (4D with last two dims equal)
        if shape.len() != 4 {
            return None;
        }

        let batch = shape[0];
        let heads = shape[1];
        let seq_out = shape[2];
        let seq_in = shape[3];

        // Must be square attention output
        if seq_out != seq_in {
            return None;
        }

        let seq = seq_out;
        let flat_size = seq * seq;

        // Memory limit: 16M elements per identity matrix (seq^4 = 16M at seq=63)
        // For seq=64, flat_size=4096, identity would be 4096^2 = 16.7M - borderline
        // For seq=128, flat_size=16384, identity would be 268M - too large
        // Limit to seq <= 64 for now
        if flat_size > 4096 {
            return None;
        }

        let batch_size = batch * heads;
        let total_elements = batch_size * flat_size * flat_size;

        // Total memory check: batch_size * flat_size^2 * 4 bytes * 2 (lower + upper)
        // Allow up to 256MB total for coefficient matrices
        let max_elements = 256 * 1024 * 1024 / 4 / 2; // ~32M elements per matrix
        if total_elements > max_elements {
            return None;
        }

        // Create identity matrix [flat_size, flat_size]
        let eye = ndarray::Array2::<f32>::eye(flat_size);

        // Stack identity matrices for each batch position
        let mut lower_a_data = Vec::with_capacity(total_elements);
        for _ in 0..batch_size {
            lower_a_data.extend(eye.iter());
        }

        // A shape: [batch, heads, flat_size, flat_size]
        let a_shape = vec![batch, heads, flat_size, flat_size];
        // Bias shape: [batch, heads, flat_size]
        let b_shape = vec![batch, heads, flat_size];

        let lower_a = ArrayD::from_shape_vec(IxDyn(&a_shape), lower_a_data.clone()).ok()?;
        let upper_a = ArrayD::from_shape_vec(IxDyn(&a_shape), lower_a_data).ok()?;

        Some(Self {
            lower_a,
            lower_b: ArrayD::zeros(IxDyn(&b_shape)),
            upper_a,
            upper_b: ArrayD::zeros(IxDyn(&b_shape)),
            input_shape: vec![batch, heads, flat_size], // Flattened shape
            output_shape: vec![batch, heads, flat_size], // Flattened shape
        })
    }

    /// Output dimension (last dimension of output shape).
    pub fn out_dim(&self) -> usize {
        *self.output_shape.last().unwrap_or(&1)
    }

    /// Input dimension (last dimension of input shape).
    pub fn in_dim(&self) -> usize {
        *self.input_shape.last().unwrap_or(&1)
    }

    /// Compose two sets of linear bounds: result = other . self
    ///
    /// If self represents: y = A1 @ x + b1 (maps x -> y)
    /// And other represents: z = A2 @ y + b2 (maps y -> z)
    /// Then the composed result is: z = (A2 @ A1) @ x + (A2 @ b1 + b2)
    ///
    /// This is used for CROWN backward propagation to compose bounds across layers.
    ///
    /// # Arguments
    /// - `other`: The outer linear bounds (maps from self's output to new output)
    ///
    /// # Returns
    /// Composed bounds that map from self's input to other's output
    ///
    /// # Shape requirements
    /// - self.lower_a shape: [...batch, out_dim_1, in_dim_1]
    /// - other.lower_a shape: [...batch, out_dim_2, out_dim_1]
    /// - Result lower_a shape: [...batch, out_dim_2, in_dim_1]
    pub fn compose(&self, other: &BatchedLinearBounds) -> Result<BatchedLinearBounds> {
        // Validate shape compatibility
        let self_shape = self.lower_a.shape();
        let other_shape = other.lower_a.shape();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "BatchedLinearBounds::compose requires at least 2D coefficient matrices"
                    .to_string(),
            ));
        }

        let self_out_dim = self_shape[self_shape.len() - 2];
        let self_in_dim = self_shape[self_shape.len() - 1];
        let other_out_dim = other_shape[other_shape.len() - 2];
        let other_in_dim = other_shape[other_shape.len() - 1];

        // other's input dim should match self's output dim
        if other_in_dim != self_out_dim {
            return Err(GammaError::shape_mismatch(
                vec![self_out_dim],
                vec![other_in_dim],
            ));
        }

        // Get batch dimensions
        let self_batch = &self_shape[..self_shape.len() - 2];
        let other_batch = &other_shape[..other_shape.len() - 2];

        // For simplicity, require matching batch dimensions
        if self_batch != other_batch {
            return Err(GammaError::shape_mismatch(
                self_batch.to_vec(),
                other_batch.to_vec(),
            ));
        }

        let batch_dims = self_batch;
        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

        // Reshape for batched matrix multiplication
        // A1: [batch_size, out_dim_1, in_dim_1]
        // A2: [batch_size, out_dim_2, out_dim_1]
        // Result: [batch_size, out_dim_2, in_dim_1]
        let a1_lower = self
            .lower_a
            .view()
            .into_shape_with_order((batch_size, self_out_dim, self_in_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape A1 for composition".to_string())
            })?;
        let a1_upper = self
            .upper_a
            .view()
            .into_shape_with_order((batch_size, self_out_dim, self_in_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape A1 for composition".to_string())
            })?;
        let a2_lower = other
            .lower_a
            .view()
            .into_shape_with_order((batch_size, other_out_dim, other_in_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape A2 for composition".to_string())
            })?;
        let a2_upper = other
            .upper_a
            .view()
            .into_shape_with_order((batch_size, other_out_dim, other_in_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape A2 for composition".to_string())
            })?;

        // Bias vectors: [batch_size, out_dim]
        let b1_lower = self
            .lower_b
            .view()
            .into_shape_with_order((batch_size, self_out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape b1 for composition".to_string())
            })?;
        let b1_upper = self
            .upper_b
            .view()
            .into_shape_with_order((batch_size, self_out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape b1 for composition".to_string())
            })?;
        let b2_lower = other
            .lower_b
            .view()
            .into_shape_with_order((batch_size, other_out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape b2 for composition".to_string())
            })?;
        let b2_upper = other
            .upper_b
            .view()
            .into_shape_with_order((batch_size, other_out_dim))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape b2 for composition".to_string())
            })?;

        // Compute composed coefficient matrices: A_composed = A2 @ A1
        // Use interval arithmetic for interval coefficient matrices.
        //
        // Important: when bounds saturate, coefficients/biases may contain +/-inf.
        // Naive arithmetic can introduce NaNs via:
        // - 0 * inf
        // - inf + (-inf)
        //
        // For sound verification we must never propagate NaNs; in ambiguous cases we
        // conservatively widen (e.g., lower=-inf, upper=+inf).
        #[inline]
        fn safe_mul_pair_for_bounds(a: f32, b: f32) -> f32 {
            if a == 0.0 || b == 0.0 {
                0.0
            } else {
                a * b
            }
        }

        #[inline]
        fn interval_mul_for_bounds(a_l: f32, a_u: f32, b_l: f32, b_u: f32) -> (f32, f32) {
            if a_l.is_nan() || a_u.is_nan() || b_l.is_nan() || b_u.is_nan() {
                return (f32::NEG_INFINITY, f32::INFINITY);
            }

            let products = [
                safe_mul_pair_for_bounds(a_l, b_l),
                safe_mul_pair_for_bounds(a_l, b_u),
                safe_mul_pair_for_bounds(a_u, b_l),
                safe_mul_pair_for_bounds(a_u, b_u),
            ];

            let mut prod_lower = f32::INFINITY;
            let mut prod_upper = f32::NEG_INFINITY;
            for p in products {
                if p.is_nan() {
                    continue;
                }
                prod_lower = prod_lower.min(p);
                prod_upper = prod_upper.max(p);
            }

            if prod_lower == f32::INFINITY || prod_upper == f32::NEG_INFINITY {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else {
                (prod_lower, prod_upper)
            }
        }

        #[inline]
        fn safe_add_lower_for_bounds(a: f32, b: f32) -> f32 {
            let s = a + b;
            if s.is_nan() {
                f32::NEG_INFINITY
            } else {
                s
            }
        }

        #[inline]
        fn safe_add_upper_for_bounds(a: f32, b: f32) -> f32 {
            let s = a + b;
            if s.is_nan() {
                f32::INFINITY
            } else {
                s
            }
        }
        let mut composed_lower_a = Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_upper_a = Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_lower_b = Array2::<f32>::zeros((batch_size, other_out_dim));
        let mut composed_upper_b = Array2::<f32>::zeros((batch_size, other_out_dim));

        for b in 0..batch_size {
            // Matrix multiply: A2[b] @ A1[b]
            // For lower bound: use A2_lower+ @ A1_lower + A2_lower- @ A1_upper
            // For upper bound: use A2_upper+ @ A1_upper + A2_upper- @ A1_lower
            for i in 0..other_out_dim {
                for j in 0..self_in_dim {
                    let mut lower_sum = 0.0_f32;
                    let mut upper_sum = 0.0_f32;

                    for k in 0..other_in_dim {
                        let a2_l = a2_lower[[b, i, k]];
                        let a2_u = a2_upper[[b, i, k]];
                        let a1_l = a1_lower[[b, k, j]];
                        let a1_u = a1_upper[[b, k, j]];

                        // Interval multiplication for the element-wise products
                        // Then sum for matrix multiply
                        // For lower bound on (A2 @ A1)[i,j]:
                        //   sum_k interval_mul(A2[i,k], A1[k,j]).lower
                        // For upper bound: sum_k interval_mul(A2[i,k], A1[k,j]).upper
                        let (prod_lower, prod_upper) =
                            interval_mul_for_bounds(a2_l, a2_u, a1_l, a1_u);
                        lower_sum = safe_add_lower_for_bounds(lower_sum, prod_lower);
                        upper_sum = safe_add_upper_for_bounds(upper_sum, prod_upper);
                    }

                    composed_lower_a[[b, i, j]] = lower_sum;
                    composed_upper_a[[b, i, j]] = upper_sum;
                }

                // Compose bias: b_composed = A2 @ b1 + b2
                // For lower bound: use A2_lower+ @ b1_lower + A2_lower- @ b1_upper + b2_lower
                let mut bias_lower = b2_lower[[b, i]];
                let mut bias_upper = b2_upper[[b, i]];

                for k in 0..other_in_dim {
                    let a2_l = a2_lower[[b, i, k]];
                    let a2_u = a2_upper[[b, i, k]];
                    let b1_l = b1_lower[[b, k]];
                    let b1_u = b1_upper[[b, k]];

                    // Interval multiplication for A2 @ b1
                    let (prod_lower, prod_upper) = interval_mul_for_bounds(a2_l, a2_u, b1_l, b1_u);
                    bias_lower = safe_add_lower_for_bounds(bias_lower, prod_lower);
                    bias_upper = safe_add_upper_for_bounds(bias_upper, prod_upper);
                }

                composed_lower_b[[b, i]] = bias_lower;
                composed_upper_b[[b, i]] = bias_upper;
            }
        }

        // Reshape back to original batch structure
        let mut output_a_shape: Vec<usize> = batch_dims.to_vec();
        output_a_shape.push(other_out_dim);
        output_a_shape.push(self_in_dim);

        let mut output_b_shape: Vec<usize> = batch_dims.to_vec();
        output_b_shape.push(other_out_dim);

        let (composed_lower_a_vec, _) = composed_lower_a.into_raw_vec_and_offset();
        let (composed_upper_a_vec, _) = composed_upper_a.into_raw_vec_and_offset();
        let (composed_lower_b_vec, _) = composed_lower_b.into_raw_vec_and_offset();
        let (composed_upper_b_vec, _) = composed_upper_b.into_raw_vec_and_offset();

        Ok(BatchedLinearBounds {
            lower_a: ArrayD::from_shape_vec(IxDyn(&output_a_shape), composed_lower_a_vec).map_err(
                |_| GammaError::InvalidSpec("Cannot reshape composed lower_a".to_string()),
            )?,
            upper_a: ArrayD::from_shape_vec(IxDyn(&output_a_shape), composed_upper_a_vec).map_err(
                |_| GammaError::InvalidSpec("Cannot reshape composed upper_a".to_string()),
            )?,
            lower_b: ArrayD::from_shape_vec(IxDyn(&output_b_shape), composed_lower_b_vec).map_err(
                |_| GammaError::InvalidSpec("Cannot reshape composed lower_b".to_string()),
            )?,
            upper_b: ArrayD::from_shape_vec(IxDyn(&output_b_shape), composed_upper_b_vec).map_err(
                |_| GammaError::InvalidSpec("Cannot reshape composed upper_b".to_string()),
            )?,
            input_shape: self.input_shape.clone(),
            output_shape: other.output_shape.clone(),
        })
    }
}

/// Safe multiplication for bound computation.
///
/// In interval arithmetic, a coefficient of 0 means no contribution,
/// so 0 * inf = 0 (not NaN). This prevents NaN propagation when
/// computing bounds with saturated (infinite) input bounds.
#[inline]
pub fn safe_mul_for_bounds(a: f32, x: f32) -> f32 {
    // Handle 0 * inf = 0 for both cases
    if a == 0.0 || x == 0.0 {
        0.0
    } else if a.is_nan() || x.is_nan() {
        // Propagate NaN explicitly to avoid hiding issues
        // NaN in coefficients means the linear bound is invalid
        f32::NAN
    } else {
        a * x
    }
}

/// Safe addition for bound computation that handles inf + (-inf) = conservative bound.
///
/// When summing bound contributions, inf + (-inf) should produce:
/// - For lower bounds being computed: -inf (conservative, sound)
/// - For upper bounds being computed: +inf (conservative, sound)
///
/// The is_lower parameter indicates which bound is being computed.
#[inline]
pub fn safe_add_for_bounds_with_polarity(sum: f32, term: f32, is_lower: bool) -> f32 {
    let result = sum + term;
    if result.is_nan() && (sum.is_infinite() || term.is_infinite()) {
        // inf + (-inf) case: use conservative bound based on polarity
        if is_lower {
            f32::NEG_INFINITY // conservative lower bound
        } else {
            f32::INFINITY // conservative upper bound
        }
    } else {
        result
    }
}

/// Safe addition for bound computation (default to positive infinity for NaN).
/// Used when the polarity isn't known (e.g., intermediate computations).
#[inline]
pub fn safe_add_for_bounds(sum: f32, term: f32) -> f32 {
    safe_add_for_bounds_with_polarity(sum, term, false) // default to upper bound (conservative)
}

/// Safe element-wise array addition that handles inf + (-inf).
///
/// For lower bounds, NaN from inf + (-inf) becomes -inf (sound lower).
/// For upper bounds, NaN from inf + (-inf) becomes +inf (sound upper).
pub fn safe_array_add(a: &ArrayD<f32>, b: &ArrayD<f32>, is_lower: bool) -> ArrayD<f32> {
    use ndarray::Zip;
    let mut result = a + b;
    let conservative = if is_lower {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &av, &bv| {
            if r.is_nan() && (av.is_infinite() || bv.is_infinite()) {
                *r = conservative;
            }
        });
    result
}

/// Batched matrix-vector multiplication with safe handling of 0 * inf.
///
/// For A with shape [..., m, n] and x with shape [..., n],
/// computes y with shape `[..., m]` where `y[...][i] = sum_j A[...][i,j] * x[...][j]`
///
/// Uses safe_mul_for_bounds to handle cases where A contains zeros and x contains
/// infinite values, which would otherwise produce NaN in standard multiplication.
pub fn batched_matvec(a: &ArrayD<f32>, x: &ArrayD<f32>) -> ArrayD<f32> {
    let a_shape = a.shape();
    let x_shape = x.shape();

    if a_shape.len() < 2 || x_shape.is_empty() {
        // Degenerate case
        return ArrayD::zeros(IxDyn(&[]));
    }

    let m = a_shape[a_shape.len() - 2];
    let n = a_shape[a_shape.len() - 1];

    assert_eq!(
        *x_shape.last().unwrap(),
        n,
        "Input dimension mismatch in batched_matvec"
    );

    // Output shape: [...batch, m]
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let mut out_shape: Vec<usize> = batch_dims.to_vec();
    out_shape.push(m);

    // Compute total batch size
    let total_batch: usize = batch_dims.iter().product();
    let total_batch = total_batch.max(1);

    // Reshape to [batch, m, n] and [batch, n]
    let a_flat = a.view().into_shape_with_order((total_batch, m, n)).unwrap();
    let x_flat = x.view().into_shape_with_order((total_batch, n)).unwrap();

    // Check if either a or x contains infinite/NaN values (need safe multiplication).
    // When coefficients (a) blow up to inf but inputs (x) are finite, the dot product
    // can produce inf + (-inf) = NaN during summation.
    let has_inf_or_nan = x.iter().any(|&v| v.is_infinite() || v.is_nan())
        || a.iter().any(|&v| v.is_infinite() || v.is_nan());

    // Compute batched matvec
    let mut result = Array2::zeros((total_batch, m));
    if has_inf_or_nan {
        // Use safe multiplication and addition to handle 0 * inf = 0 and inf + (-inf) = inf
        for b in 0..total_batch {
            for i in 0..m {
                let mut sum = 0.0f32;
                for j in 0..n {
                    let term = safe_mul_for_bounds(a_flat[[b, i, j]], x_flat[[b, j]]);
                    sum = safe_add_for_bounds(sum, term);
                }
                result[[b, i]] = sum;
            }
        }
    } else {
        // Fast path: use standard dot product when no infinities
        for b in 0..total_batch {
            let a_slice = a_flat.slice(s![b, .., ..]);
            let x_slice = x_flat.slice(s![b, ..]);
            let mut r_slice = result.slice_mut(s![b, ..]);
            r_slice.assign(&a_slice.dot(&x_slice));
        }
    }

    // Reshape back
    let (vec, _offset) = result.into_raw_vec_and_offset();
    ArrayD::from_shape_vec(IxDyn(&out_shape), vec).unwrap()
}

/// Gradient estimation method for α-CROWN optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GradientMethod {
    /// Finite differences: perturb each α individually (O(n) passes per iteration).
    /// Accurate but slow for large networks.
    FiniteDifferences,
    /// SPSA: Simultaneous Perturbation Stochastic Approximation.
    /// Perturbs all α at once with random directions (O(1) passes per iteration).
    /// Faster but noisier - compensate with more iterations.
    #[default]
    Spsa,
    /// Analytic gradients: compute exact gradients during CROWN backward pass.
    /// O(1) passes per iteration, exact gradients.
    ///
    /// **EXPERIMENTAL**: This computes local gradients at each ReLU layer but doesn't
    /// properly chain them through subsequent layers to the input. The gradient represents
    /// coefficient sensitivity, not the actual bound gradient. Use SPSA for production.
    Analytic,
    /// True analytic gradients with full chain-rule propagation.
    /// Computes ∂(output_lower)/∂α_i by propagating gradients through all downstream layers.
    ///
    /// Stores intermediate A matrices at each ReLU during backward pass, then computes
    /// true chain-rule gradients: for neuron i in ReLU layer k, gradient is
    /// Σ_j A_downstream\[j,i\] × input_contribution\[i\] where j sums over output dimensions.
    AnalyticChain,
}

/// Intermediate values stored during α-CROWN backward pass for chain-rule gradient computation.
///
/// For chain-rule gradients, we need to know the A matrix (linear bounds coefficients) at
/// each ReLU layer BEFORE the ReLU is applied. This struct stores these values.
#[derive(Debug, Clone)]
pub struct AlphaCrownIntermediate {
    /// A matrices at each ReLU layer (before ReLU applied), in forward layer order.
    /// a_at_relu\[k\] is the A matrix from output back to just before ReLU layer k.
    /// Shape of each: (num_outputs, num_neurons_at_relu_k)
    pub a_at_relu: Vec<Array2<f32>>,

    /// Pre-ReLU bounds at each ReLU layer (for determining unstable neurons).
    /// Shape: (num_relu_layers,) where each element has shape (num_neurons,)
    pub pre_relu_bounds: Vec<(Array1<f32>, Array1<f32>)>,

    /// Final linear bounds after complete backward pass.
    pub final_bounds: LinearBounds,
}

impl AlphaCrownIntermediate {
    /// Create empty intermediate storage.
    pub fn new() -> Self {
        Self {
            a_at_relu: Vec::new(),
            pre_relu_bounds: Vec::new(),
            final_bounds: LinearBounds::identity(1),
        }
    }
}

impl Default for AlphaCrownIntermediate {
    fn default() -> Self {
        Self::new()
    }
}

/// Intermediate values stored during DAG α-CROWN backward pass for chain-rule gradient computation.
///
/// Unlike `AlphaCrownIntermediate` which uses Vec for sequential networks, this uses HashMap
/// to support DAG structures where ReLU nodes are identified by name.
#[derive(Debug, Clone)]
pub struct GraphAlphaCrownIntermediate {
    /// A matrices at each ReLU node (before ReLU applied), keyed by node name.
    /// Each entry is the accumulated A matrix from output back to just before that ReLU node.
    /// Shape of each: (num_outputs, num_neurons_at_relu)
    pub a_at_relu: std::collections::HashMap<String, Array2<f32>>,

    /// Pre-ReLU bounds at each ReLU node (for determining unstable neurons).
    /// Shape: each entry is (lower, upper) arrays with shape (num_neurons,)
    pub pre_relu_bounds: std::collections::HashMap<String, (Array1<f32>, Array1<f32>)>,

    /// Final linear bounds (accumulated to input).
    pub final_bounds: LinearBounds,
}

impl GraphAlphaCrownIntermediate {
    /// Create empty intermediate storage.
    pub fn new() -> Self {
        Self {
            a_at_relu: std::collections::HashMap::new(),
            pre_relu_bounds: std::collections::HashMap::new(),
            final_bounds: LinearBounds::identity(1),
        }
    }

    /// Get the A matrix at a specific ReLU node.
    pub fn get_a_at_relu(&self, node_name: &str) -> Option<&Array2<f32>> {
        self.a_at_relu.get(node_name)
    }

    /// Get the pre-ReLU bounds at a specific ReLU node.
    pub fn get_pre_relu_bounds(&self, node_name: &str) -> Option<&(Array1<f32>, Array1<f32>)> {
        self.pre_relu_bounds.get(node_name)
    }
}

impl Default for GraphAlphaCrownIntermediate {
    fn default() -> Self {
        Self::new()
    }
}

/// Adam optimizer hyperparameters.
///
/// Bundles Adam-specific parameters to reduce function argument counts.
/// Matches PyTorch/auto_LiRPA defaults: β₁=0.9, β₂=0.999, ε=1e-8.
#[derive(Debug, Clone, Copy)]
pub struct AdamParams {
    /// Learning rate (step size)
    pub learning_rate: f32,
    /// Exponential decay rate for first moment (β₁), default: 0.9
    pub beta1: f32,
    /// Exponential decay rate for second moment (β₂), default: 0.999
    pub beta2: f32,
    /// Small constant for numerical stability (ε), default: 1e-8
    pub epsilon: f32,
    /// Iteration number (1-indexed for bias correction)
    pub t: usize,
}

impl AdamParams {
    /// Create new Adam parameters with default hyperparameters.
    ///
    /// Uses PyTorch/auto_LiRPA defaults: β₁=0.9, β₂=0.999, ε=1e-8.
    pub fn new(learning_rate: f32, t: usize) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t,
        }
    }

    /// Create Adam parameters with custom hyperparameters.
    pub fn with_hyperparams(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t,
        }
    }
}

/// Optimizer type for alpha parameter updates.
///
/// Ported from α,β-CROWN's proven configurations:
/// - α,β-CROWN default: Adam with lr=0.1, beta1=0.9, beta2=0.999
/// - This dramatically outperforms SGD+momentum for bound tightening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Optimizer {
    /// SGD with momentum. Original γ-CROWN default.
    Sgd,
    /// Adam optimizer (adaptive moment estimation).
    /// Default: matches α,β-CROWN's proven configuration.
    #[default]
    Adam,
}

/// Configuration for alpha-CROWN optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaCrownConfig {
    /// Number of optimization iterations.
    pub iterations: usize,
    /// Learning rate for alpha parameter updates.
    pub learning_rate: f32,
    /// Learning rate decay factor per iteration.
    pub lr_decay: f32,
    /// Early stop if improvement is below this threshold.
    pub tolerance: f32,
    /// Whether to use adaptive learning rate (momentum-like).
    pub use_momentum: bool,
    /// Momentum coefficient (if use_momentum is true).
    pub momentum: f32,
    /// Gradient estimation method.
    pub gradient_method: GradientMethod,
    /// Number of SPSA samples to average per iteration (reduces variance).
    pub spsa_samples: usize,
    /// Use IBP bounds for intermediate nodes (O(N)) instead of CROWN-IBP (O(N²)).
    ///
    /// When true (default), matches α,β-CROWN's `fix_interm_bounds=True` behavior:
    /// - Use cheap IBP bounds for intermediate node bounds
    /// - Only run CROWN backward pass for output node optimization
    ///
    /// This dramatically speeds up initialization for deep networks (10x+ for ResNet-4b).
    /// Set to false for tighter intermediate bounds at the cost of O(N²) initialization.
    pub fix_interm_bounds: bool,
    /// Sparse alpha optimization: only optimize the top K% most influential alphas.
    ///
    /// After the first iteration, alphas are ranked by gradient magnitude and only
    /// the top `sparse_ratio` fraction are optimized in subsequent iterations.
    /// This reduces SPSA variance and focuses optimization where it matters most.
    ///
    /// Set to 1.0 to disable sparsity (optimize all alphas).
    /// Recommended: 0.1-0.3 for deep networks with many unstable neurons.
    pub sparse_ratio: f32,
    /// Adaptive skip: automatically skip α-CROWN for deep networks where it doesn't help.
    ///
    /// For deep networks (>25 layers/nodes), bounds are often fundamentally loose and
    /// α-CROWN optimization provides no benefit while being expensive. When enabled:
    /// - Networks with more than `adaptive_skip_depth_threshold` ReLU layers skip α-CROWN
    /// - Optionally runs a 1-iteration pilot to verify α-CROWN doesn't help
    ///
    /// Set to false to always run α-CROWN regardless of network depth.
    /// Default: true (skip α-CROWN for very deep networks)
    pub adaptive_skip: bool,
    /// Depth threshold for adaptive skipping.
    ///
    /// Networks with more than this many ReLU layers will skip α-CROWN optimization
    /// if `adaptive_skip` is enabled. For ResNet-4b (10 ReLU nodes), α-CROWN provides
    /// no benefit. For ResNet-2b (6 ReLU nodes), it helps significantly.
    ///
    /// Default: 8 (skip for networks with more ReLU layers than ResNet-2b)
    pub adaptive_skip_depth_threshold: usize,
    /// Run a pilot iteration to check if α-CROWN helps before full optimization.
    ///
    /// When enabled with `adaptive_skip`, runs 1 iteration and compares the improvement
    /// to CROWN bounds. If improvement is below `pilot_improvement_threshold`, skips
    /// remaining iterations.
    ///
    /// This catches cases where depth isn't the only factor (e.g., already tight bounds).
    /// Default: true
    pub adaptive_skip_pilot: bool,
    /// Minimum improvement required from pilot iteration to continue optimization.
    ///
    /// If the first iteration improves lower bound sum by less than this amount,
    /// skip remaining iterations. The value is absolute (not relative).
    ///
    /// Default: 1e-3 (skip if pilot improvement < 0.001)
    pub pilot_improvement_threshold: f32,
    /// Optimizer to use for alpha parameter updates.
    ///
    /// Ported from α,β-CROWN: Adam significantly outperforms SGD for bound tightening.
    /// Default: Adam (matches α,β-CROWN default)
    pub optimizer: Optimizer,
    /// Adam β₁: exponential decay rate for first moment estimate.
    /// Default: 0.9 (matches α,β-CROWN and PyTorch default)
    pub adam_beta1: f32,
    /// Adam β₂: exponential decay rate for second moment estimate.
    /// Default: 0.999 (matches α,β-CROWN and PyTorch default)
    pub adam_beta2: f32,
    /// Adam ε: small constant for numerical stability.
    /// Default: 1e-8 (matches α,β-CROWN and PyTorch default)
    pub adam_epsilon: f32,
}

impl Default for AlphaCrownConfig {
    fn default() -> Self {
        Self {
            // α,β-CROWN uses 100 iterations for incomplete verifier.
            // Start with 20 iterations for faster initial testing.
            iterations: 20,
            // α,β-CROWN uses lr=0.1 with Adam optimizer.
            learning_rate: 0.1,
            // α,β-CROWN uses ExponentialLR with decay=0.98 per iteration.
            lr_decay: 0.98,
            tolerance: 1e-4,
            use_momentum: true,
            momentum: 0.9,
            gradient_method: GradientMethod::Spsa, // SPSA for robust zero-order optimization
            spsa_samples: 1, // Single sample per iteration (trades variance for speed)
            // Default to true: use IBP bounds for intermediates (fast O(N)).
            // Matches α,β-CROWN's fix_interm_bounds=True default.
            fix_interm_bounds: true,
            // Sparse optimization: focus on top 30% most influential alphas.
            // Reduces SPSA variance when perturbing fewer coordinates.
            sparse_ratio: 0.3,
            // Adaptive skip: automatically disable α-CROWN for very deep networks
            // where optimization doesn't help (bounds fundamentally loose).
            adaptive_skip: true,
            // Skip for networks with >8 ReLU layers.
            // ResNet-2b (6 ReLU nodes) benefits from α-CROWN.
            // ResNet-4b (10 ReLU nodes) does NOT benefit.
            adaptive_skip_depth_threshold: 8,
            // Run 1 pilot iteration to confirm α-CROWN doesn't help before skipping.
            adaptive_skip_pilot: true,
            // Require at least 1e-3 improvement from pilot to continue optimization.
            pilot_improvement_threshold: 1e-3,
            // Optimizer: Adam (ported from α,β-CROWN's proven configuration)
            optimizer: Optimizer::Adam,
            // Adam hyperparameters (match α,β-CROWN and PyTorch defaults)
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
        }
    }
}

impl AlphaCrownConfig {
    /// Create Adam parameters from this config for the given learning rate and iteration.
    ///
    /// This bundles the Adam hyperparameters (β₁, β₂, ε) from the config with the
    /// current learning rate (after decay) and iteration number.
    pub fn adam_params(&self, learning_rate: f32, t: usize) -> AdamParams {
        AdamParams::with_hyperparams(
            learning_rate,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
            t,
        )
    }
}

/// State for alpha-CROWN optimization.
///
/// Stores the learnable alpha parameters for unstable ReLU neurons across all layers.
/// `alpha[layer_idx][neuron_idx]` in [0, 1] is the lower bound slope for crossing ReLUs.
#[derive(Debug, Clone)]
pub struct AlphaState {
    /// Alpha values per ReLU layer. Index is the layer index in the network.
    /// Each Array1 has length equal to the number of neurons in that ReLU layer.
    /// For stable neurons (always positive or negative), alpha is unused but stored.
    pub alphas: Vec<Array1<f32>>,
    /// Mask for unstable neurons (l < 0 < u). Only these neurons have optimizable alpha.
    pub unstable_mask: Vec<Array1<bool>>,
    /// Momentum for gradient updates (velocity) - used by SGD with momentum.
    pub velocity: Vec<Array1<f32>>,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub adam_m: Vec<Array1<f32>>,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub adam_v: Vec<Array1<f32>>,
}

impl AlphaState {
    /// Initialize alpha state from pre-activation bounds.
    ///
    /// For each ReLU layer, identifies unstable neurons and initializes alpha using
    /// the adaptive heuristic: alpha = 1 if u > -l, else 0.
    pub fn from_preactivation_bounds(
        layer_bounds: &[BoundedTensor],
        relu_layer_indices: &[usize],
    ) -> Self {
        let mut alphas = Vec::with_capacity(relu_layer_indices.len());
        let mut unstable_mask = Vec::with_capacity(relu_layer_indices.len());
        let mut velocity = Vec::with_capacity(relu_layer_indices.len());

        for &layer_idx in relu_layer_indices {
            // Get the pre-activation bounds for this ReLU layer
            let pre_bounds = &layer_bounds[layer_idx];
            let pre_flat = pre_bounds.flatten();
            let num_neurons = pre_flat.len();

            let mut alpha = Array1::<f32>::zeros(num_neurons);
            let mut mask = Array1::<bool>::from_elem(num_neurons, false);

            let lower_arr = pre_flat.lower.as_slice().unwrap_or(&[]);
            let upper_arr = pre_flat.upper.as_slice().unwrap_or(&[]);

            for i in 0..num_neurons {
                let l = if i < lower_arr.len() {
                    lower_arr[i]
                } else {
                    0.0
                };
                let u = if i < upper_arr.len() {
                    upper_arr[i]
                } else {
                    0.0
                };

                if l >= 0.0 {
                    // Always positive: alpha = 1 (identity)
                    alpha[i] = 1.0;
                    mask[i] = false;
                } else if u <= 0.0 {
                    // Always negative: alpha = 0
                    alpha[i] = 0.0;
                    mask[i] = false;
                } else {
                    // Crossing: unstable, initialize with adaptive heuristic
                    // α = 1 if u > -l (more positive area), else 0 (more negative area)
                    alpha[i] = if u > -l { 1.0 } else { 0.0 };
                    mask[i] = true;
                }
            }

            alphas.push(alpha.clone());
            unstable_mask.push(mask);
            velocity.push(Array1::<f32>::zeros(num_neurons));
        }

        // Initialize Adam moment estimates
        let adam_m = alphas
            .iter()
            .map(|a| Array1::<f32>::zeros(a.len()))
            .collect();
        let adam_v = alphas
            .iter()
            .map(|a| Array1::<f32>::zeros(a.len()))
            .collect();

        Self {
            alphas,
            unstable_mask,
            velocity,
            adam_m,
            adam_v,
        }
    }

    /// Get alpha values for a specific ReLU layer (by index in relu_layer_indices).
    pub fn get_alpha(&self, relu_idx: usize) -> Option<&Array1<f32>> {
        self.alphas.get(relu_idx)
    }

    /// Update alpha values using gradient descent with optional momentum.
    ///
    /// gradient: d(loss)/d(alpha) (where loss = -lower_bound, so minimize loss = maximize lower)
    /// learning_rate: step size
    /// momentum: momentum coefficient (0 = no momentum)
    pub fn update(
        &mut self,
        relu_idx: usize,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        if relu_idx >= self.alphas.len() {
            return;
        }

        let mask = &self.unstable_mask[relu_idx];
        let alpha = &mut self.alphas[relu_idx];
        let vel = &mut self.velocity[relu_idx];

        for i in 0..alpha.len() {
            if mask[i] {
                // Update velocity with momentum
                vel[i] = momentum * vel[i] - learning_rate * gradient[i];
                // Update alpha
                alpha[i] += vel[i];
                // Clamp to [0, 1]
                alpha[i] = alpha[i].clamp(0.0, 1.0);
            }
        }
    }

    /// Count total number of unstable neurons.
    pub fn num_unstable(&self) -> usize {
        self.unstable_mask
            .iter()
            .map(|m| m.iter().filter(|&&b| b).count())
            .sum()
    }

    /// Update alpha values using Adam optimizer.
    ///
    /// Adam update rule (gradient descent to minimize loss = maximize lower bound):
    /// - m = β₁ * m + (1 - β₁) * grad
    /// - v = β₂ * v + (1 - β₂) * grad²
    /// - m_hat = m / (1 - β₁^t)  (bias correction)
    /// - v_hat = v / (1 - β₂^t)
    /// - alpha = alpha - lr * m_hat / (√v_hat + ε)
    pub fn update_adam(&mut self, relu_idx: usize, gradient: &Array1<f32>, params: &AdamParams) {
        if relu_idx >= self.alphas.len() {
            return;
        }

        let mask = &self.unstable_mask[relu_idx];
        let alpha = &mut self.alphas[relu_idx];
        let m = &mut self.adam_m[relu_idx];
        let v = &mut self.adam_v[relu_idx];

        // Bias correction factors
        let t_f = params.t.max(1) as f32;
        let bias_correction1 = 1.0 - params.beta1.powf(t_f);
        let bias_correction2 = 1.0 - params.beta2.powf(t_f);

        for i in 0..alpha.len() {
            if mask[i] {
                let g = gradient[i];

                // Update biased first moment estimate
                m[i] = params.beta1 * m[i] + (1.0 - params.beta1) * g;
                // Update biased second moment estimate
                v[i] = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;

                // Bias-corrected estimates
                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                // Adam update (gradient descent: subtract to minimize loss)
                alpha[i] -= params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);

                // Clamp to [0, 1]
                alpha[i] = alpha[i].clamp(0.0, 1.0);
            }
        }
    }
}

/// Alpha state for DAG/GraphNetwork models.
///
/// Unlike `AlphaState` which uses indices, `GraphAlphaState` uses node names
/// as keys, since DAG models have named nodes rather than sequential layer indices.
#[derive(Debug, Clone)]
pub struct GraphAlphaState {
    /// Alpha values per ReLU node. Key is the node name.
    /// Each Array1 has length equal to the number of neurons in that ReLU node.
    pub alphas: std::collections::HashMap<String, Array1<f32>>,
    /// Mask for unstable neurons (l < 0 < u). Only these neurons have optimizable alpha.
    pub unstable_mask: std::collections::HashMap<String, Array1<bool>>,
    /// Momentum for gradient updates (velocity) - used by SGD with momentum.
    pub velocity: std::collections::HashMap<String, Array1<f32>>,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub adam_m: std::collections::HashMap<String, Array1<f32>>,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub adam_v: std::collections::HashMap<String, Array1<f32>>,
}

impl GraphAlphaState {
    /// Create empty state.
    pub fn new() -> Self {
        Self {
            alphas: std::collections::HashMap::new(),
            unstable_mask: std::collections::HashMap::new(),
            velocity: std::collections::HashMap::new(),
            adam_m: std::collections::HashMap::new(),
            adam_v: std::collections::HashMap::new(),
        }
    }

    /// Initialize alpha state from pre-activation bounds for a single ReLU node.
    ///
    /// For unstable neurons (l < 0 < u), initializes alpha using the adaptive heuristic:
    /// alpha = 1 if u > -l, else 0.
    pub fn add_relu_node(&mut self, node_name: &str, pre_activation: &BoundedTensor) {
        let pre_flat = pre_activation.flatten();
        let num_neurons = pre_flat.len();

        let mut alpha = Array1::<f32>::zeros(num_neurons);
        let mut mask = Array1::<bool>::from_elem(num_neurons, false);

        let lower_arr = pre_flat.lower.as_slice().unwrap_or(&[]);
        let upper_arr = pre_flat.upper.as_slice().unwrap_or(&[]);

        for i in 0..num_neurons {
            let l = if i < lower_arr.len() {
                lower_arr[i]
            } else {
                0.0
            };
            let u = if i < upper_arr.len() {
                upper_arr[i]
            } else {
                0.0
            };

            if l >= 0.0 {
                // Always positive: alpha = 1 (identity)
                alpha[i] = 1.0;
                mask[i] = false;
            } else if u <= 0.0 {
                // Always negative: alpha = 0
                alpha[i] = 0.0;
                mask[i] = false;
            } else {
                // Crossing: unstable, initialize with adaptive heuristic
                // α = 1 if u > -l (more positive area), else 0 (more negative area)
                alpha[i] = if u > -l { 1.0 } else { 0.0 };
                mask[i] = true;
            }
        }

        self.alphas.insert(node_name.to_string(), alpha.clone());
        self.unstable_mask.insert(node_name.to_string(), mask);
        self.velocity
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_neurons));
        self.adam_m
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_neurons));
        self.adam_v
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_neurons));
    }

    /// Get alpha values for a specific ReLU node.
    pub fn get_alpha(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.alphas.get(node_name)
    }

    /// Update alpha values using gradient descent with optional momentum.
    ///
    /// gradient: d(loss)/d(alpha) (where loss = -lower_bound, so minimize loss = maximize lower)
    /// learning_rate: step size
    /// momentum: momentum coefficient (0 = no momentum)
    pub fn update(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        let Some(alpha) = self.alphas.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(vel) = self.velocity.get_mut(node_name) else {
            return;
        };

        for i in 0..alpha.len() {
            if mask[i] {
                // Update velocity with momentum
                vel[i] = momentum * vel[i] - learning_rate * gradient[i];
                // Update alpha
                alpha[i] += vel[i];
                // Clamp to [0, 1]
                alpha[i] = alpha[i].clamp(0.0, 1.0);
            }
        }
    }

    /// Count total number of unstable neurons.
    pub fn num_unstable(&self) -> usize {
        self.unstable_mask
            .values()
            .map(|m| m.iter().filter(|&&b| b).count())
            .sum()
    }

    /// Get all ReLU node names.
    pub fn relu_nodes(&self) -> impl Iterator<Item = &String> {
        self.alphas.keys()
    }

    /// Update alpha values using Adam optimizer.
    ///
    /// Adam update rule (ported from α,β-CROWN / auto_LiRPA):
    /// - m = β₁ * m + (1 - β₁) * grad
    /// - v = β₂ * v + (1 - β₂) * grad²
    /// - m_hat = m / (1 - β₁^t)  (bias correction)
    /// - v_hat = v / (1 - β₂^t)
    /// - alpha = alpha - lr * m_hat / (√v_hat + ε)
    pub fn update_adam(&mut self, node_name: &str, gradient: &Array1<f32>, params: &AdamParams) {
        let Some(alpha) = self.alphas.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(m) = self.adam_m.get_mut(node_name) else {
            return;
        };
        let Some(v) = self.adam_v.get_mut(node_name) else {
            return;
        };

        // Bias correction factors
        let t_f = params.t.max(1) as f32;
        let bias_correction1 = 1.0 - params.beta1.powf(t_f);
        let bias_correction2 = 1.0 - params.beta2.powf(t_f);

        for i in 0..alpha.len() {
            if mask[i] {
                let g = gradient[i];

                // Update biased first moment estimate
                m[i] = params.beta1 * m[i] + (1.0 - params.beta1) * g;
                // Update biased second moment estimate
                v[i] = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;

                // Bias-corrected estimates
                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                // Adam update (gradient descent: subtract to minimize loss)
                alpha[i] -= params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);

                // Clamp to [0, 1]
                alpha[i] = alpha[i].clamp(0.0, 1.0);
            }
        }
    }
}

impl Default for GraphAlphaState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, ArrayD, IxDyn};

    // =========================================================================
    // LinearBounds tests
    // =========================================================================

    #[test]
    fn test_linear_bounds_identity() {
        let bounds = LinearBounds::identity(3);

        // Check identity matrix
        assert_eq!(bounds.lower_a.nrows(), 3);
        assert_eq!(bounds.lower_a.ncols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(bounds.lower_a[[i, j]], expected);
                assert_eq!(bounds.upper_a[[i, j]], expected);
            }
        }

        // Check zero bias
        assert_eq!(bounds.lower_b, Array1::<f32>::zeros(3));
        assert_eq!(bounds.upper_b, Array1::<f32>::zeros(3));
    }

    #[test]
    fn test_linear_bounds_num_outputs_inputs() {
        let bounds = LinearBounds {
            lower_a: Array2::zeros((5, 3)),
            lower_b: Array1::zeros(5),
            upper_a: Array2::zeros((5, 3)),
            upper_b: Array1::zeros(5),
        };
        assert_eq!(bounds.num_outputs(), 5);
        assert_eq!(bounds.num_inputs(), 3);
    }

    #[test]
    fn test_linear_bounds_identity_preserves_input() {
        let identity = LinearBounds::identity(4);

        // Input bounds: [1, 2, 3, 4] to [5, 6, 7, 8]
        let input = BoundedTensor {
            lower: array![1.0_f32, 2.0, 3.0, 4.0].into_dyn(),
            upper: array![5.0_f32, 6.0, 7.0, 8.0].into_dyn(),
        };

        let output = identity.concretize(&input);

        // Identity should preserve bounds exactly
        assert_eq!(output.lower.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.upper.as_slice().unwrap(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_linear_bounds_concretize_with_positive_coeffs() {
        // y = 2*x, bounds [1, 3]
        let bounds = LinearBounds {
            lower_a: Array2::from_elem((1, 1), 2.0),
            lower_b: Array1::zeros(1),
            upper_a: Array2::from_elem((1, 1), 2.0),
            upper_b: Array1::zeros(1),
        };

        let input = BoundedTensor {
            lower: array![1.0_f32].into_dyn(),
            upper: array![3.0_f32].into_dyn(),
        };

        let output = bounds.concretize(&input);

        // y = 2*x with x in [1,3] gives y in [2, 6]
        assert!((output.lower[[0]] - 2.0).abs() < 1e-6);
        assert!((output.upper[[0]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_bounds_concretize_with_negative_coeffs() {
        // y = -2*x, bounds x in [1, 3]
        let bounds = LinearBounds {
            lower_a: Array2::from_elem((1, 1), -2.0),
            lower_b: Array1::zeros(1),
            upper_a: Array2::from_elem((1, 1), -2.0),
            upper_b: Array1::zeros(1),
        };

        let input = BoundedTensor {
            lower: array![1.0_f32].into_dyn(),
            upper: array![3.0_f32].into_dyn(),
        };

        let output = bounds.concretize(&input);

        // y = -2*x with x in [1,3] gives y in [-6, -2]
        assert!((output.lower[[0]] - (-6.0)).abs() < 1e-6);
        assert!((output.upper[[0]] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_linear_bounds_concretize_with_bias() {
        // y = x + 10
        let bounds = LinearBounds {
            lower_a: Array2::from_elem((1, 1), 1.0),
            lower_b: array![10.0_f32],
            upper_a: Array2::from_elem((1, 1), 1.0),
            upper_b: array![10.0_f32],
        };

        let input = BoundedTensor {
            lower: array![1.0_f32].into_dyn(),
            upper: array![3.0_f32].into_dyn(),
        };

        let output = bounds.concretize(&input);

        // y = x + 10 with x in [1,3] gives y in [11, 13]
        assert!((output.lower[[0]] - 11.0).abs() < 1e-6);
        assert!((output.upper[[0]] - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_bounds_concretize_mixed_coeffs() {
        // y = x1 - x2, with x1 in [1,5], x2 in [2,4]
        // lower(y) = min(x1) - max(x2) = 1 - 4 = -3
        // upper(y) = max(x1) - min(x2) = 5 - 2 = 3
        let bounds = LinearBounds {
            lower_a: array![[1.0_f32, -1.0]],
            lower_b: array![0.0_f32],
            upper_a: array![[1.0_f32, -1.0]],
            upper_b: array![0.0_f32],
        };

        let input = BoundedTensor {
            lower: array![1.0_f32, 2.0].into_dyn(),
            upper: array![5.0_f32, 4.0].into_dyn(),
        };

        let output = bounds.concretize(&input);

        assert!((output.lower[[0]] - (-3.0)).abs() < 1e-6);
        assert!((output.upper[[0]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_bounds_concretize_with_inf_and_zero_coeff() {
        // Coefficient is 0, input is infinite - should produce 0, not NaN
        let bounds = LinearBounds {
            lower_a: array![[0.0_f32]],
            lower_b: array![5.0_f32],
            upper_a: array![[0.0_f32]],
            upper_b: array![5.0_f32],
        };

        let input = BoundedTensor {
            lower: array![f32::NEG_INFINITY].into_dyn(),
            upper: array![f32::INFINITY].into_dyn(),
        };

        let output = bounds.concretize(&input);

        // 0 * inf = 0, so output = bias = 5
        assert_eq!(output.lower[[0]], 5.0);
        assert_eq!(output.upper[[0]], 5.0);
    }

    #[test]
    fn test_linear_bounds_concretize_l2_ball_basic() {
        // Identity transformation with L2 ball
        let bounds = LinearBounds::identity(2);
        let x_hat = array![1.0_f32, 2.0];
        let rho = 0.5;

        let result = bounds.concretize_l2_ball(&x_hat, rho).unwrap();

        // For identity A=I, ||a||_2 = 1 for each row
        // lower = x_hat - rho * 1 = [0.5, 1.5]
        // upper = x_hat + rho * 1 = [1.5, 2.5]
        let lower = result.lower.as_slice().unwrap();
        let upper = result.upper.as_slice().unwrap();

        assert!((lower[0] - 0.5).abs() < 1e-5);
        assert!((lower[1] - 1.5).abs() < 1e-5);
        assert!((upper[0] - 1.5).abs() < 1e-5);
        assert!((upper[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_bounds_concretize_l2_ball_zero_radius() {
        let bounds = LinearBounds::identity(2);
        let x_hat = array![3.0_f32, 4.0];

        let result = bounds.concretize_l2_ball(&x_hat, 0.0).unwrap();

        // Zero radius: bounds should equal x_hat exactly
        let lower = result.lower.as_slice().unwrap();
        let upper = result.upper.as_slice().unwrap();

        assert!((lower[0] - 3.0).abs() < 1e-5);
        assert!((lower[1] - 4.0).abs() < 1e-5);
        assert!((upper[0] - 3.0).abs() < 1e-5);
        assert!((upper[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_bounds_concretize_l2_ball_negative_rho() {
        let bounds = LinearBounds::identity(2);
        let x_hat = array![1.0_f32, 2.0];

        let result = bounds.concretize_l2_ball(&x_hat, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_bounds_concretize_l2_ball_shape_mismatch() {
        let bounds = LinearBounds::identity(3);
        let x_hat = array![1.0_f32, 2.0]; // Wrong size

        let result = bounds.concretize_l2_ball(&x_hat, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_bounds_concretize_l2_ball_with_transform() {
        // y = 2*x, x_hat = [1], rho = 0.5
        // ||a||_2 = 2
        // lower = 2*1 - 0.5*2 = 1
        // upper = 2*1 + 0.5*2 = 3
        let bounds = LinearBounds {
            lower_a: array![[2.0_f32]],
            lower_b: array![0.0_f32],
            upper_a: array![[2.0_f32]],
            upper_b: array![0.0_f32],
        };

        let x_hat = array![1.0_f32];
        let result = bounds.concretize_l2_ball(&x_hat, 0.5).unwrap();

        let lower = result.lower.as_slice().unwrap();
        let upper = result.upper.as_slice().unwrap();

        assert!((lower[0] - 1.0).abs() < 1e-5);
        assert!((upper[0] - 3.0).abs() < 1e-5);
    }

    // =========================================================================
    // BatchedLinearBounds tests
    // =========================================================================

    #[test]
    fn test_batched_linear_bounds_identity_1d() {
        let bounds = BatchedLinearBounds::identity(&[4]);

        assert_eq!(bounds.input_shape, vec![4]);
        assert_eq!(bounds.output_shape, vec![4]);
        assert_eq!(bounds.in_dim(), 4);
        assert_eq!(bounds.out_dim(), 4);

        // Check it's an identity
        let a_shape = bounds.lower_a.shape();
        assert_eq!(a_shape, &[4, 4]);
    }

    #[test]
    fn test_batched_linear_bounds_identity_2d() {
        let bounds = BatchedLinearBounds::identity(&[2, 3]);

        assert_eq!(bounds.input_shape, vec![2, 3]);
        assert_eq!(bounds.output_shape, vec![2, 3]);
        assert_eq!(bounds.in_dim(), 3);
        assert_eq!(bounds.out_dim(), 3);

        // Shape: [batch=2, out=3, in=3]
        let a_shape = bounds.lower_a.shape();
        assert_eq!(a_shape, &[2, 3, 3]);
    }

    #[test]
    fn test_batched_linear_bounds_identity_empty() {
        let bounds = BatchedLinearBounds::identity(&[]);

        // Should create minimal bounds
        assert_eq!(bounds.input_shape, vec![1]);
        assert_eq!(bounds.output_shape, vec![1]);
    }

    #[test]
    fn test_batched_linear_bounds_concretize_identity() {
        let bounds = BatchedLinearBounds::identity(&[2, 3]);

        let input = BoundedTensor {
            lower: ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
            upper: ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                .unwrap(),
        };

        let output = bounds.concretize(&input);

        // Identity should preserve bounds
        assert_eq!(
            output.lower.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            output.upper.as_slice().unwrap(),
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_batched_linear_bounds_identity_for_attention_valid() {
        // Valid attention shape: [batch=1, heads=2, seq=4, seq=4]
        let bounds = BatchedLinearBounds::identity_for_attention(&[1, 2, 4, 4]);

        assert!(bounds.is_some());
        let bounds = bounds.unwrap();

        // Flattened size = 4*4 = 16
        // A shape: [1, 2, 16, 16]
        assert_eq!(bounds.lower_a.shape(), &[1, 2, 16, 16]);
        assert_eq!(bounds.input_shape, vec![1, 2, 16]);
        assert_eq!(bounds.output_shape, vec![1, 2, 16]);
    }

    #[test]
    fn test_batched_linear_bounds_identity_for_attention_non_square() {
        // Non-square: seq_out != seq_in
        let bounds = BatchedLinearBounds::identity_for_attention(&[1, 2, 4, 5]);
        assert!(bounds.is_none());
    }

    #[test]
    fn test_batched_linear_bounds_identity_for_attention_wrong_dims() {
        // Not 4D
        let bounds = BatchedLinearBounds::identity_for_attention(&[2, 4, 4]);
        assert!(bounds.is_none());
    }

    #[test]
    fn test_batched_linear_bounds_identity_for_attention_too_large() {
        // seq=65 means flat_size=4225 > 4096 limit
        let bounds = BatchedLinearBounds::identity_for_attention(&[1, 1, 65, 65]);
        assert!(bounds.is_none());
    }

    #[test]
    fn test_batched_linear_bounds_compose_identity() {
        // Composing two identities should give identity
        let a = BatchedLinearBounds::identity(&[3]);
        let b = BatchedLinearBounds::identity(&[3]);

        let composed = a.compose(&b).unwrap();

        // Check shapes
        assert_eq!(composed.input_shape, a.input_shape);
        assert_eq!(composed.output_shape, b.output_shape);

        // Composed identity should still be identity
        // Test by concretizing
        let input = BoundedTensor {
            lower: array![1.0_f32, 2.0, 3.0].into_dyn(),
            upper: array![4.0_f32, 5.0, 6.0].into_dyn(),
        };

        let output = composed.concretize(&input);
        assert_eq!(output.lower.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(output.upper.as_slice().unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_batched_linear_bounds_compose_dimension_mismatch() {
        let a = BatchedLinearBounds::identity(&[3]);
        let b = BatchedLinearBounds::identity(&[4]); // Different dim

        let result = a.compose(&b);
        assert!(result.is_err());
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn test_safe_mul_for_bounds_normal() {
        assert_eq!(safe_mul_for_bounds(2.0, 3.0), 6.0);
        assert_eq!(safe_mul_for_bounds(-2.0, 3.0), -6.0);
        assert_eq!(safe_mul_for_bounds(0.5, 4.0), 2.0);
    }

    #[test]
    fn test_safe_mul_for_bounds_zero() {
        // 0 * anything = 0, even infinity
        assert_eq!(safe_mul_for_bounds(0.0, f32::INFINITY), 0.0);
        assert_eq!(safe_mul_for_bounds(0.0, f32::NEG_INFINITY), 0.0);
        assert_eq!(safe_mul_for_bounds(f32::INFINITY, 0.0), 0.0);
        assert_eq!(safe_mul_for_bounds(f32::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(safe_mul_for_bounds(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_safe_mul_for_bounds_nan() {
        // NaN propagates (indicates invalid bounds)
        assert!(safe_mul_for_bounds(f32::NAN, 1.0).is_nan());
        assert!(safe_mul_for_bounds(1.0, f32::NAN).is_nan());
    }

    #[test]
    fn test_safe_mul_for_bounds_infinity() {
        // Non-zero * infinity = infinity
        assert_eq!(safe_mul_for_bounds(2.0, f32::INFINITY), f32::INFINITY);
        assert_eq!(safe_mul_for_bounds(-2.0, f32::INFINITY), f32::NEG_INFINITY);
        assert_eq!(
            safe_mul_for_bounds(2.0, f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn test_safe_add_for_bounds_with_polarity_normal() {
        assert_eq!(safe_add_for_bounds_with_polarity(1.0, 2.0, true), 3.0);
        assert_eq!(safe_add_for_bounds_with_polarity(1.0, 2.0, false), 3.0);
    }

    #[test]
    fn test_safe_add_for_bounds_with_polarity_inf_minus_inf() {
        // inf + (-inf) = NaN in standard math, but we use conservative bounds
        // For lower bound: use -inf (conservative)
        let result_lower =
            safe_add_for_bounds_with_polarity(f32::INFINITY, f32::NEG_INFINITY, true);
        assert_eq!(result_lower, f32::NEG_INFINITY);

        // For upper bound: use +inf (conservative)
        let result_upper =
            safe_add_for_bounds_with_polarity(f32::INFINITY, f32::NEG_INFINITY, false);
        assert_eq!(result_upper, f32::INFINITY);
    }

    #[test]
    fn test_safe_add_for_bounds_default() {
        // Default (no polarity) should use upper bound (conservative = +inf for NaN)
        let result = safe_add_for_bounds(f32::INFINITY, f32::NEG_INFINITY);
        assert_eq!(result, f32::INFINITY);
    }

    #[test]
    fn test_safe_array_add_normal() {
        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 2.0, 3.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0_f32, 5.0, 6.0]).unwrap();

        let result_lower = safe_array_add(&a, &b, true);
        let result_upper = safe_array_add(&a, &b, false);

        assert_eq!(result_lower.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
        assert_eq!(result_upper.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_safe_array_add_with_inf() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 2.0]).unwrap();

        let result_lower = safe_array_add(&a, &b, true);
        let result_upper = safe_array_add(&a, &b, false);

        // First element: inf + (-inf) = -inf for lower, +inf for upper
        assert_eq!(result_lower[[0]], f32::NEG_INFINITY);
        assert_eq!(result_upper[[0]], f32::INFINITY);

        // Second element: 1 + 2 = 3
        assert_eq!(result_lower[[1]], 3.0);
        assert_eq!(result_upper[[1]], 3.0);
    }

    #[test]
    fn test_batched_matvec_simple() {
        // A: [1, 2, 3] matrix, x: [1, 3] vector
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32])
            .unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 1.0, 1.0_f32]).unwrap();

        let result = batched_matvec(&a, &x);

        // Row 0: 1+2+3 = 6
        // Row 1: 4+5+6 = 15
        assert_eq!(result.shape(), &[1, 2]);
        let slice = result.as_slice().unwrap();
        assert!((slice[0] - 6.0).abs() < 1e-6);
        assert!((slice[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_batched_matvec_with_inf_and_zero() {
        // Test 0 * inf = 0 handling
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.0, 1.0_f32]).unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::INFINITY, 2.0]).unwrap();

        let result = batched_matvec(&a, &x);

        // 0 * inf + 1 * 2 = 0 + 2 = 2
        assert_eq!(result[[0, 0]], 2.0);
    }

    #[test]
    fn test_batched_matvec_batched() {
        // Batch of 2, each 2x2 matrix
        let a = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                1.0, 0.0, 0.0, 1.0, // Identity batch 0
                2.0, 0.0, 0.0, 2.0, // 2*Identity batch 1
            ],
        )
        .unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![3.0, 4.0, 3.0, 4.0_f32]).unwrap();

        let result = batched_matvec(&a, &x);

        assert_eq!(result.shape(), &[2, 2]);
        // Batch 0: I @ [3,4] = [3,4]
        assert!((result[[0, 0]] - 3.0).abs() < 1e-6);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-6);
        // Batch 1: 2I @ [3,4] = [6,8]
        assert!((result[[1, 0]] - 6.0).abs() < 1e-6);
        assert!((result[[1, 1]] - 8.0).abs() < 1e-6);
    }

    // =========================================================================
    // GradientMethod tests
    // =========================================================================

    #[test]
    fn test_gradient_method_default() {
        let method = GradientMethod::default();
        assert_eq!(method, GradientMethod::Spsa);
    }

    #[test]
    fn test_gradient_method_equality() {
        assert_eq!(GradientMethod::Spsa, GradientMethod::Spsa);
        assert_eq!(
            GradientMethod::FiniteDifferences,
            GradientMethod::FiniteDifferences
        );
        assert_ne!(GradientMethod::Spsa, GradientMethod::FiniteDifferences);
    }

    #[test]
    fn test_gradient_method_clone() {
        // GradientMethod implements Copy, so we test the Clone impl via Copy behavior
        let method = GradientMethod::Spsa;
        let cloned: GradientMethod = method; // Uses Copy (which implies Clone)
        assert_eq!(method, cloned);
    }

    // =========================================================================
    // AlphaCrownConfig tests
    // =========================================================================

    #[test]
    fn test_alpha_crown_config_default() {
        let config = AlphaCrownConfig::default();

        // Default now matches α,β-CROWN's proven configuration
        assert_eq!(config.iterations, 20);
        assert_eq!(config.learning_rate, 0.1); // α,β-CROWN default
        assert_eq!(config.lr_decay, 0.98); // α,β-CROWN ExponentialLR decay
        assert_eq!(config.tolerance, 1e-4);
        assert!(config.use_momentum);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.gradient_method, GradientMethod::Spsa);
        assert_eq!(config.spsa_samples, 1);
        assert!(config.fix_interm_bounds);
        assert_eq!(config.sparse_ratio, 0.3);
        assert!(config.adaptive_skip);
        assert_eq!(config.adaptive_skip_depth_threshold, 8);
        assert!(config.adaptive_skip_pilot);
        assert_eq!(config.pilot_improvement_threshold, 1e-3);
        // Adam optimizer settings (ported from α,β-CROWN)
        assert_eq!(config.optimizer, Optimizer::Adam);
        assert_eq!(config.adam_beta1, 0.9);
        assert_eq!(config.adam_beta2, 0.999);
        assert_eq!(config.adam_epsilon, 1e-8);
    }

    #[test]
    fn test_alpha_crown_config_clone() {
        let config = AlphaCrownConfig::default();
        let cloned = config.clone();

        assert_eq!(config.iterations, cloned.iterations);
        assert_eq!(config.learning_rate, cloned.learning_rate);
        assert_eq!(config.gradient_method, cloned.gradient_method);
    }

    #[test]
    fn test_alpha_crown_config_custom() {
        let config = AlphaCrownConfig {
            iterations: 10,
            learning_rate: 0.1,
            lr_decay: 0.95,
            tolerance: 1e-6,
            use_momentum: false,
            momentum: 0.0,
            gradient_method: GradientMethod::FiniteDifferences,
            spsa_samples: 5,
            fix_interm_bounds: false,
            sparse_ratio: 1.0,
            adaptive_skip: false,
            adaptive_skip_depth_threshold: 100,
            adaptive_skip_pilot: false,
            pilot_improvement_threshold: 0.0,
            optimizer: Optimizer::Sgd,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
        };

        assert_eq!(config.iterations, 10);
        assert!(!config.use_momentum);
        assert_eq!(config.gradient_method, GradientMethod::FiniteDifferences);
        assert_eq!(config.optimizer, Optimizer::Sgd);
    }

    #[test]
    fn test_alpha_crown_config_serialization() {
        let config = AlphaCrownConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AlphaCrownConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.iterations, deserialized.iterations);
        assert_eq!(config.learning_rate, deserialized.learning_rate);
        assert_eq!(config.gradient_method, deserialized.gradient_method);
    }

    // =========================================================================
    // AlphaState tests
    // =========================================================================

    #[test]
    fn test_alpha_state_from_all_positive() {
        // All neurons always positive: no unstable
        let bounds = vec![BoundedTensor {
            lower: array![1.0_f32, 2.0, 3.0].into_dyn(),
            upper: array![4.0_f32, 5.0, 6.0].into_dyn(),
        }];
        let relu_indices = vec![0];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        assert_eq!(state.num_unstable(), 0);
        // All alphas should be 1.0 (identity for positive)
        assert_eq!(state.alphas[0].as_slice().unwrap(), &[1.0, 1.0, 1.0]);
        // Mask should be all false
        assert_eq!(
            state.unstable_mask[0].as_slice().unwrap(),
            &[false, false, false]
        );
    }

    #[test]
    fn test_alpha_state_from_all_negative() {
        // All neurons always negative: no unstable
        let bounds = vec![BoundedTensor {
            lower: array![-6.0_f32, -5.0, -4.0].into_dyn(),
            upper: array![-3.0_f32, -2.0, -1.0].into_dyn(),
        }];
        let relu_indices = vec![0];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        assert_eq!(state.num_unstable(), 0);
        // All alphas should be 0.0 (zero for negative)
        assert_eq!(state.alphas[0].as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_alpha_state_from_mixed_bounds() {
        // Mix of positive, negative, and crossing
        let bounds = vec![BoundedTensor {
            lower: array![1.0_f32, -5.0, -2.0].into_dyn(), // always pos, always neg, crossing
            upper: array![3.0_f32, -1.0, 4.0].into_dyn(),
        }];
        let relu_indices = vec![0];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        assert_eq!(state.num_unstable(), 1); // Only neuron 2 is unstable

        // Neuron 0: always positive -> alpha=1, mask=false
        assert_eq!(state.alphas[0][0], 1.0);
        assert!(!state.unstable_mask[0][0]);

        // Neuron 1: always negative -> alpha=0, mask=false
        assert_eq!(state.alphas[0][1], 0.0);
        assert!(!state.unstable_mask[0][1]);

        // Neuron 2: crossing [-2, 4], u=4 > -l=2 -> alpha=1, mask=true
        assert_eq!(state.alphas[0][2], 1.0);
        assert!(state.unstable_mask[0][2]);
    }

    #[test]
    fn test_alpha_state_from_crossing_more_negative() {
        // Crossing neuron with more negative area
        let bounds = vec![BoundedTensor {
            lower: array![-4.0_f32].into_dyn(), // crossing with u < -l
            upper: array![1.0_f32].into_dyn(),
        }];
        let relu_indices = vec![0];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        // u=1 < -l=4 -> alpha=0 (adaptive heuristic)
        assert_eq!(state.alphas[0][0], 0.0);
        assert!(state.unstable_mask[0][0]);
    }

    #[test]
    fn test_alpha_state_get_alpha() {
        let bounds = vec![BoundedTensor {
            lower: array![-1.0_f32, -1.0].into_dyn(),
            upper: array![1.0_f32, 1.0].into_dyn(),
        }];
        let relu_indices = vec![0];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        assert!(state.get_alpha(0).is_some());
        assert!(state.get_alpha(1).is_none()); // Out of range
    }

    #[test]
    fn test_alpha_state_update_without_momentum() {
        // Use asymmetric bounds where u > -l, so alpha initializes to 1
        let bounds = vec![BoundedTensor {
            lower: array![-1.0_f32, -1.0].into_dyn(),
            upper: array![2.0_f32, 2.0].into_dyn(), // u=2 > -l=1
        }];
        let relu_indices = vec![0];

        let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        // Both neurons unstable with alpha=1 (u > -l)
        // Verify initial state before update
        assert_eq!(state.alphas[0][0], 1.0);
        assert_eq!(state.alphas[0][1], 1.0);

        // Apply gradient descent (gradient points up, so alpha decreases)
        let gradient = array![0.5_f32, 0.5];
        state.update(0, &gradient, 0.1, 0.0);

        // alpha -= lr * gradient = 1.0 - 0.1 * 0.5 = 0.95
        assert!((state.alphas[0][0] - 0.95).abs() < 1e-6);
        assert!((state.alphas[0][1] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_state_update_with_clamping() {
        // Use asymmetric bounds where u > -l, so alpha initializes to 1
        let bounds = vec![BoundedTensor {
            lower: array![-1.0_f32].into_dyn(),
            upper: array![2.0_f32].into_dyn(), // u=2 > -l=1
        }];
        let relu_indices = vec![0];

        let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);
        // Alpha starts at 1.0 (since u > -l)
        assert_eq!(state.alphas[0][0], 1.0);

        // Large positive gradient should clamp alpha to 0
        let gradient = array![100.0_f32];
        state.update(0, &gradient, 1.0, 0.0);

        assert_eq!(state.alphas[0][0], 0.0); // Clamped to 0

        // Large negative gradient should clamp alpha to 1
        let gradient = array![-100.0_f32];
        state.update(0, &gradient, 1.0, 0.0);

        assert_eq!(state.alphas[0][0], 1.0); // Clamped to 1
    }

    #[test]
    fn test_alpha_state_update_skips_stable() {
        // Use asymmetric bounds for unstable neuron: u=2 > -l=1 so alpha=1
        let bounds = vec![BoundedTensor {
            lower: array![1.0_f32, -1.0].into_dyn(), // First stable (positive), second unstable
            upper: array![2.0_f32, 2.0].into_dyn(),  // Second neuron: u=2 > -l=1
        }];
        let relu_indices = vec![0];

        let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        // First neuron is stable (alpha=1, mask=false)
        // Second neuron is unstable with alpha=1 (u=2 > -l=1)
        let initial_stable = state.alphas[0][0];
        assert_eq!(state.alphas[0][1], 1.0);

        // Update shouldn't change stable neuron
        let gradient = array![10.0_f32, 0.5];
        state.update(0, &gradient, 0.1, 0.0);

        assert_eq!(state.alphas[0][0], initial_stable); // Unchanged
        assert!((state.alphas[0][1] - 0.95).abs() < 1e-6); // Updated
    }

    #[test]
    fn test_alpha_state_update_invalid_index() {
        let bounds = vec![BoundedTensor {
            lower: array![-1.0_f32].into_dyn(),
            upper: array![1.0_f32].into_dyn(),
        }];
        let relu_indices = vec![0];

        let mut state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);
        let alpha_before = state.alphas[0][0];

        // Invalid relu index - should silently return
        let gradient = array![1.0_f32];
        state.update(99, &gradient, 0.1, 0.0);

        // State unchanged
        assert_eq!(state.alphas[0][0], alpha_before);
    }

    #[test]
    fn test_alpha_state_num_unstable_multiple_layers() {
        let bounds = vec![
            BoundedTensor {
                lower: array![-1.0_f32, 1.0, -1.0].into_dyn(), // 2 unstable
                upper: array![1.0_f32, 2.0, 1.0].into_dyn(),
            },
            BoundedTensor {
                lower: array![-1.0_f32, -1.0].into_dyn(), // 2 unstable
                upper: array![1.0_f32, 1.0].into_dyn(),
            },
        ];
        let relu_indices = vec![0, 1];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        assert_eq!(state.num_unstable(), 4); // 2 + 2
    }

    // =========================================================================
    // GraphAlphaState tests
    // =========================================================================

    #[test]
    fn test_graph_alpha_state_new() {
        let state = GraphAlphaState::new();

        assert!(state.alphas.is_empty());
        assert!(state.unstable_mask.is_empty());
        assert!(state.velocity.is_empty());
        assert_eq!(state.num_unstable(), 0);
    }

    #[test]
    fn test_graph_alpha_state_default() {
        let state = GraphAlphaState::default();
        assert!(state.alphas.is_empty());
    }

    #[test]
    fn test_graph_alpha_state_add_relu_node_all_positive() {
        let mut state = GraphAlphaState::new();

        let bounds = BoundedTensor {
            lower: array![1.0_f32, 2.0].into_dyn(),
            upper: array![3.0_f32, 4.0].into_dyn(),
        };

        state.add_relu_node("relu1", &bounds);

        assert_eq!(state.num_unstable(), 0);
        assert_eq!(
            state.get_alpha("relu1").unwrap().as_slice().unwrap(),
            &[1.0, 1.0]
        );
    }

    #[test]
    fn test_graph_alpha_state_add_relu_node_mixed() {
        let mut state = GraphAlphaState::new();

        let bounds = BoundedTensor {
            lower: array![1.0_f32, -3.0, -1.0].into_dyn(), // positive, negative, crossing
            upper: array![2.0_f32, -1.0, 2.0].into_dyn(),
        };

        state.add_relu_node("relu1", &bounds);

        assert_eq!(state.num_unstable(), 1);

        let alpha = state.get_alpha("relu1").unwrap();
        assert_eq!(alpha[0], 1.0); // Positive
        assert_eq!(alpha[1], 0.0); // Negative
        assert_eq!(alpha[2], 1.0); // Crossing with u=2 > -l=1
    }

    #[test]
    fn test_graph_alpha_state_add_multiple_nodes() {
        let mut state = GraphAlphaState::new();

        state.add_relu_node(
            "relu1",
            &BoundedTensor {
                lower: array![-1.0_f32].into_dyn(),
                upper: array![1.0_f32].into_dyn(),
            },
        );

        state.add_relu_node(
            "relu2",
            &BoundedTensor {
                lower: array![-1.0_f32, -1.0].into_dyn(),
                upper: array![1.0_f32, 1.0].into_dyn(),
            },
        );

        assert_eq!(state.num_unstable(), 3); // 1 + 2

        let nodes: Vec<&String> = state.relu_nodes().collect();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_graph_alpha_state_get_alpha_missing() {
        let state = GraphAlphaState::new();
        assert!(state.get_alpha("nonexistent").is_none());
    }

    #[test]
    fn test_graph_alpha_state_update() {
        let mut state = GraphAlphaState::new();

        // Use asymmetric bounds where u > -l, so alpha initializes to 1
        state.add_relu_node(
            "relu1",
            &BoundedTensor {
                lower: array![-1.0_f32, -1.0].into_dyn(),
                upper: array![2.0_f32, 2.0].into_dyn(), // u=2 > -l=1
            },
        );

        // Both neurons unstable with alpha=1 (u > -l)
        assert_eq!(state.get_alpha("relu1").unwrap()[0], 1.0);
        assert_eq!(state.get_alpha("relu1").unwrap()[1], 1.0);

        let gradient = array![0.5_f32, 0.5];
        state.update("relu1", &gradient, 0.1, 0.0);

        let alpha = state.get_alpha("relu1").unwrap();
        // alpha -= lr * gradient = 1.0 - 0.1 * 0.5 = 0.95
        assert!((alpha[0] - 0.95).abs() < 1e-6);
        assert!((alpha[1] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_graph_alpha_state_update_missing_node() {
        let mut state = GraphAlphaState::new();

        // Should not panic, just silently return
        let gradient = array![1.0_f32];
        state.update("nonexistent", &gradient, 0.1, 0.0);

        assert_eq!(state.num_unstable(), 0);
    }

    #[test]
    fn test_graph_alpha_state_update_with_momentum() {
        let mut state = GraphAlphaState::new();

        // Use asymmetric bounds where u > -l, so alpha initializes to 1
        state.add_relu_node(
            "relu1",
            &BoundedTensor {
                lower: array![-1.0_f32].into_dyn(),
                upper: array![2.0_f32].into_dyn(), // u=2 > -l=1
            },
        );

        // Alpha starts at 1.0 (u > -l)
        assert_eq!(state.get_alpha("relu1").unwrap()[0], 1.0);

        // First update
        let gradient = array![0.5_f32];
        state.update("relu1", &gradient, 0.1, 0.9);

        // vel = 0.9 * 0 - 0.1 * 0.5 = -0.05
        // alpha = 1.0 + (-0.05) = 0.95
        let alpha1 = state.get_alpha("relu1").unwrap()[0];
        assert!((alpha1 - 0.95).abs() < 1e-6);

        // Second update - momentum should accumulate
        state.update("relu1", &gradient, 0.1, 0.9);

        // vel = 0.9 * (-0.05) - 0.1 * 0.5 = -0.045 - 0.05 = -0.095
        // alpha = 0.95 + (-0.095) = 0.855
        let alpha2 = state.get_alpha("relu1").unwrap()[0];
        assert!((alpha2 - 0.855).abs() < 1e-5);
    }

    #[test]
    fn test_graph_alpha_state_relu_nodes_iterator() {
        let mut state = GraphAlphaState::new();

        state.add_relu_node(
            "a_relu",
            &BoundedTensor {
                lower: array![-1.0_f32].into_dyn(),
                upper: array![1.0_f32].into_dyn(),
            },
        );
        state.add_relu_node(
            "b_relu",
            &BoundedTensor {
                lower: array![-1.0_f32].into_dyn(),
                upper: array![1.0_f32].into_dyn(),
            },
        );

        let mut nodes: Vec<&str> = state.relu_nodes().map(|s| s.as_str()).collect();
        nodes.sort();

        assert_eq!(nodes, vec!["a_relu", "b_relu"]);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_linear_bounds_large_matrix() {
        // Test with a larger matrix to ensure no indexing issues
        let n = 100;
        let bounds = LinearBounds::identity(n);

        let input = BoundedTensor {
            lower: ArrayD::from_elem(IxDyn(&[n]), 0.0_f32),
            upper: ArrayD::from_elem(IxDyn(&[n]), 1.0_f32),
        };

        let output = bounds.concretize(&input);

        assert_eq!(output.lower.shape(), &[n]);
        assert_eq!(output.upper.shape(), &[n]);
    }

    #[test]
    fn test_batched_matvec_single_batch() {
        // Single batch with identity-like matrix
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 0.0, 0.0, 1.0_f32]).unwrap();
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![3.0, 4.0_f32]).unwrap();

        let result = batched_matvec(&a, &x);

        assert_eq!(result.shape(), &[1, 2]);
        assert!((result[[0, 0]] - 3.0).abs() < 1e-6);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_state_empty_layers() {
        let bounds: Vec<BoundedTensor> = vec![];
        let relu_indices: Vec<usize> = vec![];

        let state = AlphaState::from_preactivation_bounds(&bounds, &relu_indices);

        assert_eq!(state.num_unstable(), 0);
        assert!(state.alphas.is_empty());
    }

    #[test]
    fn test_linear_bounds_single_element() {
        let bounds = LinearBounds::identity(1);

        let input = BoundedTensor {
            lower: array![5.0_f32].into_dyn(),
            upper: array![10.0_f32].into_dyn(),
        };

        let output = bounds.concretize(&input);

        assert_eq!(output.lower[[0]], 5.0);
        assert_eq!(output.upper[[0]], 10.0);
    }
}
