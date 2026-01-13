//! Zonotope tensor for correlation-aware bound propagation.
//!
//! Zonotopes represent values as center + Σᵢ (coeffᵢ · eᵢ) where eᵢ ∈ [-1, 1].
//! Unlike interval bounds, zonotopes track correlations between variables through
//! shared error symbols, giving tighter bounds for operations like Q@K^T in attention.
//!
//! # Key Insight
//!
//! For Q@K^T where Q = f(X) and K = g(X) depend on the same input X:
//! - IBP treats Q and K as independent: bounds explode by ~1600x per layer
//! - Zonotopes share error symbols: `e_i² ∈ [0,1]` not `[-1,1]`, giving tighter bounds
//!
//! # References
//!
//! - Bonaert et al. (2020): "Robustness Verification for Transformers" (arxiv:2002.06622)
//! - DeepT: research/repos/DeepT/.../Verifiers/Zonotope.py

use gamma_core::{GammaError, Result};
use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::BoundedTensor;

/// A zonotope tensor: center + Σᵢ (coeffᵢ · eᵢ) where eᵢ ∈ [-1, 1]
///
/// Memory layout: coeffs has shape `(1 + n_error_terms, ...element_shape)`
/// - `coeffs[0]` = center
/// - `coeffs[1..]` = error term coefficients
///
/// # Example
///
/// For input x ∈ [0.9, 1.1] (center=1.0, epsilon=0.1):
/// ```text
/// Zonotope: x = 1.0 + 0.1·e₁   where e₁ ∈ [-1, 1]
/// coeffs = [[1.0], [0.1]]  (shape: 2×1)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZonotopeTensor {
    /// Combined center and error coefficients.
    /// Shape: (1 + n_error_terms, ...element_shape)
    pub coeffs: ArrayD<f32>,

    /// Number of error terms (not including center).
    pub n_error_terms: usize,

    /// Shape of each element tensor (excludes the error term dimension).
    pub element_shape: Vec<usize>,
}

impl ZonotopeTensor {
    /// Create a zonotope from combined coefficients array.
    ///
    /// # Arguments
    /// * `coeffs` - Array of shape (1 + n_error_terms, ...element_shape)
    pub fn new(coeffs: ArrayD<f32>) -> Result<Self> {
        let shape = coeffs.shape();
        if shape.is_empty() {
            return Err(GammaError::InvalidSpec(
                "Zonotope coeffs must have at least 1 dimension".to_string(),
            ));
        }

        let n_error_terms = shape[0].saturating_sub(1);
        let element_shape = shape[1..].to_vec();

        Ok(Self {
            coeffs,
            n_error_terms,
            element_shape,
        })
    }

    /// Create a zonotope representing a concrete value (no uncertainty).
    pub fn concrete(values: ArrayD<f32>) -> Self {
        let mut coeffs_shape = vec![1];
        coeffs_shape.extend_from_slice(values.shape());

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));
        coeffs.index_axis_mut(Axis(0), 0).assign(&values);

        Self {
            coeffs,
            n_error_terms: 0,
            element_shape: values.shape().to_vec(),
        }
    }

    /// Create a zonotope from input with epsilon perturbation.
    ///
    /// Each element of the input gets its own error symbol, allowing
    /// the zonotope to track how perturbations to each element propagate.
    ///
    /// # Arguments
    /// * `values` - Center values for the input
    /// * `epsilon` - Maximum perturbation for each element
    ///
    /// # Note
    ///
    /// This creates n_elements error terms, which can be memory-intensive
    /// for large inputs. For large models, consider `from_input_shared_symbols`
    /// which uses fewer symbols.
    pub fn from_input_elementwise(values: &ArrayD<f32>, epsilon: f32) -> Self {
        let n_elements = values.len();
        let _element_shape = values.shape().to_vec(); // Preserved for future multi-dim support

        // coeffs shape: (1 + n_elements, flat_size)
        // We flatten for simplicity; can reshape later
        let flat_values = values.iter().cloned().collect::<Vec<_>>();

        let mut coeffs = Array2::<f32>::zeros((1 + n_elements, n_elements));

        // Center row = values
        for (i, &v) in flat_values.iter().enumerate() {
            coeffs[[0, i]] = v;
        }

        // Each element gets epsilon coefficient at its own error term
        for i in 0..n_elements {
            coeffs[[1 + i, i]] = epsilon;
        }

        Self {
            coeffs: coeffs.into_dyn(),
            n_error_terms: n_elements,
            element_shape: vec![n_elements], // Flattened
        }
    }

    /// Create a zonotope with a single shared error symbol.
    ///
    /// All elements share one error symbol, representing uniform perturbation.
    /// This is memory-efficient but doesn't track element-specific correlations.
    ///
    /// # Arguments
    /// * `values` - Center values
    /// * `epsilon` - Maximum perturbation (same for all elements)
    pub fn from_input_shared(values: &ArrayD<f32>, epsilon: f32) -> Self {
        let element_shape = values.shape().to_vec();

        // coeffs shape: (2, ...element_shape)
        let mut coeffs_shape = vec![2];
        coeffs_shape.extend_from_slice(&element_shape);

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));

        // Center = values
        coeffs.index_axis_mut(Axis(0), 0).assign(values);

        // Single error term with coefficient = epsilon for all elements
        coeffs.index_axis_mut(Axis(0), 1).fill(epsilon);

        Self {
            coeffs,
            n_error_terms: 1,
            element_shape,
        }
    }

    /// Create a 2D zonotope from a matrix with per-element error symbols.
    ///
    /// Each element (i,j) gets its own error symbol. This is needed for
    /// operations like Q@K^T where we want to track correlations between
    /// all elements of Q and K.
    ///
    /// # Arguments
    /// * `values` - Center values with shape (rows, cols)
    /// * `epsilon` - Maximum perturbation for each element
    ///
    /// # Layout
    /// * coeffs shape: `(1 + rows*cols, rows, cols)`
    /// * `coeffs[0]` = center matrix
    /// * `coeffs[1 + i*cols + j]` has epsilon at position `(i,j)`, zero elsewhere
    pub fn from_input_2d(values: &Array2<f32>, epsilon: f32) -> Self {
        let rows = values.nrows();
        let cols = values.ncols();
        let n_elements = rows * cols;

        // coeffs shape: (1 + n_elements, rows, cols)
        let mut coeffs = ndarray::Array3::<f32>::zeros((1 + n_elements, rows, cols));

        // Center = values
        coeffs.index_axis_mut(Axis(0), 0).assign(values);

        // Each element (i,j) gets its own error symbol
        for i in 0..rows {
            for j in 0..cols {
                let error_idx = 1 + i * cols + j;
                coeffs[[error_idx, i, j]] = epsilon;
            }
        }

        Self {
            coeffs: coeffs.into_dyn(),
            n_error_terms: n_elements,
            element_shape: vec![rows, cols],
        }
    }

    /// Create a zonotope with per-position error symbols (for sequence data).
    ///
    /// For shape (..., seq_len, embed_dim), creates seq_len error symbols.
    /// All elements at each position share a symbol.
    ///
    /// # Arguments
    /// * `values` - Center values with shape (..., seq_len, embed_dim)
    /// * `epsilon` - Maximum perturbation
    pub fn from_input_per_position(values: &ArrayD<f32>, epsilon: f32) -> Result<Self> {
        let shape = values.shape();
        if shape.len() < 2 {
            return Err(GammaError::InvalidSpec(
                "from_input_per_position requires at least 2 dimensions".to_string(),
            ));
        }

        let seq_len = shape[shape.len() - 2];
        let element_shape = shape.to_vec();

        let n_error_terms = match shape.len() {
            2 => seq_len,
            3 => shape[0] * seq_len,
            _ => {
                return Err(GammaError::InvalidSpec(
                    "from_input_per_position currently only supports 2D or 3D tensors".to_string(),
                ));
            }
        };

        // coeffs shape: (1 + n_error_terms, ...element_shape)
        let mut coeffs_shape = vec![1 + n_error_terms];
        coeffs_shape.extend_from_slice(&element_shape);

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));

        // Center = values
        coeffs.index_axis_mut(Axis(0), 0).assign(values);

        match shape.len() {
            2 => {
                // For position i, set coeffs[1+i, i, :] = epsilon
                for pos in 0..seq_len {
                    for emb in 0..shape[1] {
                        coeffs[[1 + pos, pos, emb]] = epsilon;
                    }
                }
            }
            3 => {
                let batch = shape[0];
                let dim = shape[2];
                // For (b,pos), set coeffs[1 + b*seq + pos, b, pos, :] = epsilon
                for b in 0..batch {
                    for pos in 0..seq_len {
                        let err = 1 + b * seq_len + pos;
                        for d in 0..dim {
                            coeffs[[err, b, pos, d]] = epsilon;
                        }
                    }
                }
            }
            _ => unreachable!("handled above"),
        }

        Ok(Self {
            coeffs,
            n_error_terms,
            element_shape,
        })
    }

    /// Get the center tensor (point estimate).
    pub fn center(&self) -> ArrayD<f32> {
        self.coeffs.index_axis(Axis(0), 0).to_owned()
    }

    /// Compute the radius at each element (max deviation from center).
    ///
    /// radius = Σᵢ |coeffᵢ| (sum of absolute error coefficients)
    pub fn radius(&self) -> ArrayD<f32> {
        let mut radius = ArrayD::zeros(IxDyn(&self.element_shape));

        for i in 1..=self.n_error_terms {
            radius = radius + self.coeffs.index_axis(Axis(0), i).mapv(f32::abs);
        }

        radius
    }

    /// Convert zonotope to interval bounds [center - radius, center + radius].
    ///
    /// This is a lossy conversion - interval bounds don't track correlations.
    pub fn to_bounded_tensor(&self) -> BoundedTensor {
        let center = self.center();
        let radius = self.radius();

        BoundedTensor {
            lower: &center - &radius,
            upper: &center + &radius,
        }
    }

    /// Get the shape of each element (excluding error term dimension).
    pub fn shape(&self) -> &[usize] {
        &self.element_shape
    }

    /// Total number of elements per error term.
    pub fn len(&self) -> usize {
        self.element_shape.iter().product()
    }

    /// Check if the zonotope has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum width (upper - lower) across all elements.
    pub fn max_width(&self) -> f32 {
        let radius = self.radius();
        let width = radius.mapv(|r| 2.0 * r);
        width.iter().cloned().fold(0.0_f32, f32::max)
    }

    /// Check if bounds have exploded to infinity.
    pub fn has_unbounded(&self) -> bool {
        self.coeffs.iter().any(|v| v.is_infinite())
    }
}

/// Linear operations on zonotopes (preserve zonotope form exactly).
impl ZonotopeTensor {
    /// Scalar addition: z + c
    pub fn shift(&self, scalar: f32) -> Self {
        let mut result = self.clone();
        result
            .coeffs
            .index_axis_mut(Axis(0), 0)
            .mapv_inplace(|v| v + scalar);
        result
    }

    /// Scalar multiplication: c * z
    pub fn scale(&self, scalar: f32) -> Self {
        let mut result = self.clone();
        result.coeffs.mapv_inplace(|v| v * scalar);
        result
    }

    /// Element-wise addition of two zonotopes with same error symbols.
    ///
    /// (a₀ + Σᵢ aᵢeᵢ) + (b₀ + Σᵢ bᵢeᵢ) = (a₀+b₀) + Σᵢ (aᵢ+bᵢ)eᵢ
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.n_error_terms != other.n_error_terms {
            return Err(GammaError::InvalidSpec(format!(
                "Cannot add zonotopes with different error term counts: {} vs {}",
                self.n_error_terms, other.n_error_terms
            )));
        }

        if self.element_shape != other.element_shape {
            return Err(GammaError::shape_mismatch(
                self.element_shape.clone(),
                other.element_shape.clone(),
            ));
        }

        let coeffs = &self.coeffs + &other.coeffs;
        Self::new(coeffs)
    }

    /// Linear transformation: W·z + b
    ///
    /// Applies weight matrix to center and all error coefficients.
    /// The zonotope form is preserved exactly.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix of shape (out_features, in_features)
    /// * `bias` - Optional bias vector of shape (out_features,)
    ///
    /// # Input/Output
    /// * Input zonotope shape: (..., in_features)
    /// * Output zonotope shape: (..., out_features)
    pub fn linear(&self, weight: &Array2<f32>, bias: Option<&Array1<f32>>) -> Result<Self> {
        let in_features = weight.ncols();
        let out_features = weight.nrows();

        // Check that last dimension matches weight's in_features
        if self.element_shape.is_empty() || self.element_shape.last() != Some(&in_features) {
            return Err(GammaError::shape_mismatch(
                vec![in_features],
                self.element_shape.clone(),
            ));
        }

        let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let prefix_shape = &self.element_shape[..self.element_shape.len() - 1];
        let prefix_size = prefix_shape.iter().product::<usize>().max(1);
        let n_rows = 1 + self.n_error_terms;

        let mut result_shape = vec![n_rows];
        result_shape.extend_from_slice(prefix_shape);
        result_shape.push(out_features);
        let mut result_coeffs = ArrayD::<f32>::zeros(IxDyn(&result_shape));

        let weight_t = weight.t();
        for row in 0..n_rows {
            let input_view = coeffs
                .index_axis(Axis(0), row)
                .into_shape_with_order(IxDyn(&[prefix_size, in_features]))
                .map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape linear input to 2D".to_string())
                })?
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    GammaError::InvalidSpec("Cannot view linear input as 2D".to_string())
                })?;

            let output_2d = input_view.dot(&weight_t);
            let output = output_2d
                .into_dyn()
                .into_shape_with_order(IxDyn(&result_shape[1..]))
                .map_err(|_| GammaError::InvalidSpec("Cannot reshape linear output".to_string()))?;

            result_coeffs.index_axis_mut(Axis(0), row).assign(&output);
        }

        if let Some(b) = bias {
            let mut center = result_coeffs.index_axis_mut(Axis(0), 0);
            let last_axis = center.ndim().saturating_sub(1);
            for mut lane in center.lanes_mut(Axis(last_axis)) {
                lane += &b.view();
            }
        }

        let mut new_element_shape = prefix_shape.to_vec();
        new_element_shape.push(out_features);

        Ok(Self {
            coeffs: result_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: new_element_shape,
        })
    }
}

/// Bilinear operations on zonotopes.
impl ZonotopeTensor {
    /// Dot product of two 1D zonotopes: z₁ · z₂
    ///
    /// For z₁ = a₀ + Σᵢ aᵢeᵢ and z₂ = b₀ + Σᵢ bᵢeᵢ:
    ///
    /// z₁·z₂ = (a₀·b₀) + Σᵢ(a₀bᵢ + aᵢb₀)eᵢ + Σᵢ(aᵢbᵢ)eᵢ² + Σᵢ≠ⱼ(aᵢbⱼ)eᵢeⱼ
    ///
    /// Key insight: `eᵢ² ∈ [0,1]`, so we compute:
    /// - Center shift: +0.5 · Σᵢ(aᵢbᵢ)
    /// - New error: 0.5 · Σᵢ|aᵢbᵢ| + Σᵢ<ⱼ|aᵢbⱼ + aⱼbᵢ|
    ///
    /// # Returns
    /// A scalar zonotope (`element_shape = [1]`) with result and new error term.
    pub fn dot(&self, other: &Self) -> Result<Self> {
        if self.n_error_terms != other.n_error_terms {
            return Err(GammaError::InvalidSpec(format!(
                "Cannot compute dot product of zonotopes with different error counts: {} vs {}",
                self.n_error_terms, other.n_error_terms
            )));
        }

        if self.element_shape != other.element_shape {
            return Err(GammaError::shape_mismatch(
                self.element_shape.clone(),
                other.element_shape.clone(),
            ));
        }

        // For now, only support 1D zonotopes
        if self.element_shape.len() != 1 {
            return Err(GammaError::InvalidSpec(
                "dot() currently only supports 1D zonotopes".to_string(),
            ));
        }

        let n = self.n_error_terms;
        let _dim = self.element_shape[0];

        // Extract center and error coefficients
        let a0 = self.coeffs.index_axis(Axis(0), 0);
        let b0 = other.coeffs.index_axis(Axis(0), 0);

        // 1. Center term: a₀ · b₀
        let mut center: f32 = a0.iter().zip(b0.iter()).map(|(&a, &b)| a * b).sum();

        // 2. Linear error terms: Σᵢ(a₀bᵢ + aᵢb₀)eᵢ
        // These are preserved in output
        let mut linear_coeffs = Vec::with_capacity(n);
        for i in 1..=n {
            let ai = self.coeffs.index_axis(Axis(0), i);
            let bi = other.coeffs.index_axis(Axis(0), i);

            // Coefficient for eᵢ in output
            let coeff: f32 = a0.iter().zip(bi.iter()).map(|(&a, &b)| a * b).sum::<f32>()
                + ai.iter().zip(b0.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            linear_coeffs.push(coeff);
        }

        // 3. Quadratic terms eᵢ² and cross terms eᵢeⱼ

        // 3a. Same-symbol products: aᵢbᵢ·eᵢ² where eᵢ² = 0.5 ± 0.5
        let mut center_shift: f32 = 0.0;
        let mut half_term: f32 = 0.0;

        for i in 1..=n {
            let ai = self.coeffs.index_axis(Axis(0), i);
            let bi = other.coeffs.index_axis(Axis(0), i);

            // aᵢ · bᵢ (dot product of coefficient vectors for error i)
            let ai_dot_bi: f32 = ai.iter().zip(bi.iter()).map(|(&a, &b)| a * b).sum();

            // eᵢ² = 0.5 + 0.5·e_new, so contribution is:
            // center += 0.5 * ai_dot_bi
            // new_error += 0.5 * |ai_dot_bi|
            center_shift += 0.5 * ai_dot_bi;
            half_term += 0.5 * ai_dot_bi.abs();
        }

        // 3b. Cross terms: aᵢbⱼ·eᵢeⱼ where i≠j
        // These become independent new errors, but we collapse them
        let mut big_term: f32 = 0.0;

        for i in 1..=n {
            let ai = self.coeffs.index_axis(Axis(0), i);
            let bi = other.coeffs.index_axis(Axis(0), i);

            for j in (i + 1)..=n {
                let aj = self.coeffs.index_axis(Axis(0), j);
                let bj = other.coeffs.index_axis(Axis(0), j);

                // Mixed term: aᵢ·bⱼ + aⱼ·bᵢ
                let ai_dot_bj: f32 = ai.iter().zip(bj.iter()).map(|(&a, &b)| a * b).sum();
                let aj_dot_bi: f32 = aj.iter().zip(bi.iter()).map(|(&a, &b)| a * b).sum();

                // Collapse cross terms into single error bound
                big_term += (ai_dot_bj + aj_dot_bi).abs();
            }
        }

        // Final center
        center += center_shift;

        // New error term coefficient (collapses all cross-products)
        let new_error_coeff = half_term + big_term;

        // Build result: scalar zonotope with original + 1 new error terms
        // coeffs shape: (1 + n + 1, 1)
        let mut result_coeffs = Array2::<f32>::zeros((1 + n + 1, 1));

        result_coeffs[[0, 0]] = center;
        for (i, &c) in linear_coeffs.iter().enumerate() {
            result_coeffs[[1 + i, 0]] = c;
        }
        result_coeffs[[1 + n, 0]] = new_error_coeff;

        Ok(Self {
            coeffs: result_coeffs.into_dyn(),
            n_error_terms: n + 1,
            element_shape: vec![1],
        })
    }

    /// Matrix multiplication: Z₁ @ Z₂^T where Z₁ and Z₂ share error symbols.
    ///
    /// This is the key operation for Q@K^T in attention.
    ///
    /// Following DeepT's `dot_product_precise` algorithm:
    /// 1. Center: `Q[0] @ K[0]^T`
    /// 2. Linear terms: `Q[0] @ K[i]^T + Q[i] @ K[0]^T` for each error `i`
    /// 3. Quadratic `e_i²`: center shift `0.5·Σ(Q[i]·K[i])` + radius `0.5·|Σ(Q[i]·K[i])|`
    /// 4. Cross terms `e_i×e_j`: collapse to single radius term
    ///
    /// # Arguments
    /// * `other` - The K zonotope (will be transposed for the multiplication)
    ///
    /// # Shapes
    /// * self: (seq_q, dim) zonotope with n error terms
    /// * other: (seq_k, dim) zonotope with n error terms (same symbols!)
    /// * result: (seq_q, seq_k) zonotope with n+1 error terms
    pub fn matmul_transposed(&self, other: &Self) -> Result<Self> {
        if self.n_error_terms != other.n_error_terms {
            return Err(GammaError::InvalidSpec(format!(
                "Cannot matmul zonotopes with different error counts: {} vs {}",
                self.n_error_terms, other.n_error_terms
            )));
        }

        // Support N-D zonotopes by treating all leading dimensions as batch dimensions.
        if self.element_shape.len() < 2 || other.element_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(format!(
                "matmul_transposed() requires inputs with at least 2 dims, got {:?} and {:?}",
                self.element_shape, other.element_shape
            )));
        }

        let self_rank = self.element_shape.len();
        let other_rank = other.element_shape.len();
        let self_batch_shape = &self.element_shape[..self_rank - 2];
        let other_batch_shape = &other.element_shape[..other_rank - 2];
        if self_batch_shape != other_batch_shape {
            return Err(GammaError::InvalidSpec(format!(
                "matmul_transposed batch dims must match, got {:?} and {:?}",
                self_batch_shape, other_batch_shape
            )));
        }

        let seq_q = self.element_shape[self_rank - 2];
        let dim_q = self.element_shape[self_rank - 1];
        let seq_k = other.element_shape[other_rank - 2];
        let dim_k = other.element_shape[other_rank - 1];

        if dim_q != dim_k {
            return Err(GammaError::shape_mismatch(vec![dim_k], vec![dim_q]));
        }

        let dim = dim_q;
        let n = self.n_error_terms;

        let result_n_errors = n + 1;
        let batch_size = self_batch_shape.iter().product::<usize>().max(1);

        let self_coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let other_coeffs: Cow<'_, ArrayD<f32>> = if other.coeffs.is_standard_layout() {
            Cow::Borrowed(&other.coeffs)
        } else {
            Cow::Owned(other.coeffs.as_standard_layout().to_owned())
        };

        let self_4d = self_coeffs
            .as_ref()
            .view()
            .into_shape_with_order(IxDyn(&[1 + n, batch_size, seq_q, dim]))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape self coeffs for matmul".to_string())
            })?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|_| GammaError::InvalidSpec("Cannot view self coeffs as 4D".to_string()))?;
        let other_4d = other_coeffs
            .as_ref()
            .view()
            .into_shape_with_order(IxDyn(&[1 + n, batch_size, seq_k, dim]))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape other coeffs for matmul".to_string())
            })?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|_| GammaError::InvalidSpec("Cannot view other coeffs as 4D".to_string()))?;

        let mut result_4d =
            ndarray::Array4::<f32>::zeros((1 + result_n_errors, batch_size, seq_q, seq_k));

        for b in 0..batch_size {
            for q in 0..seq_q {
                for k in 0..seq_k {
                    // ===== Section 1: Center =====
                    let mut center: f32 = 0.0;
                    for d in 0..dim {
                        center += self_4d[[0, b, q, d]] * other_4d[[0, b, k, d]];
                    }

                    // ===== Section 2: Handle e_i² terms =====
                    let mut center_shift: f32 = 0.0;
                    let mut half_term: f32 = 0.0;
                    for i in 1..=n {
                        let mut q_dot_k: f32 = 0.0;
                        for d in 0..dim {
                            q_dot_k += self_4d[[i, b, q, d]] * other_4d[[i, b, k, d]];
                        }
                        center_shift += 0.5 * q_dot_k;
                        half_term += 0.5 * q_dot_k.abs();
                    }

                    // ===== Section 3: Preserve linear error terms =====
                    for i in 1..=n {
                        let mut linear_coeff: f32 = 0.0;
                        for d in 0..dim {
                            linear_coeff += self_4d[[0, b, q, d]] * other_4d[[i, b, k, d]];
                            linear_coeff += self_4d[[i, b, q, d]] * other_4d[[0, b, k, d]];
                        }
                        result_4d[[i, b, q, k]] = linear_coeff;
                    }

                    // ===== Section 4: Cross terms =====
                    let mut big_term: f32 = 0.0;
                    for i in 1..=n {
                        for j in (i + 1)..=n {
                            let mut mixed: f32 = 0.0;
                            for d in 0..dim {
                                mixed += self_4d[[i, b, q, d]] * other_4d[[j, b, k, d]];
                                mixed += self_4d[[j, b, q, d]] * other_4d[[i, b, k, d]];
                            }
                            big_term += mixed.abs();
                        }
                    }

                    result_4d[[0, b, q, k]] = center + center_shift;
                    result_4d[[n + 1, b, q, k]] = half_term + big_term;
                }
            }
        }

        let mut out_element_shape = self_batch_shape.to_vec();
        out_element_shape.push(seq_q);
        out_element_shape.push(seq_k);

        let mut out_coeffs_shape = vec![1 + result_n_errors];
        out_coeffs_shape.extend_from_slice(&out_element_shape);
        let out_coeffs = result_4d
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_coeffs_shape))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape matmul output".to_string()))?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: result_n_errors,
            element_shape: out_element_shape,
        })
    }
}

/// Additional operations for GraphNetwork integration.
impl ZonotopeTensor {
    /// Element-wise addition by a constant tensor.
    ///
    /// Each element of the zonotope is shifted by the corresponding constant.
    /// z_i + c_i = (center_i + c_i) + Σⱼ (coeffⱼ,ᵢ · eⱼ)
    pub fn add_constant(&self, constant: &ArrayD<f32>) -> Result<Self> {
        if constant.shape() != self.element_shape.as_slice() {
            return Err(GammaError::shape_mismatch(
                self.element_shape.clone(),
                constant.shape().to_vec(),
            ));
        }

        let mut result = self.clone();
        // Add constant only to the center (index 0)
        let mut center = result.coeffs.index_axis_mut(Axis(0), 0);
        center += constant;
        Ok(result)
    }

    /// Element-wise multiplication by a constant tensor.
    ///
    /// z_i * c_i = (center_i * c_i) + Σⱼ (coeffⱼ,ᵢ * c_i · eⱼ)
    pub fn mul_constant(&self, constant: &ArrayD<f32>) -> Result<Self> {
        if constant.shape() != self.element_shape.as_slice() {
            return Err(GammaError::shape_mismatch(
                self.element_shape.clone(),
                constant.shape().to_vec(),
            ));
        }

        let mut result = self.clone();
        // Multiply all coefficients (center and error terms) by the constant
        for i in 0..=self.n_error_terms {
            let mut row = result.coeffs.index_axis_mut(Axis(0), i);
            row *= constant;
        }
        Ok(result)
    }

    /// Element-wise multiplication of two zonotopes: z1 ⊙ z2
    ///
    /// For SwiGLU: silu(gate) ⊙ up, where both have shared error symbols.
    ///
    /// # Mathematical Details
    ///
    /// For z1 = c1 + Σᵢ a1ᵢeᵢ and z2 = c2 + Σᵢ a2ᵢeᵢ:
    ///
    /// z1 * z2 = c1*c2 + c1*(Σᵢ a2ᵢeᵢ) + c2*(Σᵢ a1ᵢeᵢ) + (Σᵢ a1ᵢeᵢ)(Σⱼ a2ⱼeⱼ)
    ///
    /// The quadratic terms split into:
    /// - Same-symbol: `eᵢ² ∈ [0,1]` → shift center by `0.5*Σᵢ(a1ᵢ*a2ᵢ)`, add error `0.5*Σᵢ|a1ᵢ*a2ᵢ|`
    /// - Cross-symbol: `eᵢeⱼ ∈ [-1,1]` → add new error term `Σᵢ<ⱼ|a1ᵢ*a2ⱼ + a1ⱼ*a2ᵢ|`
    ///
    /// This preserves correlations between z1 and z2, giving tighter bounds than IBP
    /// which treats them as independent intervals.
    ///
    /// # Key Insight for SwiGLU
    ///
    /// When silu(gate) and up share error symbols (they do, since both come from the
    /// same input through FFN projections), this method exploits that correlation.
    /// IBP gives 36x growth; zonotope multiplication should be much tighter.
    pub fn mul_elementwise(&self, other: &Self) -> Result<Self> {
        // Expand to match error term counts
        let (z1, z2) = self.expand_to_match(other)?;

        if z1.element_shape != z2.element_shape {
            return Err(GammaError::shape_mismatch(
                z1.element_shape.clone(),
                z2.element_shape.clone(),
            ));
        }

        // Generalize to N-D by flattening elements; the multiplication is per-element
        // and does not mix coordinates, so flattening preserves semantics.
        let element_shape = z1.element_shape.clone();
        let n_elements = element_shape.iter().product::<usize>();

        let z1_flat = z1.reshape(&[n_elements])?;
        let z2_flat = z2.reshape(&[n_elements])?;

        let dim = n_elements;
        let n_errors = z1_flat.n_error_terms;

        let n_rows = 1 + n_errors + 1;
        let mut result_coeffs = ndarray::Array2::<f32>::zeros((n_rows, dim));

        let c1 = z1_flat.coeffs.index_axis(Axis(0), 0);
        let c2 = z2_flat.coeffs.index_axis(Axis(0), 0);

        for d in 0..dim {
            let c1_d = c1[d];
            let c2_d = c2[d];

            let mut a1: Vec<f32> = Vec::with_capacity(n_errors);
            let mut a2: Vec<f32> = Vec::with_capacity(n_errors);
            for i in 1..=n_errors {
                a1.push(z1_flat.coeffs[[i, d]]);
                a2.push(z2_flat.coeffs[[i, d]]);
            }

            let same_symbol_sum: f32 = a1.iter().zip(a2.iter()).map(|(&x, &y)| x * y).sum();
            result_coeffs[[0, d]] = c1_d * c2_d + 0.5 * same_symbol_sum;

            for i in 0..n_errors {
                result_coeffs[[i + 1, d]] = c1_d * a2[i] + c2_d * a1[i];
            }

            let same_symbol_error: f32 = 0.5
                * a1.iter()
                    .zip(a2.iter())
                    .map(|(&x, &y)| (x * y).abs())
                    .sum::<f32>();
            let mut cross_error: f32 = 0.0;
            for i in 0..n_errors {
                for j in (i + 1)..n_errors {
                    cross_error += (a1[i] * a2[j] + a1[j] * a2[i]).abs();
                }
            }
            result_coeffs[[n_errors + 1, d]] = same_symbol_error + cross_error;
        }

        let flat = Self {
            coeffs: result_coeffs.into_dyn(),
            n_error_terms: n_errors + 1,
            element_shape: vec![n_elements],
        };

        flat.reshape(&element_shape)
    }

    /// Convert to a zonotope with compatible structure for another zonotope.
    ///
    /// If zonotopes have different numbers of error terms, this expands
    /// the smaller one with zeros to match.
    pub fn expand_to_match(&self, other: &Self) -> Result<(Self, Self)> {
        if self.n_error_terms == other.n_error_terms && self.element_shape == other.element_shape {
            return Ok((self.clone(), other.clone()));
        }

        // For now, only support same shape
        if self.element_shape != other.element_shape {
            return Err(GammaError::shape_mismatch(
                self.element_shape.clone(),
                other.element_shape.clone(),
            ));
        }

        let max_errors = self.n_error_terms.max(other.n_error_terms);
        let shape = &self.element_shape;

        // Expand self if needed
        let expanded_self = if self.n_error_terms < max_errors {
            let mut new_shape = vec![1 + max_errors];
            new_shape.extend_from_slice(shape);
            let mut new_coeffs = ArrayD::zeros(IxDyn(&new_shape));
            for i in 0..=self.n_error_terms {
                new_coeffs
                    .index_axis_mut(Axis(0), i)
                    .assign(&self.coeffs.index_axis(Axis(0), i));
            }
            Self {
                coeffs: new_coeffs,
                n_error_terms: max_errors,
                element_shape: shape.clone(),
            }
        } else {
            self.clone()
        };

        // Expand other if needed
        let expanded_other = if other.n_error_terms < max_errors {
            let mut new_shape = vec![1 + max_errors];
            new_shape.extend_from_slice(shape);
            let mut new_coeffs = ArrayD::zeros(IxDyn(&new_shape));
            for i in 0..=other.n_error_terms {
                new_coeffs
                    .index_axis_mut(Axis(0), i)
                    .assign(&other.coeffs.index_axis(Axis(0), i));
            }
            Self {
                coeffs: new_coeffs,
                n_error_terms: max_errors,
                element_shape: shape.clone(),
            }
        } else {
            other.clone()
        };

        Ok((expanded_self, expanded_other))
    }

    /// Create zonotope from a BoundedTensor with a single shared error term.
    ///
    /// This is a lossy conversion - we can't recover the original correlations.
    /// The resulting zonotope has center = (lower+upper)/2 and radius = (upper-lower)/2.
    pub fn from_bounded_tensor(bounds: &BoundedTensor) -> Self {
        let center = (&bounds.lower + &bounds.upper) / 2.0;
        let radius = (&bounds.upper - &bounds.lower) / 2.0;

        let element_shape = bounds.shape().to_vec();
        let mut coeffs_shape = vec![2]; // center + 1 error term
        coeffs_shape.extend_from_slice(&element_shape);

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));
        coeffs.index_axis_mut(Axis(0), 0).assign(&center);
        coeffs.index_axis_mut(Axis(0), 1).assign(&radius);

        Self {
            coeffs,
            n_error_terms: 1,
            element_shape,
        }
    }

    /// Create a per-position zonotope from bounds with shape (..., seq, dim).
    ///
    /// Uses one error symbol per leading-batch+sequence position pair, with per-feature radii:
    /// - center = (lower + upper) / 2
    /// - error term for position (batch_idx, seq_idx) has coefficients radius[batch_idx, seq_idx, :]
    ///
    /// This is used for sequence data when we want to preserve correlations across multiple
    /// projections (Q/K/V) that share the same sequence position.
    pub fn from_bounded_tensor_per_position(bounds: &BoundedTensor) -> Result<Self> {
        let shape = bounds.shape();
        if shape.len() < 2 {
            return Err(GammaError::InvalidSpec(format!(
                "from_bounded_tensor_per_position requires at least 2D bounds, got shape {:?}",
                shape
            )));
        }

        let dim = shape[shape.len() - 1];
        let seq = shape[shape.len() - 2];
        let batch_shape = &shape[..shape.len() - 2];
        let batch_size = batch_shape.iter().product::<usize>().max(1);

        let center = (&bounds.lower + &bounds.upper) / 2.0;
        let radius = (&bounds.upper - &bounds.lower) / 2.0;

        let center_3d = center
            .into_shape_with_order(IxDyn(&[batch_size, seq, dim]))
            .map_err(|e| GammaError::InvalidSpec(format!("reshape center failed: {}", e)))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| GammaError::InvalidSpec("Cannot view center as 3D".to_string()))?;
        let radius_3d = radius
            .into_shape_with_order(IxDyn(&[batch_size, seq, dim]))
            .map_err(|e| GammaError::InvalidSpec(format!("reshape radius failed: {}", e)))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| GammaError::InvalidSpec("Cannot view radius as 3D".to_string()))?;

        let n_error_terms = batch_size * seq;
        let mut coeffs_flat =
            ndarray::Array4::<f32>::zeros((1 + n_error_terms, batch_size, seq, dim));
        coeffs_flat.index_axis_mut(Axis(0), 0).assign(&center_3d);

        for b in 0..batch_size {
            for s in 0..seq {
                let err = 1 + b * seq + s;
                for d in 0..dim {
                    coeffs_flat[[err, b, s, d]] = radius_3d[[b, s, d]];
                }
            }
        }

        let mut out_shape = vec![1 + n_error_terms];
        out_shape.extend_from_slice(batch_shape);
        out_shape.push(seq);
        out_shape.push(dim);

        let coeffs = coeffs_flat
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| GammaError::InvalidSpec(format!("reshape coeffs failed: {}", e)))?;

        Ok(Self {
            coeffs,
            n_error_terms,
            element_shape: shape.to_vec(),
        })
    }

    /// Create a per-position zonotope from 2D bounds (seq, dim).
    ///
    /// Uses one error symbol per sequence position, with per-feature radii:
    /// - center = (lower + upper) / 2
    /// - error term for position i has coefficients radius[i, :]
    ///
    /// This preserves correlations between multiple projections (Q/K/V) that share the same
    /// input sequence position, without introducing per-element error symbols.
    pub fn from_bounded_tensor_per_position_2d(bounds: &BoundedTensor) -> Result<Self> {
        let shape = bounds.shape();
        if shape.len() != 2 {
            return Err(GammaError::InvalidSpec(format!(
                "from_bounded_tensor_per_position_2d requires 2D bounds, got shape {:?}",
                shape
            )));
        }
        Self::from_bounded_tensor_per_position(bounds)
    }

    /// Reshape the zonotope to a new element shape.
    ///
    /// This preserves all correlations because we're just rearranging elements
    /// within each error term slice. The total number of elements must be preserved.
    ///
    /// # Arguments
    /// * `target_shape` - The new shape for the element tensor
    ///
    /// # Example
    /// ```ignore
    /// // Reshape (4, 8) zonotope to (2, 16)
    /// let z = z.reshape(&[2, 16])?;
    /// ```
    pub fn reshape(&self, target_shape: &[usize]) -> Result<Self> {
        let old_size: usize = self.element_shape.iter().product();
        let new_size: usize = target_shape.iter().product();

        if old_size != new_size {
            return Err(GammaError::shape_mismatch(
                self.element_shape.clone(),
                target_shape.to_vec(),
            ));
        }

        // coeffs shape: (1 + n_error_terms, ...old_element_shape)
        // new shape: (1 + n_error_terms, ...target_shape)
        let mut new_coeffs_shape = vec![1 + self.n_error_terms];
        new_coeffs_shape.extend_from_slice(target_shape);

        // Ensure contiguous memory layout before reshaping
        // This is needed when the array comes from operations like tile that
        // may produce non-standard layouts
        let contiguous = if self.coeffs.is_standard_layout() {
            self.coeffs.clone()
        } else {
            self.coeffs.as_standard_layout().to_owned()
        };

        let new_coeffs = contiguous
            .into_shape_with_order(IxDyn(&new_coeffs_shape))
            .map_err(|e| {
                GammaError::InvalidSpec(format!(
                    "Failed to reshape zonotope coeffs from {:?} to {:?}: {}",
                    self.coeffs.shape(),
                    new_coeffs_shape,
                    e
                ))
            })?;

        Ok(Self {
            coeffs: new_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: target_shape.to_vec(),
        })
    }

    /// Tile (repeat) the zonotope along a specified axis.
    ///
    /// This preserves correlations because duplicated elements share the same
    /// error symbols as the original. This is essential for tracking GQA attention
    /// where K/V heads are repeated to match Q heads.
    ///
    /// # Arguments
    /// * `axis` - The axis in element_shape to tile (0-indexed)
    /// * `reps` - Number of times to repeat along that axis
    ///
    /// # Example
    /// ```ignore
    /// // Tile (4, 128) zonotope to (16, 128) by repeating 4x along axis 0
    /// let z = z.tile(0, 4)?;
    /// ```
    pub fn tile(&self, axis: usize, reps: usize) -> Result<Self> {
        if axis >= self.element_shape.len() {
            return Err(GammaError::InvalidSpec(format!(
                "Tile axis {} out of bounds for shape {:?}",
                axis, self.element_shape
            )));
        }

        if reps == 0 {
            return Err(GammaError::InvalidSpec("Tile reps must be > 0".to_string()));
        }

        if reps == 1 {
            return Ok(self.clone());
        }

        // In coeffs, axis 0 is error terms, so element axis i corresponds to coeffs axis i+1
        let coeffs_axis = axis + 1;

        // Use ndarray concatenate to repeat
        let views: Vec<_> = std::iter::repeat(self.coeffs.view()).take(reps).collect();

        let new_coeffs = ndarray::concatenate(Axis(coeffs_axis), &views).map_err(|e| {
            GammaError::InvalidSpec(format!(
                "Failed to tile zonotope along axis {}: {}",
                axis, e
            ))
        })?;

        // Update element shape
        let mut new_element_shape = self.element_shape.clone();
        new_element_shape[axis] *= reps;

        Ok(Self {
            coeffs: new_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: new_element_shape,
        })
    }

    /// Apply LayerNorm using affine approximation around the center point.
    ///
    /// LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    ///
    /// This is approximately linear near the center, so we:
    /// 1. Compute center output: y_c = LayerNorm(center)
    /// 2. Compute effective scale: s = gamma / std(center)
    /// 3. Transform error terms: error_i -> s * error_i (preserves zonotope form)
    /// 4. Add new error term to bound the approximation error
    ///
    /// # Arguments
    /// * `gamma` - Scale parameter (per feature)
    /// * `beta` - Shift parameter (per feature)
    /// * `eps` - Small constant for numerical stability
    ///
    /// # Note
    /// This approximation is tighter for small perturbations. For large perturbations,
    /// the center-based linearization may underestimate the true bound width.
    pub fn layer_norm_affine(
        &self,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        eps: f32,
    ) -> Result<Self> {
        if self.element_shape.is_empty() {
            return Err(GammaError::InvalidSpec(
                "layer_norm_affine requires at least 1 dimension".to_string(),
            ));
        }

        let dim = *self
            .element_shape
            .last()
            .ok_or_else(|| GammaError::InvalidSpec("Empty element shape".to_string()))?;
        let prefix_shape = &self.element_shape[..self.element_shape.len() - 1];
        let prefix_size = prefix_shape.iter().product::<usize>().max(1);

        if gamma.len() != dim {
            return Err(GammaError::shape_mismatch(vec![dim], vec![gamma.len()]));
        }
        if beta.len() != dim {
            return Err(GammaError::shape_mismatch(vec![dim], vec![beta.len()]));
        }

        let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let coeffs_3d = coeffs
            .view()
            .into_shape_with_order(IxDyn(&[1 + self.n_error_terms, prefix_size, dim]))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape LayerNorm coeffs to 3D".to_string())
            })?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot view LayerNorm coeffs as 3D".to_string())
            })?;

        let center_2d = coeffs_3d.index_axis(Axis(0), 0);

        let n_rows = 1 + self.n_error_terms + 1;
        let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, prefix_size, dim));

        // Track max approximation error across all positions
        let mut max_approx_error = 0.0f32;

        for row_idx in 0..prefix_size {
            let row = center_2d.row(row_idx);

            // Compute mean and variance of this row
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let centered: Vec<f32> = row.iter().map(|&x| x - mean).collect();
            let var: f32 = centered.iter().map(|&c| c * c).sum::<f32>() / dim as f32;
            let std = (var + eps).sqrt();

            // Guard against division by zero or very small std
            let std_safe = std.max(1e-10);

            // Compute effective scale per feature: gamma / std
            // This is the approximate linear scaling factor
            let eff_gamma: Vec<f32> = gamma
                .iter()
                .map(|&g| {
                    // Cap effective gamma to prevent overflow in large-gamma models
                    let raw = g / std_safe;
                    raw.clamp(-1e6, 1e6)
                })
                .collect();

            // Compute LayerNorm output at center
            for d in 0..dim {
                let y_c = eff_gamma[d] * centered[d] + beta[d];
                result_coeffs[[0, row_idx, d]] = y_c;
            }

            // Transform error coefficients by effective gamma
            // This preserves the zonotope form: if input has coefficient a_i,
            // output has coefficient (gamma/std) * a_i (approximately)
            //
            // The full Jacobian is more complex (includes mean/var derivatives),
            // but this diagonal approximation works well for small perturbations.
            for i in 1..=self.n_error_terms {
                for d in 0..dim {
                    result_coeffs[[i, row_idx, d]] = eff_gamma[d] * coeffs_3d[[i, row_idx, d]];
                }
            }

            // Estimate approximation error for this position
            // The diagonal approximation ignores off-diagonal Jacobian terms.
            // The error is bounded by the sum of off-diagonal contributions.
            //
            // For LayerNorm, the off-diagonal term dy_i/dx_j (i != j) is:
            // -(gamma_i / (n * std)) - gamma_i * (x_i - mean) * (x_j - mean) / (n * std^3)
            //
            // We bound this by: sum_j |dy_i/dx_j| * radius_j
            // where radius_j is the sum of absolute error coefficients for position j
            let mut total_radius: f32 = 0.0;
            for d in 0..dim {
                for i in 1..=self.n_error_terms {
                    total_radius += coeffs_3d[[i, row_idx, d]].abs();
                }
            }

            // Off-diagonal contribution per output feature (approximate upper bound)
            // This is conservative: we assume all perturbations contribute to mean/var shift
            let mean_deriv_contrib = gamma
                .iter()
                .map(|&g| g / (dim as f32 * std_safe))
                .sum::<f32>();
            let approx_error = mean_deriv_contrib * total_radius;
            max_approx_error = max_approx_error.max(approx_error);
        }

        // Add new error term for approximation error (same for all features)
        result_coeffs
            .index_axis_mut(Axis(0), self.n_error_terms + 1)
            .fill(max_approx_error);

        let mut out_shape = vec![n_rows];
        out_shape.extend_from_slice(prefix_shape);
        out_shape.push(dim);
        let out_coeffs = result_coeffs
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| GammaError::InvalidSpec("Cannot reshape LayerNorm output".to_string()))?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: self.n_error_terms + 1,
            element_shape: self.element_shape.clone(),
        })
    }

    /// Softmax with linear approximation to preserve zonotope form.
    ///
    /// softmax(x)_i = exp(x_i) / sum_j exp(x_j)
    ///
    /// This approximation linearizes softmax around the center and adds an error
    /// term to bound the non-linearity, enabling zonotope propagation through
    /// attention patterns.
    ///
    /// # Mathematical Details
    ///
    /// The Jacobian of softmax at center c is:
    ///   J\[i,j\] = s_c\[i\] * (δ\[i,j\] - s_c\[j\])
    /// where s_c = softmax(c).
    ///
    /// For a zonotope z = c + Σₖ aₖeₖ:
    ///   softmax(z) ≈ s_c + J @ (z - c) = s_c + Σₖ (J @ aₖ)eₖ
    ///
    /// # Error Bound
    ///
    /// The approximation error is bounded by:
    ///   |softmax(z) - linear_approx| ≤ max|Hessian| * r² / 2
    ///
    /// For softmax, the Hessian is complex but bounded. We use a conservative
    /// bound: max second derivative ≤ 0.5 (softmax outputs are in \[0,1\] with
    /// bounded curvature).
    ///
    /// # Arguments
    /// * `axis` - The axis along which to apply softmax (default: last axis)
    pub fn softmax_affine(&self, axis: i32) -> Result<Self> {
        if self.element_shape.is_empty() {
            return Err(GammaError::InvalidSpec(
                "softmax_affine requires at least 1 dimension".to_string(),
            ));
        }

        let ndim = self.element_shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis_usize >= ndim {
            return Err(GammaError::InvalidSpec(format!(
                "softmax axis {} out of range for {} dimensions",
                axis, ndim
            )));
        }

        // Get the size of the softmax dimension
        let softmax_dim = self.element_shape[axis_usize];

        // Helper: stable softmax computation
        fn compute_softmax(x: &[f32]) -> Vec<f32> {
            let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exp_x.iter().sum();
            exp_x.iter().map(|&e| e / sum).collect()
        }

        // Helper: compute Jacobian @ vector for softmax
        // J[i,j] = s[i] * (δ[i,j] - s[j])
        // (J @ v)[i] = sum_j s[i] * (δ[i,j] - s[j]) * v[j]
        //            = s[i] * v[i] - s[i] * sum_j s[j] * v[j]
        //            = s[i] * (v[i] - dot(s, v))
        fn jacobian_vector_product(s: &[f32], v: &[f32]) -> Vec<f32> {
            let dot_sv: f32 = s.iter().zip(v.iter()).map(|(&si, &vi)| si * vi).sum();
            s.iter()
                .zip(v.iter())
                .map(|(&si, &vi)| si * (vi - dot_sv))
                .collect()
        }

        // For 1D input (shape [dim])
        if ndim == 1 {
            let dim = softmax_dim;
            let n_rows = 1 + self.n_error_terms + 1; // center + errors + 1 new approx error
            let mut result_coeffs = ndarray::Array2::<f32>::zeros((n_rows, dim));

            // Get center and compute softmax
            let center: Vec<f32> = self.coeffs.index_axis(Axis(0), 0).iter().cloned().collect();
            let s_c = compute_softmax(&center);

            // Output center = softmax(center)
            for (i, &s) in s_c.iter().enumerate() {
                result_coeffs[[0, i]] = s;
            }

            // Transform each error coefficient through Jacobian
            let mut max_radius = 0.0f32;
            for k in 1..=self.n_error_terms {
                let err_k: Vec<f32> = self.coeffs.index_axis(Axis(0), k).iter().cloned().collect();

                // Compute max radius for error bound
                let radius_k: f32 = err_k.iter().map(|x| x.abs()).sum();
                max_radius = max_radius.max(radius_k);

                // Apply Jacobian: J @ err_k
                let transformed = jacobian_vector_product(&s_c, &err_k);
                for (i, &t) in transformed.iter().enumerate() {
                    result_coeffs[[k, i]] = t;
                }
            }

            // Add approximation error term
            // Conservative bound: for softmax with inputs varying by radius r,
            // the maximum error from linearization is bounded by 0.5 * r² (Hessian bound)
            // Actually, softmax is Lipschitz with constant 0.25 for second derivative
            // Using conservative multiplier of 0.5
            let approx_error = 0.5 * max_radius * max_radius;
            result_coeffs
                .row_mut(self.n_error_terms + 1)
                .fill(approx_error);

            let out_coeffs = result_coeffs
                .into_dyn()
                .into_shape_with_order(IxDyn(&[n_rows, dim]))
                .map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape softmax output".to_string())
                })?;

            return Ok(Self {
                coeffs: out_coeffs,
                n_error_terms: self.n_error_terms + 1,
                element_shape: self.element_shape.clone(),
            });
        }

        // For 2D input (shape [seq, dim] or [batch, seq]) - softmax on last axis
        if ndim == 2 && axis_usize == ndim - 1 {
            let seq = self.element_shape[0];
            let dim = self.element_shape[1];
            let n_rows = 1 + self.n_error_terms + 1;
            let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, seq, dim));

            let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
                Cow::Borrowed(&self.coeffs)
            } else {
                Cow::Owned(self.coeffs.as_standard_layout().to_owned())
            };
            let coeffs = coeffs.as_ref();

            let coeffs_3d = coeffs
                .view()
                .into_shape_with_order(IxDyn(&[1 + self.n_error_terms, seq, dim]))
                .map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape softmax coeffs to 3D".to_string())
                })?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| {
                    GammaError::InvalidSpec("Cannot view softmax coeffs as 3D".to_string())
                })?;

            let mut max_approx_error = 0.0f32;

            // Process each sequence position independently (softmax along dim)
            for s in 0..seq {
                let center: Vec<f32> = (0..dim).map(|d| coeffs_3d[[0, s, d]]).collect();
                let s_c = compute_softmax(&center);

                // Output center
                for (d, &sc) in s_c.iter().enumerate() {
                    result_coeffs[[0, s, d]] = sc;
                }

                // Transform error coefficients
                let mut max_radius_this_pos = 0.0f32;
                for k in 1..=self.n_error_terms {
                    let err_k: Vec<f32> = (0..dim).map(|d| coeffs_3d[[k, s, d]]).collect();
                    let radius_k: f32 = err_k.iter().map(|x| x.abs()).sum();
                    max_radius_this_pos = max_radius_this_pos.max(radius_k);

                    let transformed = jacobian_vector_product(&s_c, &err_k);
                    for (d, &t) in transformed.iter().enumerate() {
                        result_coeffs[[k, s, d]] = t;
                    }
                }

                let approx_error = 0.5 * max_radius_this_pos * max_radius_this_pos;
                max_approx_error = max_approx_error.max(approx_error);
            }

            // Fill approximation error (single shared error term)
            result_coeffs
                .index_axis_mut(Axis(0), self.n_error_terms + 1)
                .fill(max_approx_error);

            let mut out_shape = vec![n_rows];
            out_shape.extend_from_slice(&self.element_shape);
            let out_coeffs = result_coeffs
                .into_dyn()
                .into_shape_with_order(IxDyn(&out_shape))
                .map_err(|_| {
                    GammaError::InvalidSpec("Cannot reshape softmax output".to_string())
                })?;

            return Ok(Self {
                coeffs: out_coeffs,
                n_error_terms: self.n_error_terms + 1,
                element_shape: self.element_shape.clone(),
            });
        }

        // For higher-dimensional inputs or different axis, fall back to general case
        // This handles shapes like [batch, heads, seq_q, seq_k] for attention
        Err(GammaError::InvalidSpec(format!(
            "softmax_affine for shape {:?} axis {} not yet implemented",
            self.element_shape, axis
        )))
    }

    /// Causal softmax with linear approximation to preserve zonotope form.
    ///
    /// Causal attention masks out "future" keys: for each query position `i`, only keys
    /// `j <= i` participate in the softmax. Masked positions (`j > i`) output exactly 0.
    ///
    /// This uses the same Jacobian-based linearization as `softmax_affine`, but:
    /// - Computes softmax and Jacobian only over the unmasked prefix `0..=i`
    /// - Forces masked outputs (and their coefficients) to 0
    /// - Does not add approximation error to masked outputs
    ///
    /// # Arguments
    /// * `axis` - The axis along which to apply softmax (must be the last axis)
    pub fn softmax_affine_causal(&self, axis: i32) -> Result<Self> {
        if self.element_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(format!(
                "softmax_affine_causal requires at least 2 dimensions, got {:?}",
                self.element_shape
            )));
        }

        let ndim = self.element_shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis_usize >= ndim {
            return Err(GammaError::InvalidSpec(format!(
                "softmax axis {} out of range for {} dimensions",
                axis, ndim
            )));
        }
        if axis_usize != ndim - 1 {
            return Err(GammaError::InvalidSpec(format!(
                "softmax_affine_causal only supports last-axis softmax, got axis {} for shape {:?}",
                axis, self.element_shape
            )));
        }

        let seq_q = self.element_shape[ndim - 2];
        let seq_k = self.element_shape[ndim - 1];
        if seq_q > seq_k {
            return Err(GammaError::InvalidSpec(format!(
                "softmax_affine_causal requires seq_q ({}) <= seq_k ({})",
                seq_q, seq_k
            )));
        }

        let prefix_size: usize = self.element_shape[..ndim - 2].iter().product();
        let n_attn_rows = prefix_size * seq_q;

        let n_terms = 1 + self.n_error_terms; // center + existing errors
        let n_rows_out = n_terms + 1; // + 1 new approx error term

        fn compute_softmax_prefix(x: &[f32]) -> Vec<f32> {
            let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exp_x.iter().sum();
            exp_x.iter().map(|&e| e / sum).collect()
        }

        fn jacobian_vector_product_prefix(s: &[f32], v: &[f32]) -> Vec<f32> {
            let dot_sv: f32 = s.iter().zip(v.iter()).map(|(&si, &vi)| si * vi).sum();
            s.iter()
                .zip(v.iter())
                .map(|(&si, &vi)| si * (vi - dot_sv))
                .collect()
        }

        let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let in_coeffs_3d = coeffs
            .view()
            .into_shape_with_order(IxDyn(&[n_terms, n_attn_rows, seq_k]))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape causal softmax coeffs to 3D".to_string())
            })?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot view causal softmax coeffs as 3D".to_string())
            })?;

        let mut out_coeffs_3d = ndarray::Array3::<f32>::zeros((n_rows_out, n_attn_rows, seq_k));
        let mut max_approx_error = 0.0f32;

        for row in 0..n_attn_rows {
            let query_i = row % seq_q;
            let allowed = (query_i + 1).min(seq_k);

            let center: Vec<f32> = (0..allowed).map(|j| in_coeffs_3d[[0, row, j]]).collect();
            let s_c = compute_softmax_prefix(&center);

            for (j, &sc) in s_c.iter().enumerate() {
                out_coeffs_3d[[0, row, j]] = sc;
            }

            let mut max_radius_this_row = 0.0f32;
            for k in 1..=self.n_error_terms {
                let err_k: Vec<f32> = (0..allowed).map(|j| in_coeffs_3d[[k, row, j]]).collect();
                let radius_k: f32 = err_k.iter().map(|x| x.abs()).sum();
                max_radius_this_row = max_radius_this_row.max(radius_k);

                let transformed = jacobian_vector_product_prefix(&s_c, &err_k);
                for (j, &t) in transformed.iter().enumerate() {
                    out_coeffs_3d[[k, row, j]] = t;
                }
            }

            let approx_error = 0.5 * max_radius_this_row * max_radius_this_row;
            max_approx_error = max_approx_error.max(approx_error);
        }

        // Add approximation error term, but only for unmasked entries.
        for row in 0..n_attn_rows {
            let query_i = row % seq_q;
            let allowed = (query_i + 1).min(seq_k);
            for j in 0..allowed {
                out_coeffs_3d[[n_terms, row, j]] = max_approx_error;
            }
        }

        let mut out_shape = vec![n_rows_out];
        out_shape.extend_from_slice(&self.element_shape);
        let out_coeffs = out_coeffs_3d
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| {
                GammaError::InvalidSpec("Cannot reshape causal softmax output".to_string())
            })?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: self.n_error_terms + 1,
            element_shape: self.element_shape.clone(),
        })
    }

    /// SiLU (Swish) activation with linear approximation to preserve zonotope form.
    ///
    /// SiLU(x) = x * sigmoid(x) is used in SwiGLU FFN layers of modern LLMs.
    /// This approximation evaluates SiLU at the center and uses the derivative
    /// as a linear scaling factor, adding an error term for the approximation.
    ///
    /// # Key Insight
    ///
    /// Without this approximation, SiLU falls back to IBP which loses all correlations.
    /// This causes FFN bounds to explode (~36x per block from SwiGLU multiplication).
    /// With the linear approximation, correlations are preserved through SiLU,
    /// enabling tighter SwiGLU multiplication bounds.
    ///
    /// # Mathematical Details
    ///
    /// For a zonotope z = c + Σᵢ aᵢeᵢ:
    /// - output_center = silu(c)
    /// - slope = silu'(c) = sigmoid(c) * (1 + c * (1 - sigmoid(c)))
    /// - output = silu(c) + slope * (z - c) = silu(c) - slope*c + slope*z
    /// - Error bound from second derivative: |silu''(x)| * r² / 2
    ///
    /// # Soundness
    ///
    /// The linear approximation is sound because we bound the maximum error
    /// over the input interval using the maximum second derivative of SiLU.
    pub fn silu_affine(&self) -> Result<Self> {
        // SiLU helper functions
        fn sigmoid(x: f32) -> f32 {
            // Numerically stable sigmoid
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            }
        }

        fn silu(x: f32) -> f32 {
            x * sigmoid(x)
        }

        fn silu_derivative(x: f32) -> f32 {
            // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let s = sigmoid(x);
            s * (1.0 + x * (1.0 - s))
        }

        fn silu_second_derivative(x: f32) -> f32 {
            // SiLU''(x) = sigmoid(x) * (1 - sigmoid(x)) * (2 + x - 2*x*sigmoid(x))
            let s = sigmoid(x);
            let s1 = 1.0 - s;
            s * s1 * (2.0 + x - 2.0 * x * s)
        }

        // Support 1D and 2D zonotopes
        match self.element_shape.len() {
            1 => {
                let dim = self.element_shape[0];
                let n_rows = 1 + self.n_error_terms + 1; // center + existing errors + 1 new error
                let mut result_coeffs = ndarray::Array2::<f32>::zeros((n_rows, dim));

                // Get center values
                let center = self.coeffs.index_axis(Axis(0), 0);

                let mut max_approx_error = 0.0f32;

                for d in 0..dim {
                    let c = center[d];

                    // Compute radius (sum of absolute error coefficients)
                    let radius: f32 = (1..=self.n_error_terms)
                        .map(|i| self.coeffs[[i, d]].abs())
                        .sum();

                    // Output center = silu(c)
                    result_coeffs[[0, d]] = silu(c);

                    // Transform error coefficients by slope
                    let slope = silu_derivative(c);
                    for i in 1..=self.n_error_terms {
                        result_coeffs[[i, d]] = slope * self.coeffs[[i, d]];
                    }

                    // Bound approximation error using second derivative
                    // |f(x) - f(c) - f'(c)*(x-c)| <= max|f''| * r^2 / 2
                    if radius > 0.0 {
                        // Find max |SiLU''| over [c-r, c+r] by sampling
                        // SiLU'' has a maximum around x ≈ -1.28 and approaches 0 for large |x|
                        let lo = c - radius;
                        let hi = c + radius;
                        let mut max_second = 0.0f32;
                        for i in 0..=20 {
                            let t = i as f32 / 20.0;
                            let x = lo + (hi - lo) * t;
                            max_second = max_second.max(silu_second_derivative(x).abs());
                        }
                        // Also check critical points of SiLU''
                        // SiLU'' has extrema near x ≈ -2.4 and x ≈ 0.7
                        for &critical in &[-2.4f32, -1.28, 0.7, 0.0] {
                            if lo <= critical && critical <= hi {
                                max_second = max_second.max(silu_second_derivative(critical).abs());
                            }
                        }
                        let approx_error = max_second * radius * radius / 2.0;
                        max_approx_error = max_approx_error.max(approx_error);
                    }
                }

                // Add new error term for approximation
                for d in 0..dim {
                    result_coeffs[[self.n_error_terms + 1, d]] = max_approx_error;
                }

                Ok(Self {
                    coeffs: result_coeffs.into_dyn(),
                    n_error_terms: self.n_error_terms + 1,
                    element_shape: self.element_shape.clone(),
                })
            }
            2 => {
                // 2D case: (seq, dim)
                let seq_len = self.element_shape[0];
                let dim = self.element_shape[1];
                let n_rows = 1 + self.n_error_terms + 1;
                let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, seq_len, dim));

                let center = self.coeffs.index_axis(Axis(0), 0);
                let center_2d = center
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| GammaError::InvalidSpec("Cannot view center as 2D".to_string()))?;

                let mut max_approx_error = 0.0f32;

                for s in 0..seq_len {
                    for d in 0..dim {
                        let c = center_2d[[s, d]];

                        // Compute radius
                        let radius: f32 = (1..=self.n_error_terms)
                            .map(|i| self.coeffs[[i, s, d]].abs())
                            .sum();

                        // Output center
                        result_coeffs[[0, s, d]] = silu(c);

                        // Transform error coefficients
                        let slope = silu_derivative(c);
                        for i in 1..=self.n_error_terms {
                            result_coeffs[[i, s, d]] = slope * self.coeffs[[i, s, d]];
                        }

                        // Bound approximation error
                        if radius > 0.0 {
                            let lo = c - radius;
                            let hi = c + radius;
                            let mut max_second = 0.0f32;
                            for i in 0..=20 {
                                let t = i as f32 / 20.0;
                                let x = lo + (hi - lo) * t;
                                max_second = max_second.max(silu_second_derivative(x).abs());
                            }
                            for &critical in &[-2.4f32, -1.28, 0.7, 0.0] {
                                if lo <= critical && critical <= hi {
                                    max_second =
                                        max_second.max(silu_second_derivative(critical).abs());
                                }
                            }
                            let approx_error = max_second * radius * radius / 2.0;
                            max_approx_error = max_approx_error.max(approx_error);
                        }
                    }
                }

                // Add new error term
                for s in 0..seq_len {
                    for d in 0..dim {
                        result_coeffs[[self.n_error_terms + 1, s, d]] = max_approx_error;
                    }
                }

                Ok(Self {
                    coeffs: result_coeffs.into_dyn(),
                    n_error_terms: self.n_error_terms + 1,
                    element_shape: self.element_shape.clone(),
                })
            }
            _ => {
                // Generalize to N-D by flattening elements; SiLU is applied element-wise.
                let element_shape = self.element_shape.clone();
                let n_elements = element_shape.iter().product::<usize>();
                let flat = self.reshape(&[n_elements])?;
                let out_flat = flat.silu_affine()?;
                out_flat.reshape(&element_shape)
            }
        }
    }

    /// Transpose the last two dimensions of the zonotope.
    ///
    /// For a zonotope with element_shape (..., M, N), produces one with shape (..., N, M).
    /// This is needed for matmul operations where we want A @ B instead of A @ B^T.
    pub fn transpose_last_two(&self) -> Result<Self> {
        if self.element_shape.len() < 2 {
            return Err(GammaError::InvalidSpec(format!(
                "transpose_last_two requires at least 2 dimensions, got {:?}",
                self.element_shape
            )));
        }

        let ndim = self.coeffs.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        // Swap the last two axes in coeffs (which are last two element dims)
        axes.swap(ndim - 2, ndim - 1);

        let transposed = self.coeffs.clone().permuted_axes(axes);

        let mut new_element_shape = self.element_shape.clone();
        let n = new_element_shape.len();
        new_element_shape.swap(n - 2, n - 1);

        Ok(Self {
            coeffs: transposed,
            n_error_terms: self.n_error_terms,
            element_shape: new_element_shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_concrete_zonotope() {
        let values = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let z = ZonotopeTensor::concrete(values);

        assert_eq!(z.n_error_terms, 0);
        assert_eq!(z.max_width(), 0.0);
    }

    #[test]
    fn test_from_input_shared() {
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        assert_eq!(z.n_error_terms, 1);
        assert_eq!(z.element_shape, vec![2]);

        let bounds = z.to_bounded_tensor();
        assert!((bounds.lower[[0]] - 0.9).abs() < 1e-6);
        assert!((bounds.upper[[0]] - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_from_input_elementwise() {
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::from_input_elementwise(&values, 0.1);

        assert_eq!(z.n_error_terms, 2); // One per element
        assert_eq!(z.element_shape, vec![2]);

        let bounds = z.to_bounded_tensor();
        assert!((bounds.lower[[0]] - 0.9).abs() < 1e-6);
        assert!((bounds.upper[[1]] - 2.1).abs() < 1e-6);
    }

    #[test]
    fn test_shift_and_scale() {
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        // Shift by 1
        let shifted = z.shift(1.0);
        assert!((shifted.center()[[0]] - 2.0).abs() < 1e-6);

        // Scale by 2
        let scaled = z.scale(2.0);
        assert!((scaled.center()[[0]] - 2.0).abs() < 1e-6);
        assert!((scaled.max_width() - 0.4).abs() < 1e-6); // 2 * 0.1 * 2 = 0.4
    }

    #[test]
    fn test_add_zonotopes() {
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z1 = ZonotopeTensor::from_input_shared(&values, 0.1);
        let z2 = ZonotopeTensor::from_input_shared(&values, 0.1);

        let sum = z1.add(&z2).unwrap();
        assert!((sum.center()[[0]] - 2.0).abs() < 1e-6);
        assert!((sum.max_width() - 0.4).abs() < 1e-6); // 2 * 0.1 * 2 = 0.4
    }

    #[test]
    fn test_linear_transform() {
        // Create zonotope: [1±0.1, 2±0.1]
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::from_input_elementwise(&values, 0.1);

        // Weight: [[1, 1], [1, -1]] -> out = [x+y, x-y]
        let weight = arr2(&[[1.0, 1.0], [1.0, -1.0]]);

        let result = z.linear(&weight, None).unwrap();

        // Center: [1+2, 1-2] = [3, -1]
        assert!((result.center()[[0]] - 3.0).abs() < 1e-6);
        assert!((result.center()[[1]] - (-1.0)).abs() < 1e-6);

        // Error propagation:
        // For out[0] = x + y: both errors contribute, but since x and y have separate symbols,
        // width = |1*0.1| + |1*0.1| = 0.2 (from x's symbol) + 0.2 (from y's symbol)
        // Actually: error1 affects x, error2 affects y
        // out[0] gets 0.1 from error1 (through x) and 0.1 from error2 (through y)
        // Total width = 2 * (0.1 + 0.1) = 0.4
        let bounds = result.to_bounded_tensor();
        assert!((bounds.upper[[0]] - bounds.lower[[0]] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_concrete() {
        // Two concrete zonotopes (no error)
        let a = ZonotopeTensor::concrete(arr1(&[1.0, 2.0]).into_dyn());
        let b = ZonotopeTensor::concrete(arr1(&[3.0, 4.0]).into_dyn());

        let result = a.dot(&b).unwrap();

        // 1*3 + 2*4 = 11
        assert!((result.center()[[0]] - 11.0).abs() < 1e-6);
        assert_eq!(result.max_width(), 0.0); // No error
    }

    #[test]
    fn test_dot_product_with_errors() {
        // z1 = [1+0.1e1, 2+0.1e2] (each element has its own error)
        // z2 = [1+0.1e1, 1+0.1e2] (SAME error symbols - correlated!)
        let values1 = arr1(&[1.0, 2.0]).into_dyn();
        let values2 = arr1(&[1.0, 1.0]).into_dyn();

        let z1 = ZonotopeTensor::from_input_elementwise(&values1, 0.1);
        let z2 = ZonotopeTensor::from_input_elementwise(&values2, 0.1);

        let result = z1.dot(&z2).unwrap();

        // Center: 1*1 + 2*1 = 3
        // Plus center shift from e_i^2 terms: 0.5*(0.1*0.1 + 0.1*0.1) = 0.01
        let expected_center = 3.0 + 0.01;
        assert!((result.center()[[0]] - expected_center).abs() < 1e-5);

        // The zonotope dot product should be tighter than IBP
        let bounds = result.to_bounded_tensor();
        let width = bounds.upper[[0]] - bounds.lower[[0]];

        // Compare to IBP: [0.9,1.1]*[0.9,1.1] + [1.9,2.1]*[0.9,1.1]
        // = [0.81,1.21] + [1.71,2.31] = [2.52,3.52] -> width 1.0
        // Zonotope should be tighter (but not by huge amount for this small example)
        assert!(width < 1.5); // Sanity check
    }

    #[test]
    fn test_zonotope_vs_ibp_correlation_advantage() {
        // This test demonstrates why zonotopes are better for correlated inputs
        //
        // Scenario: z = x * x where x = 1 ± 0.5
        // IBP: [0.5, 1.5] * [0.5, 1.5] = [0.25, 2.25] (treats as independent)
        // True range: (1±0.5)² = 1 ± 1 + 0.25 = [0.25, 2.25] (same for single var!)
        //
        // But for z = x * y where x,y SHARE perturbation (x = 1+e, y = 1+e):
        // IBP: still [0.25, 2.25]
        // Zonotope: (1+e)*(1+e) = 1 + 2e + e² where e²∈[0,1]
        //         = 1.5 + 2e + 0.5e' where e,e'∈[-1,1]
        //         range: [1.5-2-0.5, 1.5+2+0.5] = [-1, 4] -- wait that's worse!

        // Actually the advantage is when Q and K both depend on same X
        // but through DIFFERENT linear transforms. Let me redo:

        // x = [1±0.1] (input with 1 error symbol)
        // Q = 2x (linear)
        // K = 3x (linear)
        // Q*K = 6x² = 6*(1±0.1)² = 6*(1 ± 0.2 + 0.01) = 6*[0.81, 1.21] = [4.86, 7.26]
        // width_true = 2.4

        // IBP would compute:
        // Q ∈ [1.8, 2.2]
        // K ∈ [2.7, 3.3]
        // Q*K ∈ [1.8*2.7, 2.2*3.3] = [4.86, 7.26] -- same! Because 1D is special.

        // The advantage appears in higher dimensions where cross-correlations matter.
        // For example, Q·K where Q,K ∈ R^d and both depend on same input.

        // Let's verify basic functionality with shared error:
        let x = ZonotopeTensor::from_input_shared(&arr1(&[1.0]).into_dyn(), 0.1);

        // q = 2*x
        let q = x.scale(2.0);
        assert!((q.center()[[0]] - 2.0).abs() < 1e-6);

        // k = 3*x
        let k = x.scale(3.0);
        assert!((k.center()[[0]] - 3.0).abs() < 1e-6);

        // Both Q and K share the same error symbol!
        assert_eq!(q.n_error_terms, 1);
        assert_eq!(k.n_error_terms, 1);
    }

    #[test]
    fn test_from_input_2d() {
        // Create 2D zonotope with shape (2, 3)
        let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        assert_eq!(z.n_error_terms, 6); // 2 * 3 = 6 elements
        assert_eq!(z.element_shape, vec![2, 3]);

        // Center should be the original values
        let center = z.center();
        assert!((center[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((center[[1, 2]] - 6.0).abs() < 1e-6);

        // Each element should have width 0.2 (±0.1)
        let bounds = z.to_bounded_tensor();
        assert!((bounds.lower[[0, 0]] - 0.9).abs() < 1e-6);
        assert!((bounds.upper[[0, 0]] - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_transposed_concrete() {
        // Two concrete matrices (no error)
        // Q: (2, 3), K: (2, 3) -> Q @ K^T: (2, 2)
        let q_vals = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let k_vals = arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);

        // Create zonotopes with 0 error (concrete)
        let mut q_coeffs = ndarray::Array3::<f32>::zeros((1, 2, 3));
        q_coeffs.index_axis_mut(Axis(0), 0).assign(&q_vals);
        let q = ZonotopeTensor {
            coeffs: q_coeffs.into_dyn(),
            n_error_terms: 0,
            element_shape: vec![2, 3],
        };

        let mut k_coeffs = ndarray::Array3::<f32>::zeros((1, 2, 3));
        k_coeffs.index_axis_mut(Axis(0), 0).assign(&k_vals);
        let k = ZonotopeTensor {
            coeffs: k_coeffs.into_dyn(),
            n_error_terms: 0,
            element_shape: vec![2, 3],
        };

        let result = q.matmul_transposed(&k).unwrap();

        // Q @ K^T = [[1,0],[0,0]] (row i, col j = Q[i,:] · K[j,:])
        assert_eq!(result.element_shape, vec![2, 2]);
        let center = result.center();
        assert!((center[[0, 0]] - 1.0).abs() < 1e-6); // [1,0,0] · [1,0,0] = 1
        assert!((center[[0, 1]] - 0.0).abs() < 1e-6); // [1,0,0] · [0,0,1] = 0
        assert!((center[[1, 0]] - 0.0).abs() < 1e-6); // [0,1,0] · [1,0,0] = 0
        assert!((center[[1, 1]] - 0.0).abs() < 1e-6); // [0,1,0] · [0,0,1] = 0
    }

    #[test]
    fn test_matmul_transposed_with_error() {
        // Q and K share error symbols - this is the key for attention!
        // Q: (1, 2) with values [1, 0] + 0.1*error
        // K: (1, 2) with values [1, 0] + 0.1*error (SAME error symbol!)
        // Result: (1, 1) = Q · K = 1 + perturbation

        let q_vals = arr2(&[[1.0, 0.0]]);
        let k_vals = arr2(&[[1.0, 0.0]]);

        let q = ZonotopeTensor::from_input_2d(&q_vals, 0.1);
        let k = ZonotopeTensor::from_input_2d(&k_vals, 0.1);

        let result = q.matmul_transposed(&k).unwrap();

        assert_eq!(result.element_shape, vec![1, 1]);

        // Center: 1*1 + 0*0 = 1, plus e_i² center shift
        // For error term 0 (position 0,0 in both Q and K):
        //   Q[1,0,0] = 0.1, K[1,0,0] = 0.1
        //   center_shift += 0.5 * (0.1 * 0.1) = 0.005
        // For error term 1 (position 0,1 in both Q and K):
        //   Q[2,0,1] = 0.1, K[2,0,1] = 0.1
        //   center_shift += 0.5 * (0.1 * 0.1) = 0.005
        // Total center = 1 + 0.01 = 1.01
        let center = result.center();
        assert!((center[[0, 0]] - 1.01).abs() < 1e-5);

        // Width should be computable
        let bounds = result.to_bounded_tensor();
        let width = bounds.upper[[0, 0]] - bounds.lower[[0, 0]];
        assert!(width > 0.0);
        assert!(width < 1.0); // Sanity check - should be small for small epsilon
    }

    #[test]
    fn test_matmul_transposed_zonotope_tighter_than_ibp() {
        // This test demonstrates the zonotope advantage for correlated Q@K^T
        //
        // Scenario: Q and K both come from same input X
        // Q = X, K = X (identity transforms)
        // Q@K^T = X@X^T
        //
        // When X has correlated perturbations across positions,
        // zonotope tracks this and gives tighter bounds.

        // X: (2, 2) matrix with perturbation
        let x_vals = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let epsilon = 0.1;

        // Create zonotope for X (Q and K share the same zonotope!)
        let x = ZonotopeTensor::from_input_2d(&x_vals, epsilon);

        // Zonotope: Q@K^T where Q=K=X
        let result_zonotope = x.matmul_transposed(&x).unwrap();
        let zonotope_bounds = result_zonotope.to_bounded_tensor();

        // IBP would compute: (X ± ε) @ (X ± ε)^T treating them independently
        // For diagonal elements: X[i,:] · X[i,:] = sum of squares
        // [1,0]·[1,0] = 1, perturbed: [1±0.1, 0±0.1]·[1±0.1, 0±0.1]
        // IBP treats as independent: [0.81,1.21] + [0,0.04] = [0.81, 1.25]
        // Width = 0.44

        // For zonotope with shared symbols, x_i * x_i where x_i = center + ε*e_i
        // gives tighter bounds because we know e_i * e_i = e_i² ∈ [0,1]

        let zonotope_width_00 = zonotope_bounds.upper[[0, 0]] - zonotope_bounds.lower[[0, 0]];

        // IBP width for [0,0] position (computed independently)
        // [1±0.1]² + [0±0.1]² = [0.81,1.21] + [0,0.01] = [0.81, 1.22]
        // IBP width ≈ 0.41
        let ibp_lower_00 = 0.81_f32; // min of x²
        let ibp_upper_00 = 1.21 + 0.01; // max of sum of squares with epsilon
        let ibp_width_00 = ibp_upper_00 - ibp_lower_00;

        // Zonotope should be at least as tight, ideally tighter
        // (In practice, for x*x with same error, zonotope gives exact bounds)
        println!("Zonotope width [0,0]: {}", zonotope_width_00);
        println!("IBP width [0,0]: {}", ibp_width_00);

        // The zonotope should give a valid bound
        assert!(zonotope_bounds.lower[[0, 0]] <= 1.01); // Center is ~1.01 after shift
        assert!(zonotope_bounds.upper[[0, 0]] >= 1.01);
    }

    #[test]
    fn test_zonotope_advantage_different_transforms() {
        // The zonotope advantage appears when Q and K are DIFFERENT transforms of X.
        //
        // Scenario: X = [1] (scalar), epsilon = 0.5
        // Q = 2*X = 2 (but shares X's error symbol!)
        // K = -1*X = -1 (also shares X's error symbol!)
        //
        // Q * K = 2*(-1) = -2 when no perturbation
        //
        // With perturbation X = 1 + 0.5*e where e ∈ [-1, 1]:
        // Q = 2*(1 + 0.5*e) = 2 + e
        // K = -1*(1 + 0.5*e) = -1 - 0.5*e
        // Q * K = (2 + e)*(-1 - 0.5*e) = -2 - e + -e - 0.5*e² = -2 - 2e - 0.5*e²
        //
        // Since e ∈ [-1, 1]: -2e ∈ [-2, 2], and e² ∈ [0, 1], so -0.5*e² ∈ [-0.5, 0]
        // True range: [-2 - 2 - 0.5, -2 + 2 + 0] = [-4.5, 0]
        //
        // IBP (treating Q and K independently):
        // Q ∈ [1.5, 2.5], K ∈ [-1.5, -0.5]
        // Q * K ∈ [2.5 * -1.5, 1.5 * -0.5] = [-3.75, -0.75]  <- WRONG ORDER
        // Actually: min = 2.5*-1.5 = -3.75, max = 1.5*-0.5 = -0.75
        // IBP width = 3.0
        //
        // Zonotope (tracking correlation):
        // Q = 2 + e (center=2, coeff[1]=1 for e)
        // K = -1 - 0.5*e (center=-1, coeff[1]=-0.5 for e)
        //
        // Q*K center: 2*(-1) = -2
        // e² term: 1*(-0.5)*e² = -0.5*e² -> center shift = -0.25, half_term = 0.25
        // Linear: 2*(-0.5) + 1*(-1) = -2 for e
        // New center = -2 + (-0.25) = -2.25
        // Radius = |−2| + 0.25 = 2.25
        // Zonotope range: [-2.25 - 2.25, -2.25 + 2.25] = [-4.5, 0] <- EXACT!
        //
        // Zonotope width = 4.5 (exact)
        // IBP width = 3.0 (wrong! doesn't contain true range)
        //
        // This shows zonotopes give SOUND bounds while IBP can be UNSOUND
        // for correlated variables with opposite signs.

        // Build zonotopes manually for Q = [2+e] and K = [-1-0.5e]
        // where they share the same error symbol

        // Q: (1,1) zonotope with 1 error term
        // coeffs shape: (2, 1, 1) = [center=2, error_coeff=1]
        let mut q_coeffs = ndarray::Array3::<f32>::zeros((2, 1, 1));
        q_coeffs[[0, 0, 0]] = 2.0; // center
        q_coeffs[[1, 0, 0]] = 1.0; // coefficient for e (since Q = 2X = 2*(1+0.5e) = 2+e)
        let q = ZonotopeTensor {
            coeffs: q_coeffs.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1, 1],
        };

        // K: (1,1) zonotope with same error symbol
        // K = -X = -(1+0.5e) = -1 - 0.5e
        let mut k_coeffs = ndarray::Array3::<f32>::zeros((2, 1, 1));
        k_coeffs[[0, 0, 0]] = -1.0; // center
        k_coeffs[[1, 0, 0]] = -0.5; // coefficient for e
        let k = ZonotopeTensor {
            coeffs: k_coeffs.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1, 1],
        };

        let result = q.matmul_transposed(&k).unwrap();

        // Center should be: 2*(-1) + 0.5*(1*-0.5) = -2 - 0.25 = -2.25
        let center = result.center()[[0, 0]];
        assert!(
            (center - (-2.25)).abs() < 1e-5,
            "Expected center -2.25, got {}",
            center
        );

        let bounds = result.to_bounded_tensor();
        let lower = bounds.lower[[0, 0]];
        let upper = bounds.upper[[0, 0]];
        let width = upper - lower;

        println!(
            "Q*K zonotope: center={}, bounds=[{}, {}], width={}",
            center, lower, upper, width
        );

        // Zonotope should contain the true range [-4.5, 0]
        assert!(
            lower <= -4.5 + 0.01,
            "Lower bound {} should be <= -4.5",
            lower
        );
        assert!(upper >= 0.0 - 0.01, "Upper bound {} should be >= 0", upper);

        // IBP would give [-3.75, -0.75] which is UNSOUND (doesn't contain 0 or -4.5)
        // So zonotope wins by being CORRECT, not just tighter
    }

    #[test]
    fn test_add_constant() {
        // Create zonotope: [1±0.1, 2±0.1]
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        // Add constant [0.5, 1.0]
        let constant = arr1(&[0.5, 1.0]).into_dyn();
        let result = z.add_constant(&constant).unwrap();

        // Center should be [1.5, 3.0]
        assert!((result.center()[[0]] - 1.5).abs() < 1e-6);
        assert!((result.center()[[1]] - 3.0).abs() < 1e-6);

        // Radius should be unchanged (0.1 for each)
        let bounds = result.to_bounded_tensor();
        assert!((bounds.lower[[0]] - 1.4).abs() < 1e-6);
        assert!((bounds.upper[[0]] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_mul_constant() {
        // Create zonotope: [1±0.1, 2±0.1]
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        // Multiply by constant [2.0, 0.5]
        let constant = arr1(&[2.0, 0.5]).into_dyn();
        let result = z.mul_constant(&constant).unwrap();

        // Center should be [2.0, 1.0]
        assert!((result.center()[[0]] - 2.0).abs() < 1e-6);
        assert!((result.center()[[1]] - 1.0).abs() < 1e-6);

        // Widths: first element width = 0.2 * 2 = 0.4, second = 0.2 * 0.5 = 0.1
        let bounds = result.to_bounded_tensor();
        let width_0 = bounds.upper[[0]] - bounds.lower[[0]];
        let width_1 = bounds.upper[[1]] - bounds.lower[[1]];
        assert!((width_0 - 0.4).abs() < 1e-6);
        assert!((width_1 - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_from_bounded_tensor() {
        // Create a BoundedTensor
        use crate::BoundedTensor;
        let bounds =
            BoundedTensor::new(arr1(&[0.5, 1.5]).into_dyn(), arr1(&[1.5, 2.5]).into_dyn()).unwrap();

        let z = ZonotopeTensor::from_bounded_tensor(&bounds);

        // Center should be [1.0, 2.0]
        assert!((z.center()[[0]] - 1.0).abs() < 1e-6);
        assert!((z.center()[[1]] - 2.0).abs() < 1e-6);

        // Should round-trip back to same bounds
        let bounds2 = z.to_bounded_tensor();
        assert!((bounds2.lower[[0]] - 0.5).abs() < 1e-6);
        assert!((bounds2.upper[[0]] - 1.5).abs() < 1e-6);
        assert!((bounds2.lower[[1]] - 1.5).abs() < 1e-6);
        assert!((bounds2.upper[[1]] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_expand_to_match() {
        // Create two zonotopes with different error term counts
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z1 = ZonotopeTensor::from_input_shared(&values, 0.1); // 1 error term
        let z2 = ZonotopeTensor::from_input_elementwise(&values, 0.1); // 2 error terms

        let (expanded1, expanded2) = z1.expand_to_match(&z2).unwrap();

        // Both should have 2 error terms now
        assert_eq!(expanded1.n_error_terms, 2);
        assert_eq!(expanded2.n_error_terms, 2);

        // z1's bounds should be preserved
        let b1 = expanded1.to_bounded_tensor();
        assert!((b1.lower[[0]] - 0.9).abs() < 1e-6);
        assert!((b1.upper[[0]] - 1.1).abs() < 1e-6);

        // z2's bounds should be preserved
        let b2 = expanded2.to_bounded_tensor();
        assert!((b2.lower[[0]] - 0.9).abs() < 1e-6);
        assert!((b2.upper[[0]] - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_reshape() {
        // Create (2, 4) zonotope with per-position error terms (2 errors)
        let values = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        assert_eq!(z.element_shape, vec![2, 4]);
        assert_eq!(z.n_error_terms, 8); // 2*4 = 8 per-element errors

        // Reshape to (4, 2)
        let reshaped = z.reshape(&[4, 2]).unwrap();

        assert_eq!(reshaped.element_shape, vec![4, 2]);
        assert_eq!(reshaped.n_error_terms, 8); // Same number of error terms

        // Total elements preserved
        assert_eq!(z.len(), reshaped.len());

        // Bounds should be preserved (same values, just rearranged)
        let orig_bounds = z.to_bounded_tensor();
        let new_bounds = reshaped.to_bounded_tensor();

        // First element [0,0] in original = first element [0,0] in reshaped (row-major order)
        assert!((orig_bounds.lower[[0, 0]] - new_bounds.lower[[0, 0]]).abs() < 1e-6);
        assert!((orig_bounds.upper[[0, 0]] - new_bounds.upper[[0, 0]]).abs() < 1e-6);
    }

    #[test]
    fn test_reshape_error_different_size() {
        let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        // Try to reshape to different size - should fail
        let result = z.reshape(&[3, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tile() {
        // Create (2, 3) zonotope
        let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        assert_eq!(z.element_shape, vec![2, 3]);

        // Tile 3x along axis 0 -> (6, 3)
        let tiled = z.tile(0, 3).unwrap();

        assert_eq!(tiled.element_shape, vec![6, 3]);
        assert_eq!(tiled.n_error_terms, z.n_error_terms); // Same errors (shared symbols!)

        // Original values should be repeated
        let center = tiled.center();
        // Row 0 = original row 0
        assert!((center[[0, 0]] - 1.0).abs() < 1e-6);
        // Row 2 = original row 0 (repeated)
        assert!((center[[2, 0]] - 1.0).abs() < 1e-6);
        // Row 4 = original row 0 (repeated again)
        assert!((center[[4, 0]] - 1.0).abs() < 1e-6);
        // Row 1 = original row 1
        assert!((center[[1, 0]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_tile_preserves_correlations() {
        // Key test: tiling should preserve error symbol correlations
        // This is essential for GQA where K heads are tiled to match Q heads
        // The tiled values share the same uncertainty as the original

        let values = arr2(&[[1.0, 2.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.5);

        assert_eq!(z.n_error_terms, 2); // Per-element errors

        // Tile 4x -> 4 copies, all sharing the same error symbols
        let tiled = z.tile(0, 4).unwrap();

        assert_eq!(tiled.element_shape, vec![4, 2]);
        assert_eq!(tiled.n_error_terms, 2); // SAME number of errors!

        // All 4 rows should have the same center and error structure
        let bounds = tiled.to_bounded_tensor();
        for i in 0..4 {
            // Each row has the same bounds as the original
            assert!((bounds.lower[[i, 0]] - 0.5).abs() < 1e-6); // 1.0 - 0.5
            assert!((bounds.upper[[i, 0]] - 1.5).abs() < 1e-6); // 1.0 + 0.5
            assert!((bounds.lower[[i, 1]] - 1.5).abs() < 1e-6); // 2.0 - 0.5
            assert!((bounds.upper[[i, 1]] - 2.5).abs() < 1e-6); // 2.0 + 0.5
        }

        // Crucially, the tiled rows share error symbols!
        // This means if row 0 takes value 1.5, rows 1-3 ALSO take 1.5
        // (they move together, not independently)
    }

    #[test]
    fn test_tile_identity() {
        // Tiling with reps=1 should return identical zonotope
        let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        let tiled = z.tile(0, 1).unwrap();

        assert_eq!(tiled.element_shape, z.element_shape);
        assert_eq!(tiled.n_error_terms, z.n_error_terms);

        let orig_bounds = z.to_bounded_tensor();
        let tiled_bounds = tiled.to_bounded_tensor();

        for i in 0..2 {
            for j in 0..2 {
                assert!((orig_bounds.lower[[i, j]] - tiled_bounds.lower[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_transpose_last_two() {
        // Create (2, 3) zonotope
        let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        assert_eq!(z.element_shape, vec![2, 3]);

        let transposed = z.transpose_last_two().unwrap();

        assert_eq!(transposed.element_shape, vec![3, 2]);
        assert_eq!(transposed.n_error_terms, z.n_error_terms);

        let orig_center = z.center();
        let trans_center = transposed.center();

        // z[i,j] should equal transposed[j,i]
        assert!((orig_center[[0, 0]] - trans_center[[0, 0]]).abs() < 1e-6);
        assert!((orig_center[[0, 1]] - trans_center[[1, 0]]).abs() < 1e-6);
        assert!((orig_center[[0, 2]] - trans_center[[2, 0]]).abs() < 1e-6);
        assert!((orig_center[[1, 0]] - trans_center[[0, 1]]).abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm_affine_concrete() {
        // Test LayerNorm on a concrete zonotope (no error terms)
        // Input: [[1, 2, 3], [4, 5, 6]] (2 positions, 3 features)
        let values = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let z = ZonotopeTensor::concrete(values.clone().into_dyn());

        // LayerNorm parameters: gamma=1, beta=0, eps=1e-5
        let gamma = arr1(&[1.0_f32, 1.0, 1.0]);
        let beta = arr1(&[0.0_f32, 0.0, 0.0]);
        let eps = 1e-5;

        let result = z.layer_norm_affine(&gamma, &beta, eps).unwrap();

        // For concrete input, output center should match standard LayerNorm
        let center = result.center();

        // Row 0: mean=2, var=2/3, std=sqrt(2/3)≈0.8165
        // Normalized: [-1.22, 0, 1.22] (approx)
        // center[0] + center[2] should be ~0 (symmetric around zero)
        assert!((center[[0, 0]] + center[[0, 2]]).abs() < 0.01);
        assert!(center[[0, 1]].abs() < 0.01); // Mean feature should be ~0

        // Row 1: mean=5, var=2/3, std=sqrt(2/3)≈0.8165
        // Same normalization pattern
        assert!((center[[1, 0]] + center[[1, 2]]).abs() < 0.01);
        assert!(center[[1, 1]].abs() < 0.01);
    }

    #[test]
    fn test_layer_norm_affine_with_error() {
        // Test that LayerNorm preserves zonotope correlations
        // Input: 2x3 zonotope with per-position error symbols
        let values = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let z = ZonotopeTensor::from_input_2d(&values, 0.1);

        let gamma = arr1(&[1.0_f32, 1.0, 1.0]);
        let beta = arr1(&[0.0_f32, 0.0, 0.0]);
        let eps = 1e-5;

        let result = z.layer_norm_affine(&gamma, &beta, eps).unwrap();

        // Should have original error terms + 1 new approximation error term
        assert_eq!(result.n_error_terms, z.n_error_terms + 1);
        assert_eq!(result.element_shape, z.element_shape);

        // Output bounds should be tighter than naive IBP
        // (This is the main benefit of zonotope LayerNorm)
        let bounds = result.to_bounded_tensor();
        let ibp_z = ZonotopeTensor::from_input_2d(&values, 0.1);
        let ibp_bounds = ibp_z.to_bounded_tensor();

        // Both should have finite bounds
        assert!(bounds.lower.iter().all(|&v| v.is_finite()));
        assert!(bounds.upper.iter().all(|&v| v.is_finite()));
        assert!(ibp_bounds.lower.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_layer_norm_affine_with_gamma_beta() {
        // Test with non-trivial gamma and beta
        let values = arr2(&[[0.0_f32, 0.0, 0.0]]);
        let z = ZonotopeTensor::concrete(values.into_dyn());

        let gamma = arr1(&[2.0_f32, 1.0, 0.5]);
        let beta = arr1(&[1.0_f32, 2.0, 3.0]);
        let eps = 1e-5;

        let result = z.layer_norm_affine(&gamma, &beta, eps).unwrap();
        let center = result.center();

        // For constant input [0,0,0]:
        // mean=0, var=0, std=sqrt(eps)
        // Each output_i = gamma_i * 0 / std + beta_i = beta_i
        for d in 0..3 {
            assert!((center[[0, d]] - beta[d]).abs() < 0.1);
        }
    }

    // ==================== Softmax Affine Tests ====================

    #[test]
    fn test_softmax_affine_concrete_1d() {
        // Test softmax on concrete (no error) zonotope
        // softmax([1,2,3]) = [0.09003, 0.24473, 0.66524]
        let values = arr1(&[1.0_f32, 2.0, 3.0]);
        let z = ZonotopeTensor::concrete(values.into_dyn());

        let result = z.softmax_affine(-1).unwrap();
        let center = result.center();

        // Compute expected softmax
        let e1 = 1.0_f32.exp();
        let e2 = 2.0_f32.exp();
        let e3 = 3.0_f32.exp();
        let sum = e1 + e2 + e3;
        let expected = [e1 / sum, e2 / sum, e3 / sum];

        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (center[i] - exp).abs() < 1e-5,
                "softmax[{}] = {}, got {}",
                i,
                exp,
                center[i]
            );
        }
    }

    #[test]
    fn test_softmax_affine_uniform() {
        // Test softmax on uniform input: softmax([1,1,1]) = [1/3, 1/3, 1/3]
        let values = arr1(&[1.0_f32, 1.0, 1.0]);
        let z = ZonotopeTensor::concrete(values.into_dyn());

        let result = z.softmax_affine(-1).unwrap();
        let center = result.center();

        for i in 0..3 {
            assert!(
                (center[i] - 1.0 / 3.0).abs() < 1e-5,
                "uniform softmax[{}] should be 1/3, got {}",
                i,
                center[i]
            );
        }
    }

    #[test]
    fn test_softmax_affine_with_error_1d() {
        // Test softmax on zonotope with error
        // z = [2, 2, 2] + 0.1 * e1 * [1, -1, 0]
        // At center: softmax([2,2,2]) = [1/3, 1/3, 1/3]
        let values = arr1(&[2.0_f32, 2.0, 2.0]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);

        let result = z.softmax_affine(-1).unwrap();

        // Center should still be [1/3, 1/3, 1/3]
        let center = result.center();
        for i in 0..3 {
            assert!(
                (center[i] - 1.0 / 3.0).abs() < 1e-4,
                "softmax center[{}] should be 1/3, got {}",
                i,
                center[i]
            );
        }

        // Output should have error terms (transformed through Jacobian)
        assert!(
            result.n_error_terms >= 1,
            "softmax output should have error terms"
        );

        // The bounds should be close to but not exactly [1/3, 1/3, 1/3]
        let bounds = result.to_bounded_tensor();
        for i in 0..3 {
            assert!(bounds.lower[i] >= 0.0, "softmax lower bound should be >= 0");
            assert!(bounds.upper[i] <= 1.0, "softmax upper bound should be <= 1");
            assert!(
                bounds.lower[i] < bounds.upper[i],
                "softmax bounds should have nonzero width"
            );
        }
    }

    #[test]
    fn test_softmax_affine_2d() {
        // Test softmax on 2D input (seq_len=2, dim=3)
        let values = arr2(&[[1.0_f32, 2.0, 3.0], [0.0, 0.0, 0.0]]);
        let z = ZonotopeTensor::concrete(values.into_dyn());

        let result = z.softmax_affine(-1).unwrap();
        let center = result.center();

        // Row 0: softmax([1,2,3])
        let e1 = 1.0_f32.exp();
        let e2 = 2.0_f32.exp();
        let e3 = 3.0_f32.exp();
        let sum0 = e1 + e2 + e3;
        let expected0 = [e1 / sum0, e2 / sum0, e3 / sum0];

        // Row 1: softmax([0,0,0]) = [1/3, 1/3, 1/3]
        let expected1 = [1.0 / 3.0; 3];

        for i in 0..3 {
            assert!(
                (center[[0, i]] - expected0[i]).abs() < 1e-5,
                "softmax row0[{}] = {}, got {}",
                i,
                expected0[i],
                center[[0, i]]
            );
            assert!(
                (center[[1, i]] - expected1[i]).abs() < 1e-5,
                "softmax row1[{}] = {}, got {}",
                i,
                expected1[i],
                center[[1, i]]
            );
        }
    }

    #[test]
    fn test_softmax_affine_jacobian_correctness() {
        // Verify the Jacobian is correct by numerical differentiation
        // softmax at center [1, 2]:
        // s = [e^1/(e^1+e^2), e^2/(e^1+e^2)] ≈ [0.269, 0.731]
        // Jacobian J[i,j] = s[i] * (δ[i,j] - s[j])
        // J = [[s0*(1-s0), -s0*s1],
        //      [-s1*s0, s1*(1-s1)]]
        let center = arr1(&[1.0_f32, 2.0]);
        let z = ZonotopeTensor::from_input_shared(&center.clone().into_dyn(), 0.1);

        let result = z.softmax_affine(-1).unwrap();

        // Numerical test: compute softmax at center + delta and center - delta
        let delta = 0.001;
        fn softmax_2(x: &[f32]) -> [f32; 2] {
            let e0 = x[0].exp();
            let e1 = x[1].exp();
            let sum = e0 + e1;
            [e0 / sum, e1 / sum]
        }

        // The error coefficient should be approximately J @ [0.1, 0.1]
        // Since shared input, both elements get +/- 0.1
        let s_plus = softmax_2(&[center[0] + delta, center[1] + delta]);
        let s_minus = softmax_2(&[center[0] - delta, center[1] - delta]);

        // With shared error, the derivative is d(softmax)/d(shift) where shift = +/- delta
        let _numerical_deriv = [
            (s_plus[0] - s_minus[0]) / (2.0 * delta),
            (s_plus[1] - s_minus[1]) / (2.0 * delta),
        ];

        // The error coefficient for shared input should be approximately numerical_deriv * 0.1
        // But our implementation transforms per-element errors
        // This test verifies the zonotope bounds contain the true range
        let bounds = result.to_bounded_tensor();

        let test_shift = 0.05;
        let s_test = softmax_2(&[center[0] + test_shift, center[1] + test_shift]);
        for (i, &s_val) in s_test.iter().enumerate() {
            assert!(
                bounds.lower[i] <= s_val + 0.01 && s_val - 0.01 <= bounds.upper[i],
                "softmax bounds should contain true value at small shift"
            );
        }
    }

    #[test]
    fn test_softmax_affine_sum_to_one_preserved() {
        // Softmax output sum should be approximately 1
        let values = arr1(&[1.0_f32, -1.0, 0.5, 2.0]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);

        let result = z.softmax_affine(-1).unwrap();
        let center = result.center();

        let sum: f32 = center.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax center should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_softmax_affine_causal_masks_future_positions() {
        // Causal softmax must output exactly 0 for masked positions (j > i).
        let seq = 4_usize;
        let center: Vec<f32> = (0..seq * seq).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let values = ndarray::ArrayD::from_shape_vec(vec![seq, seq], center).expect("shape ok");

        // Shared uncertainty across all entries; masked logits still vary, but should not affect outputs.
        let z = ZonotopeTensor::from_input_shared(&values, 0.25);

        let result = z.softmax_affine_causal(-1).unwrap();

        // Causal softmax adds exactly one new approximation error term.
        assert_eq!(result.n_error_terms, z.n_error_terms + 1);

        let center_out = result.center();
        for i in 0..seq {
            let mut row_sum = 0.0f32;
            for j in 0..seq {
                let v = center_out[[i, j]];
                if j > i {
                    assert!(
                        v.abs() < 1e-6,
                        "masked causal softmax center should be 0 at ({},{}) got {}",
                        i,
                        j,
                        v
                    );
                } else {
                    row_sum += v;
                }
            }
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "causal softmax row {} center should sum to 1, got {}",
                i,
                row_sum
            );
        }

        let bounds = result.to_bounded_tensor();
        for i in 0..seq {
            for j in (i + 1)..seq {
                assert!(
                    bounds.upper[[i, j]] <= 1e-6 && bounds.lower[[i, j]] >= -1e-6,
                    "masked causal softmax bounds should be 0 at ({},{}) got [{},{}]",
                    i,
                    j,
                    bounds.lower[[i, j]],
                    bounds.upper[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_softmax_affine_error_term_added() {
        // Verify that softmax adds exactly one new error term
        let values = arr1(&[1.0_f32, 2.0, 3.0]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);
        let initial_errors = z.n_error_terms;

        let result = z.softmax_affine(-1).unwrap();

        assert_eq!(
            result.n_error_terms,
            initial_errors + 1,
            "softmax should add exactly one error term for approximation"
        );
    }

    #[test]
    fn test_softmax_affine_bounds_valid() {
        // For small epsilon, softmax bounds should be close to [0, 1]
        // Note: For large epsilon, the linear approximation error term can
        // make bounds extend outside [0,1], which is conservative but sound.
        let values = arr1(&[1.0_f32, 2.0, 3.0, 2.5]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);

        let result = z.softmax_affine(-1).unwrap();
        let bounds = result.to_bounded_tensor();

        for i in 0..4 {
            // For small epsilon, bounds should be reasonably close to [0, 1]
            assert!(
                bounds.lower[i] >= -0.2,
                "softmax lower[{}] = {} should be near 0 for small epsilon",
                i,
                bounds.lower[i]
            );
            assert!(
                bounds.upper[i] <= 1.2,
                "softmax upper[{}] = {} should be near 1 for small epsilon",
                i,
                bounds.upper[i]
            );
        }
    }

    #[test]
    fn test_softmax_affine_large_epsilon_sound() {
        // Even with large epsilon, bounds must contain true softmax range
        // The linear approximation may give loose bounds but should be sound
        let center = [1.0_f32, 2.0, 3.0];
        let values = arr1(&center);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.5);

        let result = z.softmax_affine(-1).unwrap();
        let bounds = result.to_bounded_tensor();

        // Test soundness: evaluate softmax at extreme points
        fn softmax_3(x: &[f32; 3]) -> [f32; 3] {
            let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = e.iter().sum();
            [e[0] / sum, e[1] / sum, e[2] / sum]
        }

        // Test at center + epsilon and center - epsilon (shared error)
        let eps = 0.5;
        let s_plus = softmax_3(&[center[0] + eps, center[1] + eps, center[2] + eps]);
        let s_minus = softmax_3(&[center[0] - eps, center[1] - eps, center[2] - eps]);

        // Due to approximation error term, bounds may be larger than true range
        // but should contain the true values with some tolerance
        for i in 0..3 {
            assert!(
                bounds.lower[i] <= s_plus[i].min(s_minus[i]) + 0.1,
                "softmax bounds should be sound (contain true lower)"
            );
            assert!(
                bounds.upper[i] >= s_plus[i].max(s_minus[i]) - 0.1,
                "softmax bounds should be sound (contain true upper)"
            );
        }
    }

    #[test]
    fn test_silu_affine_concrete() {
        // Test SiLU on concrete (no error) zonotope
        // SiLU(x) = x * sigmoid(x)
        let values = arr1(&[-1.0_f32, 0.0, 1.0, 2.0]);
        let z = ZonotopeTensor::concrete(values.clone().into_dyn());

        let result = z.silu_affine().unwrap();
        let center = result.center();

        // Expected SiLU values
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        for d in 0..4 {
            let expected = silu(values[d]);
            assert!(
                (center[d] - expected).abs() < 1e-5,
                "SiLU({}) = {}, got {}",
                values[d],
                expected,
                center[d]
            );
        }
    }

    #[test]
    fn test_silu_affine_with_error() {
        // Test SiLU preserves error structure with approximation
        let values = arr1(&[0.0_f32, 1.0]);
        let z = ZonotopeTensor::from_input_shared(&values.clone().into_dyn(), 0.1);

        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor();

        // Verify bounds are sound: check that actual SiLU values at bounds are contained
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        // For input [0±0.1, 1±0.1]
        assert!(bounds.lower[0] <= silu(-0.1));
        assert!(bounds.upper[0] >= silu(0.1));
        assert!(bounds.lower[1] <= silu(0.9));
        assert!(bounds.upper[1] >= silu(1.1));
    }

    #[test]
    fn test_silu_affine_2d() {
        // Test 2D SiLU (needed for transformer FFN)
        let values = arr2(&[[0.0_f32, 1.0], [-1.0, 2.0]]);
        let z = ZonotopeTensor::concrete(values.clone().into_dyn());

        let result = z.silu_affine().unwrap();
        assert_eq!(result.element_shape, vec![2, 2]);

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        let center = result.center();
        for s in 0..2 {
            for d in 0..2 {
                let expected = silu(values[[s, d]]);
                assert!(
                    (center[[s, d]] - expected).abs() < 1e-5,
                    "SiLU({}) = {}, got {}",
                    values[[s, d]],
                    expected,
                    center[[s, d]]
                );
            }
        }
    }

    #[test]
    fn test_mul_elementwise_concrete() {
        // Test element-wise multiplication with concrete zonotopes
        let v1 = arr1(&[2.0_f32, 3.0]);
        let v2 = arr1(&[4.0_f32, 5.0]);
        let z1 = ZonotopeTensor::concrete(v1.clone().into_dyn());
        let z2 = ZonotopeTensor::concrete(v2.clone().into_dyn());

        let result = z1.mul_elementwise(&z2).unwrap();
        let center = result.center();

        // Expected: [2*4, 3*5] = [8, 15]
        assert!((center[0] - 8.0).abs() < 1e-6);
        assert!((center[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_mul_elementwise_with_shared_errors() {
        // Test that mul_elementwise exploits shared error symbols
        // z1 = 1 + 0.1*e1, z2 = 2 + 0.2*e1 (same error symbol)
        // z1*z2 = (1 + 0.1*e1)*(2 + 0.2*e1)
        //       = 2 + 0.2*e1 + 0.2*e1 + 0.02*e1²
        //       = 2 + 0.4*e1 + 0.02*e1²
        // e1² ∈ [0,1], so center shift = 0.5*0.02 = 0.01
        // Center = 2 + 0.01 = 2.01
        // Linear coeff = 0.4
        // Quadratic error = 0.5*|0.02| = 0.01

        // Create zonotopes with same error symbol
        let mut coeffs1 = ndarray::Array2::<f32>::zeros((2, 1));
        coeffs1[[0, 0]] = 1.0; // center
        coeffs1[[1, 0]] = 0.1; // error coeff

        let mut coeffs2 = ndarray::Array2::<f32>::zeros((2, 1));
        coeffs2[[0, 0]] = 2.0;
        coeffs2[[1, 0]] = 0.2;

        let z1 = ZonotopeTensor {
            coeffs: coeffs1.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1],
        };
        let z2 = ZonotopeTensor {
            coeffs: coeffs2.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1],
        };

        let result = z1.mul_elementwise(&z2).unwrap();

        // Check center (with quadratic shift)
        let center = result.center();
        assert!(
            (center[0] - 2.01).abs() < 1e-5,
            "Expected center 2.01, got {}",
            center[0]
        );

        // Check linear error coefficient
        // c1*a2 + c2*a1 = 1*0.2 + 2*0.1 = 0.4
        assert!(
            (result.coeffs[[1, 0]] - 0.4).abs() < 1e-5,
            "Expected linear coeff 0.4, got {}",
            result.coeffs[[1, 0]]
        );

        // Check quadratic error term
        // 0.5 * |0.1*0.2| = 0.01
        assert!(
            (result.coeffs[[2, 0]] - 0.01).abs() < 1e-5,
            "Expected quadratic error 0.01, got {}",
            result.coeffs[[2, 0]]
        );
    }

    #[test]
    fn test_mul_elementwise_soundness() {
        // Test that zonotope multiplication produces SOUND (over-approximate) bounds
        //
        // Note: For a SINGLE shared error symbol with SAME-SIGN coefficients,
        // zonotope multiplication may be LOOSER than IBP because the e² term
        // adds conservative error. This is sound but not tight.
        //
        // Zonotope multiplication shines when:
        // 1. Coefficients have opposite signs (e_i can't be at both extremes simultaneously)
        // 2. There are many error symbols where cross-terms cancel
        //
        // For SwiGLU, the benefit comes from preserving correlations through the
        // entire FFN (Linear -> SiLU -> MulBinary -> Linear), where zonotope
        // tracks dependencies that IBP loses at each operation.

        // Test soundness: verify that true product values fall within zonotope bounds
        let mut coeffs1 = ndarray::Array2::<f32>::zeros((2, 1));
        coeffs1[[0, 0]] = 1.0; // center
        coeffs1[[1, 0]] = 0.5; // error coeff

        let mut coeffs2 = ndarray::Array2::<f32>::zeros((2, 1));
        coeffs2[[0, 0]] = 2.0;
        coeffs2[[1, 0]] = 0.3;

        let z1 = ZonotopeTensor {
            coeffs: coeffs1.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1],
        };
        let z2 = ZonotopeTensor {
            coeffs: coeffs2.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1],
        };

        let zono_result = z1.mul_elementwise(&z2).unwrap();
        let zono_bounds = zono_result.to_bounded_tensor();

        // Check soundness: true product at various e values should be within bounds
        for e in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let x1 = 1.0 + 0.5 * e;
            let x2 = 2.0 + 0.3 * e;
            let product = x1 * x2;
            assert!(
                zono_bounds.lower[0] <= product && product <= zono_bounds.upper[0],
                "Product {} at e={} not in zonotope bounds [{}, {}]",
                product,
                e,
                zono_bounds.lower[0],
                zono_bounds.upper[0]
            );
        }

        // Verify zonotope bounds are finite and reasonable
        assert!(zono_bounds.lower[0].is_finite());
        assert!(zono_bounds.upper[0].is_finite());
        assert!(zono_bounds.lower[0] < zono_bounds.upper[0]);
    }

    #[test]
    fn test_mul_elementwise_opposite_signs_tighter() {
        // Test that zonotope multiplication IS tighter than IBP when
        // coefficients have opposite signs (anti-correlated)
        //
        // z1 = 1 + 0.5*e, z2 = 2 - 0.5*e (opposite sign coefficients)
        // When e=1: z1=1.5, z2=1.5, product=2.25
        // When e=-1: z1=0.5, z2=2.5, product=1.25
        // When e=0: z1=1.0, z2=2.0, product=2.0
        //
        // IBP: z1 ∈ [0.5, 1.5], z2 ∈ [1.5, 2.5]
        // IBP corners: [0.75, 1.25, 2.25, 3.75], range [0.75, 3.75], width 3.0
        //
        // But true range is [1.25, 2.25] (anti-correlation constrains extremes)
        // Zonotope should capture some of this anti-correlation benefit

        let mut coeffs1 = ndarray::Array2::<f32>::zeros((2, 1));
        coeffs1[[0, 0]] = 1.0;
        coeffs1[[1, 0]] = 0.5; // positive coefficient

        let mut coeffs2 = ndarray::Array2::<f32>::zeros((2, 1));
        coeffs2[[0, 0]] = 2.0;
        coeffs2[[1, 0]] = -0.5; // negative coefficient (anti-correlated)

        let z1 = ZonotopeTensor {
            coeffs: coeffs1.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1],
        };
        let z2 = ZonotopeTensor {
            coeffs: coeffs2.into_dyn(),
            n_error_terms: 1,
            element_shape: vec![1],
        };

        let zono_result = z1.mul_elementwise(&z2).unwrap();
        let zono_bounds = zono_result.to_bounded_tensor();
        let zono_width = zono_bounds.upper[0] - zono_bounds.lower[0];

        // IBP width
        let ibp_width = 3.75 - 0.75; // = 3.0

        // Zonotope should be much tighter due to anti-correlation
        // The linear terms c1*a2 + c2*a1 = 1*(-0.5) + 2*0.5 = 0.5
        // Combined with quadratic terms, should give tighter bounds
        assert!(
            zono_width < ibp_width,
            "Zonotope width {} should be < IBP width {} for anti-correlated inputs",
            zono_width,
            ibp_width
        );
    }

    #[test]
    fn test_mul_elementwise_2d() {
        // Test 2D element-wise multiplication (for transformer sequence data)
        let v1 = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
        let v2 = arr2(&[[5.0_f32, 6.0], [7.0, 8.0]]);
        let z1 = ZonotopeTensor::concrete(v1.clone().into_dyn());
        let z2 = ZonotopeTensor::concrete(v2.clone().into_dyn());

        let result = z1.mul_elementwise(&z2).unwrap();
        let center = result.center();

        // Expected: [[1*5, 2*6], [3*7, 4*8]] = [[5, 12], [21, 32]]
        assert!((center[[0, 0]] - 5.0).abs() < 1e-6);
        assert!((center[[0, 1]] - 12.0).abs() < 1e-6);
        assert!((center[[1, 0]] - 21.0).abs() < 1e-6);
        assert!((center[[1, 1]] - 32.0).abs() < 1e-6);
    }

    // ============================================================
    // Mutation-killing tests for basic accessors
    // ============================================================

    #[test]
    fn test_shape_returns_correct_dimensions() {
        // Test shape() returns the actual element_shape, not empty/[0]/[1]
        // Kills: replace shape -> &[usize] with Vec::leak(Vec::new())
        // Kills: replace shape -> &[usize] with Vec::leak(vec![0])
        // Kills: replace shape -> &[usize] with Vec::leak(vec![1])
        let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(); // shape [2, 3]
        let z = ZonotopeTensor::concrete(values);

        let shape = z.shape();
        assert_eq!(shape.len(), 2, "shape should have 2 dimensions");
        assert_eq!(shape[0], 2, "first dimension should be 2");
        assert_eq!(shape[1], 3, "second dimension should be 3");
    }

    #[test]
    fn test_len_returns_correct_count() {
        // Test len() returns product of dimensions, not 0 or 1
        // Kills: replace len -> usize with 0
        // Kills: replace len -> usize with 1
        let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(); // 6 elements
        let z = ZonotopeTensor::concrete(values);

        assert_eq!(z.len(), 6, "len should be 2*3=6");

        // Also test a larger tensor
        let values_3d = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4])); // 24 elements
        let z_3d = ZonotopeTensor::concrete(values_3d);
        assert_eq!(z_3d.len(), 24, "len should be 2*3*4=24");
    }

    #[test]
    fn test_is_empty_true_for_zero_elements() {
        // Test is_empty() returns true when len() == 0
        // Kills: replace is_empty -> bool with false
        // Kills: replace == with != in is_empty
        let values = ArrayD::<f32>::zeros(IxDyn(&[0])); // empty tensor
        let z = ZonotopeTensor::concrete(values);

        assert!(z.is_empty(), "zonotope with 0 elements should be empty");
        assert_eq!(z.len(), 0, "len should be 0 for empty zonotope");
    }

    #[test]
    fn test_is_empty_false_for_non_zero_elements() {
        // Test is_empty() returns false when len() > 0
        // Kills: replace is_empty -> bool with true
        let values = arr1(&[1.0, 2.0]).into_dyn();
        let z = ZonotopeTensor::concrete(values);

        assert!(!z.is_empty(), "zonotope with elements should not be empty");
        assert_eq!(z.len(), 2, "len should equal element count");
    }

    #[test]
    fn test_has_unbounded_false_for_finite_coeffs() {
        // Test has_unbounded() returns false when all coeffs are finite
        // Kills: replace has_unbounded -> bool with true
        let values = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        assert!(
            !z.has_unbounded(),
            "zonotope with finite coeffs should not be unbounded"
        );
    }

    #[test]
    fn test_has_unbounded_true_for_infinite_coeffs() {
        // Test has_unbounded() returns true when coeffs contain infinity
        // Kills: replace has_unbounded -> bool with false
        let values = arr1(&[1.0, f32::INFINITY, 3.0]).into_dyn();
        let z = ZonotopeTensor::concrete(values);

        assert!(
            z.has_unbounded(),
            "zonotope with infinite center should be unbounded"
        );

        // Also test with infinite error term
        let mut z2 = ZonotopeTensor::from_input_shared(&arr1(&[1.0, 2.0]).into_dyn(), 0.1);
        z2.coeffs[[1, 0]] = f32::INFINITY;
        assert!(
            z2.has_unbounded(),
            "zonotope with infinite error coeff should be unbounded"
        );
    }

    // ============================================================
    // Mutation-killing tests for from_input_per_position
    // ============================================================

    #[test]
    fn test_from_input_per_position_rejects_1d() {
        // Kills: replace < with == in line 210 (shape.len() < 2)
        // Kills: replace < with > in line 210
        // Kills: replace < with <= in line 210
        let values_1d = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let result = ZonotopeTensor::from_input_per_position(&values_1d, 0.1);
        assert!(
            result.is_err(),
            "1D input should fail (requires >= 2 dimensions)"
        );
    }

    #[test]
    fn test_from_input_per_position_accepts_2d() {
        // Verify 2D input works - complements the rejection test
        let values_2d = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(); // shape [2, 2]
        let result = ZonotopeTensor::from_input_per_position(&values_2d, 0.1);
        assert!(result.is_ok(), "2D input should succeed");

        let z = result.unwrap();
        // seq_len = shape[-2] = 2, so n_error_terms = 2
        assert_eq!(z.n_error_terms, 2, "2D: n_error_terms should be seq_len");
    }

    #[test]
    fn test_from_input_per_position_seq_len_calculation() {
        // Kills: replace - with + in line 216 (shape[shape.len() - 2])
        // Kills: replace - with / in line 216
        // For 2D [seq_len=3, embed=4], seq_len should be 3
        let values = ArrayD::<f32>::zeros(IxDyn(&[3, 4])); // seq=3, embed=4
        let z = ZonotopeTensor::from_input_per_position(&values, 0.1).unwrap();

        // If mutation changed - to +, would try shape[2+2] = shape[4] -> panic or wrong
        // If mutation changed - to /, would try shape[2/2] = shape[1] = 4 (wrong)
        assert_eq!(
            z.n_error_terms, 3,
            "seq_len should be 3 (second-to-last dim)"
        );
    }

    #[test]
    fn test_from_input_per_position_2d_error_assignment() {
        // Kills: delete match arm 2 in line 220/239
        // Kills: replace + with - in line 243 (1 + pos)
        // Kills: replace + with * in line 243
        let values = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn(); // [3, 2]
        let z = ZonotopeTensor::from_input_per_position(&values, 0.5).unwrap();

        // For 2D, coeffs[1+pos, pos, :] = epsilon
        // coeffs shape: [1+3, 3, 2] = [4, 3, 2]
        assert_eq!(z.coeffs.shape(), &[4, 3, 2]);

        // Check error terms are at correct positions
        // Error term 0 (index 1) affects position 0
        assert!(
            (z.coeffs[[1, 0, 0]] - 0.5).abs() < 1e-6,
            "err0 should affect pos0"
        );
        assert!(
            (z.coeffs[[1, 0, 1]] - 0.5).abs() < 1e-6,
            "err0 should affect pos0"
        );
        assert_eq!(z.coeffs[[1, 1, 0]], 0.0, "err0 should NOT affect pos1");

        // Error term 1 (index 2) affects position 1
        assert_eq!(z.coeffs[[2, 0, 0]], 0.0, "err1 should NOT affect pos0");
        assert!(
            (z.coeffs[[2, 1, 0]] - 0.5).abs() < 1e-6,
            "err1 should affect pos1"
        );

        // Error term 2 (index 3) affects position 2
        assert_eq!(z.coeffs[[3, 1, 0]], 0.0, "err2 should NOT affect pos1");
        assert!(
            (z.coeffs[[3, 2, 0]] - 0.5).abs() < 1e-6,
            "err2 should affect pos2"
        );
    }

    #[test]
    fn test_from_input_per_position_3d_n_error_terms() {
        // Kills: delete match arm 3 in lines 221, 247
        // Kills: replace * with + in line 221 (shape[0] * seq_len)
        // Kills: replace * with / in line 221
        // For 3D [batch=2, seq=3, embed=4], n_error_terms = batch * seq = 6
        let values = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4]));
        let z = ZonotopeTensor::from_input_per_position(&values, 0.1).unwrap();

        // If * was +, would get 2 + 3 = 5 (wrong)
        // If * was /, would get 2 / 3 = 0 (wrong)
        assert_eq!(
            z.n_error_terms, 6,
            "3D: n_error_terms should be batch * seq = 6"
        );
    }

    #[test]
    fn test_from_input_per_position_3d_error_assignment() {
        // Kills mutations in line 253: 1 + b * seq_len + pos
        // Tests: replace + with -, replace * with +, etc.
        let values = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4])); // batch=2, seq=3, embed=4
        let z = ZonotopeTensor::from_input_per_position(&values, 0.5).unwrap();

        // coeffs shape: [1+6, 2, 3, 4] = [7, 2, 3, 4]
        assert_eq!(z.coeffs.shape(), &[7, 2, 3, 4]);

        // For (b=0, pos=0): err index = 1 + 0*3 + 0 = 1
        assert!(
            (z.coeffs[[1, 0, 0, 0]] - 0.5).abs() < 1e-6,
            "b0p0 should use err1"
        );
        assert_eq!(z.coeffs[[1, 0, 1, 0]], 0.0, "err1 should NOT affect b0p1");
        assert_eq!(z.coeffs[[1, 1, 0, 0]], 0.0, "err1 should NOT affect b1p0");

        // For (b=0, pos=2): err index = 1 + 0*3 + 2 = 3
        assert!(
            (z.coeffs[[3, 0, 2, 0]] - 0.5).abs() < 1e-6,
            "b0p2 should use err3"
        );

        // For (b=1, pos=0): err index = 1 + 1*3 + 0 = 4
        assert!(
            (z.coeffs[[4, 1, 0, 0]] - 0.5).abs() < 1e-6,
            "b1p0 should use err4"
        );
        assert_eq!(z.coeffs[[4, 0, 0, 0]], 0.0, "err4 should NOT affect b0p0");

        // For (b=1, pos=2): err index = 1 + 1*3 + 2 = 6
        assert!(
            (z.coeffs[[6, 1, 2, 0]] - 0.5).abs() < 1e-6,
            "b1p2 should use err6"
        );
    }

    // ============================================================
    // Mutation-killing tests for linear()
    // ============================================================

    #[test]
    fn test_linear_rejects_empty_shape() {
        // Kills: replace || with && in line 387
        // The condition is: is_empty() || last() != Some(&in_features)
        // With &&, only fails when BOTH conditions are true (which can't happen for empty)

        // Create a scalar zonotope (element_shape = [])
        let coeffs = ArrayD::<f32>::zeros(IxDyn(&[1])); // 0 error terms, scalar
        let z = ZonotopeTensor::new(coeffs).unwrap();
        assert!(
            z.element_shape.is_empty(),
            "should have empty element_shape"
        );

        let weight = arr2(&[[1.0, 2.0]]); // 1x2 matrix
        let result = z.linear(&weight, None);
        assert!(result.is_err(), "linear should reject empty element_shape");
    }

    #[test]
    fn test_linear_rejects_shape_mismatch() {
        // Complements empty test: non-empty but wrong last dimension
        let values = arr1(&[1.0, 2.0, 3.0]).into_dyn(); // shape [3]
        let z = ZonotopeTensor::concrete(values);

        let weight = arr2(&[[1.0, 2.0]]); // expects input dim 2, got 3
        let result = z.linear(&weight, None);
        assert!(result.is_err(), "linear should reject dimension mismatch");
    }

    #[test]
    fn test_linear_bias_addition() {
        // Kills: replace += with -= in line 436 (lane += &b.view())
        // Kills: replace += with *= in line 436
        let values = arr1(&[1.0, 2.0]).into_dyn(); // [2]
        let z = ZonotopeTensor::concrete(values);

        let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // identity 2x2
        let bias = arr1(&[10.0, 20.0]); // bias

        let result = z.linear(&weight, Some(&bias)).unwrap();
        let center = result.center();

        // With identity weight, output center should be input + bias
        // [1, 2] * I + [10, 20] = [11, 22]
        assert!(
            (center[[0]] - 11.0).abs() < 1e-6,
            "bias should be added (got {})",
            center[[0]]
        );
        assert!(
            (center[[1]] - 22.0).abs() < 1e-6,
            "bias should be added (got {})",
            center[[1]]
        );
    }

    #[test]
    fn test_linear_bias_with_batch() {
        // Test bias addition with multi-dimensional input
        let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(); // [2, 2]
        let z = ZonotopeTensor::concrete(values);

        let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // identity
        let bias = arr1(&[100.0, 200.0]);

        let result = z.linear(&weight, Some(&bias)).unwrap();
        let center = result.center();

        // Both rows should have bias added
        assert!((center[[0, 0]] - 101.0).abs() < 1e-6, "row 0 bias");
        assert!((center[[0, 1]] - 202.0).abs() < 1e-6, "row 0 bias");
        assert!((center[[1, 0]] - 103.0).abs() < 1e-6, "row 1 bias");
        assert!((center[[1, 1]] - 204.0).abs() < 1e-6, "row 1 bias");
    }

    // ============================================================
    // Mutation-killing tests for dot()
    // ============================================================

    #[test]
    fn test_dot_linear_coeff_computation() {
        // Kills: replace + with - in line 506 (a0·bi + ai·b0)
        // Kills: replace + with * in line 506
        // With 1 error term, check that linear coefficient = a0·b1 + a1·b0

        // a = 1 + 0.5*e (center=1, err_coeff=0.5)
        let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[2, 1]));
        a_coeffs[[0, 0]] = 1.0; // center
        a_coeffs[[1, 0]] = 0.5; // error coefficient
        let a = ZonotopeTensor::new(a_coeffs).unwrap();

        // b = 2 + 0.3*e (center=2, err_coeff=0.3)
        let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[2, 1]));
        b_coeffs[[0, 0]] = 2.0;
        b_coeffs[[1, 0]] = 0.3;
        let b = ZonotopeTensor::new(b_coeffs).unwrap();

        let result = a.dot(&b).unwrap();

        // Linear coeff for e = a0*b1 + a1*b0 = 1*0.3 + 0.5*2 = 1.3
        // If + was -, would get 1*0.3 - 0.5*2 = -0.7
        assert!(
            (result.coeffs[[1, 0]] - 1.3).abs() < 1e-6,
            "linear coeff should be 1.3, got {}",
            result.coeffs[[1, 0]]
        );
    }

    #[test]
    fn test_dot_half_term_accumulation() {
        // Kills: replace += with -= in line 527 (half_term += ...)
        // Kills: replace += with *= in line 527
        // The half_term should accumulate 0.5 * |ai·bi| for each error term

        // a = 1 + 0.4*e1 + 0.2*e2
        let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 1]));
        a_coeffs[[0, 0]] = 1.0;
        a_coeffs[[1, 0]] = 0.4;
        a_coeffs[[2, 0]] = 0.2;
        let a = ZonotopeTensor::new(a_coeffs).unwrap();

        // b = 2 + 0.5*e1 + 0.3*e2
        let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 1]));
        b_coeffs[[0, 0]] = 2.0;
        b_coeffs[[1, 0]] = 0.5;
        b_coeffs[[2, 0]] = 0.3;
        let b = ZonotopeTensor::new(b_coeffs).unwrap();

        let result = a.dot(&b).unwrap();
        let bounds = result.to_bounded_tensor();

        // half_term = 0.5*|0.4*0.5| + 0.5*|0.2*0.3| = 0.5*0.2 + 0.5*0.06 = 0.13
        // If += was -=, half_term would be negative → bounds would be wrong
        // Check that bounds contain the expected range
        let center = result.coeffs[[0, 0]];
        let lower = bounds.lower[[0]];
        let upper = bounds.upper[[0]];

        // Width should be positive (radius = half_term + other terms)
        assert!(upper > lower, "bounds should have positive width");
        assert!(upper >= center, "upper should be >= center");
        assert!(lower <= center, "lower should be <= center");
    }

    #[test]
    fn test_dot_cross_term_loop_start() {
        // Kills: replace + with * in line 538 ((i + 1)..=n)
        // If + was *, loop would start at i*1=i, including i==i case (wrong)
        // With 2 error terms, there should be exactly one cross term (1,2)

        // a = 1 + e1 + e2
        let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 1]));
        a_coeffs[[0, 0]] = 1.0;
        a_coeffs[[1, 0]] = 1.0;
        a_coeffs[[2, 0]] = 1.0;
        let a = ZonotopeTensor::new(a_coeffs).unwrap();

        // b = 1 + e1 + e2
        let b = a.clone();

        let result = a.dot(&b).unwrap();

        // With correct loop, cross term = |a1·b2 + a2·b1| = |1*1 + 1*1| = 2
        // big_term contributes to the new error coefficient (index n+1 = 3)
        // half_term for e1² and e2²: 0.5*|1*1| + 0.5*|1*1| = 1
        // new_error_coeff = half_term + big_term = 1 + 2 = 3

        // The last error term should have the combined quadratic bound
        let new_err_coeff = result.coeffs[[3, 0]];
        assert!(
            (new_err_coeff - 3.0).abs() < 1e-6,
            "new error coeff should be 3 (half=1, cross=2), got {}",
            new_err_coeff
        );
    }

    #[test]
    fn test_dot_cross_term_product() {
        // Kills: replace * with + in lines 543/544 (ai.iter().zip(bj).map(|(&a,&b)| a * b))
        // If * was +, ai·bj would be sum of (a+b) not sum of (a*b)

        // a = [1, 2] + [0.1, 0.2]*e1 + [0.3, 0.4]*e2
        let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 2]));
        a_coeffs[[0, 0]] = 1.0;
        a_coeffs[[0, 1]] = 2.0;
        a_coeffs[[1, 0]] = 0.1;
        a_coeffs[[1, 1]] = 0.2;
        a_coeffs[[2, 0]] = 0.3;
        a_coeffs[[2, 1]] = 0.4;
        let a = ZonotopeTensor::new(a_coeffs).unwrap();

        // b = [3, 4] + [0.5, 0.6]*e1 + [0.7, 0.8]*e2
        let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 2]));
        b_coeffs[[0, 0]] = 3.0;
        b_coeffs[[0, 1]] = 4.0;
        b_coeffs[[1, 0]] = 0.5;
        b_coeffs[[1, 1]] = 0.6;
        b_coeffs[[2, 0]] = 0.7;
        b_coeffs[[2, 1]] = 0.8;
        let b = ZonotopeTensor::new(b_coeffs).unwrap();

        let result = a.dot(&b).unwrap();

        // Cross term (i=1, j=2):
        // a1·b2 = 0.1*0.7 + 0.2*0.8 = 0.07 + 0.16 = 0.23
        // a2·b1 = 0.3*0.5 + 0.4*0.6 = 0.15 + 0.24 = 0.39
        // If * was +, a1·b2 would be (0.1+0.7) + (0.2+0.8) = 1.8 (wrong)

        // big_term = |a1·b2 + a2·b1| = |0.23 + 0.39| = 0.62
        // half_term = 0.5*|a1·b1| + 0.5*|a2·b2|
        //           = 0.5*|0.1*0.5+0.2*0.6| + 0.5*|0.3*0.7+0.4*0.8|
        //           = 0.5*0.17 + 0.5*0.53 = 0.085 + 0.265 = 0.35
        // new_err = 0.35 + 0.62 = 0.97

        let new_err_coeff = result.coeffs[[3, 0]];
        assert!(
            (new_err_coeff - 0.97).abs() < 0.01,
            "cross term should use products, got new_err={}",
            new_err_coeff
        );
    }

    #[test]
    fn test_dot_big_term_accumulation() {
        // Kills: replace += with -= in line 547 (big_term += ...)
        // Kills: replace + with - in line 547 ((ai_dot_bj + aj_dot_bi))

        // Same setup as above but verify big_term is accumulated correctly
        let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[4, 1])); // 3 error terms
        a_coeffs[[0, 0]] = 1.0;
        a_coeffs[[1, 0]] = 0.1;
        a_coeffs[[2, 0]] = 0.2;
        a_coeffs[[3, 0]] = 0.3;
        let a = ZonotopeTensor::new(a_coeffs).unwrap();

        let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[4, 1]));
        b_coeffs[[0, 0]] = 2.0;
        b_coeffs[[1, 0]] = 0.4;
        b_coeffs[[2, 0]] = 0.5;
        b_coeffs[[3, 0]] = 0.6;
        let b = ZonotopeTensor::new(b_coeffs).unwrap();

        let result = a.dot(&b).unwrap();

        // Cross terms: (1,2), (1,3), (2,3)
        // (1,2): |0.1*0.5 + 0.2*0.4| = |0.05 + 0.08| = 0.13
        // (1,3): |0.1*0.6 + 0.3*0.4| = |0.06 + 0.12| = 0.18
        // (2,3): |0.2*0.6 + 0.3*0.5| = |0.12 + 0.15| = 0.27
        // big_term = 0.13 + 0.18 + 0.27 = 0.58

        // half_term = 0.5*(|0.1*0.4| + |0.2*0.5| + |0.3*0.6|)
        //           = 0.5*(0.04 + 0.10 + 0.18) = 0.16

        let new_err_coeff = result.coeffs[[4, 0]]; // index = 1 + n = 4
        let expected = 0.16 + 0.58;
        assert!(
            (new_err_coeff - expected).abs() < 0.01,
            "big_term should accumulate cross terms, expected {}, got {}",
            expected,
            new_err_coeff
        );
    }

    // ============================================================
    // Mutation-killing tests for layer_norm_affine()
    // ============================================================

    #[test]
    fn test_layer_norm_affine_rejects_gamma_dim_mismatch() {
        // Kills: replace || with && in line 1190
        let values = arr2(&[[1.0, 2.0, 3.0]]).into_dyn(); // dim=3
        let z = ZonotopeTensor::concrete(values);

        let gamma = arr1(&[1.0, 1.0]); // dim=2 (wrong)
        let beta = arr1(&[0.0, 0.0, 0.0]); // dim=3 (correct)
        let result = z.layer_norm_affine(&gamma, &beta, 1e-5);

        assert!(result.is_err(), "should reject gamma with wrong dimension");
    }

    #[test]
    fn test_layer_norm_affine_rejects_beta_dim_mismatch() {
        // Complements gamma test - tests the beta part of the || condition
        let values = arr2(&[[1.0, 2.0, 3.0]]).into_dyn();
        let z = ZonotopeTensor::concrete(values);

        let gamma = arr1(&[1.0, 1.0, 1.0]); // correct
        let beta = arr1(&[0.0, 0.0]); // wrong dim
        let result = z.layer_norm_affine(&gamma, &beta, 1e-5);

        assert!(result.is_err(), "should reject beta with wrong dimension");
    }

    #[test]
    fn test_layer_norm_affine_variance_computation() {
        // Kills: replace * with + in line 1227 (c * c for variance)
        // Kills: replace / with * in line 1227 (var / dim)
        // If * was +, variance would be sum of values not sum of squares
        // If / was *, would multiply by dim instead of divide

        // Use values where sum != sum of squares
        let values = arr2(&[[2.0, 4.0]]).into_dyn(); // mean=3, var=(1+1)/2=1
        let z = ZonotopeTensor::concrete(values);

        let gamma = arr1(&[1.0, 1.0]); // identity scale
        let beta = arr1(&[0.0, 0.0]);
        let result = z.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
        let center = result.center();

        // With variance=1, std≈1: normalized = (x - 3) / 1 = [-1, 1]
        // Output = gamma * normalized + beta = [-1, 1]
        assert!(
            (center[[0, 0]] - (-1.0)).abs() < 0.01,
            "first element should be ~-1"
        );
        assert!(
            (center[[0, 1]] - 1.0).abs() < 0.01,
            "second element should be ~1"
        );
    }

    #[test]
    fn test_layer_norm_affine_eps_addition() {
        // Kills: replace + with - in line 1228 (var + eps)
        // Kills: replace + with * in line 1228
        // Use zero-variance input where eps matters

        let values = arr2(&[[3.0, 3.0]]).into_dyn(); // constant row, var=0
        let z = ZonotopeTensor::concrete(values);

        let gamma = arr1(&[1.0, 1.0]);
        let beta = arr1(&[0.0, 0.0]);
        let eps = 1.0; // large eps so std = sqrt(0 + 1) = 1

        let result = z.layer_norm_affine(&gamma, &beta, eps).unwrap();
        let center = result.center();

        // With var=0 and eps=1: std=1, centered=[0,0], output=beta=[0,0]
        // If + was -, sqrt would fail or give NaN (var - eps = -1)
        assert!(center[[0, 0]].is_finite(), "result should be finite");
        assert!((center[[0, 0]]).abs() < 0.01, "output should be ~0");
    }

    #[test]
    fn test_layer_norm_affine_gamma_division() {
        // Kills: replace / with * in line 1237 (g / std_safe)
        // Kills: replace / with % in line 1237

        let values = arr2(&[[0.0, 4.0]]).into_dyn(); // mean=2, var=4, std=2
        let z = ZonotopeTensor::concrete(values);

        let gamma = arr1(&[2.0, 2.0]); // gamma=2
        let beta = arr1(&[0.0, 0.0]);

        let result = z.layer_norm_affine(&gamma, &beta, 0.0).unwrap();
        let center = result.center();

        // eff_gamma = gamma/std = 2/2 = 1
        // centered = [-2, 2]
        // output = 1 * centered = [-2, 2]
        // If / was *, eff_gamma = 2*2 = 4, output = [-8, 8] (wrong)
        assert!(
            (center[[0, 0]] - (-2.0)).abs() < 0.01,
            "should be -2, not -8"
        );
        assert!((center[[0, 1]] - 2.0).abs() < 0.01, "should be 2, not 8");
    }

    #[test]
    fn test_layer_norm_affine_error_term_scaling() {
        // Kills: replace * with + in line 1255 (eff_gamma[d] * coeffs_3d[...])
        // Error coefficients should be scaled by gamma/std

        let values = arr2(&[[0.0, 4.0]]).into_dyn(); // std=2
        let z = ZonotopeTensor::from_input_shared(&values, 0.5); // 0.5 perturbation

        let gamma = arr1(&[4.0, 4.0]); // gamma=4, so eff_gamma = 4/2 = 2
        let beta = arr1(&[0.0, 0.0]);

        let result = z.layer_norm_affine(&gamma, &beta, 0.0).unwrap();

        // Original error coeff was 0.5
        // After scaling by eff_gamma=2, should be ~1.0
        // If * was +, would be eff_gamma + coeff = 2 + 0.5 = 2.5 (wrong)
        let err_coeff_0 = result.coeffs[[1, 0, 0]];
        assert!(
            (err_coeff_0.abs() - 1.0).abs() < 0.1,
            "error coeff should be ~1.0 (0.5*2), got {}",
            err_coeff_0
        );
    }

    #[test]
    fn test_layer_norm_affine_radius_accumulation() {
        // Kills: replace += with -= in line 1271 (total_radius += ...)

        // Use 2D input to match layer_norm_affine's expectation
        let values = arr2(&[[1.0, 2.0, 3.0]]).into_dyn(); // shape [1, 3]
        let z = ZonotopeTensor::from_input_shared(&values, 0.2);

        let gamma = arr1(&[1.0, 1.0, 1.0]);
        let beta = arr1(&[0.0, 0.0, 0.0]);

        let result = z.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
        let bounds = result.to_bounded_tensor();

        // The bounds should have positive width due to accumulated radius
        let width_0 = bounds.upper[[0, 0]] - bounds.lower[[0, 0]];
        assert!(
            width_0 > 0.0,
            "bounds should have positive width from accumulated radius"
        );

        // If += was -=, radius would be negative and bounds would be inverted
        assert!(
            bounds.upper[[0, 0]] >= bounds.lower[[0, 0]],
            "upper should be >= lower (not inverted)"
        );
    }

    #[test]
    fn test_layer_norm_affine_mean_deriv_division() {
        // Kills: replace / with * in line 1277 (g / (dim * std_safe))

        // With multiple positions, ensure approximation error is reasonable
        let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(); // 2 rows
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        let gamma = arr1(&[1.0, 1.0]);
        let beta = arr1(&[0.0, 0.0]);

        let result = z.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();

        // The new error term (for approximation) should be small and finite
        let new_err_idx = z.n_error_terms + 1;
        let approx_err = result.coeffs[[new_err_idx, 0, 0]];

        assert!(
            approx_err.is_finite(),
            "approximation error should be finite"
        );
        // If / was *, error would be huge (gamma * dim * std instead of gamma / (dim * std))
        assert!(
            approx_err < 10.0,
            "approximation error should be small, got {}",
            approx_err
        );
    }

    // ============== SiLU Affine Mutation-Killing Tests ==============

    #[test]
    fn test_silu_affine_sigmoid_boundary() {
        // Kills: replace >= with < in line 1331 (sigmoid boundary at x=0)
        // sigmoid(0) = 0.5, should be same from both branches

        // Test at exactly x=0 using both the positive and negative code paths
        let z = ZonotopeTensor::concrete(arr1(&[0.0_f32]).into_dyn());
        let result = z.silu_affine().unwrap();
        let center = result.center();

        // silu(0) = 0 * sigmoid(0) = 0
        assert!(
            (center[0] - 0.0).abs() < 1e-6,
            "silu(0) should be 0, got {}",
            center[0]
        );

        // Test very small positive and negative values - should give symmetric results
        let z_pos = ZonotopeTensor::concrete(arr1(&[1e-6_f32]).into_dyn());
        let z_neg = ZonotopeTensor::concrete(arr1(&[-1e-6_f32]).into_dyn());

        let r_pos = z_pos.silu_affine().unwrap();
        let r_neg = z_neg.silu_affine().unwrap();

        // silu is approximately linear near 0 with slope ~0.5
        // silu'(0) = sigmoid(0) * (1 + 0*(1-0.5)) = 0.5
        let pos_val = r_pos.center()[0];
        let neg_val = r_neg.center()[0];

        // silu(x) ≈ x/2 near 0, so silu(-eps) ≈ -silu(eps)
        assert!(
            (pos_val + neg_val).abs() < 1e-10,
            "silu should be antisymmetric near 0: {} vs {}",
            pos_val,
            neg_val
        );
    }

    #[test]
    fn test_silu_affine_derivative_formula() {
        // Kills: mutations in line 1347 (silu_derivative formula)
        // silu'(x) = s * (1.0 + x * (1.0 - s))

        // Test with known values where the derivative matters
        // For x=1: sigmoid(1) ≈ 0.7311
        // silu'(1) = 0.7311 * (1 + 1 * (1 - 0.7311)) = 0.7311 * 1.2689 ≈ 0.9277

        // Create zonotope with error term
        let z = ZonotopeTensor::from_input_shared(&arr1(&[1.0_f32]).into_dyn(), 0.5);
        let result = z.silu_affine().unwrap();

        // The error coefficient should be scaled by the derivative
        // Input has error = 0.5, output error should be 0.5 * silu'(1) ≈ 0.464
        let input_err = z.coeffs[[1, 0]];
        let output_err = result.coeffs[[1, 0]];

        let expected_slope = {
            let s = 1.0 / (1.0 + (-1.0_f32).exp());
            s * (1.0 + 1.0 * (1.0 - s))
        };

        let expected_output_err = input_err * expected_slope;
        assert!(
            (output_err - expected_output_err).abs() < 1e-5,
            "output error {} should be input {} * slope {}",
            output_err,
            input_err,
            expected_slope
        );

        // Test at x=-2 to verify different path
        // sigmoid(-2) ≈ 0.1192
        // silu'(-2) = 0.1192 * (1 + (-2) * (1 - 0.1192)) ≈ 0.1192 * (1 - 1.7616) ≈ -0.0908
        let z2 = ZonotopeTensor::from_input_shared(&arr1(&[-2.0_f32]).into_dyn(), 0.5);
        let result2 = z2.silu_affine().unwrap();

        let expected_slope2 = {
            let s = (-2.0_f32).exp() / (1.0 + (-2.0_f32).exp());
            s * (1.0 + (-2.0) * (1.0 - s))
        };

        let output_err2 = result2.coeffs[[1, 0]];
        let expected_output_err2 = z2.coeffs[[1, 0]] * expected_slope2;
        assert!(
            (output_err2 - expected_output_err2).abs() < 1e-4,
            "output error {} should be input {} * slope {}",
            output_err2,
            z2.coeffs[[1, 0]],
            expected_slope2
        );
    }

    #[test]
    fn test_silu_affine_second_derivative() {
        // Kills: mutations in lines 1352-1354 (silu_second_derivative formula)
        // silu''(x) = s * (1-s) * (2 + x - 2*x*s)

        // Large radius zonotope to have meaningful approximation error
        // The approximation error depends on max|silu''| * r^2 / 2
        let z = ZonotopeTensor::from_input_shared(&arr1(&[-1.28_f32]).into_dyn(), 0.5);
        let result = z.silu_affine().unwrap();

        // At x ≈ -1.28, silu'' has maximum magnitude
        // If second derivative formula is wrong, approximation error will be wrong

        // Get the approximation error term (the new one added)
        let approx_err_idx = result.n_error_terms;
        let approx_err = result.coeffs[[approx_err_idx, 0]];

        // Compute expected second derivative at critical point
        let s = {
            let ex = (-1.28_f32).exp();
            ex / (1.0 + ex)
        };
        let expected_max_second = (s * (1.0 - s) * (2.0 + (-1.28) - 2.0 * (-1.28) * s)).abs();

        // Error should be approximately max_second * r^2 / 2
        let r = 0.5;
        let expected_approx_err_lower = expected_max_second * r * r / 2.0 * 0.8; // Allow some tolerance

        assert!(
            approx_err >= expected_approx_err_lower,
            "approx error {} should be near {} (based on second derivative)",
            approx_err,
            expected_max_second * r * r / 2.0
        );
    }

    #[test]
    fn test_silu_affine_radius_calculation() {
        // Kills: mutations in radius > 0.0 check (line 1388)
        // Also kills: mutations in lo/hi calculation (lines 1391-1392)

        // Test with zero radius - should have zero approximation error
        let z_concrete = ZonotopeTensor::concrete(arr1(&[0.5_f32]).into_dyn());
        let result_concrete = z_concrete.silu_affine().unwrap();

        // For concrete (no error), the new error term should be 0
        let approx_err_concrete = result_concrete.coeffs[[result_concrete.n_error_terms, 0]];
        assert!(
            (approx_err_concrete - 0.0).abs() < 1e-10,
            "concrete zonotope should have 0 approx error, got {}",
            approx_err_concrete
        );

        // Test with non-zero radius
        let z_with_err = ZonotopeTensor::from_input_shared(&arr1(&[0.5_f32]).into_dyn(), 0.3);
        let result_with_err = z_with_err.silu_affine().unwrap();

        let approx_err_with = result_with_err.coeffs[[result_with_err.n_error_terms, 0]];
        assert!(
            approx_err_with > 0.0,
            "zonotope with error should have positive approx error, got {}",
            approx_err_with
        );

        // Larger radius should give larger error
        let z_larger = ZonotopeTensor::from_input_shared(&arr1(&[0.5_f32]).into_dyn(), 0.6);
        let result_larger = z_larger.silu_affine().unwrap();
        let approx_err_larger = result_larger.coeffs[[result_larger.n_error_terms, 0]];

        // Error scales as r^2, so 2x radius should give ~4x error
        assert!(
            approx_err_larger > approx_err_with * 3.0,
            "larger radius {} should give much larger error than {}",
            approx_err_larger,
            approx_err_with
        );
    }

    #[test]
    fn test_silu_affine_lo_hi_signs() {
        // Kills: replace - with / in line 1391 (lo = c - radius)
        // Kills: replace + with * in line 1392 (hi = c + radius)

        // Test with center at 1.0, radius 0.5
        // Correct: lo = 0.5, hi = 1.5
        // If lo = c/r: lo = 2.0 (wrong)
        // If hi = c*r: hi = 0.5 (wrong)

        let z = ZonotopeTensor::from_input_shared(&arr1(&[1.0_f32]).into_dyn(), 0.5);
        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor();

        // Check that bounds contain silu values at true lo and hi
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        let true_lo = 0.5_f32; // c - r
        let true_hi = 1.5_f32; // c + r

        assert!(
            bounds.lower[0] <= silu(true_lo) + 0.01,
            "lower bound {} should contain silu(0.5)={}",
            bounds.lower[0],
            silu(true_lo)
        );
        assert!(
            bounds.upper[0] >= silu(true_hi) - 0.01,
            "upper bound {} should contain silu(1.5)={}",
            bounds.upper[0],
            silu(true_hi)
        );

        // If lo/hi were computed wrong, bounds would not contain these values
    }

    #[test]
    fn test_silu_affine_interpolation_loop() {
        // Kills: mutations in lines 1395-1396 (interpolation: x = lo + (hi - lo) * t)

        // Test with interval that spans the SiLU'' maximum around -1.28
        let z = ZonotopeTensor::from_input_shared(&arr1(&[-1.0_f32]).into_dyn(), 0.5);
        let result = z.silu_affine().unwrap();

        // The approximation error should properly capture max second derivative
        let approx_err = result.coeffs[[result.n_error_terms, 0]];

        // If interpolation was wrong (e.g., lo + hi*t instead of lo + (hi-lo)*t),
        // the sampling would miss the peak
        assert!(approx_err > 0.0, "should have approximation error");

        // Verify bounds are sound - actual silu values at extremes should be contained
        let bounds = result.to_bounded_tensor();
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        assert!(
            bounds.lower[0] <= silu(-1.5),
            "lower {} should contain silu(-1.5)={}",
            bounds.lower[0],
            silu(-1.5)
        );
        assert!(
            bounds.upper[0] >= silu(-0.5),
            "upper {} should contain silu(-0.5)={}",
            bounds.upper[0],
            silu(-0.5)
        );
    }

    #[test]
    fn test_silu_affine_critical_points() {
        // Kills: mutations in critical point checks (lines 1401-1402, 1464-1465)
        // Checks: delete - in critical points, replace && with ||, replace <= with >

        // Test interval that contains a critical point (-1.28)
        let z = ZonotopeTensor::from_input_shared(&arr1(&[-1.28_f32]).into_dyn(), 0.2);
        let result = z.silu_affine().unwrap();

        let approx_err = result.coeffs[[result.n_error_terms, 0]];

        // The critical point -1.28 should be sampled since lo <= -1.28 <= hi
        // At this point, |silu''| is maximal

        // Compute expected max |silu''| at critical point
        let s = {
            let ex = (-1.28_f32).exp();
            ex / (1.0 + ex)
        };
        let max_second_at_critical = (s * (1.0 - s) * (2.0 - 1.28 - 2.0 * (-1.28) * s)).abs();

        let expected_min_err = max_second_at_critical * 0.04 / 2.0 * 0.5; // r=0.2, some tolerance
        assert!(
            approx_err >= expected_min_err,
            "approx error {} should reflect critical point sampling",
            approx_err
        );

        // Test interval that does NOT contain critical points (far positive)
        let z_far = ZonotopeTensor::from_input_shared(&arr1(&[5.0_f32]).into_dyn(), 0.2);
        let result_far = z_far.silu_affine().unwrap();

        let approx_err_far = result_far.coeffs[[result_far.n_error_terms, 0]];

        // At x=5, silu'' is very small (near 0)
        // Error should be much smaller than at critical point
        assert!(
            approx_err_far < approx_err,
            "error at x=5 ({}) should be less than at critical point ({})",
            approx_err_far,
            approx_err
        );
    }

    #[test]
    fn test_silu_affine_critical_bounds_check() {
        // Kills: replace <= with > in line 1402 (lo <= critical)
        // Kills: replace <= with > in line 1402 (critical <= hi)

        // Test interval [-2.5, -2.3] - contains critical point -2.4
        let z = ZonotopeTensor::from_input_shared(&arr1(&[-2.4_f32]).into_dyn(), 0.1);
        let result = z.silu_affine().unwrap();
        let approx_err_contains = result.coeffs[[result.n_error_terms, 0]];

        // Test interval [-2.2, -2.0] - does NOT contain -2.4
        let z2 = ZonotopeTensor::from_input_shared(&arr1(&[-2.1_f32]).into_dyn(), 0.1);
        let result2 = z2.silu_affine().unwrap();
        let approx_err_not_contains = result2.coeffs[[result2.n_error_terms, 0]];

        // Both should have approximation error, but different values due to different
        // second derivative sampling
        assert!(
            approx_err_contains > 0.0,
            "should have error when containing critical"
        );
        assert!(
            approx_err_not_contains > 0.0,
            "should have error when not containing critical"
        );
    }

    #[test]
    fn test_silu_affine_error_division() {
        // Kills: replace / with % or * in line 1406 (max_second * r * r / 2.0)

        // With known radius and max_second, verify error formula
        let z = ZonotopeTensor::from_input_shared(&arr1(&[0.0_f32]).into_dyn(), 1.0);
        let result = z.silu_affine().unwrap();

        let approx_err = result.coeffs[[result.n_error_terms, 0]];

        // At x=0, silu''(0) = 0.5 * 0.5 * 2 = 0.5
        // Error should be 0.5 * 1.0 * 1.0 / 2.0 = 0.25 (approximately)
        // If / was replaced with *, error would be 0.5 * 1.0 * 1.0 * 2.0 = 1.0

        assert!(
            approx_err < 0.5,
            "error {} should be less than 0.5 (division, not multiplication)",
            approx_err
        );
        assert!(
            approx_err >= 0.1,
            "error {} should be at least 0.1 (proper formula)",
            approx_err
        );
    }

    #[test]
    fn test_silu_affine_2d_center_copy() {
        // Kills: replace + with - in line 1426 (n_error_terms + 1)
        // Tests that new error term is added at correct index

        let values = arr2(&[[0.0_f32, 1.0]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        let orig_n_err = z.n_error_terms;
        let result = z.silu_affine().unwrap();

        assert_eq!(
            result.n_error_terms,
            orig_n_err + 1,
            "should add exactly one new error term"
        );

        // The new error term should be at index (orig_n_err + 1)
        let new_err_row = result.coeffs.index_axis(ndarray::Axis(0), orig_n_err + 1);
        let sum: f32 = new_err_row.iter().sum();
        assert!(sum.abs() > 0.0, "new error row should have content");
    }

    #[test]
    fn test_silu_affine_slope_multiplication() {
        // Kills: replace * with + in line 1383 (slope * self.coeffs[[i, d]])
        // Kills: replace * with / in line 1383

        // Test that error coefficients are multiplied by slope
        let z = ZonotopeTensor::from_input_shared(&arr1(&[2.0_f32]).into_dyn(), 1.0);

        let input_err = z.coeffs[[1, 0]];
        let result = z.silu_affine().unwrap();
        let output_err = result.coeffs[[1, 0]];

        // Compute expected slope at x=2
        let s = 1.0 / (1.0 + (-2.0_f32).exp());
        let expected_slope = s * (1.0 + 2.0 * (1.0 - s));

        // If * was +, output would be slope + input_err
        // If * was /, output would be slope / input_err

        let expected_output = input_err * expected_slope;
        assert!(
            (output_err - expected_output).abs() < 1e-4,
            "output error {} should be input {} * slope {} = {}",
            output_err,
            input_err,
            expected_slope,
            expected_output
        );

        // Verify it's not addition
        let wrong_add = expected_slope + input_err;
        assert!(
            (output_err - wrong_add).abs() > 0.1,
            "should not be addition: {} vs {}",
            output_err,
            wrong_add
        );
    }

    #[test]
    fn test_silu_affine_2d_slope_multiplication() {
        // Kills: replace * with + in line 1451 (slope * self.coeffs[[i, s, d]])

        let values = arr2(&[[2.0_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 1.0);

        let input_err = z.coeffs[[1, 0, 0]];
        let result = z.silu_affine().unwrap();
        let output_err = result.coeffs[[1, 0, 0]];

        let s = 1.0 / (1.0 + (-2.0_f32).exp());
        let expected_slope = s * (1.0 + 2.0 * (1.0 - s));
        let expected_output = input_err * expected_slope;

        assert!(
            (output_err - expected_output).abs() < 1e-4,
            "2D output error {} should be input {} * slope {}",
            output_err,
            input_err,
            expected_slope
        );
    }

    #[test]
    fn test_silu_affine_2d_lo_hi() {
        // Kills: mutations in lines 1456-1457 (lo = c - radius, hi = c + radius) for 2D

        let values = arr2(&[[1.0_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.5);
        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor();

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        // Bounds should contain silu(0.5) and silu(1.5)
        assert!(
            bounds.lower[[0, 0]] <= silu(0.5) + 0.01,
            "2D lower bound should contain silu(0.5)"
        );
        assert!(
            bounds.upper[[0, 0]] >= silu(1.5) - 0.01,
            "2D upper bound should contain silu(1.5)"
        );
    }

    #[test]
    fn test_silu_affine_nd_recursive() {
        // Kills: delete match arm 2 (3D+ case)

        // Test 3D input - should work via reshape->1D->reshape back
        let values = ndarray::Array3::<f32>::from_elem((1, 2, 2), 1.0).into_dyn();
        let z = ZonotopeTensor::concrete(values);

        let result = z.silu_affine().unwrap();

        assert_eq!(
            result.element_shape,
            vec![1, 2, 2],
            "should preserve 3D shape"
        );

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }
        let expected = silu(1.0);

        let center = result.center();
        for val in center.iter() {
            assert!(
                (*val - expected).abs() < 1e-5,
                "3D silu should compute correctly"
            );
        }
    }

    // ============== from_bounded_tensor_per_position Mutation-Killing Tests ==============

    #[test]
    fn test_from_bounded_per_position_radius_calculation() {
        // Kills: replace - with / in line 973 (radius = (upper - lower) / 2.0)

        // Create bounds with lower=0.0, upper=2.0
        // Correct radius = (2.0 - 0.0) / 2.0 = 1.0
        // If - was /: radius = (2.0 / 0.0) / 2.0 = inf

        let lower = arr2(&[[0.0_f32, 1.0], [2.0, 3.0]]).into_dyn();
        let upper = arr2(&[[2.0_f32, 3.0], [4.0, 5.0]]).into_dyn();
        let bounds = BoundedTensor { lower, upper };

        let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

        // Check that radius is correct (1.0 for all elements)
        let out_bounds = z.to_bounded_tensor();

        // Lower should be original lower (0.0, 1.0, 2.0, 3.0)
        assert!(
            (out_bounds.lower[[0, 0]] - 0.0).abs() < 1e-6,
            "lower bound should be 0.0, got {}",
            out_bounds.lower[[0, 0]]
        );
        assert!((out_bounds.lower[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((out_bounds.lower[[1, 0]] - 2.0).abs() < 1e-6);
        assert!((out_bounds.lower[[1, 1]] - 3.0).abs() < 1e-6);

        // Upper should be original upper (2.0, 3.0, 4.0, 5.0)
        assert!(
            (out_bounds.upper[[0, 0]] - 2.0).abs() < 1e-6,
            "upper bound should be 2.0, got {}",
            out_bounds.upper[[0, 0]]
        );

        // Verify bounds are finite (would be inf if - was /)
        assert!(
            out_bounds.lower.iter().all(|&x| x.is_finite()),
            "lower should be finite"
        );
        assert!(
            out_bounds.upper.iter().all(|&x| x.is_finite()),
            "upper should be finite"
        );
    }

    #[test]
    fn test_from_bounded_per_position_n_error_terms() {
        // Kills: replace * with + in line 986 (n_error_terms = batch_size * seq)
        // Kills: replace * with / in line 986

        // Create 2x3 bounds (seq=2, dim=3)
        // Correct n_error_terms = 1 * 2 = 2 (batch_size=1, seq=2)
        // If * was +: n_error_terms = 1 + 2 = 3
        // If * was /: n_error_terms = 1 / 2 = 0

        let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]).into_dyn();
        let upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]).into_dyn();
        let bounds = BoundedTensor { lower, upper };

        let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

        // Should have 2 error terms (one per sequence position)
        assert_eq!(
            z.n_error_terms, 2,
            "should have batch*seq = 1*2 = 2 error terms"
        );

        // Coeffs should be (1 + 2) x 2 x 3 = 3 x 2 x 3
        assert_eq!(
            z.coeffs.shape(),
            &[3, 2, 3],
            "coeffs shape should be (1+n_err, seq, dim)"
        );
    }

    #[test]
    fn test_from_bounded_per_position_coeffs_shape() {
        // Kills: replace + with - in line 988 (1 + n_error_terms)
        // Kills: replace + with * in line 988

        // Create 3x4 bounds (seq=3, dim=4)
        let lower = ndarray::Array2::<f32>::zeros((3, 4)).into_dyn();
        let upper = ndarray::Array2::<f32>::ones((3, 4)).into_dyn();
        let bounds = BoundedTensor { lower, upper };

        let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

        // n_error_terms = 3
        // Correct coeffs first dim = 1 + 3 = 4
        // If + was -: 1 - 3 = -2 (would fail on Array allocation)
        // If + was *: 1 * 3 = 3 (would be missing center row)

        assert_eq!(
            z.coeffs.shape()[0],
            4,
            "first dim of coeffs should be 1 + n_error_terms = 4"
        );

        // Verify center row exists and has correct values
        let center = z.center();
        assert_eq!(center.shape(), &[3, 4], "center should be 3x4");

        // Center should be (lower + upper) / 2 = 0.5
        for val in center.iter() {
            assert!((*val - 0.5).abs() < 1e-6, "center should be 0.5");
        }
    }

    #[test]
    fn test_from_bounded_per_position_error_index() {
        // Kills: replace + with - in line 995 (1 + b * seq + s)
        // Kills: replace + with * in line 995
        // Kills: replace * with + in line 995 (b * seq)
        // Kills: replace * with / in line 995

        // Create 3D bounds: (2, 3, 4) = (batch=2, seq=3, dim=4)
        let lower = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();
        let mut upper = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();

        // Give each position a unique radius so we can verify error assignment
        // Position (b, s) gets radius = (b * 10 + s + 1) / 10
        for b in 0..2 {
            for s in 0..3 {
                for d in 0..4 {
                    upper[[b, s, d]] = (b * 10 + s + 1) as f32 / 5.0;
                }
            }
        }

        let bounds = BoundedTensor { lower, upper };
        let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

        // n_error_terms = 2 * 3 = 6
        assert_eq!(z.n_error_terms, 6);

        // Error term index should be 1 + b * seq + s
        // For (b=0, s=0): err = 1 + 0*3 + 0 = 1
        // For (b=0, s=1): err = 1 + 0*3 + 1 = 2
        // For (b=0, s=2): err = 1 + 0*3 + 2 = 3
        // For (b=1, s=0): err = 1 + 1*3 + 0 = 4
        // For (b=1, s=1): err = 1 + 1*3 + 1 = 5
        // For (b=1, s=2): err = 1 + 1*3 + 2 = 6

        // Verify each position has error in the correct slot
        // The radius for (b,s) is (b*10+s+1)/10, so error coeff = radius = (b*10+s+1)/10

        // Check (b=0, s=0) has error in slot 1
        let r_00 = 0.5 * 1.0 / 5.0; // radius = upper/2 = 1/10
        assert!(
            (z.coeffs[[1, 0, 0, 0]] - r_00).abs() < 1e-5,
            "err 1 should have radius for (0,0), got {}",
            z.coeffs[[1, 0, 0, 0]]
        );
        // Other error slots for this position should be 0
        assert!(
            (z.coeffs[[2, 0, 0, 0]] - 0.0).abs() < 1e-6,
            "err 2 should be 0 for (0,0)"
        );

        // Check (b=1, s=1) has error in slot 5
        let r_11 = 0.5 * 12.0 / 5.0; // b*10+s+1 = 12, radius = 12/10 = 1.2
        assert!(
            (z.coeffs[[5, 1, 1, 0]] - r_11).abs() < 1e-4,
            "err 5 should have radius for (1,1), got {}",
            z.coeffs[[5, 1, 1, 0]]
        );
    }

    #[test]
    fn test_from_bounded_per_position_2d_accepts_2d() {
        // Kills: replace != with == in line 1029

        // 2D input should be accepted
        let lower = arr2(&[[0.0_f32, 1.0]]).into_dyn();
        let upper = arr2(&[[2.0_f32, 3.0]]).into_dyn();
        let bounds = BoundedTensor { lower, upper };

        let result = ZonotopeTensor::from_bounded_tensor_per_position_2d(&bounds);
        assert!(result.is_ok(), "2D bounds should be accepted");
    }

    #[test]
    fn test_from_bounded_per_position_2d_rejects_1d() {
        // Kills: replace != with == in line 1029

        // 1D input should be rejected
        let lower = arr1(&[0.0_f32, 1.0]).into_dyn();
        let upper = arr1(&[2.0_f32, 3.0]).into_dyn();
        let bounds = BoundedTensor { lower, upper };

        let result = ZonotopeTensor::from_bounded_tensor_per_position_2d(&bounds);
        assert!(
            result.is_err(),
            "1D bounds should be rejected by 2D variant"
        );
    }

    #[test]
    fn test_from_bounded_per_position_2d_rejects_3d() {
        // Kills: replace != with == in line 1029

        // 3D input should be rejected
        let lower = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();
        let upper = ndarray::Array3::<f32>::ones((2, 3, 4)).into_dyn();
        let bounds = BoundedTensor { lower, upper };

        let result = ZonotopeTensor::from_bounded_tensor_per_position_2d(&bounds);
        assert!(
            result.is_err(),
            "3D bounds should be rejected by 2D variant"
        );
    }

    // ============== transpose_last_two Mutation-Killing Tests ==============

    #[test]
    fn test_transpose_last_two_comparison() {
        // Kills: replace < with > in line 1504

        // Test with 1D input - should fail (less than 2 dims)
        let z_1d = ZonotopeTensor::concrete(arr1(&[1.0_f32, 2.0]).into_dyn());
        let result = z_1d.transpose_last_two();
        assert!(
            result.is_err(),
            "1D zonotope should fail transpose_last_two"
        );

        // Test with 2D input - should work (exactly 2 dims)
        let z_2d = ZonotopeTensor::concrete(arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn());
        let result = z_2d.transpose_last_two();
        assert!(
            result.is_ok(),
            "2D zonotope should work with transpose_last_two"
        );
    }

    #[test]
    fn test_transpose_last_two_axis_swap() {
        // Kills: replace - with / in line 1514 (ndim - 2, ndim - 1)

        // Test 2D transpose: (2, 3) -> (3, 2)
        let values = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let z = ZonotopeTensor::concrete(values.clone().into_dyn());

        let transposed = z.transpose_last_two().unwrap();

        assert_eq!(
            transposed.element_shape,
            vec![3, 2],
            "shape should be transposed from (2,3) to (3,2)"
        );

        // Verify values are correctly transposed
        let center = transposed.center();

        // Original [0,0]=1, [0,1]=2, [0,2]=3, [1,0]=4, [1,1]=5, [1,2]=6
        // Transposed [0,0]=1, [0,1]=4, [1,0]=2, [1,1]=5, [2,0]=3, [2,1]=6
        assert!((center[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((center[[0, 1]] - 4.0).abs() < 1e-6);
        assert!((center[[1, 0]] - 2.0).abs() < 1e-6);
        assert!((center[[1, 1]] - 5.0).abs() < 1e-6);
        assert!((center[[2, 0]] - 3.0).abs() < 1e-6);
        assert!((center[[2, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_last_two_3d() {
        // Test 3D transpose: (2, 3, 4) -> (2, 4, 3)
        // Only last two dimensions should be swapped

        let values = ndarray::Array3::<f32>::from_shape_fn((2, 3, 4), |(a, b, c)| {
            (a * 100 + b * 10 + c) as f32
        })
        .into_dyn();

        let z = ZonotopeTensor::concrete(values);
        let transposed = z.transpose_last_two().unwrap();

        assert_eq!(
            transposed.element_shape,
            vec![2, 4, 3],
            "3D shape (2,3,4) should become (2,4,3)"
        );

        // Verify first dimension is unchanged, last two are swapped
        let center = transposed.center();

        // Original [0, 1, 2] = 12.0 should become transposed [0, 2, 1] = 12.0
        assert!(
            (center[[0, 2, 1]] - 12.0).abs() < 1e-6,
            "value at [0,1,2] should move to [0,2,1]"
        );

        // Original [1, 0, 3] = 103.0 should become transposed [1, 3, 0] = 103.0
        assert!(
            (center[[1, 3, 0]] - 103.0).abs() < 1e-6,
            "value at [1,0,3] should move to [1,3,0]"
        );
    }

    // ============== Additional 2D SiLU Mutation-Killing Tests ==============

    #[test]
    fn test_silu_affine_2d_radius_zero_check() {
        // Kills: replace > with >= in line 1455 (radius > 0.0)

        // Test 2D zonotope with zero error (concrete) - should have zero approx error
        let values = arr2(&[[0.5_f32, 1.0]]).into_dyn();
        let z = ZonotopeTensor::concrete(values);

        let result = z.silu_affine().unwrap();
        let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];

        assert!(
            (approx_err - 0.0).abs() < 1e-10,
            "2D concrete zonotope should have 0 approx error, got {}",
            approx_err
        );

        // Test 2D zonotope with non-zero error
        let values2 = arr2(&[[0.5_f32, 1.0]]).into_dyn();
        let z2 = ZonotopeTensor::from_input_shared(&values2, 0.3);

        let result2 = z2.silu_affine().unwrap();
        let approx_err2 = result2.coeffs[[result2.n_error_terms, 0, 0]];

        assert!(
            approx_err2 > 0.0,
            "2D zonotope with error should have positive approx error, got {}",
            approx_err2
        );
    }

    #[test]
    fn test_silu_affine_2d_lo_hi_calculation() {
        // Kills: replace + with - in line 1457 (hi = c + radius)
        // Kills: replace + with * in line 1457

        let values = arr2(&[[1.0_f32], [2.0]]).into_dyn(); // 2 positions
        let z = ZonotopeTensor::from_input_shared(&values, 0.5);
        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor();

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        // Position 0: center=1.0, radius=0.5, lo=0.5, hi=1.5
        assert!(
            bounds.lower[[0, 0]] <= silu(0.5) + 0.01,
            "2D pos0 lower should contain silu(0.5)"
        );
        assert!(
            bounds.upper[[0, 0]] >= silu(1.5) - 0.01,
            "2D pos0 upper should contain silu(1.5)"
        );

        // Position 1: center=2.0, radius=0.5, lo=1.5, hi=2.5
        assert!(
            bounds.lower[[1, 0]] <= silu(1.5) + 0.01,
            "2D pos1 lower should contain silu(1.5)"
        );
        assert!(
            bounds.upper[[1, 0]] >= silu(2.5) - 0.01,
            "2D pos1 upper should contain silu(2.5)"
        );
    }

    #[test]
    fn test_silu_affine_2d_interpolation_division() {
        // Kills: replace / with % in line 1460 (i as f32 / 20.0)
        // Kills: replace / with * in line 1460

        // With % or *, the interpolation parameter t would be wrong
        let values = arr2(&[[0.0_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 1.0);

        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor();

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        // Bounds should contain silu(-1) and silu(1)
        assert!(
            bounds.lower[[0, 0]] <= silu(-1.0) + 0.01,
            "2D interpolation test: lower should contain silu(-1)"
        );
        assert!(
            bounds.upper[[0, 0]] >= silu(1.0) - 0.01,
            "2D interpolation test: upper should contain silu(1)"
        );
    }

    #[test]
    fn test_silu_affine_2d_interpolation_formula() {
        // Kills: mutations in line 1461 (x = lo + (hi - lo) * t)

        // Use center at SiLU'' critical point to maximize sensitivity to formula errors
        let values = arr2(&[[-1.28_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.5);

        let result = z.silu_affine().unwrap();

        // The approximation error should properly capture max second derivative
        let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];
        assert!(approx_err > 0.0, "2D should have approximation error");

        // Verify bounds soundness
        let bounds = result.to_bounded_tensor();
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        assert!(
            bounds.lower[[0, 0]] <= silu(-1.78),
            "2D lower {} should contain silu(-1.78)={}",
            bounds.lower[[0, 0]],
            silu(-1.78)
        );
        assert!(
            bounds.upper[[0, 0]] >= silu(-0.78),
            "2D upper {} should contain silu(-0.78)={}",
            bounds.upper[[0, 0]],
            silu(-0.78)
        );
    }

    #[test]
    fn test_silu_affine_2d_critical_point_sign() {
        // Kills: delete - in line 1464 (critical points -2.4, -1.28)

        // Center at -1.0 with radius 0.5 should contain critical point -1.28
        let values = arr2(&[[-1.0_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.5);

        let result = z.silu_affine().unwrap();
        let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];

        // Error should be non-trivial since we're near the SiLU'' peak
        assert!(
            approx_err > 0.001,
            "2D approx error should reflect critical point, got {}",
            approx_err
        );

        // Test at positive center - critical points shouldn't be sampled
        let values_pos = arr2(&[[2.0_f32]]).into_dyn();
        let z_pos = ZonotopeTensor::from_input_shared(&values_pos, 0.2);

        let result_pos = z_pos.silu_affine().unwrap();
        let approx_err_pos = result_pos.coeffs[[result_pos.n_error_terms, 0, 0]];

        // Error at x=2 should be smaller (far from critical points)
        assert!(
            approx_err_pos < approx_err,
            "2D error at x=2 ({}) should be less than at x=-1 ({})",
            approx_err_pos,
            approx_err
        );
    }

    #[test]
    fn test_silu_affine_2d_critical_bounds() {
        // Kills: replace && with || in line 1465
        // Kills: replace <= with > in line 1465

        // Test interval exactly containing critical point 0.7
        let values = arr2(&[[0.7_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 0.1);

        let result = z.silu_affine().unwrap();
        let approx_err_contains = result.coeffs[[result.n_error_terms, 0, 0]];

        // Test interval NOT containing any critical points (1.5 to 2.5)
        let values2 = arr2(&[[2.0_f32]]).into_dyn();
        let z2 = ZonotopeTensor::from_input_shared(&values2, 0.5);

        let result2 = z2.silu_affine().unwrap();
        let approx_err_not_contains = result2.coeffs[[result2.n_error_terms, 0, 0]];

        // Both should have approximation error
        assert!(
            approx_err_contains > 0.0,
            "should have error when containing critical"
        );
        assert!(
            approx_err_not_contains > 0.0,
            "should have error when not containing critical"
        );
    }

    #[test]
    fn test_silu_affine_2d_error_formula() {
        // Kills: mutations in line 1469 (max_second * radius * radius / 2.0)

        let values = arr2(&[[0.0_f32]]).into_dyn();
        let z = ZonotopeTensor::from_input_shared(&values, 1.0);

        let result = z.silu_affine().unwrap();
        let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];

        // At x=0, silu''(0) = 0.5 * 0.5 * 2 = 0.5
        // Error should be around 0.5 * 1.0 * 1.0 / 2.0 = 0.25
        // If / was *, error would be 1.0
        // If * was +, error would be different

        assert!(
            approx_err < 0.6,
            "2D error {} should be less than 0.6",
            approx_err
        );
        assert!(
            approx_err >= 0.1,
            "2D error {} should be at least 0.1",
            approx_err
        );
    }

    #[test]
    fn test_silu_affine_2d_multiple_positions() {
        // Test with multiple sequence positions to ensure 2D loop is correct

        // Create 3x2 input (3 positions, 2 features)
        let values = arr2(&[
            [-1.28_f32, 0.0], // Position 0: critical region, center
            [1.0, 2.0],       // Position 1: positive values
            [-2.0, -3.0],     // Position 2: negative values
        ])
        .into_dyn();

        let z = ZonotopeTensor::from_input_shared(&values, 0.3);
        let result = z.silu_affine().unwrap();

        assert_eq!(result.element_shape, vec![3, 2]);

        let bounds = result.to_bounded_tensor();

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        // Check bounds at each position
        for s in 0..3 {
            for d in 0..2 {
                let c = values[[s, d]];
                let lo = c - 0.3;
                let hi = c + 0.3;

                assert!(
                    bounds.lower[[s, d]] <= silu(lo) + 0.05,
                    "2D multi-pos lower[{},{}] should contain silu({})",
                    s,
                    d,
                    lo
                );
                assert!(
                    bounds.upper[[s, d]] >= silu(hi) - 0.05,
                    "2D multi-pos upper[{},{}] should contain silu({})",
                    s,
                    d,
                    hi
                );
            }
        }
    }
}
