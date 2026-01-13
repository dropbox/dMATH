//! Network types and graph representations for bound propagation.
//!
//! This module contains the core network abstractions:
//! - `Network`: Sequential layer-based network
//! - `GraphNetwork`: DAG-based computation graph for attention patterns
//! - `GraphNode`: A single node in a computation graph
//! - `AttentionGraphBuilder`: Builder for constructing attention patterns

use crate::bounds::{
    AlphaCrownConfig, AlphaCrownIntermediate, AlphaState, BatchedLinearBounds, GradientMethod,
    GraphAlphaCrownIntermediate, GraphAlphaState, LinearBounds, Optimizer,
};
use crate::domain_clip::DomainClipper;
use crate::layers::{
    AddLayer, BoundPropagation, GELULayer, Layer, LayerNormLayer, LinearLayer, MatMulLayer,
    ReLULayer, SoftmaxLayer,
};
use crate::types::{
    BlockBoundsInfo, BlockProgress, BlockWiseResult, LayerByLayerResult, LayerProgress,
    NodeBoundsInfo, VerificationCheckpoint,
};

use gamma_core::{GammaError, GemmEngine, Result};
use gamma_tensor::{BoundedTensor, ZonotopeTensor};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::borrow::Cow;
use tracing::{debug, info, instrument, warn};

/// Compute the broadcast shape for two shapes according to NumPy/ONNX broadcasting rules.
///
/// Returns None if the shapes are not broadcastable.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    // Pad shorter shape with 1s from the left
    let a_padded: Vec<usize> = std::iter::repeat(1)
        .take(max_len - a.len())
        .chain(a.iter().copied())
        .collect();
    let b_padded: Vec<usize> = std::iter::repeat(1)
        .take(max_len - b.len())
        .chain(b.iter().copied())
        .collect();

    for (da, db) in a_padded.iter().zip(b_padded.iter()) {
        if *da == *db {
            result.push(*da);
        } else if *da == 1 {
            result.push(*db);
        } else if *db == 1 {
            result.push(*da);
        } else {
            return None; // Not broadcastable
        }
    }

    Some(result)
}
/// A neural network represented as a sequence of layers.
#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    /// Create an empty network.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer to the network.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Enable or disable forward mode for all LayerNorm layers in the network.
    ///
    /// Returns the number of LayerNorm layers modified.
    pub fn set_layernorm_forward_mode(&mut self, enabled: bool) -> usize {
        let mut count = 0;
        for layer in &mut self.layers {
            if let Layer::LayerNorm(ref mut ln) = layer {
                ln.forward_mode = enabled;
                count += 1;
            }
        }
        count
    }

    /// Create a copy of this network with forward mode enabled for all LayerNorm layers.
    pub fn with_layernorm_forward_mode(mut self, enabled: bool) -> Self {
        self.set_layernorm_forward_mode(enabled);
        self
    }

    /// Propagate bounds through the entire network using IBP.
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut current = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            debug!("IBP propagating through layer {}", i);
            current = layer
                .propagate_ibp(&current)
                .map_err(|e| GammaError::LayerError {
                    layer_index: i,
                    layer_type: layer.layer_type().to_string(),
                    source: Box::new(e),
                })?;
        }
        Ok(current)
    }

    /// Propagate bounds with strict soundness guarantees.
    ///
    /// This method applies directed rounding after each layer to ensure
    /// mathematical soundness of interval arithmetic. Lower bounds are
    /// rounded DOWN (toward -∞) and upper bounds are rounded UP (toward +∞).
    ///
    /// # Performance
    /// Adds ~1 ULP of bound width per layer. For a network with N layers,
    /// final bounds will be wider by ~N ULPs compared to `propagate_ibp()`.
    /// This is negligible compared to relaxation approximation errors.
    ///
    /// # When to Use
    /// - When strict mathematical soundness is required
    /// - For formal verification applications
    /// - When comparing bounds against reference implementations
    #[inline]
    pub fn propagate_ibp_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut current = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            debug!("IBP (sound) propagating through layer {}", i);
            current = layer
                .propagate_ibp(&current)
                .map_err(|e| GammaError::LayerError {
                    layer_index: i,
                    layer_type: layer.layer_type().to_string(),
                    source: Box::new(e),
                })?;
            // Apply directed rounding: lower bounds down, upper bounds up
            current.round_for_soundness_inplace();
        }
        Ok(current)
    }

    /// Run IBP forward pass and collect bounds at each layer.
    ///
    /// Returns a vector of bounds, where `bounds\[i\]` is the output of layer i.
    /// The input bounds are NOT included in the returned vector.
    pub fn collect_ibp_bounds(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        let n = self.layers.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let mut bounds = Vec::with_capacity(n);
        let mut current = input.clone();

        // Process all but last layer with cloning
        for (i, layer) in self.layers[..n - 1].iter().enumerate() {
            current = layer
                .propagate_ibp(&current)
                .map_err(|e| GammaError::LayerError {
                    layer_index: i,
                    layer_type: layer.layer_type().to_string(),
                    source: Box::new(e),
                })?;
            bounds.push(current.clone());
        }

        // Process last layer without cloning (move ownership)
        let last_layer = &self.layers[n - 1];
        current = last_layer
            .propagate_ibp(&current)
            .map_err(|e| GammaError::LayerError {
                layer_index: n - 1,
                layer_type: last_layer.layer_type().to_string(),
                source: Box::new(e),
            })?;
        bounds.push(current); // Move, not clone

        Ok(bounds)
    }

    /// Run CROWN-IBP to collect tighter intermediate bounds.
    ///
    /// This method computes tighter bounds than pure IBP by using CROWN backward
    /// propagation for each intermediate layer. For each layer k, it:
    /// 1. Runs CROWN backward from layer k to the input
    /// 2. Takes the intersection (tighter) of IBP and CROWN bounds
    ///
    /// This is more expensive than pure IBP (O(n) CROWN passes) but produces
    /// significantly tighter intermediate bounds, which leads to tighter ReLU
    /// relaxations and ultimately tighter final bounds.
    ///
    /// Returns a vector of bounds, where `bounds[i]` is the (tightened) output of layer i.
    pub fn collect_crown_ibp_bounds(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        self.collect_crown_ibp_bounds_with_engine(input, None)
    }

    pub fn collect_crown_ibp_bounds_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<BoundedTensor>> {
        let n = self.layers.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Step 1: Run IBP forward to get initial bounds
        let ibp_bounds = self.collect_ibp_bounds(input)?;

        // Step 2: For each layer, try to tighten bounds using CROWN
        let mut crown_ibp_bounds = Vec::with_capacity(n);

        for (k, ibp_bound) in ibp_bounds.iter().enumerate() {
            // Create partial network from layers 0..=k
            let partial_layers = &self.layers[0..=k];

            // Try to compute CROWN bounds for this partial network
            let crown_bounds = self.propagate_crown_partial_with_engine(
                input,
                partial_layers,
                &crown_ibp_bounds,
                engine,
            );
            let tightened = match crown_bounds {
                Ok(cb) => {
                    // Only intersect if shapes match; otherwise fall back to IBP
                    if cb.shape() == ibp_bound.shape() {
                        ibp_bound.intersection(&cb)
                    } else {
                        // Shape mismatch (e.g., batch dimensions) - use IBP
                        ibp_bound.clone()
                    }
                }
                Err(_) => ibp_bound.clone(), // Fall back to IBP if CROWN fails
            };

            crown_ibp_bounds.push(tightened);
        }

        Ok(crown_ibp_bounds)
    }

    /// Propagate CROWN bounds through a partial network (subset of layers).
    ///
    /// Used by collect_crown_ibp_bounds to compute bounds at intermediate layers.
    fn propagate_crown_partial_with_engine(
        &self,
        input: &BoundedTensor,
        layers: &[Layer],
        prior_bounds: &[BoundedTensor],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if layers.is_empty() {
            return Ok(input.clone());
        }

        // Get output dimension from last layer
        // We need to compute IBP bounds just for the last layer to know the output dim
        let last_layer = layers.last().unwrap();
        let pre_last = if layers.len() == 1 {
            input.clone()
        } else if prior_bounds.len() >= layers.len() - 1 {
            prior_bounds[layers.len() - 2].clone()
        } else {
            // Compute IBP for intermediate if not available
            let mut current = input.clone();
            for layer in layers.iter().take(layers.len() - 1) {
                current = layer.propagate_ibp(&current)?;
            }
            current
        };
        let output_bounds = last_layer.propagate_ibp(&pre_last)?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        // Initialize linear bounds at output
        let mut linear_bounds = LinearBounds::identity(output_dim);

        // Propagate backward through each layer
        for (i, layer) in layers.iter().enumerate().rev() {
            // Get pre-activation bounds for this layer
            let pre_activation = if i == 0 {
                input
            } else if i - 1 < prior_bounds.len() {
                &prior_bounds[i - 1]
            } else {
                // Compute IBP if not available (shouldn't happen in normal use)
                return Err(GammaError::InvalidSpec(
                    "Missing prior bounds for CROWN-IBP".to_string(),
                ));
            };

            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear_with_engine(&linear_bounds, engine)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Conv1d(c) => {
                    let input_shape = pre_activation.shape();
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else {
                        return Err(GammaError::InvalidSpec(
                            "Conv1d input shape too small for CROWN".to_string(),
                        ));
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_length(in_len);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Conv2d(c) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        return Err(GammaError::InvalidSpec(
                            "Conv2d input shape too small for CROWN".to_string(),
                        ));
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Softmax(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::GELU(g) => {
                    linear_bounds =
                        g.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LayerNorm(ln) => {
                    linear_bounds =
                        ln.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Transpose(t) => {
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let next = transpose_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Flatten(f) => {
                    let next = f.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Reshape(r) => {
                    let next = r.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AddConstant(a) => {
                    let next = a.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::SubConstant(s) => {
                    let next = s.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::MulConstant(m) => {
                    let next = m.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::DivConstant(d) => {
                    let next = d.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::BatchNorm(bn) => {
                    linear_bounds =
                        bn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tanh(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sigmoid(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softplus(sp) => {
                    linear_bounds =
                        sp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LeakyReLU(lr) => {
                    linear_bounds =
                        lr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Clip(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSigmoid(hs) => {
                    linear_bounds =
                        hs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Elu(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Selu(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PRelu(pr) => {
                    linear_bounds =
                        pr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSwish(hw) => {
                    linear_bounds =
                        hw.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Celu(ce) => {
                    linear_bounds =
                        ce.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Mish(mi) => {
                    linear_bounds =
                        mi.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Exp(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Log(lg) => {
                    linear_bounds =
                        lg.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softsign(ss) => {
                    linear_bounds =
                        ss.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sin(sn) => {
                    linear_bounds =
                        sn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Cos(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Reciprocal(rc) => {
                    linear_bounds =
                        rc.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sqrt(sq) => {
                    linear_bounds =
                        sq.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Abs(ab) => {
                    linear_bounds =
                        ab.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PowConstant(p) => {
                    linear_bounds =
                        p.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ThresholdedRelu(tr) => {
                    linear_bounds =
                        tr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Shrink(sh) => {
                    linear_bounds =
                        sh.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LogSoftmax(ls) => {
                    linear_bounds =
                        ls.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ReduceSum(rs) => {
                    linear_bounds =
                        rs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ReduceMean(rm) => {
                    linear_bounds =
                        rm.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tile(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Slice(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::CausalSoftmax(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::AveragePool(ap) => {
                    linear_bounds =
                        ap.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MaxPool2d(mp) => {
                    linear_bounds =
                        mp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Floor(f) => {
                    linear_bounds =
                        f.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Ceil(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Round(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sign(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                // Unsupported layers for CROWN - return error to fall back to IBP
                Layer::Add(_)
                | Layer::Concat(_)
                | Layer::Sub(_)
                | Layer::MulBinary(_)
                | Layer::Div(_)
                | Layer::MatMul(_)
                | Layer::Where(_)
                | Layer::NonZero(_) => {
                    return Err(GammaError::NotSupported(format!(
                        "CROWN not supported for layer type: {}",
                        layer.layer_type()
                    )));
                }
            }
        }

        // Concretize linear bounds with input bounds and reshape to the IBP output shape.
        //
        // LinearBounds::concretize returns a flat 1D tensor. For ONNX models (and any model
        // with non-1D activations), this causes a shape mismatch against IBP bounds and
        // prevents CROWN-IBP from intersecting/tightening intermediate bounds.
        linear_bounds.concretize(input).reshape(&output_shape)
    }

    /// Propagate bounds through the entire network using CROWN.
    ///
    /// CROWN (Convex Relaxation based perturbation ON-the-fly Network) provides
    /// tighter bounds than IBP by representing bounds as linear functions of the input.
    /// This implementation matches Auto-LiRPA's "backward" method.
    ///
    /// Algorithm:
    /// 1. Run CROWN-IBP to collect tighter pre-activation bounds
    ///    - For Linear layer outputs: CROWN backward from that layer to input
    ///    - For ReLU layer outputs: IBP from the tighter linear bounds
    /// 2. Initialize linear bounds at output: A = I, b = 0 (output = output)
    /// 3. Propagate backward through each layer using tighter intermediate bounds
    /// 4. Concretize final linear bounds using input bounds
    pub fn propagate_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine(input, None)
    }

    #[inline]
    #[instrument(skip(self, input, engine), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        // Step 1: Collect CROWN-IBP intermediate bounds
        // This matches Auto-LiRPA's "backward" method which uses CROWN to compute
        // tighter pre-ReLU bounds (not just IBP). For deep networks (3+ layers),
        // this produces significantly tighter bounds than using pure IBP.
        //
        // CROWN-IBP for pre-Linear bounds: run CROWN backward from that layer to input
        // IBP for post-ReLU bounds: ReLU output bounds are max(0, lower), max(0, upper)
        let layer_bounds = self.collect_crown_ibp_bounds_with_engine(input, engine)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| GammaError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 2: Initialize linear bounds at output
        // output >= 1 * output + 0, output <= 1 * output + 0
        let mut linear_bounds = LinearBounds::identity(output_dim);

        // Step 3: Propagate backward through each layer
        for (i, layer) in self.layers.iter().enumerate().rev() {
            debug!(
                "CROWN: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds (bounds before this layer)
            // For layer i, pre-activation bounds are:
            // - layer_bounds[i-1] for i > 0
            // - input for i == 0
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear_with_engine(&linear_bounds, engine)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Conv1d(c) => {
                    // Conv1d CROWN: clone layer, set input_length, propagate
                    let input_shape = pre_activation.shape();
                    // Input shape: (in_channels, length) or (batch, in_channels, length)
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else {
                        debug!(
                            "CROWN: Conv1d input shape too small: {:?}, using IBP",
                            input_shape
                        );
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_length(in_len);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Conv2d(c) => {
                    // Conv2d CROWN: clone layer, set input_shape, propagate
                    let input_shape = pre_activation.shape();
                    // Input shape: (channels, height, width) or (batch, channels, height, width)
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        debug!(
                            "CROWN: Conv2d input shape too small: {:?}, using IBP",
                            input_shape
                        );
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AveragePool(ap) => {
                    linear_bounds =
                        ap.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MaxPool2d(mp) => {
                    linear_bounds =
                        mp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softmax(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::CausalSoftmax(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::GELU(g) => {
                    linear_bounds =
                        g.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LayerNorm(ln) => {
                    linear_bounds =
                        ln.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::BatchNorm(bn) => {
                    linear_bounds =
                        bn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Transpose(t) => {
                    // Transpose is a linear operation (permutation)
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let next = transpose_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AddConstant(ac) => {
                    let next = ac.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Reshape(r) => {
                    let next = r.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Flatten(f) => {
                    let next = f.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::MulConstant(m) => {
                    let next = m.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Abs(ab) => {
                    linear_bounds =
                        ab.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::DivConstant(d) => {
                    let next = d.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::SubConstant(s) => {
                    let next = s.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Sqrt(sq) => {
                    linear_bounds =
                        sq.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PowConstant(p) => {
                    match p.propagate_linear_with_bounds(&linear_bounds, pre_activation) {
                        Ok(next) => linear_bounds = next,
                        Err(err) => {
                            debug!(
                                "CROWN: PowConstant not supported ({}), using IBP bounds",
                                err
                            );
                            return self.propagate_ibp(input);
                        }
                    }
                }
                Layer::ReduceMean(rm) => {
                    linear_bounds =
                        rm.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ReduceSum(rs) => {
                    linear_bounds =
                        rs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tanh(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sigmoid(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softplus(sp) => {
                    linear_bounds =
                        sp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LeakyReLU(lr) => {
                    linear_bounds =
                        lr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Clip(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSigmoid(hs) => {
                    linear_bounds =
                        hs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Elu(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Selu(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PRelu(pr) => {
                    linear_bounds =
                        pr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSwish(hw) => {
                    linear_bounds =
                        hw.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Celu(ce) => {
                    linear_bounds =
                        ce.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Mish(mi) => {
                    linear_bounds =
                        mi.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Exp(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Log(lg) => {
                    linear_bounds =
                        lg.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softsign(ss) => {
                    linear_bounds =
                        ss.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sin(sn) => {
                    linear_bounds =
                        sn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Cos(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Reciprocal(rc) => {
                    linear_bounds =
                        rc.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ThresholdedRelu(tr) => {
                    linear_bounds =
                        tr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Shrink(sh) => {
                    linear_bounds =
                        sh.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tile(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Slice(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LogSoftmax(ls) => {
                    linear_bounds =
                        ls.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Floor(f) => {
                    linear_bounds =
                        f.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Ceil(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Round(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sign(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::Add(_)
                | Layer::Concat(_)
                | Layer::Sub(_)
                | Layer::Div(_)
                | Layer::Where(_)
                | Layer::NonZero(_) => {
                    // Binary/ternary/data-dependent ops can't be handled in sequential CROWN, fall back to IBP
                    debug!(
                        "CROWN: Binary/ternary/data-dependent ops not supported in sequential network, using IBP bounds"
                    );
                    return self.propagate_ibp(input);
                }
            }
        }

        // Step 4: Concretize using input bounds
        debug!("CROWN: Concretizing linear bounds with input");
        linear_bounds.concretize(input).reshape(&output_shape)
    }

    /// Propagate bounds using fast CROWN with IBP intermediate bounds.
    ///
    /// This method is 3-10x faster than standard CROWN by using simple IBP bounds
    /// for intermediate layers instead of running CROWN-IBP tightening passes.
    /// The tradeoff is slightly looser bounds, but this is often acceptable for:
    /// - Initial bound computation before α-optimization
    /// - Cases where verification succeeds easily
    /// - Performance-critical code paths
    ///
    /// Algorithm:
    /// 1. Run IBP forward to collect intermediate bounds (fast)
    /// 2. Run CROWN backward using IBP bounds for ReLU relaxation
    /// 3. Concretize final linear bounds using input bounds
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_fast(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        // Step 1: Collect IBP intermediate bounds (fast, no CROWN-IBP overhead)
        let layer_bounds = self.collect_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| GammaError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "CROWN-fast: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 2: Initialize linear bounds at output
        let mut linear_bounds = LinearBounds::identity(output_dim);

        // Step 3: Propagate backward through each layer using IBP bounds
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Conv1d(c) => {
                    let input_shape = pre_activation.shape();
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else {
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_length(in_len);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Conv2d(c) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AveragePool(ap) => {
                    linear_bounds =
                        ap.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MaxPool2d(mp) => {
                    linear_bounds =
                        mp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softmax(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::CausalSoftmax(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::GELU(g) => {
                    linear_bounds =
                        g.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LayerNorm(ln) => {
                    linear_bounds =
                        ln.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::BatchNorm(bn) => {
                    linear_bounds =
                        bn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Transpose(t) => {
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let next = transpose_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AddConstant(ac) => {
                    let next = ac.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Reshape(r) => {
                    let next = r.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Flatten(f) => {
                    let next = f.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::MulConstant(m) => {
                    let next = m.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Abs(ab) => {
                    linear_bounds =
                        ab.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::DivConstant(d) => {
                    let next = d.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::SubConstant(s) => {
                    let next = s.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Sqrt(sq) => {
                    linear_bounds =
                        sq.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PowConstant(p) => {
                    match p.propagate_linear_with_bounds(&linear_bounds, pre_activation) {
                        Ok(next) => linear_bounds = next,
                        Err(_) => return self.propagate_ibp(input),
                    }
                }
                Layer::ReduceMean(rm) => {
                    linear_bounds =
                        rm.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ReduceSum(rs) => {
                    linear_bounds =
                        rs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tanh(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sigmoid(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softplus(sp) => {
                    linear_bounds =
                        sp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LeakyReLU(lr) => {
                    linear_bounds =
                        lr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Clip(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSigmoid(hs) => {
                    linear_bounds =
                        hs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Elu(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Selu(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PRelu(pr) => {
                    linear_bounds =
                        pr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSwish(hw) => {
                    linear_bounds =
                        hw.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Celu(ce) => {
                    linear_bounds =
                        ce.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Mish(mi) => {
                    linear_bounds =
                        mi.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Exp(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Log(lg) => {
                    linear_bounds =
                        lg.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softsign(ss) => {
                    linear_bounds =
                        ss.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sin(sn) => {
                    linear_bounds =
                        sn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Cos(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Reciprocal(rc) => {
                    linear_bounds =
                        rc.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ThresholdedRelu(tr) => {
                    linear_bounds =
                        tr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Shrink(sh) => {
                    linear_bounds =
                        sh.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tile(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Slice(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LogSoftmax(ls) => {
                    linear_bounds =
                        ls.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Floor(f) => {
                    linear_bounds =
                        f.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Ceil(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Round(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sign(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::Add(_)
                | Layer::Concat(_)
                | Layer::Sub(_)
                | Layer::Div(_)
                | Layer::Where(_)
                | Layer::NonZero(_) => {
                    return self.propagate_ibp(input);
                }
            }
        }

        // Step 4: Concretize using input bounds
        linear_bounds.concretize(input).reshape(&output_shape)
    }

    /// Propagate bounds using CROWN with CROWN-IBP intermediate bounds.
    ///
    /// This method produces tighter bounds than standard CROWN by computing
    /// tighter intermediate bounds using CROWN backward propagation at each layer.
    /// The intermediate bounds are used for ReLU relaxation, resulting in
    /// tighter final bounds.
    ///
    /// Algorithm:
    /// 1. Run CROWN-IBP to collect tighter intermediate bounds
    /// 2. Run CROWN backward using the tighter bounds for ReLU relaxation
    /// 3. Concretize final linear bounds using input bounds
    ///
    /// This is more expensive than standard CROWN but produces significantly
    /// tighter bounds, especially for deep networks.
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        // Step 1: Collect CROWN-IBP intermediate bounds
        let layer_bounds = self.collect_crown_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| GammaError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "CROWN-IBP: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 2: Initialize linear bounds at output
        let mut linear_bounds = LinearBounds::identity(output_dim);

        // Step 3: Propagate backward through each layer using CROWN-IBP bounds
        for (i, layer) in self.layers.iter().enumerate().rev() {
            debug!(
                "CROWN-IBP: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds (using CROWN-IBP tightened bounds)
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Conv1d(c) => {
                    let input_shape = pre_activation.shape();
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else {
                        debug!(
                            "CROWN-IBP: Conv1d input shape too small: {:?}, using IBP",
                            input_shape
                        );
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_length(in_len);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Conv2d(c) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        debug!(
                            "CROWN-IBP: Conv2d input shape too small: {:?}, using IBP",
                            input_shape
                        );
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let next = conv_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AveragePool(ap) => {
                    linear_bounds =
                        ap.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MaxPool2d(mp) => {
                    linear_bounds =
                        mp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Abs(ab) => {
                    linear_bounds =
                        ab.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softmax(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::CausalSoftmax(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::GELU(g) => {
                    linear_bounds =
                        g.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LayerNorm(ln) => {
                    linear_bounds =
                        ln.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::BatchNorm(bn) => {
                    linear_bounds =
                        bn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Transpose(t) => {
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let next = transpose_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AddConstant(ac) => {
                    let next = ac.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Reshape(r) => {
                    let next = r.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Flatten(f) => {
                    let next = f.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::MulConstant(m) => {
                    let next = m.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::DivConstant(d) => {
                    let next = d.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::SubConstant(s) => {
                    let next = s.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Sqrt(sq) => {
                    linear_bounds =
                        sq.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PowConstant(p) => {
                    match p.propagate_linear_with_bounds(&linear_bounds, pre_activation) {
                        Ok(next) => linear_bounds = next,
                        Err(err) => {
                            debug!(
                                "CROWN-IBP: PowConstant not supported ({}), using IBP bounds",
                                err
                            );
                            return self.propagate_ibp(input);
                        }
                    }
                }
                Layer::ReduceMean(rm) => {
                    linear_bounds =
                        rm.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ReduceSum(rs) => {
                    linear_bounds =
                        rs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tanh(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sigmoid(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softplus(sp) => {
                    linear_bounds =
                        sp.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LeakyReLU(lr) => {
                    linear_bounds =
                        lr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Clip(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSigmoid(hs) => {
                    linear_bounds =
                        hs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Elu(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Selu(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::PRelu(pr) => {
                    linear_bounds =
                        pr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::HardSwish(hw) => {
                    linear_bounds =
                        hw.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Celu(ce) => {
                    linear_bounds =
                        ce.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Mish(mi) => {
                    linear_bounds =
                        mi.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Exp(e) => {
                    linear_bounds =
                        e.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Log(lg) => {
                    linear_bounds =
                        lg.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Softsign(ss) => {
                    linear_bounds =
                        ss.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sin(sn) => {
                    linear_bounds =
                        sn.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Cos(cs) => {
                    linear_bounds =
                        cs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Reciprocal(rc) => {
                    linear_bounds =
                        rc.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::ThresholdedRelu(tr) => {
                    linear_bounds =
                        tr.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Shrink(sh) => {
                    linear_bounds =
                        sh.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Tile(t) => {
                    linear_bounds =
                        t.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Slice(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LogSoftmax(ls) => {
                    linear_bounds =
                        ls.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Floor(f) => {
                    linear_bounds =
                        f.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Ceil(c) => {
                    linear_bounds =
                        c.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Round(r) => {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Sign(s) => {
                    linear_bounds =
                        s.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::Add(_)
                | Layer::Concat(_)
                | Layer::Sub(_)
                | Layer::Div(_)
                | Layer::Where(_)
                | Layer::NonZero(_) => {
                    debug!(
                        "CROWN-IBP: Binary/ternary/data-dependent ops not supported in sequential network, using IBP bounds"
                    );
                    return self.propagate_ibp(input);
                }
            }
        }

        // Step 4: Concretize using input bounds
        debug!("CROWN-IBP: Concretizing linear bounds with input");
        linear_bounds.concretize(input).reshape(&output_shape)
    }

    /// Propagate bounds through a Linear/ReLU network using SDP-CROWN offsets for an ℓ2 input set.
    ///
    /// This implements the SDP-CROWN tightening for ReLU layers (arXiv:2506.06665) by:
    /// - Running standard IBP on the ℓ∞ box `x_hat ± rho` (contains the ℓ2 ball) to obtain
    ///   elementwise pre-activation bounds for the usual CROWN slopes.
    /// - Replacing the box relaxation offset at each ReLU with the SDP-CROWN offset valid for
    ///   `||z - z_hat||_2 <= rho_z` at that layer.
    /// - Concretizing the final linear bounds over the input ℓ2 ball.
    ///
    /// Current limitations:
    /// - Only supports sequential networks consisting of `Linear` and `ReLU` layers.
    pub fn propagate_sdp_crown(
        &self,
        input: &BoundedTensor,
        x_hat: &Array1<f32>,
        rho: f32,
    ) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        if rho < 0.0 {
            return Err(GammaError::InvalidSpec(format!(
                "SDP-CROWN: rho must be >= 0 (got {rho})"
            )));
        }

        let input_flat = input.flatten();
        if input_flat.len() != x_hat.len() {
            return Err(GammaError::shape_mismatch(
                vec![input_flat.len()],
                vec![x_hat.len()],
            ));
        }

        // Step 1: IBP forward on the box relaxation (needed for ReLU slopes).
        let layer_bounds = self.collect_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| GammaError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        // Step 2: Precompute ℓ2 ball centers/radii for each ReLU pre-activation.
        // Use Lipschitz propagation: ReLU is 1-Lipschitz, Linear scales by spectral norm.
        let mut relu_preactivation: Vec<Option<(Array1<f32>, f32)>> = vec![None; self.layers.len()];

        let mut center = x_hat.clone();
        let mut radius = rho;
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Linear(l) => {
                    let mut next = l.weight.dot(&center);
                    if let Some(b) = &l.bias {
                        next += b;
                    }
                    center = next;
                    radius *= l.spectral_norm();
                }
                Layer::ReLU(_) => {
                    relu_preactivation[i] = Some((center.clone(), radius));
                    center.mapv_inplace(|v| v.max(0.0));
                }
                other => {
                    return Err(GammaError::InvalidSpec(format!(
                        "SDP-CROWN currently supports Linear/ReLU networks only (saw {other:?})"
                    )));
                }
            }
        }

        debug!(
            "SDP-CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 3: Backward CROWN pass with SDP offsets at ReLUs.
        let mut linear_bounds = LinearBounds::identity(output_dim);
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };
            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    let (z_hat, z_rho) = relu_preactivation[i].as_ref().ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "SDP-CROWN: missing pre-activation ball for ReLU layer {i}"
                        ))
                    })?;
                    linear_bounds = r.propagate_linear_with_bounds_sdp(
                        &linear_bounds,
                        pre_activation,
                        z_hat,
                        *z_rho,
                    )?;
                }
                _ => unreachable!("validated above"),
            }
        }

        // Step 4: Concretize over input ℓ2 ball.
        linear_bounds
            .concretize_l2_ball(x_hat, rho)?
            .reshape(&output_shape)
    }

    /// Propagate bounds through the network using batched CROWN.
    ///
    /// This version preserves N-D shape structure (e.g., [batch, seq, hidden]) throughout
    /// propagation, unlike regular CROWN which flattens to 1D. This is essential for
    /// transformer verification where position-wise operations need to maintain structure.
    ///
    /// Algorithm:
    /// 1. Run IBP forward to collect pre-activation bounds
    /// 2. Initialize batched linear bounds at output: A = I, b = 0 per position
    /// 3. Propagate backward through each layer using batched operations
    /// 4. Concretize final linear bounds using input bounds
    ///
    /// Currently supports: Linear, ReLU, GELU layers.
    /// Other layers fall back to regular CROWN.
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        // Check if we can use batched CROWN (Linear, ReLU, GELU, Softmax, LayerNorm, Conv1d supported)
        // Note: Conv2d has propagate_linear_batched but requires shape transformation for network use
        let can_use_batched = self.layers.iter().all(|layer| {
            matches!(
                layer,
                Layer::Linear(_)
                    | Layer::ReLU(_)
                    | Layer::GELU(_)
                    | Layer::Softmax(_)
                    | Layer::LayerNorm(_)
                    | Layer::Conv1d(_)
            )
        });

        if !can_use_batched {
            debug!("Batched CROWN: Falling back to regular CROWN (unsupported layers)");
            return self.propagate_crown(input);
        }

        // Step 1: Run IBP forward to collect bounds at each layer
        // Note: Using IBP instead of CROWN-IBP for batched mode due to shape complexities
        // with multi-dimensional tensors. The regular propagate_crown() uses CROWN-IBP.
        let layer_bounds = self.collect_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| GammaError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "Batched CROWN: Starting backward propagation with shape {:?}",
            output_shape
        );

        // Step 2: Initialize batched linear bounds at output
        // A = I (identity) for each position, b = 0
        let mut batched_bounds = BatchedLinearBounds::identity(&output_shape);

        // Step 3: Propagate backward through each layer
        for (i, layer) in self.layers.iter().enumerate().rev() {
            debug!(
                "Batched CROWN: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            batched_bounds = match layer {
                Layer::Linear(l) => l.propagate_linear_batched(&batched_bounds)?,
                Layer::ReLU(r) => {
                    r.propagate_linear_batched_with_bounds(&batched_bounds, pre_activation)?
                }
                Layer::GELU(g) => {
                    g.propagate_linear_batched_with_bounds(&batched_bounds, pre_activation)?
                }
                Layer::Softmax(s) => {
                    s.propagate_linear_batched_with_bounds(&batched_bounds, pre_activation)?
                }
                Layer::LayerNorm(ln) => {
                    ln.propagate_linear_batched_with_bounds(&batched_bounds, pre_activation)?
                }
                Layer::Conv1d(c) => {
                    // Conv1d batched CROWN: clone layer, set input_length, propagate
                    let input_shape = pre_activation.shape();
                    let in_len = input_shape[input_shape.len() - 1];
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_length(in_len);
                    conv_with_shape.propagate_linear_batched(&batched_bounds)?
                }
                _ => {
                    // Should not reach here due to earlier check
                    debug!("Batched CROWN: Unsupported layer, falling back to regular CROWN");
                    return self.propagate_crown(input);
                }
            };
        }

        // Step 4: Concretize using input bounds
        debug!("Batched CROWN: Concretizing linear bounds with input");
        let concrete = batched_bounds.concretize(input);

        // Output should match the expected output shape
        if concrete.shape() != output_shape.as_slice() {
            concrete.reshape(&output_shape)
        } else {
            Ok(concrete)
        }
    }

    /// Propagate bounds through the network using α-CROWN with optimized parameters.
    ///
    /// α-CROWN extends CROWN by making the lower bound slope (α) for unstable ReLUs
    /// learnable and optimizing it via gradient descent to tighten bounds.
    ///
    /// Algorithm:
    /// 1. Run IBP to collect pre-activation bounds
    /// 2. Initialize α state (from heuristic)
    /// 3. For each optimization iteration:
    ///    a. Run CROWN backward with current α values
    ///    b. Concretize to get bounds
    ///    c. Compute gradients ∂bounds/∂α
    ///    d. Update α via gradient descent
    /// 4. Return the tightest bounds found
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_alpha_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_engine(input, None)
    }

    /// α-CROWN with optional GEMM acceleration engine.
    #[inline]
    #[instrument(skip(self, input, engine), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_alpha_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine(
            input,
            &AlphaCrownConfig::default(),
            engine,
        )
    }

    /// α-CROWN with custom configuration (no acceleration engine).
    #[instrument(skip(self, input, config), fields(num_layers = self.layers.len(), iterations = config.iterations))]
    pub fn propagate_alpha_crown_with_config(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine(input, config, None)
    }

    /// α-CROWN with custom configuration and optional GEMM acceleration engine.
    #[instrument(skip(self, input, config, engine), fields(num_layers = self.layers.len(), iterations = config.iterations))]
    pub fn propagate_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }

        // Step 1: Run CROWN-IBP to collect tighter bounds at each layer
        // This matches the intermediate bound computation used in propagate_crown()
        let layer_bounds = self.collect_crown_ibp_bounds_with_engine(input, engine)?;
        let output_dim = layer_bounds.last().unwrap().len();

        // Check for Conv2d/MaxPool2d layers - fall back to CROWN for now
        for layer in &self.layers {
            if matches!(layer, Layer::Conv2d(_) | Layer::MaxPool2d(_)) {
                debug!("α-CROWN: Conv2d/MaxPool2d detected, falling back to CROWN");
                return self.propagate_crown_with_engine(input, engine);
            }
        }

        // Step 2: Identify ReLU layers and initialize α state
        let relu_layer_indices: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, Layer::ReLU(_)))
            .map(|(i, _)| i)
            .collect();

        if relu_layer_indices.is_empty() {
            // No ReLU layers, just use CROWN
            return self.propagate_crown_with_engine(input, engine);
        }

        // Map from layer index to relu_layer_indices position
        let mut layer_to_relu_idx: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (relu_idx, &layer_idx) in relu_layer_indices.iter().enumerate() {
            layer_to_relu_idx.insert(layer_idx, relu_idx);
        }

        // Initialize alpha state
        // We need pre-activation bounds for ReLU layers
        // For ReLU at layer i, pre-activation is layer_bounds[i-1] (or input if i==0)
        let pre_activation_bounds: Vec<BoundedTensor> = relu_layer_indices
            .iter()
            .map(|&i| {
                if i == 0 {
                    input.clone()
                } else {
                    layer_bounds[i - 1].clone()
                }
            })
            .collect();

        let mut alpha_state = AlphaState::from_preactivation_bounds(
            &pre_activation_bounds,
            &(0..relu_layer_indices.len()).collect::<Vec<_>>(),
        );

        let num_unstable = alpha_state.num_unstable();
        if num_unstable == 0 {
            // No unstable neurons, α-CROWN won't help
            debug!("α-CROWN: No unstable neurons, using CROWN");
            return self.propagate_crown_with_engine(input, engine);
        }

        debug!(
            "α-CROWN: Starting optimization with {} unstable neurons across {} ReLU layers",
            num_unstable,
            relu_layer_indices.len()
        );

        // Step 3: Optimization loop
        // Track element-wise best bounds across iterations:
        // - best_lower: maximum lower bound seen for each output dimension
        // - best_upper: minimum upper bound seen for each output dimension
        // Initialize from CROWN bounds to ensure α-CROWN never returns worse bounds.
        let crown_bounds = self.propagate_crown_with_engine(input, engine)?;
        let mut best_lower: ArrayD<f32> = crown_bounds.lower.clone();
        let mut best_upper: ArrayD<f32> = crown_bounds.upper.clone();
        let mut best_lower_sum: f32 = crown_bounds
            .lower
            .as_slice()
            .map(|s| s.iter().sum())
            .unwrap_or(f32::NEG_INFINITY);
        let mut prev_best_lower_sum = best_lower_sum;
        let mut no_improve_iters = 0usize;
        let mut lr = config.learning_rate;

        for iter in 0..config.iterations {
            // Run CROWN backward with current alpha values
            let mut linear_bounds = LinearBounds::identity(output_dim);
            let mut gradients: Vec<Array1<f32>> = Vec::with_capacity(relu_layer_indices.len());

            // Initialize gradient storage for each ReLU layer
            for &relu_idx in &relu_layer_indices {
                let pre_act = if relu_idx == 0 {
                    input
                } else {
                    &layer_bounds[relu_idx - 1]
                };
                gradients.push(Array1::zeros(pre_act.len()));
            }

            // Backward pass through layers
            for (i, layer) in self.layers.iter().enumerate().rev() {
                let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

                match layer {
                    Layer::Linear(l) => {
                        let next = l.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::ReLU(r) => {
                        // Get the relu index for this layer
                        if let Some(&relu_idx) = layer_to_relu_idx.get(&i) {
                            let alpha = alpha_state.get_alpha(relu_idx).ok_or_else(|| {
                                GammaError::InvalidSpec(format!(
                                    "Missing alpha for ReLU layer {}",
                                    i
                                ))
                            })?;

                            let (new_bounds, grad) = r.propagate_linear_with_alpha(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                            )?;

                            // Accumulate gradient
                            gradients[relu_idx] = grad;
                            linear_bounds = new_bounds;
                        } else {
                            linear_bounds =
                                r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                        }
                    }
                    Layer::Conv1d(_)
                    | Layer::Conv2d(_)
                    | Layer::AveragePool(_)
                    | Layer::MaxPool2d(_) => {
                        // Should not happen due to early check
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    Layer::Softmax(_)
                    | Layer::CausalSoftmax(_)
                    | Layer::GELU(_)
                    | Layer::LayerNorm(_)
                    | Layer::BatchNorm(_) => {
                        // Transformer/normalization ops not supported in α-CROWN, fall back to CROWN
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    Layer::Transpose(t) => {
                        // Transpose is a linear operation
                        let next = t.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::AddConstant(ac) => {
                        // AddConstant is a linear operation (adds constant to bias)
                        let next = ac.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::Reshape(r) => {
                        // Reshape is a linear operation (index permutation)
                        let next = r.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::Flatten(f) => {
                        // Flatten is a linear operation (dimension collapsing)
                        let next = f.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::MulConstant(m) => {
                        // MulConstant is a linear operation (scaling)
                        let next = m.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::Abs(_) => {
                        // Abs is nonlinear, fall back to CROWN/IBP
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    Layer::DivConstant(d) => {
                        // DivConstant is a linear operation (inverse scaling)
                        let next = d.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::SubConstant(s) => {
                        // SubConstant is a linear operation
                        let next = s.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::Sqrt(_) | Layer::PowConstant(_) => {
                        // Nonlinear ops, fall back to CROWN/IBP
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    Layer::ReduceMean(rm) => {
                        linear_bounds =
                            rm.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                    Layer::ReduceSum(rs) => {
                        linear_bounds =
                            rs.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                    Layer::Tanh(_)
                    | Layer::Sigmoid(_)
                    | Layer::Softplus(_)
                    | Layer::LeakyReLU(_)
                    | Layer::Clip(_)
                    | Layer::Elu(_)
                    | Layer::Selu(_)
                    | Layer::PRelu(_)
                    | Layer::HardSigmoid(_)
                    | Layer::HardSwish(_)
                    | Layer::Exp(_)
                    | Layer::Log(_)
                    | Layer::Celu(_)
                    | Layer::Mish(_)
                    | Layer::LogSoftmax(_)
                    | Layer::ThresholdedRelu(_)
                    | Layer::Shrink(_)
                    | Layer::Softsign(_)
                    | Layer::Floor(_)
                    | Layer::Ceil(_)
                    | Layer::Round(_)
                    | Layer::Sign(_)
                    | Layer::Reciprocal(_) => {
                        // Activation ops not supported in α-CROWN, fall back to CROWN
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    Layer::Sin(_) | Layer::Cos(_) | Layer::Tile(_) | Layer::Slice(_) => {
                        // Trigonometric ops, Tile, and Slice not supported in α-CROWN
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    Layer::MatMul(_)
                    | Layer::MulBinary(_)
                    | Layer::Add(_)
                    | Layer::Concat(_)
                    | Layer::Sub(_)
                    | Layer::Div(_)
                    | Layer::Where(_)
                    | Layer::NonZero(_) => {
                        // Binary/ternary/data-dependent ops not supported in sequential α-CROWN
                        return self.propagate_crown_with_engine(input, engine);
                    }
                }
            }

            // Concretize to get actual bounds
            let concrete_bounds = linear_bounds.concretize(input);

            // Update element-wise best bounds using flat iteration to handle any array shape:
            // - best_lower[i] = max(best_lower[i], concrete_bounds.lower[i])
            // - best_upper[i] = min(best_upper[i], concrete_bounds.upper[i])
            if let (Some(best_l_slice), Some(curr_l_slice)) =
                (best_lower.as_slice_mut(), concrete_bounds.lower.as_slice())
            {
                for (best, &curr) in best_l_slice.iter_mut().zip(curr_l_slice.iter()) {
                    if curr > *best {
                        *best = curr;
                    }
                }
            }
            if let (Some(best_u_slice), Some(curr_u_slice)) =
                (best_upper.as_slice_mut(), concrete_bounds.upper.as_slice())
            {
                for (best, &curr) in best_u_slice.iter_mut().zip(curr_u_slice.iter()) {
                    if curr < *best {
                        *best = curr;
                    }
                }
            }

            let lower_sum: f32 = concrete_bounds
                .lower
                .as_slice()
                .map(|s| s.iter().sum())
                .unwrap_or(0.0);

            // Track best lower_sum for early stopping
            if lower_sum > best_lower_sum {
                best_lower_sum = lower_sum;
            }

            // Early stopping check (compare best improvement since last iteration).
            let best_improvement = best_lower_sum - prev_best_lower_sum;
            if best_improvement < config.tolerance {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
            }
            if iter > 0 && no_improve_iters >= 3 {
                debug!(
                    "α-CROWN: Converged at iteration {} (best improvement < {} for {} iters)",
                    iter, config.tolerance, no_improve_iters
                );
                break;
            }

            // Compute gradient using configured method
            let eps = 1e-3; // Perturbation magnitude
            let num_relus = relu_layer_indices.len();
            let numerical_gradients: Vec<Array1<f32>> = match config.gradient_method {
                GradientMethod::Spsa => {
                    // SPSA: Simultaneous Perturbation Stochastic Approximation
                    // Perturb ALL parameters at once with random directions
                    // Only requires 2 forward passes per sample (vs 2*n for finite diff)
                    use rand::Rng;
                    let mut rng = rand::rng();

                    let mut avg_grads: Vec<Array1<f32>> = (0..num_relus)
                        .map(|relu_idx| Array1::zeros(alpha_state.alphas[relu_idx].len()))
                        .collect();

                    // Save original alpha values for restoration
                    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();

                    // Average over multiple samples to reduce variance
                    for _sample in 0..config.spsa_samples {
                        // Generate random Bernoulli perturbation (+1 or -1) for each α
                        let perturbations: Vec<Array1<f32>> = (0..num_relus)
                            .map(|relu_idx| {
                                let n = alpha_state.alphas[relu_idx].len();
                                Array1::from_iter((0..n).map(|i| {
                                    if alpha_state.unstable_mask[relu_idx][i] {
                                        if rng.random_bool(0.5) {
                                            1.0
                                        } else {
                                            -1.0
                                        }
                                    } else {
                                        0.0 // Don't perturb stable neurons
                                    }
                                }))
                            })
                            .collect();

                        // Apply +ε perturbation from original
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                alpha_state.alphas[relu_idx][i] = (original_alphas[relu_idx][i]
                                    + eps * perturbations[relu_idx][i])
                                    .clamp(0.0, 1.0);
                            }
                        }
                        let bounds_plus = self
                            .propagate_alpha_crown_single_pass(
                                input,
                                &layer_bounds,
                                &alpha_state,
                                engine,
                            )
                            .unwrap_or_else(|_| concrete_bounds.clone());
                        let lower_plus: f32 = bounds_plus
                            .lower
                            .as_slice()
                            .map(|s| s.iter().sum())
                            .unwrap_or(0.0);

                        // Apply -ε perturbation from original
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                alpha_state.alphas[relu_idx][i] = (original_alphas[relu_idx][i]
                                    - eps * perturbations[relu_idx][i])
                                    .clamp(0.0, 1.0);
                            }
                        }
                        let bounds_minus = self
                            .propagate_alpha_crown_single_pass(
                                input,
                                &layer_bounds,
                                &alpha_state,
                                engine,
                            )
                            .unwrap_or_else(|_| concrete_bounds.clone());
                        let lower_minus: f32 = bounds_minus
                            .lower
                            .as_slice()
                            .map(|s| s.iter().sum())
                            .unwrap_or(0.0);

                        // SPSA gradient estimate: g_i = (f+ - f-) / (2 * eps * Δ_i)
                        let diff = lower_plus - lower_minus;
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                if alpha_state.unstable_mask[relu_idx][i]
                                    && perturbations[relu_idx][i].abs() > 0.5
                                {
                                    avg_grads[relu_idx][i] +=
                                        diff / (2.0 * eps * perturbations[relu_idx][i]);
                                }
                            }
                        }
                    }

                    // Restore original alpha values
                    for (alpha, original) in
                        alpha_state.alphas.iter_mut().zip(original_alphas.iter())
                    {
                        alpha.assign(original);
                    }

                    // Average the gradients
                    let num_samples = config.spsa_samples as f32;
                    for grad in &mut avg_grads {
                        *grad /= num_samples;
                    }
                    avg_grads
                }
                GradientMethod::FiniteDifferences => {
                    // Finite differences: perturb each α individually
                    let mut grads = Vec::with_capacity(num_relus);
                    for relu_idx in 0..num_relus {
                        let num_neurons = alpha_state.alphas[relu_idx].len();
                        let mut grad = Array1::<f32>::zeros(num_neurons);

                        // Only compute gradient for unstable neurons
                        for neuron_idx in 0..num_neurons {
                            if !alpha_state.unstable_mask[relu_idx][neuron_idx] {
                                continue;
                            }

                            let orig_alpha = alpha_state.alphas[relu_idx][neuron_idx];

                            // Compute f(α + ε)
                            alpha_state.alphas[relu_idx][neuron_idx] =
                                (orig_alpha + eps).clamp(0.0, 1.0);
                            let bounds_plus = self
                                .propagate_alpha_crown_single_pass(
                                    input,
                                    &layer_bounds,
                                    &alpha_state,
                                    engine,
                                )
                                .unwrap_or_else(|_| concrete_bounds.clone());
                            let lower_plus: f32 = bounds_plus
                                .lower
                                .as_slice()
                                .map(|s| s.iter().sum())
                                .unwrap_or(0.0);

                            // Compute f(α - ε)
                            alpha_state.alphas[relu_idx][neuron_idx] =
                                (orig_alpha - eps).clamp(0.0, 1.0);
                            let bounds_minus = self
                                .propagate_alpha_crown_single_pass(
                                    input,
                                    &layer_bounds,
                                    &alpha_state,
                                    engine,
                                )
                                .unwrap_or_else(|_| concrete_bounds.clone());
                            let lower_minus: f32 = bounds_minus
                                .lower
                                .as_slice()
                                .map(|s| s.iter().sum())
                                .unwrap_or(0.0);

                            // Restore original alpha
                            alpha_state.alphas[relu_idx][neuron_idx] = orig_alpha;

                            // Central difference gradient
                            grad[neuron_idx] = (lower_plus - lower_minus) / (2.0 * eps);
                        }
                        grads.push(grad);
                    }
                    grads
                }
                GradientMethod::Analytic => {
                    // Analytic gradients: use the gradients computed during the backward pass.
                    // The gradient for α_i is the sum of incoming A coefficients where the
                    // lower relaxation (y >= α*x) is used, i.e., where A[j,i] >= 0.
                    // This is already computed by propagate_linear_with_alpha().
                    //
                    // No additional CROWN passes needed - O(1) per iteration.
                    gradients.clone()
                }
                GradientMethod::AnalyticChain => {
                    // True chain-rule gradients: compute ∂(output_lower)/∂α_i by
                    // properly chaining gradients through all downstream layers.
                    //
                    // This stores intermediate A matrices at each ReLU and uses them
                    // to compute how each alpha affects the final output bounds.
                    let intermediate = self
                        .propagate_alpha_crown_with_intermediates(
                            input,
                            &layer_bounds,
                            &alpha_state,
                            engine,
                        )
                        .unwrap_or_else(|_| AlphaCrownIntermediate::default());

                    if intermediate.a_at_relu.is_empty() {
                        // Fall back to local gradients if intermediate storage failed
                        debug!("AnalyticChain: intermediate storage failed, falling back to local gradients");
                        gradients.clone()
                    } else {
                        self.compute_chain_rule_gradients(
                            input,
                            &layer_bounds,
                            &alpha_state,
                            &intermediate,
                        )
                    }
                }
            };

            // Debug output for first iteration (only when RUST_LOG=debug)
            if iter == 0 {
                for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                    let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                    debug!(
                        "α-CROWN iter 0: ReLU layer {} gradient L2 norm={:.6} ({:?})",
                        relu_idx, grad_norm, config.gradient_method
                    );
                }
            }

            // Update alpha using numerical gradient (gradient ascent to maximize lower bound)
            // For gradient ascent, negate the gradient (we want to maximize lower bound).
            let adam_params = config.adam_params(lr, iter + 1);
            for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                let neg_grad = grad.mapv(|v| -v);
                match config.optimizer {
                    Optimizer::Adam => {
                        // Adam optimizer: adaptive moment estimation
                        alpha_state.update_adam(relu_idx, &neg_grad, &adam_params);
                    }
                    Optimizer::Sgd => {
                        // SGD with momentum
                        let momentum = if config.use_momentum {
                            config.momentum
                        } else {
                            0.0
                        };
                        alpha_state.update(relu_idx, &neg_grad, lr, momentum);
                    }
                }
            }

            // Learning rate decay
            lr *= config.lr_decay;

            if iter % 5 == 0 {
                debug!(
                    "α-CROWN iter {}: lower_sum = {:.6}, lr = {:.6}",
                    iter, lower_sum, lr
                );
            }

            prev_best_lower_sum = best_lower_sum;
        }

        // Return element-wise best bounds found across all iterations.
        // If no valid bounds were found, fall back to CROWN.
        let has_valid_bounds =
            best_lower.iter().all(|&v| v.is_finite()) && best_upper.iter().all(|&v| v.is_finite());

        if has_valid_bounds {
            Ok(BoundedTensor::new(best_lower, best_upper).unwrap_or_else(|_| input.clone()))
        } else {
            // Fall back to CROWN if no valid bounds were found
            self.propagate_crown_with_engine(input, engine)
        }
    }

    /// Single forward+backward pass with given alpha state (for numerical gradient).
    fn propagate_alpha_crown_single_pass(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let output_dim = layer_bounds.last().map(|b| b.len()).unwrap_or(input.len());

        // Build map from layer index to alpha_state index
        let relu_layer_indices: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, Layer::ReLU(_)))
            .map(|(i, _)| i)
            .collect();
        let mut layer_to_relu_idx: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (relu_idx, &layer_idx) in relu_layer_indices.iter().enumerate() {
            layer_to_relu_idx.insert(layer_idx, relu_idx);
        }

        // Backward pass
        let mut linear_bounds = LinearBounds::identity(output_dim);
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear_with_engine(&linear_bounds, engine)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    if let Some(&relu_idx) = layer_to_relu_idx.get(&i) {
                        if let Some(alpha) = alpha_state.get_alpha(relu_idx) {
                            let (new_bounds, _) = r.propagate_linear_with_alpha(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                            )?;
                            linear_bounds = new_bounds;
                        } else {
                            linear_bounds =
                                r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                        }
                    } else {
                        linear_bounds =
                            r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                }
                Layer::Transpose(t) => {
                    let next = t.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AddConstant(ac) => {
                    let next = ac.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Reshape(r) => {
                    let next = r.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::MulConstant(m) => {
                    let next = m.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::DivConstant(d) => {
                    let next = d.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::SubConstant(s) => {
                    let next = s.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Flatten(f) => {
                    let next = f.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                _ => {
                    // For unsupported layers, fall back to CROWN
                    return self.propagate_crown_with_engine(input, engine);
                }
            }
        }

        Ok(linear_bounds.concretize(input))
    }

    /// Single forward+backward pass with given alpha state, storing intermediate A matrices
    /// for chain-rule gradient computation.
    ///
    /// This is similar to `propagate_alpha_crown_single_pass` but additionally stores
    /// the A matrix at each ReLU layer BEFORE the ReLU is applied, enabling computation
    /// of true chain-rule gradients.
    fn propagate_alpha_crown_with_intermediates(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<AlphaCrownIntermediate> {
        let output_dim = layer_bounds.last().map(|b| b.len()).unwrap_or(input.len());

        // Build map from layer index to alpha_state index
        let relu_layer_indices: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, Layer::ReLU(_)))
            .map(|(i, _)| i)
            .collect();
        let mut layer_to_relu_idx: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (relu_idx, &layer_idx) in relu_layer_indices.iter().enumerate() {
            layer_to_relu_idx.insert(layer_idx, relu_idx);
        }

        let mut a_at_relu: Vec<Array2<f32>> = Vec::new();
        let mut pre_relu_bounds: Vec<(Array1<f32>, Array1<f32>)> = Vec::new();

        // Backward pass
        let mut linear_bounds = LinearBounds::identity(output_dim);
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear_with_engine(&linear_bounds, engine)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    if let Some(&relu_idx) = layer_to_relu_idx.get(&i) {
                        // Store A matrix BEFORE this ReLU is applied
                        // This is the A matrix from output to this ReLU's input
                        a_at_relu.push(linear_bounds.lower_a.clone());

                        // Store pre-ReLU bounds for this layer
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
                        pre_relu_bounds.push((lower, upper));

                        // Apply ReLU with alpha
                        if let Some(alpha) = alpha_state.get_alpha(relu_idx) {
                            let (new_bounds, _) = r.propagate_linear_with_alpha(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                            )?;
                            linear_bounds = new_bounds;
                        } else {
                            linear_bounds =
                                r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                        }
                    } else {
                        linear_bounds =
                            r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                }
                Layer::Transpose(t) => {
                    let next = t.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::AddConstant(ac) => {
                    let next = ac.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Reshape(rsh) => {
                    let next = rsh.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::MulConstant(m) => {
                    let next = m.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::DivConstant(d) => {
                    let next = d.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::SubConstant(s) => {
                    let next = s.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::Flatten(f) => {
                    let next = f.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                _ => {
                    // For unsupported layers, return empty intermediate with CROWN bounds
                    let crown_bounds = self.propagate_crown_with_engine(input, engine)?;
                    return Ok(AlphaCrownIntermediate {
                        a_at_relu: Vec::new(),
                        pre_relu_bounds: Vec::new(),
                        final_bounds: LinearBounds::identity(crown_bounds.len()),
                    });
                }
            }
        }

        // Reverse to get forward layer order (we collected in backward order)
        a_at_relu.reverse();
        pre_relu_bounds.reverse();

        Ok(AlphaCrownIntermediate {
            a_at_relu,
            pre_relu_bounds,
            final_bounds: linear_bounds,
        })
    }

    /// Compute true chain-rule gradients for alpha parameters.
    ///
    /// For each unstable neuron i in ReLU layer k:
    /// ∂(output_lower_sum)/∂α_k[i] = Σ_j A_to_relu[j,i] × dRelax/dα × input_sensitivity[i]
    ///
    /// Where:
    /// - A_to_relu[j,i] is the coefficient from output j to neuron i (before ReLU k)
    /// - dRelax/dα depends on whether we're using lower or upper relaxation
    /// - input_sensitivity captures how the neuron value affects downstream computation
    ///
    /// This properly chains gradients through all downstream layers.
    fn compute_chain_rule_gradients(
        &self,
        input: &BoundedTensor,
        _layer_bounds: &[BoundedTensor], // Reserved for future: computing input contribution
        alpha_state: &AlphaState,
        intermediate: &AlphaCrownIntermediate,
    ) -> Vec<Array1<f32>> {
        if intermediate.a_at_relu.is_empty() {
            // Fall back to empty gradients if no intermediates were stored
            return alpha_state
                .alphas
                .iter()
                .map(|a| Array1::zeros(a.len()))
                .collect();
        }

        // Input bounds for future use in computing full input contribution
        let input_flat = input.flatten();
        let _x_lower: Array1<f32> = input_flat
            .lower
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap_or_else(|_| Array1::zeros(input.len()));
        let _x_upper: Array1<f32> = input_flat
            .upper
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap_or_else(|_| Array1::zeros(input.len()));

        let num_relus = intermediate.a_at_relu.len();
        let mut gradients: Vec<Array1<f32>> = Vec::with_capacity(num_relus);

        // For each ReLU layer
        for relu_idx in 0..num_relus {
            let a_at_relu = &intermediate.a_at_relu[relu_idx];
            let (pre_lower, pre_upper) = &intermediate.pre_relu_bounds[relu_idx];
            let n_neurons = pre_lower.len();

            let mut grad = Array1::<f32>::zeros(n_neurons);

            // For each neuron in this ReLU layer
            for i in 0..n_neurons {
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Only unstable neurons (l < 0 < u) have non-zero gradient
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Compute gradient contribution from all output dimensions
                // The gradient of the lower bound w.r.t. α_i depends on:
                // 1. A_to_relu[j,i] - how output j depends on neuron i's value
                // 2. The relaxation: for lower bound with A[j,i] >= 0, we use y >= α*x
                //    The contribution to lower bound is: A[j,i] * α * x
                //    So ∂(lower)/∂α = A[j,i] * x (where x is the pre-ReLU value)
                //
                // For maximizing the lower bound:
                // For lower relaxation y >= α*x with x ∈ [l, u] where l < 0 < u:
                // - Contribution to lower bound = A[j,i] * α * min(x) = A[j,i] * α * l
                // - Gradient ∂bound/∂α = A[j,i] * l
                // Note: l < 0 for unstable neurons, so gradient is typically negative
                // when A[j,i] > 0, meaning increasing α decreases the lower bound.

                let num_outputs = a_at_relu.nrows();
                let mut grad_i = 0.0f32;

                for j in 0..num_outputs {
                    let a_ji = a_at_relu[[j, i]];

                    // When A >= 0, lower relaxation uses y >= α*x
                    // The binding point is x = l (lower bound), not u
                    // because we minimize α*x over [l,u] with α >= 0 and l < 0
                    if a_ji > 0.0 {
                        // Lower relaxation active: y >= α*x
                        // Contribution to lower bound: A[j,i] * α * l
                        // Gradient w.r.t. α: A[j,i] * l
                        grad_i += a_ji * l;
                    }
                    // When A < 0, upper relaxation y <= (u/(u-l))*(x-l) is used
                    // This doesn't depend on α, so gradient is 0
                }

                grad[i] = grad_i;
            }

            gradients.push(grad);
        }

        gradients
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GraphNetwork: DAG-based computation graph for attention patterns
// =============================================================================

/// A node in a computation graph.
///
/// Each node represents a single operation that takes one or more inputs
/// and produces a single output. Nodes can reference other nodes' outputs
/// or the network input.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub name: String,
    /// The layer/operation to apply.
    pub layer: Layer,
    /// Names of input nodes. For unary ops, this has 1 element.
    /// For binary ops (MatMul, Add), this has 2 elements.
    /// Special value: "_input" refers to the network input.
    pub inputs: Vec<String>,
}

impl GraphNode {
    /// Create a new graph node.
    pub fn new(name: impl Into<String>, layer: Layer, inputs: Vec<String>) -> Self {
        Self {
            name: name.into(),
            layer,
            inputs,
        }
    }

    /// Create a node that takes network input.
    pub fn from_input(name: impl Into<String>, layer: Layer) -> Self {
        Self::new(name, layer, vec!["_input".to_string()])
    }

    /// Create a binary operation node.
    pub fn binary(
        name: impl Into<String>,
        layer: Layer,
        input_a: impl Into<String>,
        input_b: impl Into<String>,
    ) -> Self {
        Self::new(name, layer, vec![input_a.into(), input_b.into()])
    }
}

/// A neural network represented as a directed acyclic graph (DAG).
///
/// Unlike `Network` which is sequential, `GraphNetwork` can represent
/// branching computations like attention (Q/K/V projections, matmul, softmax).
///
/// # Example: Simplified Attention
/// ```text
///              input
///             /  |  \
///          Q_proj K_proj V_proj
///             \  |      |
///              \ |      |
///           matmul(Q,K^T)
///                |      |
///             softmax   |
///                \     /
///              matmul(attn, V)
///                   |
///                output
/// ```
#[derive(Debug, Clone)]
pub struct GraphNetwork {
    /// All nodes in the graph, keyed by name.
    pub(crate) nodes: std::collections::HashMap<String, GraphNode>,
    /// Order of node names for iteration.
    node_order: Vec<String>,
    /// Name of the output node.
    output_node: String,
}

impl GraphNetwork {
    /// Decide whether to use the expensive O(N^2) CROWN-IBP intermediate tightening pass.
    ///
    /// For CNN-style DAGs (e.g., ResNets), CROWN-IBP intermediates dramatically improve
    /// ReLU relaxations across skip connections.
    ///
    /// For transformer-style graphs (MatMul/Softmax/LayerNorm/GELU/etc.), we prefer the
    /// forward IBP pass (`collect_node_bounds`) which also contains transformer-specific
    /// tightening (e.g., attention matmul bounds) and avoids unsupported ops in the
    /// CROWN-to-node tightening pass.
    fn should_use_crown_ibp_intermediates(&self) -> bool {
        !self.nodes.values().any(|node| {
            matches!(
                node.layer,
                Layer::MatMul(_)
                    | Layer::Softmax(_)
                    | Layer::CausalSoftmax(_)
                    | Layer::LayerNorm(_)
                    | Layer::GELU(_)
                    | Layer::MulBinary(_)
            )
        })
    }

    /// Create a new empty graph network.
    pub fn new() -> Self {
        Self {
            nodes: std::collections::HashMap::new(),
            node_order: Vec::new(),
            output_node: String::new(),
        }
    }

    /// Add a node to the graph.
    ///
    /// Nodes should be added in topological order (dependencies before dependents).
    pub fn add_node(&mut self, node: GraphNode) {
        let name = node.name.clone();
        self.node_order.push(name.clone());
        self.nodes.insert(name, node);
    }

    /// Set the output node.
    pub fn set_output(&mut self, name: impl Into<String>) {
        self.output_node = name.into();
    }

    /// Get the output node name.
    pub fn get_output_name(&self) -> &str {
        &self.output_node
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<&GraphNode> {
        self.nodes.get(name)
    }

    /// Get all node names in topological order.
    pub fn node_names(&self) -> &[String] {
        &self.node_order
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Enable or disable forward mode for all LayerNorm nodes in the graph.
    ///
    /// Forward mode uses the center point (midpoint of bounds) for mean/std
    /// computation, dramatically reducing bound explosion (up to 80x tighter
    /// bounds) but may not be perfectly sound for large perturbations.
    ///
    /// Returns the number of LayerNorm nodes modified.
    pub fn set_layernorm_forward_mode(&mut self, enabled: bool) -> usize {
        let mut count = 0;
        for node in self.nodes.values_mut() {
            if let Layer::LayerNorm(ref mut ln) = node.layer {
                ln.forward_mode = enabled;
                count += 1;
            }
        }
        count
    }

    /// Create a copy of this graph with forward mode enabled for all LayerNorm nodes.
    pub fn with_layernorm_forward_mode(mut self, enabled: bool) -> Self {
        self.set_layernorm_forward_mode(enabled);
        self
    }

    /// Perform topological sort to get valid execution order.
    ///
    /// Returns node names in order such that all dependencies come before dependents.
    /// Returns an error if the graph contains cycles.
    pub fn topological_sort(&self) -> Result<Vec<String>> {
        let mut visited = std::collections::HashSet::with_capacity(self.nodes.len());
        let mut temp_mark = std::collections::HashSet::with_capacity(self.nodes.len());
        let mut result = Vec::with_capacity(self.nodes.len());

        fn visit(
            name: &str,
            nodes: &std::collections::HashMap<String, GraphNode>,
            visited: &mut std::collections::HashSet<String>,
            temp_mark: &mut std::collections::HashSet<String>,
            result: &mut Vec<String>,
        ) -> Result<()> {
            if visited.contains(name) {
                return Ok(());
            }
            if temp_mark.contains(name) {
                return Err(GammaError::InvalidSpec(format!(
                    "Cycle detected in graph at node: {}",
                    name
                )));
            }
            if name == "_input" {
                return Ok(());
            }

            temp_mark.insert(name.to_string());

            if let Some(node) = nodes.get(name) {
                for input_name in &node.inputs {
                    visit(input_name, nodes, visited, temp_mark, result)?;
                }
            }

            temp_mark.remove(name);
            visited.insert(name.to_string());
            result.push(name.to_string());
            Ok(())
        }

        // Sort keys for deterministic topological ordering
        // (HashMap iteration order is non-deterministic)
        let mut sorted_keys: Vec<&String> = self.nodes.keys().collect();
        sorted_keys.sort();
        for name in sorted_keys {
            visit(name, &self.nodes, &mut visited, &mut temp_mark, &mut result)?;
        }

        Ok(result)
    }

    /// Propagate bounds through the graph using IBP.
    ///
    /// Executes nodes in topological order, storing intermediate bounds
    /// and using them for downstream operations.
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        fn summarize_bounds(bounds: &BoundedTensor) -> (f32, f32, bool, bool, bool) {
            const MAX_BOUND: f32 = f32::MAX / 2.0;
            let mut max_width = 0.0_f32;
            let mut max_abs = 0.0_f32;
            let mut saturated = false;
            let mut has_nan = false;
            let mut has_non_finite = false;

            for (&l, &u) in bounds.lower.iter().zip(bounds.upper.iter()) {
                if l.is_nan() || u.is_nan() {
                    has_nan = true;
                }
                if !l.is_finite() || !u.is_finite() {
                    has_non_finite = true;
                }
                let width = u - l;
                if width.is_finite() {
                    max_width = max_width.max(width);
                } else {
                    has_non_finite = true;
                }
                max_abs = max_abs.max(l.abs()).max(u.abs());
                if l <= -0.999 * MAX_BOUND || u >= 0.999 * MAX_BOUND {
                    saturated = true;
                }
            }

            (max_width, max_abs, saturated, has_nan, has_non_finite)
        }

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Store bounds for each node's output
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // Process nodes in topological order
        for node_name in &exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            debug!(
                "GraphNetwork IBP: processing node {} ({})",
                node_name,
                node.layer.layer_type()
            );

            let output_bounds = match &node.layer {
                Layer::Where(w) => {
                    // If WhereLayer has embedded constants, use them with single condition input
                    if w.has_embedded_constants() {
                        if node.inputs.is_empty() {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} with embedded constants requires 1 input (condition)",
                                node_name
                            )));
                        }
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        w.propagate_ibp_with_condition(cond)?
                    } else {
                        // Standard Where: needs 3 graph inputs
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs, got {}",
                                node_name,
                                node.inputs.len()
                            )));
                        }
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        let x = self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                        let y = self.get_bounds_ref(&node.inputs[2], input, &bounds_cache)?;
                        w.propagate_ibp_ternary(cond, x, y)?
                    }
                }
                _ if matches!(&node.layer, Layer::Concat(_)) => {
                    // Concat: N-ary operation (2+ inputs)
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Concat node {} requires at least 2 inputs, got {}",
                            node_name,
                            node.inputs.len()
                        )));
                    }

                    if let Layer::Concat(concat) = &node.layer {
                        // Collect bounds for all inputs, using stored constants if available
                        let input_bounds: Vec<&BoundedTensor> = node
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(i, inp_name)| {
                                concat.get_constant_input(i).map(Ok).unwrap_or_else(|| {
                                    self.get_bounds_ref(inp_name, input, &bounds_cache)
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;

                        // Use N-ary propagation
                        concat.propagate_ibp_nary(&input_bounds)?
                    } else {
                        unreachable!()
                    }
                }
                _ if node.layer.is_binary() => {
                    // Binary operation: get two inputs
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs, got {}",
                            node_name,
                            node.inputs.len()
                        )));
                    }

                    match &node.layer {
                        Layer::MatMul(matmul) if matmul.transpose_b => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            if let Some(tighter) = self.try_attention_matmul_bounds_zonotope(
                                node,
                                input,
                                &bounds_cache,
                            )? {
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(input_a, input_b)?
                            }
                        }
                        _ => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            node.layer.propagate_ibp_binary(input_a, input_b)?
                        }
                    }
                }
                _ => {
                    // Unary operation: get single input
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }

                    let node_input = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                    node.layer.propagate_ibp(node_input)?
                }
            };

            let (max_width, max_abs, saturated, has_nan, has_non_finite) =
                summarize_bounds(&output_bounds);
            debug!(
                "GraphNetwork IBP: {} ({}) shape {:?} max_width {:.2e} max_abs {:.2e} saturated={} nan={} non_finite={}",
                node_name,
                node.layer.layer_type(),
                output_bounds.shape(),
                max_width,
                max_abs,
                saturated,
                has_nan,
                has_non_finite
            );
            if saturated || has_nan || has_non_finite {
                debug!(
                    "GraphNetwork IBP: WARNING: bounds degraded at {} ({})",
                    node_name,
                    node.layer.layer_type()
                );
            }

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        // Return the output node's bounds
        if self.output_node.is_empty() {
            // Use the last node in exec order as output
            let last_name = exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?;
            bounds_cache.remove(last_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!("Output bounds not found for {}", last_name))
            })
        } else {
            bounds_cache.remove(&self.output_node).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "Output node {} not found in results",
                    self.output_node
                ))
            })
        }
    }

    /// Collect activation statistics from a concrete forward pass.
    ///
    /// Runs a concrete forward pass through the network using the center values
    /// of the input bounds, collecting per-layer activation statistics for use
    /// with domain clipping.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (center values will be used)
    /// * `clipper` - Domain clipper to store statistics in
    ///
    /// # Example
    /// ```ignore
    /// let mut clipper = DomainClipper::default();
    /// // Collect statistics from multiple samples
    /// for sample in samples {
    ///     let input = BoundedTensor::concrete(sample);
    ///     graph.collect_activation_statistics(&input, &mut clipper)?;
    /// }
    /// // Use clipper for tighter bounds
    /// let bounds = graph.propagate_ibp_with_clipper(&input, &mut clipper)?;
    /// ```
    pub fn collect_activation_statistics(
        &self,
        input: &BoundedTensor,
        clipper: &mut DomainClipper,
    ) -> Result<()> {
        if self.nodes.is_empty() {
            return Ok(());
        }

        // Use center of input bounds as concrete value
        let center = (&input.lower + &input.upper) / 2.0;

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Store concrete values for each node
        let mut value_cache: std::collections::HashMap<String, ArrayD<f32>> =
            std::collections::HashMap::new();

        // Process nodes in topological order
        for node_name in &exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get input value(s)
            let output_value = match &node.layer {
                Layer::Where(w) => {
                    if w.has_embedded_constants() {
                        let cond =
                            self.get_concrete_value(&node.inputs[0], &center, &value_cache)?;
                        let cond_bt = BoundedTensor::concrete(cond);
                        let out = w.propagate_ibp_with_condition(&cond_bt)?;
                        out.lower
                    } else {
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs",
                                node_name
                            )));
                        }

                        let cond =
                            self.get_concrete_value(&node.inputs[0], &center, &value_cache)?;
                        let x = self.get_concrete_value(&node.inputs[1], &center, &value_cache)?;
                        let y = self.get_concrete_value(&node.inputs[2], &center, &value_cache)?;

                        let cond_bt = BoundedTensor::concrete(cond);
                        let x_bt = BoundedTensor::concrete(x);
                        let y_bt = BoundedTensor::concrete(y);
                        let out = w.propagate_ibp_ternary(&cond_bt, &x_bt, &y_bt)?;
                        out.lower
                    }
                }
                _ if node.layer.is_binary() => {
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs",
                            node_name
                        )));
                    }
                    let input_a =
                        self.get_concrete_value(&node.inputs[0], &center, &value_cache)?;
                    let input_b =
                        self.get_concrete_value(&node.inputs[1], &center, &value_cache)?;

                    // Create bounded tensor from concrete values for propagation
                    let bounds_a = BoundedTensor::concrete(input_a);
                    let bounds_b = BoundedTensor::concrete(input_b);
                    let output = node.layer.propagate_ibp_binary(&bounds_a, &bounds_b)?;
                    // Use center (lower == upper for concrete)
                    output.lower
                }
                _ => {
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }
                    let node_input =
                        self.get_concrete_value(&node.inputs[0], &center, &value_cache)?;

                    // Create bounded tensor from concrete value for propagation
                    let bounds_input = BoundedTensor::concrete(node_input);
                    let output = node.layer.propagate_ibp(&bounds_input)?;
                    output.lower
                }
            };

            // Record statistics for this layer
            clipper.observe(node_name, &output_value)?;

            // Store for downstream nodes
            value_cache.insert(node_name.clone(), output_value);
        }

        Ok(())
    }

    /// Helper to get concrete value from cache or input.
    fn get_concrete_value(
        &self,
        input_name: &str,
        center: &ArrayD<f32>,
        cache: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        if input_name == "_input" {
            Ok(center.clone())
        } else {
            cache.get(input_name).cloned().ok_or_else(|| {
                GammaError::InvalidSpec(format!("Concrete value not found for {}", input_name))
            })
        }
    }

    /// Propagate bounds through the graph using IBP with domain clipping.
    ///
    /// Similar to `propagate_ibp`, but applies domain clipping after each layer
    /// using statistics collected by the clipper. This can significantly tighten
    /// bounds for deep networks where bound explosion is a problem.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor
    /// * `clipper` - Domain clipper with pre-collected statistics
    ///
    /// # Returns
    /// Output bounds after propagation with clipping applied.
    ///
    /// # Example
    /// ```ignore
    /// // First collect statistics from concrete samples
    /// let mut clipper = DomainClipper::default();
    /// for sample in samples {
    ///     graph.collect_activation_statistics(&BoundedTensor::concrete(sample), &mut clipper)?;
    /// }
    ///
    /// // Then propagate with clipping
    /// let input = BoundedTensor::from_epsilon(center, epsilon);
    /// let bounds = graph.propagate_ibp_with_clipper(&input, &mut clipper)?;
    /// ```
    pub fn propagate_ibp_with_clipper(
        &self,
        input: &BoundedTensor,
        clipper: &mut DomainClipper,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Store bounds for each node's output
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // Track total clipping effect
        let mut total_reduction = 0.0_f32;

        // Process nodes in topological order
        for node_name in &exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Compute output bounds
            let output_bounds = match &node.layer {
                Layer::Where(w) => {
                    if w.has_embedded_constants() {
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        w.propagate_ibp_with_condition(cond)?
                    } else {
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs",
                                node_name
                            )));
                        }
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        let x = self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                        let y = self.get_bounds_ref(&node.inputs[2], input, &bounds_cache)?;
                        w.propagate_ibp_ternary(cond, x, y)?
                    }
                }
                _ if node.layer.is_binary() => {
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs",
                            node_name
                        )));
                    }
                    let input_a = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                    let input_b = self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                    node.layer.propagate_ibp_binary(input_a, input_b)?
                }
                _ => {
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }
                    let node_input = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                    node.layer.propagate_ibp(node_input)?
                }
            };

            // Apply domain clipping
            let (clipped_bounds, reduction) = clipper.clip_bounds(node_name, &output_bounds)?;
            total_reduction += reduction;

            bounds_cache.insert(node_name.clone(), clipped_bounds);
        }

        if total_reduction > 0.0 {
            debug!(
                "Domain clipping reduced total bound width by {:.2e}",
                total_reduction
            );
        }

        // Return the output node's bounds
        if self.output_node.is_empty() {
            let last_name = exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?;
            bounds_cache.remove(last_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!("Output bounds not found for {}", last_name))
            })
        } else {
            bounds_cache.remove(&self.output_node).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "Output node {} not found in results",
                    self.output_node
                ))
            })
        }
    }

    /// Propagate bounds through the graph using IBP, returning detailed per-node information.
    ///
    /// This is useful for layer-by-layer verification to track bound growth through the
    /// network and identify where bounds saturate or degrade.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor
    /// * `epsilon` - Input perturbation epsilon (for reporting)
    ///
    /// # Returns
    /// A `LayerByLayerResult` containing detailed bounds information for each node.
    pub fn propagate_ibp_detailed(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<LayerByLayerResult> {
        self.propagate_ibp_detailed_with_progress(input, epsilon, None::<fn(LayerProgress)>)
    }

    /// Propagate bounds through the graph using IBP, returning detailed per-node information,
    /// with optional progress callback.
    ///
    /// Same as `propagate_ibp_detailed`, but calls the provided callback after each node is
    /// processed. This is useful for long-running layer-by-layer runs on large graphs.
    pub fn propagate_ibp_detailed_with_progress<F>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
    ) -> Result<LayerByLayerResult>
    where
        F: Fn(LayerProgress),
    {
        let start_time = std::time::Instant::now();

        if self.nodes.is_empty() {
            return Ok(LayerByLayerResult {
                nodes: vec![],
                input_epsilon: epsilon,
                final_width: input.max_width(),
                degraded_at_node: None,
                total_nodes: 0,
            });
        }

        const MAX_BOUND: f32 = f32::MAX / 2.0;

        // Get execution order
        let exec_order = self.topological_sort()?;
        let total_nodes = exec_order.len();

        // Store bounds for each node's output
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::with_capacity(total_nodes);

        // Track per-node information
        let mut node_infos: Vec<NodeBoundsInfo> = Vec::with_capacity(total_nodes);
        let mut degraded_at_node: Option<usize> = None;
        let mut degraded_so_far: usize = 0;
        let mut max_sensitivity_so_far: f32 = 1.0;

        // Process nodes in topological order
        for (node_index, node_name) in exec_order.iter().enumerate() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get input bounds width.
            //
            // For DAG nodes (especially binary ops like Add/MatMul), use the max width across
            // all inputs to avoid underestimating the node's effective input uncertainty.
            let input_width = if node.inputs.is_empty() {
                input.max_width()
            } else {
                node.inputs
                    .iter()
                    .map(|inp| {
                        if inp == "_input" {
                            input.max_width()
                        } else {
                            bounds_cache
                                .get(inp)
                                .map(|b| b.max_width())
                                .unwrap_or(input.max_width())
                        }
                    })
                    .fold(0.0_f32, f32::max)
            };

            let output_bounds = match &node.layer {
                Layer::Where(w) => {
                    if w.has_embedded_constants() {
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        w.propagate_ibp_with_condition(cond)?
                    } else {
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs",
                                node_name
                            )));
                        }
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        let x = self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                        let y = self.get_bounds_ref(&node.inputs[2], input, &bounds_cache)?;
                        w.propagate_ibp_ternary(cond, x, y)?
                    }
                }
                _ if matches!(&node.layer, Layer::Concat(_)) => {
                    // Concat: N-ary operation (2+ inputs)
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Concat node {} requires at least 2 inputs, got {}",
                            node_name,
                            node.inputs.len()
                        )));
                    }

                    if let Layer::Concat(concat) = &node.layer {
                        let input_bounds: Vec<&BoundedTensor> = node
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(i, inp_name)| {
                                concat.get_constant_input(i).map(Ok).unwrap_or_else(|| {
                                    self.get_bounds_ref(inp_name, input, &bounds_cache)
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        concat.propagate_ibp_nary(&input_bounds)?
                    } else {
                        unreachable!()
                    }
                }
                _ if node.layer.is_binary() => {
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs",
                            node_name
                        )));
                    }
                    match &node.layer {
                        Layer::MatMul(matmul) if matmul.transpose_b => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            if let Some(tighter) = self.try_attention_matmul_bounds_zonotope(
                                node,
                                input,
                                &bounds_cache,
                            )? {
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(input_a, input_b)?
                            }
                        }
                        Layer::MulBinary(_) => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            // Try zonotope tightening for SwiGLU pattern (up * silu(gate))
                            if let Some(tighter) =
                                self.try_ffn_swiglu_bounds_zonotope(node, input, &bounds_cache)?
                            {
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(input_a, input_b)?
                            }
                        }
                        _ => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            node.layer.propagate_ibp_binary(input_a, input_b)?
                        }
                    }
                }
                _ => {
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }
                    let node_input = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                    node.layer.propagate_ibp(node_input)?
                }
            };

            // Collect statistics
            let output_width = output_bounds.max_width();
            let mut min_bound = f32::INFINITY;
            let mut max_bound = f32::NEG_INFINITY;
            let mut saturated = false;
            let mut has_nan = false;
            let mut has_infinite = false;

            for (&l, &u) in output_bounds.lower.iter().zip(output_bounds.upper.iter()) {
                if l.is_nan() || u.is_nan() {
                    has_nan = true;
                }
                if !l.is_finite() || !u.is_finite() {
                    has_infinite = true;
                }
                min_bound = min_bound.min(l);
                max_bound = max_bound.max(u);
                if l <= -0.999 * MAX_BOUND || u >= 0.999 * MAX_BOUND {
                    saturated = true;
                }
            }

            let sensitivity = if input_width > 0.0 && input_width.is_finite() {
                output_width / input_width
            } else if output_width == 0.0 {
                1.0
            } else {
                f32::INFINITY
            };

            let node_info = NodeBoundsInfo {
                name: node_name.clone(),
                layer_type: node.layer.layer_type().to_string(),
                input_width,
                output_width,
                sensitivity,
                output_shape: output_bounds.shape().to_vec(),
                min_bound,
                max_bound,
                saturated,
                has_nan,
                has_infinite,
            };

            // Track first degraded node
            if degraded_at_node.is_none() && node_info.has_degraded() {
                degraded_at_node = Some(node_infos.len());
            }

            if node_info.has_degraded() {
                degraded_so_far += 1;
            }
            if sensitivity.is_finite() {
                max_sensitivity_so_far = max_sensitivity_so_far.max(sensitivity);
            } else {
                max_sensitivity_so_far = f32::INFINITY;
            }

            if let Some(ref callback) = progress_callback {
                callback(LayerProgress {
                    node_index,
                    total_nodes,
                    node_name: node_info.name.clone(),
                    layer_type: node_info.layer_type.clone(),
                    elapsed: start_time.elapsed(),
                    current_max_sensitivity: max_sensitivity_so_far,
                    degraded_so_far,
                });
            }

            node_infos.push(node_info);
            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        // Get final output width
        let output_node_name = if self.output_node.is_empty() {
            exec_order.last().map(|s| s.as_str()).unwrap_or("")
        } else {
            &self.output_node
        };
        let final_width = bounds_cache
            .get(output_node_name)
            .map(|b| b.max_width())
            .unwrap_or(f32::INFINITY);

        Ok(LayerByLayerResult {
            nodes: node_infos,
            input_epsilon: epsilon,
            final_width,
            degraded_at_node,
            total_nodes,
        })
    }

    /// Propagate bounds through the graph block-wise with zonotope reset per block.
    ///
    /// This method processes transformer blocks independently, resetting bounds
    /// at each block boundary. This prevents bound explosion from propagating
    /// through the entire network and allows zonotope tightening to be effective
    /// for each block's Q@K^T attention.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    ///
    /// # Block Detection
    /// Blocks are detected by node name prefixes (e.g., "layer0_", "layer1_").
    /// Each block runs from attn_norm through add2 (the second residual add).
    ///
    /// # Returns
    /// A `BlockWiseResult` containing per-block sensitivity analysis.
    pub fn propagate_ibp_block_wise(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<BlockWiseResult> {
        self.propagate_ibp_block_wise_with_progress(input, epsilon, None::<fn(BlockProgress)>)
    }

    /// Block-wise IBP propagation with progress reporting callback.
    ///
    /// Same as `propagate_ibp_block_wise`, but calls the provided callback after each
    /// block is processed. This is useful for long-running verifications on large models.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    /// * `progress_callback` - Optional callback called after each block with progress info
    ///
    /// # Example
    /// ```ignore
    /// graph.propagate_ibp_block_wise_with_progress(&input, epsilon, Some(|p| {
    ///     eprintln!("Block {}/{}: {}", p.block_index + 1, p.total_blocks, p.block_name);
    /// }))
    /// ```
    pub fn propagate_ibp_block_wise_with_progress<F>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
    {
        self.propagate_ibp_block_wise_with_options(input, epsilon, progress_callback, 0)
    }

    /// Block-wise IBP propagation with progress and max_blocks limit.
    ///
    /// Same as `propagate_ibp_block_wise_with_progress`, but can limit verification to
    /// the first `max_blocks` blocks (0 = all blocks).
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    /// * `progress_callback` - Optional callback called after each block with progress info
    /// * `max_blocks` - Maximum number of blocks to verify (0 = all blocks)
    pub fn propagate_ibp_block_wise_with_options<F>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
        max_blocks: usize,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
    {
        let start_time = std::time::Instant::now();

        if self.nodes.is_empty() {
            return Ok(BlockWiseResult {
                blocks: vec![],
                block_epsilon: epsilon,
                total_blocks: 0,
                max_sensitivity: 1.0,
                degraded_blocks: 0,
            });
        }

        const MAX_BOUND: f32 = f32::MAX / 2.0;

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Identify block boundaries: nodes with "layerN_" prefix
        // Group nodes by their block prefix
        let mut block_nodes: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut non_block_nodes: Vec<String> = Vec::with_capacity(exec_order.len());

        for node_name in &exec_order {
            // Try to parse block index from node name (e.g., "layer0_attn_norm" -> 0)
            if let Some(block_idx) = Self::parse_block_index(node_name) {
                block_nodes
                    .entry(block_idx)
                    .or_default()
                    .push(node_name.clone());
            } else {
                non_block_nodes.push(node_name.clone());
            }
        }

        if block_nodes.is_empty() {
            // No blocks detected, fall back to standard layer-by-layer
            debug!("No transformer blocks detected in graph, falling back to layer-by-layer");
            let layer_result = self.propagate_ibp_detailed(input, epsilon)?;
            return Ok(BlockWiseResult {
                blocks: vec![BlockBoundsInfo {
                    block_index: 0,
                    block_name: "all_layers".to_string(),
                    nodes: layer_result.nodes,
                    input_width: epsilon * 2.0,
                    output_width: layer_result.final_width,
                    sensitivity: layer_result.final_width / (epsilon * 2.0),
                    qk_matmul_width: None,
                    swiglu_width: None,
                    degraded: layer_result.degraded_at_node.is_some(),
                }],
                block_epsilon: epsilon,
                total_blocks: 1,
                max_sensitivity: layer_result.final_width / (epsilon * 2.0),
                degraded_blocks: if layer_result.degraded_at_node.is_some() {
                    1
                } else {
                    0
                },
            });
        }

        let actual_total_blocks = block_nodes.len();
        let total_blocks = if max_blocks > 0 && max_blocks < actual_total_blocks {
            max_blocks
        } else {
            actual_total_blocks
        };
        let mut blocks: Vec<BlockBoundsInfo> = Vec::with_capacity(total_blocks);
        let mut max_sensitivity: f32 = 0.0;
        let mut degraded_blocks: usize = 0;

        // Process each block independently (limited by max_blocks if set)
        for (&block_idx, nodes_in_block) in block_nodes.iter().take(total_blocks) {
            let block_name = format!("layer{}", block_idx);

            // Create fresh input bounds for this block
            // Use the shape of the input to the first node in the block
            let first_node_name = &nodes_in_block[0];
            let first_node = self.nodes.get(first_node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!("Node not found: {}", first_node_name))
            })?;

            // Get block input shape from the first node's input
            let block_input_shape =
                if first_node.inputs.is_empty() || first_node.inputs[0] == "_input" {
                    input.shape().to_vec()
                } else {
                    // Find the node that feeds into this block (should be from previous block or _input)
                    let input_node_name = &first_node.inputs[0];
                    // If it's from a previous block, use that shape; otherwise use default
                    if let Some(_prev_node) = self.nodes.get(input_node_name) {
                        // Estimate output shape - for now use input shape
                        input.shape().to_vec()
                    } else {
                        input.shape().to_vec()
                    }
                };

            // Create fresh bounded tensor with epsilon perturbation
            let block_input_center = ArrayD::zeros(IxDyn(&block_input_shape));
            let block_input = BoundedTensor::from_epsilon(block_input_center, epsilon);
            let block_input_width = epsilon * 2.0;

            // Create bounds cache for this block
            let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
                std::collections::HashMap::with_capacity(nodes_in_block.len());

            // Track per-node information for this block
            let mut node_infos: Vec<NodeBoundsInfo> = Vec::with_capacity(nodes_in_block.len());
            let mut qk_matmul_width: Option<f32> = None;
            let mut swiglu_width: Option<f32> = None;
            let mut block_degraded = false;
            let mut block_output_width = block_input_width;

            // Process nodes in this block
            for node_name in nodes_in_block {
                let node = self.nodes.get(node_name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!("Node not found: {}", node_name))
                })?;

                // Get input bounds width.
                // For binary nodes inside a block, consider both inputs. If an input is produced
                // outside the current block, approximate it using `block_input_width`.
                let input_width = if node.inputs.is_empty() {
                    block_input_width
                } else {
                    node.inputs
                        .iter()
                        .map(|inp| {
                            if inp == "_input" {
                                block_input_width
                            } else if let Some(cached) = bounds_cache.get(inp) {
                                cached.max_width()
                            } else {
                                block_input_width
                            }
                        })
                        .fold(0.0_f32, f32::max)
                };

                // Propagate bounds
                let output_bounds = if node.layer.is_binary() {
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs",
                            node_name
                        )));
                    }
                    let input_a =
                        self.get_bounds_for_block(&node.inputs[0], &block_input, &bounds_cache)?;
                    let input_b =
                        self.get_bounds_for_block(&node.inputs[1], &block_input, &bounds_cache)?;
                    match &node.layer {
                        Layer::MatMul(matmul) if matmul.transpose_b => {
                            // Try zonotope tightening for Q@K^T
                            if let Some(tighter) = self.try_attention_matmul_bounds_zonotope_block(
                                node,
                                &block_input,
                                &bounds_cache,
                            )? {
                                // Record Q@K^T width
                                qk_matmul_width = Some(tighter.max_width());
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(&input_a, &input_b)?
                            }
                        }
                        Layer::MulBinary(_) => {
                            // Try zonotope tightening for SwiGLU (up * silu(gate))
                            if let Some(tighter) = self.try_ffn_swiglu_bounds_zonotope_block(
                                node,
                                &block_input,
                                &bounds_cache,
                            )? {
                                // Record SwiGLU width
                                swiglu_width = Some(tighter.max_width());
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(&input_a, &input_b)?
                            }
                        }
                        _ => node.layer.propagate_ibp_binary(&input_a, &input_b)?,
                    }
                } else if let Layer::Where(w) = &node.layer {
                    if w.has_embedded_constants() {
                        let cond = self.get_bounds_for_block(
                            &node.inputs[0],
                            &block_input,
                            &bounds_cache,
                        )?;
                        w.propagate_ibp_with_condition(&cond)?
                    } else {
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs",
                                node_name
                            )));
                        }
                        let cond = self.get_bounds_for_block(
                            &node.inputs[0],
                            &block_input,
                            &bounds_cache,
                        )?;
                        let x = self.get_bounds_for_block(
                            &node.inputs[1],
                            &block_input,
                            &bounds_cache,
                        )?;
                        let y = self.get_bounds_for_block(
                            &node.inputs[2],
                            &block_input,
                            &bounds_cache,
                        )?;
                        w.propagate_ibp_ternary(&cond, &x, &y)?
                    }
                } else {
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }

                    // Try zonotope tightening for ffn_down Linear nodes
                    if let Layer::Linear(_) = &node.layer {
                        if let Some(tighter) =
                            self.try_ffn_down_zonotope_block(node, &block_input, &bounds_cache)?
                        {
                            // FFN down zonotope tightening succeeded
                            debug!("FFN down zonotope applied for {}", node_name);
                            tighter
                        } else {
                            let node_input = self.get_bounds_for_block(
                                &node.inputs[0],
                                &block_input,
                                &bounds_cache,
                            )?;
                            node.layer.propagate_ibp(&node_input)?
                        }
                    } else {
                        let node_input = self.get_bounds_for_block(
                            &node.inputs[0],
                            &block_input,
                            &bounds_cache,
                        )?;
                        node.layer.propagate_ibp(&node_input)?
                    }
                };

                // Collect statistics
                let output_width = output_bounds.max_width();
                let mut min_bound = f32::INFINITY;
                let mut max_bound = f32::NEG_INFINITY;
                let mut saturated = false;
                let mut has_nan = false;
                let mut has_infinite = false;

                for (&l, &u) in output_bounds.lower.iter().zip(output_bounds.upper.iter()) {
                    if l.is_nan() || u.is_nan() {
                        has_nan = true;
                    }
                    if !l.is_finite() || !u.is_finite() {
                        has_infinite = true;
                    }
                    min_bound = min_bound.min(l);
                    max_bound = max_bound.max(u);
                    if l <= -0.999 * MAX_BOUND || u >= 0.999 * MAX_BOUND {
                        saturated = true;
                    }
                }

                let sensitivity = if input_width > 0.0 && input_width.is_finite() {
                    output_width / input_width
                } else if output_width == 0.0 {
                    1.0
                } else {
                    f32::INFINITY
                };

                let node_info = NodeBoundsInfo {
                    name: node_name.clone(),
                    layer_type: node.layer.layer_type().to_string(),
                    input_width,
                    output_width,
                    sensitivity,
                    output_shape: output_bounds.shape().to_vec(),
                    min_bound,
                    max_bound,
                    saturated,
                    has_nan,
                    has_infinite,
                };

                if node_info.has_degraded() {
                    block_degraded = true;
                }

                node_infos.push(node_info);
                block_output_width = output_width;
                bounds_cache.insert(node_name.clone(), output_bounds);
            }

            let block_sensitivity = if block_input_width > 0.0 {
                block_output_width / block_input_width
            } else {
                f32::INFINITY
            };

            if block_sensitivity > max_sensitivity {
                max_sensitivity = block_sensitivity;
            }

            if block_degraded {
                degraded_blocks += 1;
            }

            // Report progress before pushing (block_name will be moved)
            // Note: degraded_blocks was already incremented above if needed
            if let Some(ref callback) = progress_callback {
                callback(BlockProgress {
                    block_index: block_idx,
                    total_blocks,
                    block_name: block_name.clone(),
                    elapsed: start_time.elapsed(),
                    current_max_sensitivity: max_sensitivity,
                    degraded_so_far: degraded_blocks,
                });
            }

            blocks.push(BlockBoundsInfo {
                block_index: block_idx,
                block_name,
                nodes: node_infos,
                input_width: block_input_width,
                output_width: block_output_width,
                sensitivity: block_sensitivity,
                qk_matmul_width,
                swiglu_width,
                degraded: block_degraded,
            });
        }

        Ok(BlockWiseResult {
            total_blocks: blocks.len(),
            blocks,
            block_epsilon: epsilon,
            max_sensitivity,
            degraded_blocks,
        })
    }

    /// Block-wise IBP propagation with checkpoint support for resumable verification.
    ///
    /// Supports resuming from a previous checkpoint and saving progress after each block.
    /// Useful for multi-hour verification of very large models (32B+).
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    /// * `progress_callback` - Optional callback called after each block with progress info
    /// * `checkpoint_callback` - Optional callback called after each block to save checkpoint
    /// * `max_blocks` - Maximum number of blocks to verify (0 = all blocks)
    /// * `resume_from` - Optional checkpoint to resume from (skips completed blocks)
    pub fn propagate_ibp_block_wise_with_checkpoint<F, G>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
        checkpoint_callback: Option<G>,
        max_blocks: usize,
        resume_from: Option<&VerificationCheckpoint>,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
        G: Fn(&BlockBoundsInfo, u64, usize), // (block_info, elapsed_ms, total_blocks)
    {
        let start_time = std::time::Instant::now();

        // Get prior elapsed time if resuming
        let prior_elapsed_ms = resume_from.map(|c| c.elapsed_ms).unwrap_or(0);

        if self.nodes.is_empty() {
            return Ok(BlockWiseResult {
                blocks: vec![],
                block_epsilon: epsilon,
                total_blocks: 0,
                max_sensitivity: 1.0,
                degraded_blocks: 0,
            });
        }

        const MAX_BOUND: f32 = f32::MAX / 2.0;

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Identify block boundaries: nodes with "layerN_" prefix
        // Group nodes by their block prefix
        let mut block_nodes: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut non_block_nodes: Vec<String> = Vec::with_capacity(exec_order.len());

        for node_name in &exec_order {
            // Try to parse block index from node name (e.g., "layer0_attn_norm" -> 0)
            if let Some(block_idx) = Self::parse_block_index(node_name) {
                block_nodes
                    .entry(block_idx)
                    .or_default()
                    .push(node_name.clone());
            } else {
                non_block_nodes.push(node_name.clone());
            }
        }

        if block_nodes.is_empty() {
            // No blocks detected, fall back to standard layer-by-layer
            debug!("No transformer blocks detected in graph, falling back to layer-by-layer");
            let layer_result = self.propagate_ibp_detailed(input, epsilon)?;
            return Ok(BlockWiseResult {
                blocks: vec![BlockBoundsInfo {
                    block_index: 0,
                    block_name: "all_layers".to_string(),
                    nodes: layer_result.nodes,
                    input_width: epsilon * 2.0,
                    output_width: layer_result.final_width,
                    sensitivity: layer_result.final_width / (epsilon * 2.0),
                    qk_matmul_width: None,
                    swiglu_width: None,
                    degraded: layer_result.degraded_at_node.is_some(),
                }],
                block_epsilon: epsilon,
                total_blocks: 1,
                max_sensitivity: layer_result.final_width / (epsilon * 2.0),
                degraded_blocks: if layer_result.degraded_at_node.is_some() {
                    1
                } else {
                    0
                },
            });
        }

        // Initialize from checkpoint or start fresh
        let (mut blocks, mut max_sensitivity, mut degraded_blocks, skip_blocks) =
            if let Some(checkpoint) = resume_from {
                (
                    checkpoint.completed_blocks.clone(),
                    checkpoint.max_sensitivity,
                    checkpoint.degraded_blocks,
                    checkpoint.next_block_index,
                )
            } else {
                (Vec::new(), 0.0f32, 0usize, 0usize)
            };

        let actual_total_blocks = block_nodes.len();
        let total_blocks = if max_blocks > 0 && max_blocks < actual_total_blocks {
            max_blocks
        } else {
            actual_total_blocks
        };

        // Process each block independently (limited by max_blocks if set)
        for (&block_idx, nodes_in_block) in block_nodes.iter().take(total_blocks) {
            // Skip already-completed blocks
            if block_idx < skip_blocks {
                continue;
            }

            let block_name = format!("layer{}", block_idx);

            // Create fresh input bounds for this block
            // Use the shape of the input to the first node in the block
            let first_node_name = &nodes_in_block[0];
            let first_node = self.nodes.get(first_node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!("Node not found: {}", first_node_name))
            })?;

            // Get block input shape from the first node's input
            let block_input_shape =
                if first_node.inputs.is_empty() || first_node.inputs[0] == "_input" {
                    input.shape().to_vec()
                } else {
                    // Find the node that feeds into this block (should be from previous block or _input)
                    let input_node_name = &first_node.inputs[0];
                    // If it's from a previous block, use that shape; otherwise use default
                    if let Some(_prev_node) = self.nodes.get(input_node_name) {
                        // Estimate output shape - for now use input shape
                        input.shape().to_vec()
                    } else {
                        input.shape().to_vec()
                    }
                };

            // Create fresh bounded tensor with epsilon perturbation
            let block_input_center = ArrayD::zeros(IxDyn(&block_input_shape));
            let block_input = BoundedTensor::from_epsilon(block_input_center, epsilon);
            let block_input_width = epsilon * 2.0;

            // Create bounds cache for this block
            let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
                std::collections::HashMap::with_capacity(nodes_in_block.len());

            // Track per-node information for this block
            let mut node_infos: Vec<NodeBoundsInfo> = Vec::with_capacity(nodes_in_block.len());
            let mut qk_matmul_width: Option<f32> = None;
            let mut swiglu_width: Option<f32> = None;
            let mut block_degraded = false;
            let mut block_output_width = block_input_width;

            // Process nodes in this block
            for node_name in nodes_in_block {
                let node = self.nodes.get(node_name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!("Node not found: {}", node_name))
                })?;

                // Get input bounds width.
                // For binary nodes inside a block, consider both inputs. If an input is produced
                // outside the current block, approximate it using `block_input_width`.
                let input_width = if node.inputs.is_empty() {
                    block_input_width
                } else {
                    node.inputs
                        .iter()
                        .map(|inp| {
                            if inp == "_input" {
                                block_input_width
                            } else if let Some(cached) = bounds_cache.get(inp) {
                                cached.max_width()
                            } else {
                                block_input_width
                            }
                        })
                        .fold(0.0_f32, f32::max)
                };

                // Propagate bounds
                let output_bounds = if node.layer.is_binary() {
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs",
                            node_name
                        )));
                    }
                    let input_a =
                        self.get_bounds_for_block(&node.inputs[0], &block_input, &bounds_cache)?;
                    let input_b =
                        self.get_bounds_for_block(&node.inputs[1], &block_input, &bounds_cache)?;
                    match &node.layer {
                        Layer::MatMul(matmul) if matmul.transpose_b => {
                            // Try zonotope tightening for Q@K^T
                            if let Some(tighter) = self.try_attention_matmul_bounds_zonotope_block(
                                node,
                                &block_input,
                                &bounds_cache,
                            )? {
                                // Record Q@K^T width
                                qk_matmul_width = Some(tighter.max_width());
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(&input_a, &input_b)?
                            }
                        }
                        Layer::MulBinary(_) => {
                            // Try zonotope tightening for SwiGLU (up * silu(gate))
                            if let Some(tighter) = self.try_ffn_swiglu_bounds_zonotope_block(
                                node,
                                &block_input,
                                &bounds_cache,
                            )? {
                                // Record SwiGLU width
                                swiglu_width = Some(tighter.max_width());
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(&input_a, &input_b)?
                            }
                        }
                        _ => node.layer.propagate_ibp_binary(&input_a, &input_b)?,
                    }
                } else if let Layer::Where(w) = &node.layer {
                    if w.has_embedded_constants() {
                        let cond = self.get_bounds_for_block(
                            &node.inputs[0],
                            &block_input,
                            &bounds_cache,
                        )?;
                        w.propagate_ibp_with_condition(&cond)?
                    } else {
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs",
                                node_name
                            )));
                        }
                        let cond = self.get_bounds_for_block(
                            &node.inputs[0],
                            &block_input,
                            &bounds_cache,
                        )?;
                        let x = self.get_bounds_for_block(
                            &node.inputs[1],
                            &block_input,
                            &bounds_cache,
                        )?;
                        let y = self.get_bounds_for_block(
                            &node.inputs[2],
                            &block_input,
                            &bounds_cache,
                        )?;
                        w.propagate_ibp_ternary(&cond, &x, &y)?
                    }
                } else {
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }

                    // Try zonotope tightening for ffn_down Linear nodes
                    if let Layer::Linear(_) = &node.layer {
                        if let Some(tighter) =
                            self.try_ffn_down_zonotope_block(node, &block_input, &bounds_cache)?
                        {
                            // FFN down zonotope tightening succeeded
                            debug!("FFN down zonotope applied for {}", node_name);
                            tighter
                        } else {
                            let node_input = self.get_bounds_for_block(
                                &node.inputs[0],
                                &block_input,
                                &bounds_cache,
                            )?;
                            node.layer.propagate_ibp(&node_input)?
                        }
                    } else {
                        let node_input = self.get_bounds_for_block(
                            &node.inputs[0],
                            &block_input,
                            &bounds_cache,
                        )?;
                        node.layer.propagate_ibp(&node_input)?
                    }
                };

                // Collect statistics
                let output_width = output_bounds.max_width();
                let mut min_bound = f32::INFINITY;
                let mut max_bound = f32::NEG_INFINITY;
                let mut saturated = false;
                let mut has_nan = false;
                let mut has_infinite = false;

                for (&l, &u) in output_bounds.lower.iter().zip(output_bounds.upper.iter()) {
                    if l.is_nan() || u.is_nan() {
                        has_nan = true;
                    }
                    if !l.is_finite() || !u.is_finite() {
                        has_infinite = true;
                    }
                    min_bound = min_bound.min(l);
                    max_bound = max_bound.max(u);
                    if l <= -0.999 * MAX_BOUND || u >= 0.999 * MAX_BOUND {
                        saturated = true;
                    }
                }

                let sensitivity = if input_width > 0.0 && input_width.is_finite() {
                    output_width / input_width
                } else if output_width == 0.0 {
                    1.0
                } else {
                    f32::INFINITY
                };

                let node_info = NodeBoundsInfo {
                    name: node_name.clone(),
                    layer_type: node.layer.layer_type().to_string(),
                    input_width,
                    output_width,
                    sensitivity,
                    output_shape: output_bounds.shape().to_vec(),
                    min_bound,
                    max_bound,
                    saturated,
                    has_nan,
                    has_infinite,
                };

                if node_info.has_degraded() {
                    block_degraded = true;
                }

                node_infos.push(node_info);
                block_output_width = output_width;
                bounds_cache.insert(node_name.clone(), output_bounds);
            }

            let block_sensitivity = if block_input_width > 0.0 {
                block_output_width / block_input_width
            } else {
                f32::INFINITY
            };

            if block_sensitivity > max_sensitivity {
                max_sensitivity = block_sensitivity;
            }

            if block_degraded {
                degraded_blocks += 1;
            }

            let block_info = BlockBoundsInfo {
                block_index: block_idx,
                block_name: block_name.clone(),
                nodes: node_infos,
                input_width: block_input_width,
                output_width: block_output_width,
                sensitivity: block_sensitivity,
                qk_matmul_width,
                swiglu_width,
                degraded: block_degraded,
            };

            // Report progress
            if let Some(ref callback) = progress_callback {
                callback(BlockProgress {
                    block_index: block_idx,
                    total_blocks,
                    block_name: block_name.clone(),
                    elapsed: start_time.elapsed(),
                    current_max_sensitivity: max_sensitivity,
                    degraded_so_far: degraded_blocks,
                });
            }

            // Save checkpoint
            let elapsed_ms = prior_elapsed_ms + start_time.elapsed().as_millis() as u64;
            if let Some(ref callback) = checkpoint_callback {
                callback(&block_info, elapsed_ms, total_blocks);
            }

            blocks.push(block_info);
        }

        Ok(BlockWiseResult {
            total_blocks: blocks.len(),
            blocks,
            block_epsilon: epsilon,
            max_sensitivity,
            degraded_blocks,
        })
    }

    /// Parse block index from node name (e.g., "layer0_attn_norm" -> Some(0)).
    pub(crate) fn parse_block_index(node_name: &str) -> Option<usize> {
        let after_layer = node_name.strip_prefix("layer")?;
        let num_end = after_layer.find(|c: char| !c.is_ascii_digit())?;
        let num_str = &after_layer[..num_end];
        num_str.parse().ok()
    }

    /// Get bounds from cache or create fresh for block-wise verification.
    fn get_bounds_for_block(
        &self,
        input_name: &str,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<BoundedTensor> {
        if input_name == "_input" {
            Ok(block_input.clone())
        } else if let Some(cached) = bounds_cache.get(input_name) {
            Ok(cached.clone())
        } else {
            // Input is from outside this block - use fresh bounds
            // This shouldn't happen often for well-structured blocks
            Ok(block_input.clone())
        }
    }

    /// Try to apply zonotope tightening for Q@K^T in block-wise mode.
    fn try_attention_matmul_bounds_zonotope_block(
        &self,
        matmul_node: &GraphNode,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        // Delegate to the main zonotope function with fresh block input
        self.try_attention_matmul_bounds_zonotope(matmul_node, block_input, bounds_cache)
    }

    /// Propagate bounds through the graph using zonotopes for correlation-aware attention.
    ///
    /// Zonotopes track correlations between Q and K through shared error symbols,
    /// giving tighter bounds for Q@K^T in attention than IBP.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (center values)
    /// * `epsilon` - Perturbation epsilon for input
    ///
    /// # Returns
    /// Bounds as a BoundedTensor (converted from zonotope at output).
    ///
    /// # Supported Operations
    /// - Linear layers (preserves zonotope form exactly)
    /// - AddConstant, MulConstant (preserves zonotope form exactly)
    /// - MatMul for Q@K^T patterns (uses matmul_transposed)
    /// - Other ops: falls back to IBP and converts to zonotope (loses correlations)
    ///
    /// # Limitations
    /// - Best effort for tensors with >=2 dims (..., seq, dim)
    /// - Larger inputs may require significant memory for error symbols
    #[inline]
    pub fn propagate_zonotope(
        &self,
        input: &BoundedTensor,
        _epsilon: f32,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            // Empty graph: nothing to do.
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Zonotope propagation works best on 2D, but we support batched sequence tensors too.
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            debug!(
                "GraphNetwork zonotope: input shape {:?} has <2 dims, falling back to IBP",
                input_shape
            );
            return self.propagate_ibp(input);
        }

        // Create input zonotope with per-position error symbols derived from the current bounds.
        // This ensures zonotope propagation remains compatible with compositional bounds where
        // per-element radii may not equal the original epsilon.
        let input_zonotope = ZonotopeTensor::from_bounded_tensor_per_position(input)?;
        debug!(
            "GraphNetwork zonotope: created input zonotope with {} error terms, shape {:?}",
            input_zonotope.n_error_terms, input_zonotope.element_shape
        );

        // Store zonotopes for each node's output
        let mut zonotope_cache: std::collections::HashMap<String, ZonotopeTensor> =
            std::collections::HashMap::new();

        // Also store interval bounds for fallback
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // Track which nodes have zonotope representations
        let mut has_zonotope: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Process nodes in topological order
        for node_name in &exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            debug!(
                "GraphNetwork zonotope: processing node {} ({})",
                node_name,
                node.layer.layer_type()
            );

            // Try to propagate as zonotope; fall back to IBP if not supported
            let result = self.propagate_zonotope_node(
                node,
                &input_zonotope,
                input,
                &zonotope_cache,
                &bounds_cache,
                &has_zonotope,
            );

            match result {
                Ok(z) => {
                    debug!(
                        "GraphNetwork zonotope: node {} output zonotope with {} error terms, max_width {}",
                        node_name,
                        z.n_error_terms,
                        z.max_width()
                    );
                    bounds_cache.insert(node_name.clone(), z.to_bounded_tensor());
                    zonotope_cache.insert(node_name.clone(), z);
                    has_zonotope.insert(node_name.clone());
                }
                Err(e) => {
                    // Fall back to IBP for this node
                    debug!(
                        "GraphNetwork zonotope: node {} falling back to IBP: {}",
                        node_name, e
                    );

                    let ibp_bounds = match &node.layer {
                        Layer::Where(w) => {
                            if w.has_embedded_constants() {
                                let cond =
                                    self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                                w.propagate_ibp_with_condition(cond)?
                            } else {
                                if node.inputs.len() < 3 {
                                    return Err(GammaError::InvalidSpec(format!(
                                        "Where node {} requires 3 inputs",
                                        node_name
                                    )));
                                }
                                let cond =
                                    self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                                let x =
                                    self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                                let y =
                                    self.get_bounds_ref(&node.inputs[2], input, &bounds_cache)?;
                                w.propagate_ibp_ternary(cond, x, y)?
                            }
                        }
                        _ if node.layer.is_binary() => {
                            if node.inputs.len() < 2 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Binary node {} requires 2 inputs",
                                    node_name
                                )));
                            }
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            node.layer.propagate_ibp_binary(input_a, input_b)?
                        }
                        _ => {
                            let node_input =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            node.layer.propagate_ibp(node_input)?
                        }
                    };

                    // Convert IBP bounds to zonotope (loses correlation info)
                    let z = ZonotopeTensor::from_bounded_tensor(&ibp_bounds);
                    bounds_cache.insert(node_name.clone(), ibp_bounds);
                    zonotope_cache.insert(node_name.clone(), z);
                    // Don't mark as has_zonotope - it's derived from IBP
                }
            }
        }

        // Get output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        // Return bounds from output zonotope
        let output = zonotope_cache.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;

        info!(
            "GraphNetwork zonotope: final output has {} error terms, max_width {}",
            output.n_error_terms,
            output.max_width()
        );

        Ok(output.to_bounded_tensor())
    }

    /// Try to propagate a single node using zonotopes.
    ///
    /// Returns Ok(ZonotopeTensor) if the operation is supported, Err otherwise.
    fn propagate_zonotope_node(
        &self,
        node: &GraphNode,
        input_zonotope: &ZonotopeTensor,
        _input_bounds: &BoundedTensor,
        zonotope_cache: &std::collections::HashMap<String, ZonotopeTensor>,
        _bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
        _has_zonotope: &std::collections::HashSet<String>,
    ) -> Result<ZonotopeTensor> {
        // Helper to get zonotope for an input
        let get_zonotope = |name: &str| -> Result<ZonotopeTensor> {
            if name == "_input" {
                Ok(input_zonotope.clone())
            } else if let Some(z) = zonotope_cache.get(name) {
                Ok(z.clone())
            } else {
                Err(GammaError::InvalidSpec(format!(
                    "No zonotope found for node {}",
                    name
                )))
            }
        };

        match &node.layer {
            // Linear layer: z = W @ z + b
            Layer::Linear(linear) => {
                let input_z = get_zonotope(&node.inputs[0])?;
                input_z.linear(&linear.weight, linear.bias.as_ref())
            }

            // AddConstant: z + c
            Layer::AddConstant(add_const) => {
                let input_z = get_zonotope(&node.inputs[0])?;
                input_z.add_constant(&add_const.constant)
            }

            // MulConstant: z * c
            Layer::MulConstant(mul_const) => {
                let input_z = get_zonotope(&node.inputs[0])?;
                input_z.mul_constant(&mul_const.constant)
            }

            // MatMul: Q @ K^T - the key operation for attention
            Layer::MatMul(matmul) => {
                if node.inputs.len() < 2 {
                    return Err(GammaError::InvalidSpec(
                        "MatMul requires 2 inputs".to_string(),
                    ));
                }

                let q_z = get_zonotope(&node.inputs[0])?;
                let k_z = get_zonotope(&node.inputs[1])?;

                // Check if Q and K share error symbols
                if q_z.n_error_terms != k_z.n_error_terms {
                    debug!(
                        "MatMul zonotope: Q has {} errors, K has {}, need to expand",
                        q_z.n_error_terms, k_z.n_error_terms
                    );
                    let (q_expanded, k_expanded) = q_z.expand_to_match(&k_z)?;
                    return q_expanded.matmul_transposed(&k_expanded);
                }

                // Both have same error symbols - this is the ideal case for Q@K^T
                if !matmul.transpose_b {
                    return Err(GammaError::InvalidSpec(
                        "MatMul without transpose_b not yet supported for zonotope".to_string(),
                    ));
                }

                let out = q_z.matmul_transposed(&k_z)?;
                if let Some(scale) = matmul.scale {
                    let scale_tensor = ndarray::ArrayD::from_elem(out.element_shape.clone(), scale);
                    out.mul_constant(&scale_tensor)
                } else {
                    Ok(out)
                }
            }

            // Add: z1 + z2 (element-wise)
            Layer::Add(_) => {
                if node.inputs.len() < 2 {
                    return Err(GammaError::InvalidSpec("Add requires 2 inputs".to_string()));
                }

                let a_z = get_zonotope(&node.inputs[0])?;
                let b_z = get_zonotope(&node.inputs[1])?;

                // Expand to match if needed
                let (a_expanded, b_expanded) = a_z.expand_to_match(&b_z)?;
                a_expanded.add(&b_expanded)
            }

            // Reshape: preserves correlations perfectly (just rearranges elements)
            Layer::Reshape(reshape_layer) => {
                let input_z = get_zonotope(&node.inputs[0])?;

                // Compute output shape from reshape layer
                let input_shape = &input_z.element_shape;
                let output_shape = reshape_layer.compute_output_shape(input_shape)?;

                input_z.reshape(&output_shape)
            }

            // Tile: preserves correlations (duplicated elements share error symbols)
            // Essential for GQA where K/V heads are tiled to match Q heads
            Layer::Tile(tile_layer) => {
                let input_z = get_zonotope(&node.inputs[0])?;

                // Compute actual axis (handle negative indexing)
                let ndim = input_z.element_shape.len();
                let axis = if tile_layer.axis < 0 {
                    (ndim as i32 + tile_layer.axis) as usize
                } else {
                    tile_layer.axis as usize
                };

                input_z.tile(axis, tile_layer.reps)
            }

            // Transpose: preserves correlations (just permutes axes)
            Layer::Transpose(transpose_layer) => {
                let input_z = get_zonotope(&node.inputs[0])?;

                // Check if it's a simple swap of last two dimensions
                let ndim = transpose_layer.axes.len();
                if ndim >= 2
                    && transpose_layer.axes[ndim - 2] == ndim - 1
                    && transpose_layer.axes[ndim - 1] == ndim - 2
                {
                    input_z.transpose_last_two()
                } else {
                    Err(GammaError::InvalidSpec(format!(
                        "Zonotope transpose only supports swapping last two dims, got axes {:?}",
                        transpose_layer.axes
                    )))
                }
            }

            // LayerNorm: use affine approximation to preserve correlations
            Layer::LayerNorm(ln) => {
                let input_z = get_zonotope(&node.inputs[0])?;

                input_z.layer_norm_affine(&ln.gamma, &ln.beta, ln.eps)
            }

            // GELU/SiLU: use affine approximation to preserve correlations
            // SiLU is mapped to GELU in gamma-onnx, and both have similar shapes
            // that benefit from the linear approximation approach
            Layer::GELU(_) => {
                let input_z = get_zonotope(&node.inputs[0])?;
                input_z.silu_affine()
            }

            // MulBinary: element-wise multiplication (needed for SwiGLU)
            // z1 ⊙ z2 = silu(gate) * up, exploits shared error symbols
            Layer::MulBinary(_) => {
                if node.inputs.len() < 2 {
                    return Err(GammaError::InvalidSpec(
                        "MulBinary requires 2 inputs".to_string(),
                    ));
                }

                let z1 = get_zonotope(&node.inputs[0])?;
                let z2 = get_zonotope(&node.inputs[1])?;

                z1.mul_elementwise(&z2)
            }

            // Softmax: use affine approximation to preserve correlations
            // This linearizes softmax around the center and adds an error term for soundness
            Layer::Softmax(s) => {
                let input_z = get_zonotope(&node.inputs[0])?;
                input_z.softmax_affine(s.axis)
            }

            // CausalSoftmax: use affine approximation (same as Softmax for now)
            Layer::CausalSoftmax(cs) => {
                let input_z = get_zonotope(&node.inputs[0])?;
                input_z.softmax_affine_causal(cs.axis)
            }

            // Operations that don't preserve zonotope form well
            Layer::ReLU(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
            | Layer::Sqrt(_)
            | Layer::PowConstant(_) => {
                // These operations break zonotope form - fall back to IBP
                Err(GammaError::InvalidSpec(format!(
                    "Operation {} not supported for zonotope propagation",
                    node.layer.layer_type()
                )))
            }

            // Other operations - not yet implemented
            _ => Err(GammaError::InvalidSpec(format!(
                "Operation {} not yet implemented for zonotope propagation",
                node.layer.layer_type()
            ))),
        }
    }

    /// Propagate bounds through the graph using CROWN.
    ///
    /// CROWN provides tighter bounds than IBP by representing bounds as linear functions
    /// of the input. This implementation supports DAG structures including binary operations
    /// (MatMul, Add) which are essential for attention patterns.
    ///
    /// Algorithm:
    /// 1. Run IBP forward to collect bounds at each node (needed for ReLU/GELU relaxation)
    /// 2. Initialize linear bounds per node (output node gets identity)
    /// 3. Propagate backward: accumulate linear bounds from downstream consumers
    /// 4. For binary ops: split bounds to both inputs
    /// 5. Concretize final bounds at network input
    pub fn propagate_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine(input, None)
    }

    #[inline]
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Step 1: Collect bounds at each node for nonlinear relaxations.
        //
        // - CNN-style DAGs: use expensive CROWN-IBP intermediates for much tighter ReLU relaxations.
        // - Transformer-style graphs: use IBP forward bounds (includes transformer-specific tightening).
        let use_crown_ibp = self.should_use_crown_ibp_intermediates();
        let node_bounds = if use_crown_ibp {
            self.collect_crown_ibp_bounds_dag(input)?
        } else {
            self.collect_node_bounds(input)?
        };

        // Determine output node and dimension
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "GraphNetwork DAG-CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 2: Initialize linear bounds per node
        // Each node tracks the accumulated linear bounds from all its consumers
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();

        // Output node starts with identity bounds
        node_linear_bounds.insert(output_node_name.clone(), LinearBounds::identity(output_dim));

        // Also track linear bounds for "_input" (will be accumulated from all paths)
        // Initialize with zeros (will be added to)
        let input_dim = input.len();
        let mut input_accumulated = false;

        // Step 3: Propagate backward through nodes in reverse order
        for node_name in exec_order.iter().rev() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get this node's accumulated linear bounds.
            // We can move it out of the map because reverse-topological traversal guarantees
            // all consumers have already contributed their bounds.
            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    // This shouldn't happen in well-formed graphs
                    debug!(
                        "GraphNetwork DAG-CROWN: node {} has no consumers, skipping",
                        node_name
                    );
                    continue;
                }
            };

            debug!(
                "GraphNetwork DAG-CROWN: backward through {} ({}) with {} outputs",
                node_name,
                node.layer.layer_type(),
                node_lb.num_outputs()
            );

            // Get pre-activation bounds for this node
            let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        node.inputs[0]
                    ))
                })?
            };

            // Handle different layer types
            match &node.layer {
                Layer::Linear(l) => {
                    // Check for dimension mismatch before attempting CROWN backward
                    // This can happen after ReduceMean backward expands sequence dimensions
                    let expected_inputs = l.out_features();
                    let got_inputs = node_lb.num_inputs();
                    if got_inputs != expected_inputs {
                        debug!(
                            "GraphNetwork DAG-CROWN: Linear dimension mismatch at {}: expected {} inputs, got {}. Falling back to IBP.",
                            node_name, expected_inputs, got_inputs
                        );
                        return self.propagate_ibp(input);
                    }

                    let new_lb = l
                        .propagate_linear_with_engine(&node_lb, engine)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (Linear): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    let new_lb = r
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Transpose(t) => {
                    // Clone transpose and set input_shape for proper column permutation
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let new_lb = transpose_with_shape
                        .propagate_linear(&node_lb)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (Transpose): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::GELU(g) => {
                    let new_lb = g
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (GELU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::LayerNorm(ln) => {
                    let new_lb = ln
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (LayerNorm): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
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
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    // Add propagates the same bounds to both inputs
                    let (lb_a, lb_b) = add.propagate_linear_binary(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN failed at node '{}' (Add): {}",
                            node_name, e
                        ))
                    })?;

                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Concat(concat) => {
                    // Concat splits linear bounds to N inputs based on input shapes.
                    // Support 2+ inputs (not just binary).
                    // Check constant inputs first (CLS token, etc.), then stored shapes, then node_bounds.
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
                                input.shape().to_vec()
                            } else if let Some(shape) = concat.get_input_shape(i) {
                                shape.clone()
                            } else {
                                node_bounds
                                    .get(inp_name)
                                    .map(|b| b.shape().to_vec())
                                    .unwrap_or_else(|| vec![pre_activation.len()])
                            }
                        })
                        .collect();

                    // Use N-ary propagation
                    let bounds_vec = concat
                        .propagate_linear_nary(&node_lb, &input_shapes)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (Concat): {}",
                                node_name, e
                            ))
                        })?;

                    // Accumulate bounds to each input (skip constant inputs)
                    for (i, (inp_name, lb)) in
                        node.inputs.iter().zip(bounds_vec.into_iter()).enumerate()
                    {
                        // Skip constant inputs - they have no gradient to propagate
                        if concat.get_constant_input(i).is_some() {
                            continue;
                        }
                        self.accumulate_bounds_to_input(
                            inp_name,
                            lb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                    }
                }
                Layer::MatMul(matmul) => {
                    // Get bounds on both inputs
                    let input_a_bounds = if node.inputs[0] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "Bounds for {} not found",
                                node.inputs[0]
                            ))
                        })?
                    };

                    let input_b_bounds = if node.inputs[1] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[1]).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "Bounds for {} not found",
                                node.inputs[1]
                            ))
                        })?
                    };

                    let (lb_a, lb_b) = matmul
                        .propagate_linear_binary(&node_lb, input_a_bounds, input_b_bounds)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (MatMul): {}",
                                node_name, e
                            ))
                        })?;

                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv1d(c) => {
                    // Conv1d DAG-CROWN: clone layer, set input_length, propagate
                    let input_shape = pre_activation.shape();
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else {
                        debug!(
                            "GraphNetwork DAG-CROWN: Conv1d input shape too small: {:?}, falling back to IBP",
                            input_shape
                        );
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_length(in_len);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN failed at node '{}' (Conv1d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv2d(c) => {
                    // Conv2d DAG-CROWN: clone layer, set input_shape, propagate
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        debug!(
                            "GraphNetwork DAG-CROWN: Conv2d input shape too small: {:?}, falling back to IBP",
                            input_shape
                        );
                        return self.propagate_ibp(input);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN failed at node '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::AveragePool(ap) => {
                    let new_lb = ap
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (AveragePool): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::MaxPool2d(mp) => {
                    let new_lb = mp
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (MaxPool2d): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Softmax(s) => {
                    let new_lb = s
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (Softmax): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::CausalSoftmax(cs) => {
                    let new_lb = cs.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::AddConstant(ac) => {
                    let new_lb = ac.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Reshape(r) => {
                    let new_lb = r.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::MulConstant(m) => {
                    let new_lb = m.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Abs(ab) => {
                    let new_lb = ab.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::DivConstant(d) => {
                    let new_lb = d.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::SubConstant(s) => {
                    let new_lb = s.propagate_linear(&node_lb)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Sqrt(sq) => {
                    let new_lb = sq.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::PowConstant(p) => {
                    match p.propagate_linear_with_bounds(&node_lb, pre_activation) {
                        Ok(new_lb) => {
                            self.accumulate_bounds_to_input(
                                &node.inputs[0],
                                new_lb,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                        }
                        Err(err) => {
                            debug!(
                                "GraphNetwork DAG-CROWN: PowConstant not supported ({}), falling back to IBP",
                                err
                            );
                            return self.propagate_ibp(input);
                        }
                    }
                }
                Layer::ReduceMean(rm) => {
                    let new_lb = rm.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::ReduceSum(rs) => {
                    let new_lb = rs.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Tanh(t) => {
                    let new_lb = t.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Sigmoid(s) => {
                    let new_lb = s.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Softplus(sp) => {
                    let new_lb = sp.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::LeakyReLU(lr) => {
                    let new_lb = lr.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Clip(c) => {
                    let new_lb = c.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::HardSigmoid(hs) => {
                    let new_lb = hs.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Elu(e) => {
                    let new_lb = e.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Selu(s) => {
                    let new_lb = s.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::PRelu(pr) => {
                    let new_lb = pr.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::HardSwish(hw) => {
                    let new_lb = hw.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Celu(ce) => {
                    let new_lb = ce.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Mish(mi) => {
                    let new_lb = mi.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Exp(e) => {
                    let new_lb = e.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Log(lg) => {
                    let new_lb = lg.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Softsign(ss) => {
                    let new_lb = ss.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Sin(sn) => {
                    let new_lb = sn.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Cos(cs) => {
                    let new_lb = cs.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Reciprocal(rc) => {
                    let new_lb = rc.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::ThresholdedRelu(tr) => {
                    let new_lb = tr.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Shrink(sh) => {
                    let new_lb = sh.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::LogSoftmax(ls) => {
                    let new_lb = ls.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Sub(sub) => {
                    // Sub propagates bounds to both inputs (second input negated)
                    let (lb_a, lb_b) = sub.propagate_linear_binary(&node_lb)?;

                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Div(_) => {
                    // Div CROWN not supported (nonlinear), fall back to IBP
                    debug!("GraphNetwork DAG-CROWN: Div not supported, falling back to IBP");
                    return self.propagate_ibp(input);
                }
                Layer::Tile(t) => {
                    let new_lb = t.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Slice(s) => {
                    let new_lb = s.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Where(_) => {
                    // Where is ternary and data-dependent; treat its output as an independently-bounded
                    // intermediate variable instead of falling back the entire graph to IBP.
                    let where_bounds = node_bounds.get(node_name).ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Where output bounds for {} not found during DAG-CROWN",
                            node_name
                        ))
                    })?;

                    // If condition is provably all-true or all-false, route bounds to the chosen branch.
                    if node.inputs.len() >= 3 {
                        let cond_bounds =
                            self.get_bounds_ref(&node.inputs[0], input, &node_bounds)?;
                        let cond_all_true = cond_bounds.lower.iter().all(|&v| v >= 0.5);
                        let cond_all_false = cond_bounds.upper.iter().all(|&v| v <= 0.5);

                        if cond_all_true {
                            self.accumulate_bounds_to_input(
                                &node.inputs[1],
                                node_lb,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                            continue;
                        } else if cond_all_false {
                            self.accumulate_bounds_to_input(
                                &node.inputs[2],
                                node_lb,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                            continue;
                        }
                    }

                    // Otherwise, concretize at this node and accumulate as a constant contribution.
                    let concrete = node_lb.concretize(where_bounds);
                    let lower_b = concrete
                        .lower
                        .into_dimensionality::<ndarray::Ix1>()
                        .map_err(|_| {
                            GammaError::InvalidSpec(
                                "Where concretization produced non-1D output".to_string(),
                            )
                        })?;
                    let upper_b = concrete
                        .upper
                        .into_dimensionality::<ndarray::Ix1>()
                        .map_err(|_| {
                            GammaError::InvalidSpec(
                                "Where concretization produced non-1D output".to_string(),
                            )
                        })?;

                    let zeros = ndarray::Array2::<f32>::zeros((output_dim, input_dim));
                    let const_lb = LinearBounds {
                        lower_a: zeros.clone(),
                        lower_b,
                        upper_a: zeros,
                        upper_b,
                    };

                    self.accumulate_bounds_to_input(
                        "_input",
                        const_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    continue;
                }
                Layer::NonZero(_) => {
                    // NonZero (data-dependent output) CROWN not supported, fall back to IBP
                    debug!("GraphNetwork DAG-CROWN: NonZero not supported, falling back to IBP");
                    return self.propagate_ibp(input);
                }
                Layer::Floor(f) => {
                    let new_lb = f.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Ceil(c) => {
                    let new_lb = c.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Round(r) => {
                    let new_lb = r.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Sign(s) => {
                    let new_lb = s.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::MulBinary(mul) => {
                    // MulBinary needs bounds on both inputs for McCormick relaxation
                    let input_a_bounds = if node.inputs[0] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "MulBinary input A '{}' not found",
                                node.inputs[0]
                            ))
                        })?
                    };
                    let input_b_bounds = if node.inputs[1] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[1]).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "MulBinary input B '{}' not found",
                                node.inputs[1]
                            ))
                        })?
                    };

                    // Try McCormick CROWN propagation
                    match mul.propagate_linear_binary(&node_lb, input_a_bounds, input_b_bounds) {
                        Ok((lb_a, lb_b)) => {
                            debug!(
                                "GraphNetwork DAG-CROWN: MulBinary '{}' McCormick CROWN succeeded",
                                node_name
                            );
                            self.accumulate_bounds_to_input(
                                &node.inputs[0],
                                lb_a,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                            self.accumulate_bounds_to_input(
                                &node.inputs[1],
                                lb_b,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                        }
                        Err(e) => {
                            // McCormick failed (e.g., infinite bounds), fall back to IBP
                            debug!(
                                "GraphNetwork DAG-CROWN: MulBinary '{}' McCormick failed ({}), falling back to IBP",
                                node_name, e
                            );
                            return self.propagate_ibp(input);
                        }
                    }
                }
            };
        }

        // Step 4: Concretize using input bounds
        // The final linear bounds should be in node_linear_bounds["_input"]
        let final_bounds = node_linear_bounds
            .get("_input")
            .ok_or_else(|| GammaError::InvalidSpec("No path to network input found".to_string()))?;

        debug!(
            "GraphNetwork DAG-CROWN: Concretizing {} outputs from {} inputs",
            final_bounds.num_outputs(),
            final_bounds.num_inputs()
        );

        let mut crown_output = final_bounds.concretize(input).reshape(&output_shape)?;

        let has_softmax = exec_order.iter().any(|node_name| {
            self.nodes
                .get(node_name)
                .map(|n| matches!(n.layer, Layer::Softmax(_) | Layer::CausalSoftmax(_)))
                .unwrap_or(false)
        });

        // Softmax relaxations can be numerically conservative; enforce that CROWN is never looser
        // than forward IBP at the output by intersecting with the IBP interval.
        if has_softmax && crown_output.shape() == output_bounds.shape() {
            for ((cl, cu), (il, iu)) in crown_output
                .lower
                .iter_mut()
                .zip(crown_output.upper.iter_mut())
                .zip(output_bounds.lower.iter().zip(output_bounds.upper.iter()))
            {
                let tightened_lower = (*cl).max(*il);
                let tightened_upper = (*cu).min(*iu);
                if tightened_lower <= tightened_upper {
                    *cl = tightened_lower;
                    *cu = tightened_upper;
                } else {
                    // Intersection empty: fall back to IBP for this element.
                    *cl = *il;
                    *cu = *iu;
                }
            }
        }

        Ok(crown_output)
    }

    /// Specification-guided CROWN backward propagation.
    ///
    /// Instead of computing bounds on each output independently, this method computes
    /// bounds on linear combinations of outputs defined by a specification matrix `C`.
    /// This preserves correlation information and produces much tighter bounds for
    /// verification properties like "output_0 > output_1".
    ///
    /// # Arguments
    /// * `input` - Input bounds
    /// * `spec_matrix` - Specification matrix of shape [num_specs, output_dim]
    ///   Each row defines a linear combination of outputs to bound.
    ///
    /// # Returns
    /// BoundedTensor with shape \[num_specs\], where bounds\[i\] are bounds on spec_matrix\[i\] @ outputs.
    ///
    /// # Example
    /// For property "class_0 > class_1", use spec_matrix = [[1, -1, 0, ...]]
    /// to get bounds on output_0 - output_1 directly.
    pub fn propagate_crown_with_specs(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_with_specs_and_engine(input, spec_matrix, None)
    }

    /// Specification-guided CROWN with optional GPU engine.
    #[instrument(skip(self, input, spec_matrix, engine), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_engine(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        let num_specs = spec_matrix.nrows();
        let spec_output_dim = spec_matrix.ncols();

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Step 1: Collect bounds at each node for nonlinear relaxations.
        let use_crown_ibp = self.should_use_crown_ibp_intermediates();
        let node_bounds = if use_crown_ibp {
            self.collect_crown_ibp_bounds_dag(input)?
        } else {
            self.collect_node_bounds(input)?
        };

        // Determine output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();

        // Validate spec matrix dimensions
        if spec_output_dim != output_dim {
            return Err(GammaError::shape_mismatch(
                vec![output_dim],
                vec![spec_output_dim],
            ));
        }

        debug!(
            "GraphNetwork spec-guided CROWN: {} specs, {} outputs",
            num_specs, output_dim
        );

        // Step 2: Initialize linear bounds with specification matrix instead of identity
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();

        // Output node starts with spec matrix (not identity)
        node_linear_bounds.insert(
            output_node_name.clone(),
            LinearBounds::from_spec_matrix(spec_matrix.clone()),
        );

        let input_dim = input.len();
        let mut input_accumulated = false;

        // Step 3: Propagate backward through nodes in reverse order
        // This is the same as propagate_crown_with_engine but uses num_specs as output_dim
        for node_name in exec_order.iter().rev() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => continue,
            };

            debug!(
                "GraphNetwork spec-guided CROWN: backward through {} ({}) with {} outputs",
                node_name,
                node.layer.layer_type(),
                node_lb.num_outputs()
            );

            let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        node.inputs[0]
                    ))
                })?
            };

            // Handle different layer types (same as propagate_crown_with_engine)
            match &node.layer {
                Layer::Linear(l) => {
                    let expected_inputs = l.out_features();
                    let got_inputs = node_lb.num_inputs();
                    if got_inputs != expected_inputs {
                        debug!(
                            "Spec-guided CROWN: Linear dimension mismatch at {}: expected {}, got {}. Falling back to IBP.",
                            node_name, expected_inputs, got_inputs
                        );
                        return self.propagate_crown_with_specs_fallback_ibp(
                            input,
                            spec_matrix,
                            &node_bounds,
                            output_node_name,
                        );
                    }
                    let new_lb = l.propagate_linear_with_engine(&node_lb, engine)?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    let new_lb = r.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv2d(c) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        debug!(
                            "Spec-guided CROWN: Conv2d input shape too small, falling back to IBP"
                        );
                        return self.propagate_crown_with_specs_fallback_ibp(
                            input,
                            spec_matrix,
                            &node_bounds,
                            output_node_name,
                        );
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = match conv_with_shape.propagate_linear(&node_lb) {
                        Ok(Cow::Owned(lb)) => lb,
                        Ok(Cow::Borrowed(_)) => node_lb,
                        Err(e) => {
                            debug!("Spec-guided CROWN: Conv2d backward failed ({}), falling back to IBP", e);
                            return self.propagate_crown_with_specs_fallback_ibp(
                                input,
                                spec_matrix,
                                &node_bounds,
                                output_node_name,
                            );
                        }
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv1d(conv) => {
                    let input_shape = pre_activation.shape();
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else {
                        debug!(
                            "Spec-guided CROWN: Conv1d input shape too small, falling back to IBP"
                        );
                        return self.propagate_crown_with_specs_fallback_ibp(
                            input,
                            spec_matrix,
                            &node_bounds,
                            output_node_name,
                        );
                    };
                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_length(in_len);
                    let new_lb = match conv_with_shape.propagate_linear(&node_lb) {
                        Ok(Cow::Owned(lb)) => lb,
                        Ok(Cow::Borrowed(_)) => node_lb,
                        Err(e) => {
                            debug!("Spec-guided CROWN: Conv1d backward failed ({}), falling back to IBP", e);
                            return self.propagate_crown_with_specs_fallback_ibp(
                                input,
                                spec_matrix,
                                &node_bounds,
                                output_node_name,
                            );
                        }
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    let (lb_a, lb_b) = add.propagate_linear_binary(&node_lb)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = match f.propagate_linear(&node_lb) {
                        Ok(Cow::Owned(lb)) => lb,
                        Ok(Cow::Borrowed(_)) => node_lb,
                        Err(e) => {
                            debug!("Spec-guided CROWN: Flatten backward failed ({}), falling back to IBP", e);
                            return self.propagate_crown_with_specs_fallback_ibp(
                                input,
                                spec_matrix,
                                &node_bounds,
                                output_node_name,
                            );
                        }
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::MaxPool2d(mp) => {
                    let new_lb = mp.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::AveragePool(ap) => {
                    let new_lb = ap.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::BatchNorm(bn) => {
                    let new_lb = bn.propagate_linear_with_bounds(&node_lb, pre_activation)?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Transpose(t) => {
                    let input_shape = pre_activation.shape().to_vec();
                    let mut t_clone = t.clone();
                    t_clone.set_input_shape(input_shape);
                    let new_lb = match t_clone.propagate_linear(&node_lb) {
                        Ok(Cow::Owned(lb)) => lb,
                        Ok(Cow::Borrowed(_)) => node_lb,
                        Err(_) => {
                            return self.propagate_crown_with_specs_fallback_ibp(
                                input,
                                spec_matrix,
                                &node_bounds,
                                output_node_name,
                            )
                        }
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Reshape(r) => {
                    let new_lb = match r.propagate_linear(&node_lb) {
                        Ok(Cow::Owned(lb)) => lb,
                        Ok(Cow::Borrowed(_)) => node_lb,
                        Err(_) => {
                            return self.propagate_crown_with_specs_fallback_ibp(
                                input,
                                spec_matrix,
                                &node_bounds,
                                output_node_name,
                            )
                        }
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                // For unsupported layers, fall back to IBP-based spec computation
                _ => {
                    debug!(
                        "Spec-guided CROWN: unsupported layer {} ({}), falling back to IBP",
                        node_name,
                        node.layer.layer_type()
                    );
                    return self.propagate_crown_with_specs_fallback_ibp(
                        input,
                        spec_matrix,
                        &node_bounds,
                        output_node_name,
                    );
                }
            };
        }

        // Step 4: Concretize using input bounds
        let final_bounds = node_linear_bounds
            .get("_input")
            .ok_or_else(|| GammaError::InvalidSpec("No path to network input found".to_string()))?;

        debug!(
            "GraphNetwork spec-guided CROWN: Concretizing {} specs from {} inputs",
            final_bounds.num_outputs(),
            final_bounds.num_inputs()
        );

        let crown_output = final_bounds.concretize(input);

        // Output shape is [num_specs]
        crown_output.reshape(&[num_specs])
    }

    /// Fallback for spec-guided CROWN: compute spec bounds from IBP output bounds.
    fn propagate_crown_with_specs_fallback_ibp(
        &self,
        _input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        output_node_name: &str,
    ) -> Result<BoundedTensor> {
        // Get IBP output bounds
        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!(
                "Output node {} not found for IBP fallback",
                output_node_name
            ))
        })?;

        // Compute spec bounds via interval arithmetic (same as current objective_bounds_vec)
        let flat = output_bounds.flatten();
        let num_specs = spec_matrix.nrows();
        let mut lower = ndarray::Array1::<f32>::zeros(num_specs);
        let mut upper = ndarray::Array1::<f32>::zeros(num_specs);

        for (i, spec_row) in spec_matrix.rows().into_iter().enumerate() {
            let mut l = 0.0f32;
            let mut u = 0.0f32;
            for (j, &c) in spec_row.iter().enumerate() {
                let input_l = flat.lower[[j]];
                let input_u = flat.upper[[j]];
                if c >= 0.0 {
                    l += c * input_l;
                    u += c * input_u;
                } else {
                    l += c * input_u;
                    u += c * input_l;
                }
            }
            lower[i] = l;
            upper[i] = u;
        }

        Ok(BoundedTensor {
            lower: lower.into_dyn(),
            upper: upper.into_dyn(),
        })
    }

    /// Helper to accumulate linear bounds to a node's input.
    fn accumulate_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: LinearBounds,
        node_linear_bounds: &mut std::collections::HashMap<String, LinearBounds>,
        _output_dim: usize,
        _input_dim: usize,
        input_accumulated: &mut bool,
    ) {
        if input_name == "_input" {
            // Accumulate to network input
            if *input_accumulated {
                // Add to existing bounds with safe infinity handling
                if let Some(existing) = node_linear_bounds.get_mut("_input") {
                    existing.lower_a =
                        Self::safe_add_2d(&existing.lower_a, &new_bounds.lower_a, true);
                    existing.lower_b =
                        Self::safe_add_1d(&existing.lower_b, &new_bounds.lower_b, true);
                    existing.upper_a =
                        Self::safe_add_2d(&existing.upper_a, &new_bounds.upper_a, false);
                    existing.upper_b =
                        Self::safe_add_1d(&existing.upper_b, &new_bounds.upper_b, false);
                }
            } else {
                // First contribution to input
                node_linear_bounds.insert("_input".to_string(), new_bounds);
                *input_accumulated = true;
            }
        } else {
            // Accumulate to intermediate node
            if let Some(existing) = node_linear_bounds.get_mut(input_name) {
                existing.lower_a = Self::safe_add_2d(&existing.lower_a, &new_bounds.lower_a, true);
                existing.lower_b = Self::safe_add_1d(&existing.lower_b, &new_bounds.lower_b, true);
                existing.upper_a = Self::safe_add_2d(&existing.upper_a, &new_bounds.upper_a, false);
                existing.upper_b = Self::safe_add_1d(&existing.upper_b, &new_bounds.upper_b, false);
            } else {
                node_linear_bounds.insert(input_name.to_string(), new_bounds);
            }
        }
    }

    /// Safe addition for 2D arrays that handles inf + (-inf) = NaN.
    ///
    /// For lower bounds: inf + (-inf) -> -inf (conservative lower)
    /// For upper bounds: inf + (-inf) -> +inf (conservative upper)
    fn safe_add_2d(existing: &Array2<f32>, new: &Array2<f32>, is_lower: bool) -> Array2<f32> {
        use ndarray::Zip;
        let mut result = existing + new;
        Zip::from(&mut result)
            .and(existing)
            .and(new)
            .for_each(|r, &e, &n| {
                if r.is_nan() {
                    // inf + (-inf) case: use conservative bound
                    *r = if is_lower {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                } else if e.is_nan() || n.is_nan() {
                    // Propagate NaN if either input was already NaN
                    *r = f32::NAN;
                }
            });
        result
    }

    /// Safe addition for 1D arrays that handles inf + (-inf) = NaN.
    fn safe_add_1d(existing: &Array1<f32>, new: &Array1<f32>, is_lower: bool) -> Array1<f32> {
        use ndarray::Zip;
        let mut result = existing + new;
        Zip::from(&mut result)
            .and(existing)
            .and(new)
            .for_each(|r, &e, &n| {
                if r.is_nan() {
                    // inf + (-inf) case: use conservative bound
                    *r = if is_lower {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                } else if e.is_nan() || n.is_nan() {
                    // Propagate NaN if either input was already NaN
                    *r = f32::NAN;
                }
            });
        result
    }

    /// Safe addition for bound accumulation that handles inf + (-inf) = NaN.
    ///
    /// When accumulating bounds from multiple paths (e.g., residual connections),
    /// if one path contributes +inf and another contributes -inf, standard addition
    /// produces NaN. For sound verification:
    /// - For lower bounds: inf + (-inf) -> -inf (conservative lower)
    /// - For upper bounds: inf + (-inf) -> +inf (conservative upper)
    fn safe_add_bounds(existing: &ArrayD<f32>, new: &ArrayD<f32>, is_lower: bool) -> ArrayD<f32> {
        use ndarray::Zip;
        let mut result = existing + new;
        Zip::from(&mut result)
            .and(existing)
            .and(new)
            .for_each(|r, &e, &n| {
                if r.is_nan() {
                    // inf + (-inf) case: use conservative bound
                    *r = if is_lower {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                } else if e.is_nan() || n.is_nan() {
                    // Propagate NaN if either input was already NaN
                    *r = f32::NAN;
                }
            });
        result
    }

    pub(crate) fn sanitize_bounds_for_fallback(bounds: &BoundedTensor) -> BoundedTensor {
        use ndarray::Zip;

        let mut lower = bounds.lower.clone();
        let mut upper = bounds.upper.clone();

        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| {
            if !l.is_finite() {
                *l = f32::NEG_INFINITY;
            }
            if !u.is_finite() {
                *u = f32::INFINITY;
            }
            if *l > *u {
                *l = f32::NEG_INFINITY;
                *u = f32::INFINITY;
            }
        });

        BoundedTensor { lower, upper }
    }

    /// Helper to accumulate batched linear bounds to a node's input (for N-D CROWN).
    fn accumulate_batched_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: BatchedLinearBounds,
        node_linear_bounds: &mut std::collections::HashMap<String, BatchedLinearBounds>,
        input_accumulated: &mut bool,
    ) {
        if input_name == "_input" {
            // Accumulate to network input
            if *input_accumulated {
                // Add to existing bounds with safe infinity handling
                if let Some(existing) = node_linear_bounds.get_mut("_input") {
                    existing.lower_a =
                        Self::safe_add_bounds(&existing.lower_a, &new_bounds.lower_a, true);
                    existing.lower_b =
                        Self::safe_add_bounds(&existing.lower_b, &new_bounds.lower_b, true);
                    existing.upper_a =
                        Self::safe_add_bounds(&existing.upper_a, &new_bounds.upper_a, false);
                    existing.upper_b =
                        Self::safe_add_bounds(&existing.upper_b, &new_bounds.upper_b, false);
                }
            } else {
                // First contribution to input
                node_linear_bounds.insert("_input".to_string(), new_bounds);
                *input_accumulated = true;
            }
        } else {
            // Accumulate to intermediate node
            if let Some(existing) = node_linear_bounds.get_mut(input_name) {
                existing.lower_a =
                    Self::safe_add_bounds(&existing.lower_a, &new_bounds.lower_a, true);
                existing.lower_b =
                    Self::safe_add_bounds(&existing.lower_b, &new_bounds.lower_b, true);
                existing.upper_a =
                    Self::safe_add_bounds(&existing.upper_a, &new_bounds.upper_a, false);
                existing.upper_b =
                    Self::safe_add_bounds(&existing.upper_b, &new_bounds.upper_b, false);
            } else {
                node_linear_bounds.insert(input_name.to_string(), new_bounds);
            }
        }
    }

    /// Propagate bounds through the graph using N-D batched CROWN.
    ///
    /// This preserves tensor shape structure throughout propagation instead of
    /// flattening to 1D. Essential for transformer models where operations
    /// like attention have cross-position interactions.
    ///
    /// Supported layers:
    /// - Linear, ReLU, GELU, Softmax, LayerNorm (full batched support)
    /// - Conv1d (transposed convolution backward)
    /// - Add, MatMul (binary) - MatMul uses McCormick envelope relaxation
    /// - Transpose, MulConstant, DivConstant, AddConstant, SubConstant
    ///
    /// Note: Conv2d has propagate_linear_batched at the layer level but requires
    /// shape transformation for network integration. Falls back to IBP for Conv2d.
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Step 1: Collect bounds at each node for nonlinear relaxations.
        //
        // Batched CROWN is primarily used for transformer-style graphs; avoid CROWN-IBP
        // intermediate tightening unless the graph is CNN-style and supported.
        let node_bounds = if self.should_use_crown_ibp_intermediates() {
            self.collect_crown_ibp_bounds_dag(input)?
        } else {
            self.collect_node_bounds(input)?
        };

        // Determine output node and shape
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "GraphNetwork batched CROWN: Starting backward propagation from {:?}",
            output_shape
        );

        // Step 2: Initialize batched linear bounds per node
        let mut node_linear_bounds: std::collections::HashMap<String, BatchedLinearBounds> =
            std::collections::HashMap::new();

        // Output node starts with identity bounds
        node_linear_bounds.insert(
            output_node_name.clone(),
            BatchedLinearBounds::identity(&output_shape),
        );

        // Track if we've accumulated bounds at the input
        let mut input_accumulated = false;

        // Step 3: Propagate backward through nodes in reverse order
        for node_name in exec_order.iter().rev() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get this node's accumulated linear bounds
            let node_lb = match node_linear_bounds.get(node_name) {
                Some(lb) => lb.clone(),
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    continue;
                }
            };

            // Get pre-activation bounds for this node
            let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        node.inputs[0]
                    ))
                })?
            };

            // Handle different layer types
            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (Linear): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    let new_lb = r
                        .propagate_linear_batched_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Batched CROWN failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::GELU(g) => {
                    let new_lb = g
                        .propagate_linear_batched_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Batched CROWN failed at node '{}' (GELU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Softmax(s) => {
                    let new_lb = s
                        .propagate_linear_batched_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Batched CROWN failed at node '{}' (Softmax): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::LayerNorm(ln) => {
                    let new_lb = ln
                        .propagate_linear_batched_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Batched CROWN failed at node '{}' (LayerNorm): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    let (lb_a, lb_b) =
                        add.propagate_linear_batched_binary(&node_lb).map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Batched CROWN failed at node '{}' (Add): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Transpose(t) => {
                    let new_lb = t.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (Transpose): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::MulConstant(m) => {
                    let new_lb = m.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (MulConstant): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::DivConstant(d) => {
                    let new_lb = d.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (DivConstant): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::AddConstant(ac) => {
                    let new_lb = ac.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (AddConstant): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::SubConstant(sc) => {
                    let new_lb = sc.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (SubConstant): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv1d(c) => {
                    let new_lb = c.propagate_linear_batched(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (Conv1d): {}",
                            node_name, e
                        ))
                    })?;
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(_) => {
                    // Flatten is an index rearrangement that doesn't change the linear coefficients.
                    // Just propagate the bounds unchanged to the input.
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        node_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Reshape(_) => {
                    // Reshape is an index rearrangement that doesn't change the linear coefficients.
                    // Just propagate the bounds unchanged to the input.
                    self.accumulate_batched_bounds_to_input(
                        &node.inputs[0],
                        node_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                // Note: Conv2d has propagate_linear_batched at layer level but is not
                // integrated here yet due to BatchedLinearBounds shape transformation needs.
                // Falls through to IBP fallback below.
                Layer::MatMul(matmul) => {
                    // MatMul is binary: needs IBP bounds for both inputs
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "MatMul node '{}' requires 2 inputs, got {}",
                            node_name,
                            node.inputs.len()
                        )));
                    }

                    let input_a_name = &node.inputs[0];
                    let input_b_name = &node.inputs[1];

                    // Get IBP bounds for both inputs
                    let input_a_bounds = if input_a_name == "_input" {
                        input
                    } else {
                        node_bounds.get(input_a_name).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "IBP bounds for MatMul input A '{}' not found",
                                input_a_name
                            ))
                        })?
                    };

                    let input_b_bounds = if input_b_name == "_input" {
                        input
                    } else {
                        node_bounds.get(input_b_name).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "IBP bounds for MatMul input B '{}' not found",
                                input_b_name
                            ))
                        })?
                    };

                    let (lb_a, lb_b) = match matmul.propagate_linear_batched_binary(
                        &node_lb,
                        input_a_bounds,
                        input_b_bounds,
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            // First attempt failed. Check if this is an attention-shaped MatMul
                            // where we can try using identity_for_attention for tighter bounds.
                            let matmul_ibp_bounds =
                                node_bounds.get(node_name).ok_or_else(|| {
                                    GammaError::InvalidSpec(format!(
                                        "IBP bounds for MatMul node '{}' not found",
                                        node_name
                                    ))
                                })?;
                            let matmul_output_shape = matmul_ibp_bounds.shape();

                            // Try attention-specific CROWN for attention-shaped outputs [batch, heads, seq, seq]
                            // with seq <= 64 (memory limit from identity_for_attention)
                            if let Some(attention_identity) =
                                BatchedLinearBounds::identity_for_attention(matmul_output_shape)
                            {
                                debug!(
                                    "GraphNetwork batched CROWN: MatMul '{}' trying attention identity ({})",
                                    node_name, e
                                );

                                // Retry with flattened attention identity bounds
                                match matmul.propagate_linear_batched_binary(
                                    &attention_identity,
                                    input_a_bounds,
                                    input_b_bounds,
                                ) {
                                    Ok((_lb_a_attn, _lb_b_attn)) => {
                                        debug!(
                                            "GraphNetwork batched CROWN: MatMul '{}' attention CROWN succeeded",
                                            node_name
                                        );

                                        // Attention CROWN succeeded! The McCormick relaxation computed
                                        // linear bounds (lb_a_attn, lb_b_attn) that express the attention
                                        // output in terms of Q and K inputs.
                                        //
                                        // However, directly concretizing these McCormick bounds produces
                                        // WIDER bounds than IBP because McCormick's linear relaxation is
                                        // optimized for CROWN backward propagation composition, not for
                                        // immediate interval concretization.
                                        //
                                        // For full McCormick benefit, we would need to compose
                                        // node_lb (downstream bounds) with lb_a_attn/lb_b_attn.
                                        //
                                        // SHAPE MISMATCH LIMITATION (Worker #192):
                                        // - node_lb has per-position structure: A shape [b,h,s,s,s]
                                        //   where it operates only on the last dim of attention output.
                                        // - lb_a_attn/lb_b_attn have flattened structure: A shape [b,h,flat,q]
                                        //   where flat = s*s represents the entire attention matrix.
                                        //
                                        // Per-position bounds cannot express cross-position correlations
                                        // needed for McCormick composition. A different representation
                                        // would be needed (e.g., zonotopes or flattened CROWN which
                                        // doesn't scale to large models).
                                        //
                                        // The compose() method (Worker #191) works for compatible shapes
                                        // but not for this per-position vs flattened mismatch.
                                        //
                                        // For now, use partial CROWN with IBP bounds at attention layer:
                                        let partial_crown_with_attention =
                                            node_lb.concretize(matmul_ibp_bounds);

                                        if partial_crown_with_attention.shape()
                                            != output_shape.as_slice()
                                        {
                                            return partial_crown_with_attention
                                                .reshape(&output_shape);
                                        } else {
                                            return Ok(partial_crown_with_attention);
                                        }
                                    }
                                    Err(e2) => {
                                        debug!(
                                            "GraphNetwork batched CROWN: MatMul '{}' attention CROWN also failed ({}), using partial CROWN",
                                            node_name, e2
                                        );
                                        // Fall through to partial CROWN below
                                    }
                                }
                            } else {
                                debug!(
                                    "GraphNetwork batched CROWN: MatMul '{}' not supported ({}), using partial CROWN",
                                    node_name, e
                                );
                            }

                            // Partial CROWN: concretize the accumulated bounds using the IBP bounds
                            // at this MatMul's output. This gives CROWN benefits for layers after
                            // the MatMul while using IBP for this MatMul and earlier layers.
                            //
                            // However, if the IBP bounds are infinite/NaN, concretize() can produce
                            // NaN from (-inf + inf) additions. In that case, just return IBP bounds.
                            let has_inf_or_nan = matmul_ibp_bounds
                                .lower
                                .iter()
                                .chain(matmul_ibp_bounds.upper.iter())
                                .any(|&v| v.is_infinite() || v.is_nan());

                            let partial_crown_bounds = if has_inf_or_nan {
                                debug!(
                                    "GraphNetwork batched CROWN: MatMul '{}' has infinite IBP bounds, returning IBP",
                                    node_name
                                );
                                Self::sanitize_bounds_for_fallback(matmul_ibp_bounds)
                            } else {
                                node_lb.concretize(matmul_ibp_bounds)
                            };

                            // Ensure output shape matches expected
                            if partial_crown_bounds.shape() != output_shape.as_slice() {
                                return partial_crown_bounds.reshape(&output_shape);
                            } else {
                                return Ok(partial_crown_bounds);
                            }
                        }
                    };

                    self.accumulate_batched_bounds_to_input(
                        input_a_name,
                        lb_a,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                    self.accumulate_batched_bounds_to_input(
                        input_b_name,
                        lb_b,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::MulBinary(mul) => {
                    // MulBinary needs bounds on both inputs for McCormick relaxation
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "MulBinary node '{}' requires 2 inputs, got {}",
                            node_name,
                            node.inputs.len()
                        )));
                    }

                    let input_a_name = &node.inputs[0];
                    let input_b_name = &node.inputs[1];

                    // Get IBP bounds for both inputs
                    let input_a_bounds = if input_a_name == "_input" {
                        input
                    } else {
                        node_bounds.get(input_a_name).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "IBP bounds for MulBinary input A '{}' not found",
                                input_a_name
                            ))
                        })?
                    };

                    let input_b_bounds = if input_b_name == "_input" {
                        input
                    } else {
                        node_bounds.get(input_b_name).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "IBP bounds for MulBinary input B '{}' not found",
                                input_b_name
                            ))
                        })?
                    };

                    // Try McCormick CROWN propagation
                    match mul.propagate_linear_batched_binary(
                        &node_lb,
                        input_a_bounds,
                        input_b_bounds,
                    ) {
                        Ok((lb_a, lb_b)) => {
                            // Check if returned bounds contain inf/NaN
                            let lb_a_has_bad = lb_a
                                .lower_a
                                .iter()
                                .chain(lb_a.upper_a.iter())
                                .chain(lb_a.lower_b.iter())
                                .chain(lb_a.upper_b.iter())
                                .any(|&v| v.is_infinite() || v.is_nan());
                            let lb_b_has_bad = lb_b
                                .lower_a
                                .iter()
                                .chain(lb_b.upper_a.iter())
                                .chain(lb_b.lower_b.iter())
                                .chain(lb_b.upper_b.iter())
                                .any(|&v| v.is_infinite() || v.is_nan());

                            if lb_a_has_bad || lb_b_has_bad {
                                // McCormick produced inf/NaN, fall back to partial CROWN
                                debug!(
                                    "GraphNetwork batched CROWN: MulBinary '{}' McCormick CROWN produced inf/NaN, using partial CROWN",
                                    node_name
                                );
                                let mul_ibp_bounds =
                                    node_bounds.get(node_name).ok_or_else(|| {
                                        GammaError::InvalidSpec(format!(
                                            "IBP bounds for MulBinary node '{}' not found",
                                            node_name
                                        ))
                                    })?;
                                let partial_crown_bounds = node_lb.concretize(mul_ibp_bounds);

                                // Final NaN check
                                let has_nan = partial_crown_bounds
                                    .lower
                                    .iter()
                                    .chain(partial_crown_bounds.upper.iter())
                                    .any(|&v| v.is_nan());

                                let result = if has_nan {
                                    Self::sanitize_bounds_for_fallback(mul_ibp_bounds)
                                } else {
                                    partial_crown_bounds
                                };

                                if result.shape() != output_shape.as_slice() {
                                    return result.reshape(&output_shape);
                                } else {
                                    return Ok(result);
                                }
                            }

                            debug!(
                                "GraphNetwork batched CROWN: MulBinary '{}' McCormick CROWN succeeded",
                                node_name
                            );
                            self.accumulate_batched_bounds_to_input(
                                input_a_name,
                                lb_a,
                                &mut node_linear_bounds,
                                &mut input_accumulated,
                            );
                            self.accumulate_batched_bounds_to_input(
                                input_b_name,
                                lb_b,
                                &mut node_linear_bounds,
                                &mut input_accumulated,
                            );
                        }
                        Err(e) => {
                            // MulBinary CROWN failed, use partial CROWN
                            debug!(
                                "GraphNetwork batched CROWN: MulBinary '{}' failed ({}), using partial CROWN",
                                node_name, e
                            );
                            let mul_ibp_bounds = node_bounds.get(node_name).ok_or_else(|| {
                                GammaError::InvalidSpec(format!(
                                    "IBP bounds for MulBinary node '{}' not found",
                                    node_name
                                ))
                            })?;

                            // Check for inf/NaN in IBP bounds
                            let has_inf_or_nan = mul_ibp_bounds
                                .lower
                                .iter()
                                .chain(mul_ibp_bounds.upper.iter())
                                .any(|&v| v.is_infinite() || v.is_nan());

                            let partial_crown_bounds = if has_inf_or_nan {
                                debug!(
                                    "GraphNetwork batched CROWN: MulBinary '{}' has infinite IBP bounds, returning IBP",
                                    node_name
                                );
                                Self::sanitize_bounds_for_fallback(mul_ibp_bounds)
                            } else {
                                node_lb.concretize(mul_ibp_bounds)
                            };

                            if partial_crown_bounds.shape() != output_shape.as_slice() {
                                return partial_crown_bounds.reshape(&output_shape);
                            } else {
                                return Ok(partial_crown_bounds);
                            }
                        }
                    }
                }
                // Layers that don't yet have batched support - use partial CROWN
                _ => {
                    debug!(
                        "GraphNetwork batched CROWN: unsupported layer '{}' ({}), using partial CROWN",
                        node_name,
                        node.layer.layer_type()
                    );
                    // Partial CROWN: concretize the accumulated bounds using the IBP bounds
                    // at this node's output. This gives CROWN benefits for layers after
                    // the unsupported layer while using IBP for this layer and earlier.
                    let node_ibp_bounds = node_bounds.get(node_name).ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "IBP bounds for node '{}' not found",
                            node_name
                        ))
                    })?;

                    // Check IBP bounds AND linear bound coefficients for inf/NaN.
                    // If either contains inf/NaN, concretize() can produce NaN.
                    let ibp_has_inf_or_nan = node_ibp_bounds
                        .lower
                        .iter()
                        .chain(node_ibp_bounds.upper.iter())
                        .any(|&v| v.is_infinite() || v.is_nan());

                    let lb_has_inf_or_nan = node_lb
                        .lower_a
                        .iter()
                        .chain(node_lb.upper_a.iter())
                        .chain(node_lb.lower_b.iter())
                        .chain(node_lb.upper_b.iter())
                        .any(|&v| v.is_infinite() || v.is_nan());

                    let partial_crown_bounds = if ibp_has_inf_or_nan || lb_has_inf_or_nan {
                        debug!(
                            "GraphNetwork batched CROWN: node '{}' has infinite/NaN bounds (IBP:{}, LB:{}), returning IBP",
                            node_name, ibp_has_inf_or_nan, lb_has_inf_or_nan
                        );
                        Self::sanitize_bounds_for_fallback(node_ibp_bounds)
                    } else {
                        let concretized = node_lb.concretize(node_ibp_bounds);
                        // Final check: if concretize produced NaN, fall back to IBP
                        let has_nan = concretized
                            .lower
                            .iter()
                            .chain(concretized.upper.iter())
                            .any(|&v| v.is_nan());
                        if has_nan {
                            debug!(
                                "GraphNetwork batched CROWN: node '{}' concretize produced NaN, returning IBP",
                                node_name
                            );
                            Self::sanitize_bounds_for_fallback(node_ibp_bounds)
                        } else {
                            concretized
                        }
                    };

                    if partial_crown_bounds.shape() != output_shape.as_slice() {
                        return partial_crown_bounds.reshape(&output_shape);
                    } else {
                        return Ok(partial_crown_bounds);
                    }
                }
            };
        }

        // Step 4: Concretize using input bounds
        let final_bounds = node_linear_bounds
            .get("_input")
            .ok_or_else(|| GammaError::InvalidSpec("No path to network input found".to_string()))?;

        // Check if final bounds coefficients contain inf/NaN
        let lb_has_inf_or_nan = final_bounds
            .lower_a
            .iter()
            .chain(final_bounds.upper_a.iter())
            .chain(final_bounds.lower_b.iter())
            .chain(final_bounds.upper_b.iter())
            .any(|&v| v.is_infinite() || v.is_nan());

        if lb_has_inf_or_nan {
            debug!(
                "GraphNetwork batched CROWN: final linear bounds contain inf/NaN, falling back to IBP"
            );
            let ibp = self.propagate_ibp(input)?;
            return Ok(Self::sanitize_bounds_for_fallback(&ibp));
        }

        debug!(
            "GraphNetwork batched CROWN: Concretizing with input shape {:?}, output shape {:?}",
            final_bounds.input_shape, final_bounds.output_shape
        );

        let crown_output = final_bounds.concretize(input);

        // Final safety check: if concretize produced NaN, fall back to IBP
        let has_nan = crown_output
            .lower
            .iter()
            .chain(crown_output.upper.iter())
            .any(|&v| v.is_nan());

        if has_nan {
            debug!(
                "GraphNetwork batched CROWN: final concretize produced NaN, falling back to IBP"
            );
            let ibp = self.propagate_ibp(input)?;
            return Ok(Self::sanitize_bounds_for_fallback(&ibp));
        }

        // Ensure output shape matches expected
        if crown_output.shape() != output_shape.as_slice() {
            crown_output.reshape(&output_shape)
        } else {
            Ok(crown_output)
        }
    }

    /// Propagate bounds through the graph using α-CROWN with optimized parameters.
    ///
    /// α-CROWN extends CROWN by making the lower bound slope (α) for unstable ReLUs
    /// learnable and optimizing it via gradient descent to tighten bounds.
    ///
    /// Algorithm:
    /// 1. Run IBP to collect pre-activation bounds at each node
    /// 2. Identify ReLU nodes and initialize α state
    /// 3. For each optimization iteration:
    ///    a. Run CROWN backward with current α values
    ///    b. Concretize to get bounds
    ///    c. Compute gradients ∂bounds/∂α
    ///    d. Update α via gradient descent
    /// 4. Return the tightest bounds found
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_alpha_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_engine(input, None)
    }

    /// α-CROWN with optional GEMM acceleration engine.
    #[inline]
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_alpha_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine(
            input,
            &AlphaCrownConfig::default(),
            engine,
        )
    }

    /// α-CROWN with custom configuration (no acceleration engine).
    #[instrument(skip(self, input, config), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    pub fn propagate_alpha_crown_with_config(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine(input, config, None)
    }

    /// α-CROWN with custom configuration and optional GEMM acceleration engine.
    #[instrument(skip(self, input, config, engine), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    pub fn propagate_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Check if graph is sequential
        let is_sequential = self.is_sequential_graph(&exec_order);
        if !is_sequential {
            // Use DAG α-CROWN for non-sequential graphs
            debug!("GraphNetwork α-CROWN: non-sequential graph, using DAG α-CROWN");
            return self.propagate_dag_alpha_crown_with_config_and_engine(input, config, engine);
        }

        // Step 1: Run IBP forward to collect bounds at each node
        let node_bounds = self.collect_node_bounds(input)?;

        // Determine output dimension
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();

        // Step 2: Identify ReLU nodes and their pre-activation bounds
        let relu_nodes: Vec<(String, usize)> = exec_order
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::ReLU(_)))
                    .unwrap_or(false)
            })
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        if relu_nodes.is_empty() {
            // No ReLU nodes, just use CROWN
            return self.propagate_crown_with_engine(input, engine);
        }

        // Check for operations that need DAG α-CROWN path (has better layer support)
        for node_name in &exec_order {
            if let Some(node) = self.nodes.get(node_name) {
                match &node.layer {
                    // Conv2d, MaxPool2d, BatchNorm, Concat: use DAG α-CROWN which handles these
                    // Concat requires special handling (splitting linear bounds) supported by DAG CROWN
                    Layer::Conv2d(_)
                    | Layer::MaxPool2d(_)
                    | Layer::BatchNorm(_)
                    | Layer::Concat(_) => {
                        debug!(
                            "GraphNetwork α-CROWN: {} detected, using DAG α-CROWN (better layer support)",
                            node.layer.layer_type()
                        );
                        return self.propagate_dag_alpha_crown_with_config_and_engine(
                            input, config, engine,
                        );
                    }
                    // Softmax, LayerNorm, MatMul, Add: use CROWN (not yet in DAG path)
                    Layer::Softmax(_) | Layer::LayerNorm(_) | Layer::MatMul(_) | Layer::Add(_) => {
                        debug!(
                            "GraphNetwork α-CROWN: unsupported op {}, using CROWN",
                            node.layer.layer_type()
                        );
                        return self.propagate_crown_with_engine(input, engine);
                    }
                    _ => {} // Linear, ReLU, Transpose, GELU are supported
                }
            }
        }

        // Build map from node name to execution index (unused currently but kept for future optimization)
        let _node_to_idx: std::collections::HashMap<String, usize> = exec_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Get pre-activation bounds for each ReLU node
        let pre_activation_bounds: Vec<BoundedTensor> = relu_nodes
            .iter()
            .map(|(name, _)| {
                let node = self.nodes.get(name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!("ReLU node not found: {}", name))
                })?;
                if node.inputs.is_empty() || node.inputs[0] == "_input" {
                    Ok(input.clone())
                } else {
                    Ok(node_bounds
                        .get(&node.inputs[0])
                        .cloned()
                        .unwrap_or_else(|| input.clone()))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Initialize alpha state
        let mut alpha_state = AlphaState::from_preactivation_bounds(
            &pre_activation_bounds,
            &(0..relu_nodes.len()).collect::<Vec<_>>(),
        );

        let num_unstable = alpha_state.num_unstable();
        if num_unstable == 0 {
            debug!("GraphNetwork α-CROWN: No unstable neurons, using CROWN");
            return self.propagate_crown_with_engine(input, engine);
        }

        debug!(
            "GraphNetwork α-CROWN: Starting optimization with {} unstable neurons across {} ReLU nodes",
            num_unstable,
            relu_nodes.len()
        );

        // Map from node name to ReLU index
        let relu_name_to_idx: std::collections::HashMap<String, usize> = relu_nodes
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();

        // Step 3: Optimization loop
        // Track element-wise best bounds across iterations:
        // - best_lower: maximum lower bound seen for each output dimension
        // - best_upper: minimum upper bound seen for each output dimension
        // Initialize from CROWN bounds to ensure α-CROWN never returns worse bounds.
        let crown_bounds = self.propagate_crown_with_engine(input, engine)?;
        let mut best_lower: ArrayD<f32> = crown_bounds.lower.clone();
        let mut best_upper: ArrayD<f32> = crown_bounds.upper.clone();
        let mut best_lower_sum: f32 = crown_bounds
            .lower
            .as_slice()
            .map(|s| s.iter().sum())
            .unwrap_or(f32::NEG_INFINITY);
        let mut prev_best_lower_sum = best_lower_sum;
        let mut no_improve_iters = 0usize;
        let mut lr = config.learning_rate;

        for iter in 0..config.iterations {
            // Run CROWN backward with current alpha values
            let mut linear_bounds = LinearBounds::identity(output_dim);
            let mut gradients: Vec<Array1<f32>> = relu_nodes
                .iter()
                .map(|(name, _)| {
                    // Safe: relu_nodes was built from self.nodes, so node exists
                    let node = self
                        .nodes
                        .get(name)
                        .expect("ReLU node should exist in graph");
                    let pre_act = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[0]).unwrap_or(input)
                    };
                    Ok(Array1::zeros(pre_act.len()))
                })
                .collect::<Result<Vec<_>>>()?;

            // Backward pass through nodes in reverse order
            for node_name in exec_order.iter().rev() {
                let node = self.nodes.get(node_name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!("Node not found: {}", node_name))
                })?;

                let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                    input
                } else {
                    node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Pre-activation bounds for {} not found",
                            node.inputs[0]
                        ))
                    })?
                };

                match &node.layer {
                    Layer::Linear(l) => {
                        let next = l.propagate_linear_with_engine(&linear_bounds, engine)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::ReLU(r) => {
                        if let Some(&relu_idx) = relu_name_to_idx.get(node_name) {
                            let alpha = alpha_state.get_alpha(relu_idx).ok_or_else(|| {
                                GammaError::InvalidSpec(format!(
                                    "Missing alpha for ReLU node {}",
                                    node_name
                                ))
                            })?;

                            let (new_bounds, grad) = r.propagate_linear_with_alpha(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                            )?;

                            gradients[relu_idx] = grad;
                            linear_bounds = new_bounds;
                        } else {
                            linear_bounds =
                                r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                        }
                    }
                    Layer::Transpose(t) => {
                        // Clone transpose and set input_shape for proper column permutation
                        let input_shape = pre_activation.shape().to_vec();
                        let mut transpose_with_shape = t.clone();
                        transpose_with_shape.set_input_shape(input_shape);
                        let next = transpose_with_shape.propagate_linear(&linear_bounds)?;
                        if let Cow::Owned(next) = next {
                            linear_bounds = next;
                        }
                    }
                    Layer::GELU(g) => {
                        linear_bounds =
                            g.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                    Layer::LayerNorm(ln) => {
                        linear_bounds =
                            ln.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                    _ => {
                        // Should not reach here due to earlier check
                        return self.propagate_crown_with_engine(input, engine);
                    }
                }
            }

            // Concretize to get actual bounds
            let concrete_bounds = linear_bounds.concretize(input);

            // Update element-wise best bounds using flat iteration to handle any array shape:
            // - best_lower[i] = max(best_lower[i], concrete_bounds.lower[i])
            // - best_upper[i] = min(best_upper[i], concrete_bounds.upper[i])
            if let (Some(best_l_slice), Some(curr_l_slice)) =
                (best_lower.as_slice_mut(), concrete_bounds.lower.as_slice())
            {
                for (best, &curr) in best_l_slice.iter_mut().zip(curr_l_slice.iter()) {
                    if curr > *best {
                        *best = curr;
                    }
                }
            }
            if let (Some(best_u_slice), Some(curr_u_slice)) =
                (best_upper.as_slice_mut(), concrete_bounds.upper.as_slice())
            {
                for (best, &curr) in best_u_slice.iter_mut().zip(curr_u_slice.iter()) {
                    if curr < *best {
                        *best = curr;
                    }
                }
            }

            let lower_sum: f32 = concrete_bounds
                .lower
                .as_slice()
                .map(|s| s.iter().sum())
                .unwrap_or(0.0);

            // Track best lower_sum for early stopping
            if lower_sum > best_lower_sum {
                best_lower_sum = lower_sum;
            }

            // Early stopping check (compare best improvement since last iteration).
            let best_improvement = best_lower_sum - prev_best_lower_sum;
            if best_improvement < config.tolerance {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
            }
            if iter > 0 && no_improve_iters >= 3 {
                debug!(
                    "GraphNetwork α-CROWN: Converged at iteration {} (best improvement < {} for {} iters)",
                    iter, config.tolerance, no_improve_iters
                );
                break;
            }

            // Compute gradient using configured method.
            //
            // NOTE: GraphNetwork α-CROWN historically ignored `config.gradient_method` and always
            // used the per-ReLU local gradients returned by `propagate_linear_with_alpha` (equivalent
            // to `GradientMethod::Analytic`). Honor the config so the default (`Spsa`) is actually
            // used for graph models.
            let eps = 1e-3;
            let num_relus = relu_nodes.len();
            let numerical_gradients: Vec<Array1<f32>> = match config.gradient_method {
                GradientMethod::Spsa => {
                    use rand::Rng;
                    let mut rng = rand::rng();

                    let mut avg_grads: Vec<Array1<f32>> = (0..num_relus)
                        .map(|relu_idx| Array1::zeros(alpha_state.alphas[relu_idx].len()))
                        .collect();

                    // Save original alpha values for restoration.
                    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();

                    // Average over multiple samples to reduce variance.
                    for _sample in 0..config.spsa_samples {
                        // Generate random Bernoulli perturbation (+1/-1) for each α.
                        let perturbations: Vec<Array1<f32>> = (0..num_relus)
                            .map(|relu_idx| {
                                let n = alpha_state.alphas[relu_idx].len();
                                Array1::from_iter((0..n).map(|i| {
                                    if alpha_state.unstable_mask[relu_idx][i] {
                                        if rng.random_bool(0.5) {
                                            1.0
                                        } else {
                                            -1.0
                                        }
                                    } else {
                                        0.0
                                    }
                                }))
                            })
                            .collect();

                        // Apply +ε perturbation from original.
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                alpha_state.alphas[relu_idx][i] = (original_alphas[relu_idx][i]
                                    + eps * perturbations[relu_idx][i])
                                    .clamp(0.0, 1.0);
                            }
                        }
                        let bounds_plus = self
                            .propagate_alpha_crown_single_pass_sequential_graph(
                                input,
                                &node_bounds,
                                &exec_order,
                                output_dim,
                                &relu_name_to_idx,
                                &alpha_state,
                                engine,
                            )
                            .unwrap_or_else(|_| concrete_bounds.clone());
                        let lower_plus: f32 = bounds_plus
                            .lower
                            .as_slice()
                            .map(|s| s.iter().sum())
                            .unwrap_or(0.0);

                        // Apply -ε perturbation from original.
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                alpha_state.alphas[relu_idx][i] = (original_alphas[relu_idx][i]
                                    - eps * perturbations[relu_idx][i])
                                    .clamp(0.0, 1.0);
                            }
                        }
                        let bounds_minus = self
                            .propagate_alpha_crown_single_pass_sequential_graph(
                                input,
                                &node_bounds,
                                &exec_order,
                                output_dim,
                                &relu_name_to_idx,
                                &alpha_state,
                                engine,
                            )
                            .unwrap_or_else(|_| concrete_bounds.clone());
                        let lower_minus: f32 = bounds_minus
                            .lower
                            .as_slice()
                            .map(|s| s.iter().sum())
                            .unwrap_or(0.0);

                        // SPSA gradient estimate.
                        let diff = lower_plus - lower_minus;
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                if alpha_state.unstable_mask[relu_idx][i]
                                    && perturbations[relu_idx][i].abs() > 0.5
                                {
                                    avg_grads[relu_idx][i] +=
                                        diff / (2.0 * eps * perturbations[relu_idx][i]);
                                }
                            }
                        }
                    }

                    // Restore original alpha values.
                    for (alpha, original) in
                        alpha_state.alphas.iter_mut().zip(original_alphas.iter())
                    {
                        alpha.assign(original);
                    }

                    // Average the gradients.
                    let num_samples = config.spsa_samples.max(1) as f32;
                    for grad in &mut avg_grads {
                        *grad /= num_samples;
                    }
                    avg_grads
                }
                GradientMethod::FiniteDifferences => {
                    let mut grads = Vec::with_capacity(num_relus);
                    for relu_idx in 0..num_relus {
                        let num_neurons = alpha_state.alphas[relu_idx].len();
                        let mut grad = Array1::<f32>::zeros(num_neurons);

                        for neuron_idx in 0..num_neurons {
                            if !alpha_state.unstable_mask[relu_idx][neuron_idx] {
                                continue;
                            }

                            let orig_alpha = alpha_state.alphas[relu_idx][neuron_idx];

                            // f(α + ε)
                            alpha_state.alphas[relu_idx][neuron_idx] =
                                (orig_alpha + eps).clamp(0.0, 1.0);
                            let bounds_plus = self
                                .propagate_alpha_crown_single_pass_sequential_graph(
                                    input,
                                    &node_bounds,
                                    &exec_order,
                                    output_dim,
                                    &relu_name_to_idx,
                                    &alpha_state,
                                    engine,
                                )
                                .unwrap_or_else(|_| concrete_bounds.clone());
                            let lower_plus: f32 = bounds_plus
                                .lower
                                .as_slice()
                                .map(|s| s.iter().sum())
                                .unwrap_or(0.0);

                            // f(α - ε)
                            alpha_state.alphas[relu_idx][neuron_idx] =
                                (orig_alpha - eps).clamp(0.0, 1.0);
                            let bounds_minus = self
                                .propagate_alpha_crown_single_pass_sequential_graph(
                                    input,
                                    &node_bounds,
                                    &exec_order,
                                    output_dim,
                                    &relu_name_to_idx,
                                    &alpha_state,
                                    engine,
                                )
                                .unwrap_or_else(|_| concrete_bounds.clone());
                            let lower_minus: f32 = bounds_minus
                                .lower
                                .as_slice()
                                .map(|s| s.iter().sum())
                                .unwrap_or(0.0);

                            // Restore original alpha.
                            alpha_state.alphas[relu_idx][neuron_idx] = orig_alpha;

                            grad[neuron_idx] = (lower_plus - lower_minus) / (2.0 * eps);
                        }

                        grads.push(grad);
                    }
                    grads
                }
                GradientMethod::Analytic => {
                    // Local gradients from CROWN backward pass
                    gradients.clone()
                }
                GradientMethod::AnalyticChain => {
                    // True chain-rule gradients using intermediate A matrices.
                    // Run backward pass that stores A matrices at each ReLU node.
                    let input_dim = input.len();
                    let mut scratch: Vec<Array1<f32>> =
                        gradients.iter().map(|g| Array1::zeros(g.len())).collect();

                    match self.dag_alpha_backward_pass_with_intermediates(
                        input,
                        &node_bounds,
                        &exec_order,
                        output_dim,
                        input_dim,
                        &relu_name_to_idx,
                        &alpha_state,
                        &mut scratch,
                        engine,
                    ) {
                        Ok((_bounds, intermediate)) => {
                            // Compute chain-rule gradients from stored A matrices
                            let relu_names: Vec<String> =
                                relu_nodes.iter().map(|(name, _)| name.clone()).collect();
                            self.compute_graph_chain_rule_gradients(
                                input,
                                &relu_names,
                                &intermediate,
                            )
                        }
                        Err(e) => {
                            // Fall back to local gradients if intermediate storage failed
                            if iter == 0 {
                                debug!(
                                    "GraphNetwork: AnalyticChain failed ({}), using local gradients",
                                    e
                                );
                            }
                            gradients.clone()
                        }
                    }
                }
            };

            // Update alpha values using numerical gradient (gradient ascent to maximize lower bound).
            let adam_params = config.adam_params(lr, iter + 1);
            for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                let neg_grad = grad.mapv(|v| -v);
                match config.optimizer {
                    Optimizer::Adam => {
                        alpha_state.update_adam(relu_idx, &neg_grad, &adam_params);
                    }
                    Optimizer::Sgd => {
                        let momentum = if config.use_momentum {
                            config.momentum
                        } else {
                            0.0
                        };
                        alpha_state.update(relu_idx, &neg_grad, lr, momentum);
                    }
                }
            }

            // Learning rate decay
            lr *= config.lr_decay;

            if iter % 5 == 0 {
                debug!(
                    "GraphNetwork α-CROWN iter {}: lower_sum = {:.6}, lr = {:.6}",
                    iter, lower_sum, lr
                );
            }

            prev_best_lower_sum = best_lower_sum;
        }

        // Return element-wise best bounds found across all iterations.
        // If no valid bounds were found, fall back to CROWN.
        let has_valid_bounds =
            best_lower.iter().all(|&v| v.is_finite()) && best_upper.iter().all(|&v| v.is_finite());

        if has_valid_bounds {
            Ok(BoundedTensor::new(best_lower, best_upper).unwrap_or_else(|_| input.clone()))
        } else {
            // Fall back to CROWN if no valid bounds were found
            self.propagate_crown_with_engine(input, engine)
        }
    }

    /// Single backward pass for sequential GraphNetwork α-CROWN, used for numerical gradients.
    #[allow(clippy::too_many_arguments)]
    fn propagate_alpha_crown_single_pass_sequential_graph(
        &self,
        input: &BoundedTensor,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        relu_name_to_idx: &std::collections::HashMap<String, usize>,
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let mut linear_bounds = LinearBounds::identity(output_dim);

        for node_name in exec_order.iter().rev() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        node.inputs[0]
                    ))
                })?
            };

            match &node.layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear_with_engine(&linear_bounds, engine)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    if let Some(&relu_idx) = relu_name_to_idx.get(node_name) {
                        if let Some(alpha) = alpha_state.get_alpha(relu_idx) {
                            let (new_bounds, _grad) = r.propagate_linear_with_alpha(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                            )?;
                            linear_bounds = new_bounds;
                            continue;
                        }
                    }
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::Transpose(t) => {
                    // Clone transpose and set input_shape for proper column permutation.
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let next = transpose_with_shape.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::GELU(g) => {
                    linear_bounds =
                        g.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                Layer::LayerNorm(ln) => {
                    linear_bounds =
                        ln.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
                _ => {
                    return Err(GammaError::InvalidSpec(format!(
                        "Unsupported op {} in sequential GraphNetwork α-CROWN single pass",
                        node.layer.layer_type()
                    )));
                }
            }
        }

        Ok(linear_bounds.concretize(input))
    }

    /// Check if the graph is essentially sequential (a linear chain).
    fn is_sequential_graph(&self, exec_order: &[String]) -> bool {
        if exec_order.is_empty() {
            return true;
        }

        // Count how many times each node is used as input (consumer count)
        let mut consumer_count: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        consumer_count.insert("_input".to_string(), 0);

        for name in exec_order {
            if let Some(node) = self.nodes.get(name) {
                // Check for binary ops
                if node.layer.is_binary() {
                    return false;
                }
                // Count inputs
                for input_name in &node.inputs {
                    *consumer_count.entry(input_name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Check that no node has more than one consumer (except _input which can have one)
        for (name, count) in &consumer_count {
            if name == "_input" {
                if *count > 1 {
                    return false; // Input used by multiple nodes
                }
            } else if *count > 1 {
                return false; // Intermediate node used by multiple nodes (branching)
            }
        }

        true
    }

    /// α-CROWN for DAG (non-sequential) graphs like ResNet with optional GEMM acceleration.
    ///
    /// This handles graphs with skip connections (Add operations) and multiple paths.
    /// The backward pass accumulates linear bounds from all consumers of each node.
    #[instrument(skip(self, input, config, engine), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    fn propagate_dag_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        // Step 1: Run CROWN-IBP forward to collect tighter bounds at each node
        // This is critical for DAGs (like ResNets) where IBP bounds explode through
        // skip connections. CROWN-IBP runs backward CROWN for each intermediate layer
        // and intersects with IBP bounds, producing significantly tighter bounds.
        let node_bounds = self.collect_crown_ibp_bounds_dag(input)?;

        // Get execution order
        let exec_order = self.topological_sort()?;

        // Determine output dimension
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();
        let input_dim = input.len();

        // Step 2: Identify ReLU nodes and their pre-activation bounds
        let relu_nodes: Vec<(String, usize)> = exec_order
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::ReLU(_)))
                    .unwrap_or(false)
            })
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        if relu_nodes.is_empty() {
            // No ReLU nodes, just use CROWN
            debug!("DAG α-CROWN: No ReLU nodes, falling back to CROWN");
            return self.propagate_crown_with_engine(input, engine);
        }

        // Get pre-activation bounds for each ReLU node
        let pre_activation_bounds: Vec<BoundedTensor> = relu_nodes
            .iter()
            .map(|(name, _)| {
                let node = self.nodes.get(name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!("ReLU node not found: {}", name))
                })?;
                if node.inputs.is_empty() || node.inputs[0] == "_input" {
                    Ok(input.clone())
                } else {
                    Ok(node_bounds
                        .get(&node.inputs[0])
                        .cloned()
                        .unwrap_or_else(|| input.clone()))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Initialize alpha state
        let mut alpha_state = AlphaState::from_preactivation_bounds(
            &pre_activation_bounds,
            &(0..relu_nodes.len()).collect::<Vec<_>>(),
        );

        let num_unstable = alpha_state.num_unstable();
        if num_unstable == 0 {
            debug!("DAG α-CROWN: No unstable neurons, using CROWN");
            return self.propagate_crown_with_engine(input, engine);
        }

        debug!(
            "DAG α-CROWN: Starting optimization with {} unstable neurons across {} ReLU nodes",
            num_unstable,
            relu_nodes.len()
        );

        // Adaptive skip: check if network is too deep for α-CROWN to help
        if config.adaptive_skip && relu_nodes.len() > config.adaptive_skip_depth_threshold {
            info!(
                "DAG α-CROWN: Adaptive skip triggered - {} ReLU nodes > threshold {}. \
                 For deep networks, bounds are often fundamentally loose and α-CROWN optimization \
                 provides no benefit. Falling back to CROWN.",
                relu_nodes.len(),
                config.adaptive_skip_depth_threshold
            );
            return self.propagate_crown_with_engine(input, engine);
        }

        if tracing::enabled!(tracing::Level::DEBUG) {
            fn bound_width_summary(bounds: &BoundedTensor) -> (f32, f32, f32, usize) {
                let mut min_w = f32::INFINITY;
                let mut max_w = 0.0f32;
                let mut sum_w = 0.0f32;
                let mut count = 0usize;
                let mut invalid = 0usize;

                for (&l, &u) in bounds.lower.iter().zip(bounds.upper.iter()) {
                    let w = u - l;
                    if !w.is_finite() {
                        invalid += 1;
                        continue;
                    }
                    min_w = min_w.min(w);
                    max_w = max_w.max(w);
                    sum_w += w;
                    count += 1;
                }

                let mean_w = if count > 0 {
                    sum_w / (count as f32)
                } else {
                    f32::NAN
                };
                (mean_w, max_w, min_w, invalid)
            }

            fn unstable_summary(bounds: &BoundedTensor) -> (usize, usize, f32, f32) {
                let mut unstable = 0usize;
                let mut total = 0usize;
                let mut max_w = 0.0f32;
                let mut mean_w_sum = 0.0f32;
                let mut mean_w_count = 0usize;

                for (&l, &u) in bounds.lower.iter().zip(bounds.upper.iter()) {
                    total += 1;
                    if l < 0.0 && u > 0.0 {
                        unstable += 1;
                    }
                    let w = u - l;
                    if w.is_finite() {
                        max_w = max_w.max(w);
                        mean_w_sum += w;
                        mean_w_count += 1;
                    }
                }

                let mean_w = if mean_w_count > 0 {
                    mean_w_sum / (mean_w_count as f32)
                } else {
                    f32::NAN
                };
                (unstable, total, mean_w, max_w)
            }

            let mut per_node: Vec<(f32, f32, String, String, usize)> = Vec::new();
            for name in &exec_order {
                if let Some(bounds) = node_bounds.get(name) {
                    let (mean_w, max_w, _min_w, invalid) = bound_width_summary(bounds);
                    let layer_type = self
                        .nodes
                        .get(name)
                        .map(|n| n.layer.layer_type().to_string())
                        .unwrap_or_else(|| "<missing>".to_string());
                    per_node.push((max_w, mean_w, name.clone(), layer_type, invalid));
                }
            }
            per_node.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            if let (Some((max_w, _, max_name, max_ty, _)), Some((_, mean_w, _, _, _))) = (
                per_node.first(),
                per_node.get(per_node.len().saturating_sub(1)),
            ) {
                debug!(
                    "DAG α-CROWN: IBP width stats across {} nodes: widest={} ({} max_w={:.3e}), narrowest_mean_w≈{:.3e}",
                    per_node.len(),
                    max_name,
                    max_ty,
                    max_w,
                    mean_w
                );
            }

            for (rank, (max_w, mean_w, name, ty, invalid)) in per_node.iter().take(20).enumerate() {
                debug!(
                    "DAG α-CROWN: widest#{} node='{}' type={} mean_w={:.3e} max_w={:.3e} invalid={} ",
                    rank,
                    name,
                    ty,
                    mean_w,
                    max_w,
                    invalid
                );
            }

            for (rank, (name, _)) in relu_nodes.iter().take(20).enumerate() {
                if let Some(node) = self.nodes.get(name) {
                    let pre = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[0]).unwrap_or(input)
                    };
                    let (unstable, total, mean_w, max_w) = unstable_summary(pre);
                    let ratio = if total > 0 {
                        (unstable as f32) / (total as f32)
                    } else {
                        0.0
                    };
                    debug!(
                        "DAG α-CROWN: ReLU preact#{} node='{}' unstable={}/{} ({:.1}%) mean_w={:.3e} max_w={:.3e}",
                        rank,
                        name,
                        unstable,
                        total,
                        ratio * 100.0,
                        mean_w,
                        max_w
                    );
                }
            }
        }

        // Map from node name to ReLU index
        let relu_name_to_idx: std::collections::HashMap<String, usize> = relu_nodes
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();

        // Step 3: Optimization loop
        // Track element-wise best bounds across iterations:
        // - best_lower: maximum lower bound seen for each output dimension
        // - best_upper: minimum upper bound seen for each output dimension
        // Initialize from CROWN bounds to ensure α-CROWN never returns worse bounds.
        let crown_bounds = self.propagate_crown_with_engine(input, engine)?;
        let mut best_lower: ArrayD<f32> = crown_bounds.lower.clone();
        let mut best_upper: ArrayD<f32> = crown_bounds.upper.clone();
        let mut best_lower_sum: f32 = crown_bounds
            .lower
            .as_slice()
            .map(|s| s.iter().sum())
            .unwrap_or(f32::NEG_INFINITY);
        let mut prev_best_lower_sum = best_lower_sum;
        let mut no_improve_iters = 0usize;
        let mut lr = config.learning_rate;

        for iter in 0..config.iterations {
            // Initialize gradients for each ReLU node
            let mut gradients: Vec<Array1<f32>> = relu_nodes
                .iter()
                .map(|(name, _)| {
                    let node = self
                        .nodes
                        .get(name)
                        .expect("ReLU node should exist in graph");
                    let pre_act = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[0]).unwrap_or(input)
                    };
                    Array1::zeros(pre_act.len())
                })
                .collect();

            // Run backward pass through DAG with alpha values
            let concrete_bounds = self.dag_alpha_backward_pass_with_engine(
                input,
                &node_bounds,
                &exec_order,
                output_dim,
                input_dim,
                &relu_name_to_idx,
                &alpha_state,
                &mut gradients,
                engine,
            )?;

            // Compute loss (negative sum of lower bounds - we want to maximize)
            // Update element-wise best bounds using flat iteration to handle any array shape:
            // - best_lower[i] = max(best_lower[i], concrete_bounds.lower[i])
            // - best_upper[i] = min(best_upper[i], concrete_bounds.upper[i])
            // This ensures we never return worse bounds than any individual iteration.
            if let (Some(best_l_slice), Some(curr_l_slice)) =
                (best_lower.as_slice_mut(), concrete_bounds.lower.as_slice())
            {
                for (best, &curr) in best_l_slice.iter_mut().zip(curr_l_slice.iter()) {
                    if curr > *best {
                        *best = curr;
                    }
                }
            }
            if let (Some(best_u_slice), Some(curr_u_slice)) =
                (best_upper.as_slice_mut(), concrete_bounds.upper.as_slice())
            {
                for (best, &curr) in best_u_slice.iter_mut().zip(curr_u_slice.iter()) {
                    if curr < *best {
                        *best = curr;
                    }
                }
            }

            let lower_sum: f32 = concrete_bounds
                .lower
                .as_slice()
                .map(|s| s.iter().sum())
                .unwrap_or(0.0);

            // Track best lower_sum for early stopping
            if lower_sum > best_lower_sum {
                best_lower_sum = lower_sum;
            }

            // Early stopping check (compare best improvement since last iteration).
            let best_improvement = best_lower_sum - prev_best_lower_sum;
            if best_improvement < config.tolerance {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
            }
            if iter > 0 && no_improve_iters >= 3 {
                debug!(
                    "DAG α-CROWN: Converged at iteration {} (best improvement < {} for {} iters)",
                    iter, config.tolerance, no_improve_iters
                );
                break;
            }

            // Pilot iteration check: after first iteration, verify α-CROWN helps
            // This catches cases where network depth isn't the only factor
            if iter == 0 && config.adaptive_skip && config.adaptive_skip_pilot {
                // Compute improvement over initial CROWN bounds
                let initial_lower_sum: f32 = crown_bounds
                    .lower
                    .as_slice()
                    .map(|s| s.iter().sum())
                    .unwrap_or(f32::NEG_INFINITY);
                let pilot_improvement = best_lower_sum - initial_lower_sum;

                if pilot_improvement < config.pilot_improvement_threshold {
                    info!(
                        "DAG α-CROWN: Pilot iteration improvement ({:.3e}) < threshold ({:.3e}). \
                         α-CROWN optimization is not helping, skipping remaining iterations.",
                        pilot_improvement, config.pilot_improvement_threshold
                    );
                    // Return best bounds found so far (CROWN or pilot iteration bounds)
                    return Ok(BoundedTensor::new(best_lower.clone(), best_upper.clone())
                        .unwrap_or_else(|_| crown_bounds.clone()));
                } else {
                    debug!(
                        "DAG α-CROWN: Pilot iteration improvement ({:.3e}) >= threshold ({:.3e}). \
                         Continuing optimization.",
                        pilot_improvement, config.pilot_improvement_threshold
                    );
                }
            }

            // Compute gradient using configured method.
            //
            // NOTE: DAG α-CROWN historically ignored `config.gradient_method` and always used the
            // per-ReLU local gradients returned by `propagate_linear_with_alpha`. Honor the config
            // so the default (`Spsa`) is actually used on ResNet-like graphs with skip connections.
            let eps = 1e-3;
            let num_relus = relu_nodes.len();
            let numerical_gradients: Vec<Array1<f32>> = match config.gradient_method {
                GradientMethod::Spsa => {
                    use rand::Rng;
                    let mut rng = rand::rng();

                    let mut avg_grads: Vec<Array1<f32>> = (0..num_relus)
                        .map(|relu_idx| Array1::zeros(alpha_state.alphas[relu_idx].len()))
                        .collect();

                    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();

                    // Scratch gradient buffers (required by DAG pass signature).
                    let mut scratch: Vec<Array1<f32>> =
                        gradients.iter().map(|g| Array1::zeros(g.len())).collect();

                    for _sample in 0..config.spsa_samples {
                        let perturbations: Vec<Array1<f32>> = (0..num_relus)
                            .map(|relu_idx| {
                                let n = alpha_state.alphas[relu_idx].len();
                                Array1::from_iter((0..n).map(|i| {
                                    if alpha_state.unstable_mask[relu_idx][i] {
                                        if rng.random_bool(0.5) {
                                            1.0
                                        } else {
                                            -1.0
                                        }
                                    } else {
                                        0.0
                                    }
                                }))
                            })
                            .collect();

                        // +eps
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                alpha_state.alphas[relu_idx][i] = (original_alphas[relu_idx][i]
                                    + eps * perturbations[relu_idx][i])
                                    .clamp(0.0, 1.0);
                            }
                        }
                        scratch.iter_mut().for_each(|g| g.fill(0.0));
                        let bounds_plus = self
                            .dag_alpha_backward_pass_with_engine(
                                input,
                                &node_bounds,
                                &exec_order,
                                output_dim,
                                input_dim,
                                &relu_name_to_idx,
                                &alpha_state,
                                &mut scratch,
                                engine,
                            )
                            .unwrap_or_else(|_| concrete_bounds.clone());
                        let lower_plus: f32 = bounds_plus
                            .lower
                            .as_slice()
                            .map(|s| s.iter().sum())
                            .unwrap_or(0.0);

                        // -eps
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                alpha_state.alphas[relu_idx][i] = (original_alphas[relu_idx][i]
                                    - eps * perturbations[relu_idx][i])
                                    .clamp(0.0, 1.0);
                            }
                        }
                        scratch.iter_mut().for_each(|g| g.fill(0.0));
                        let bounds_minus = self
                            .dag_alpha_backward_pass_with_engine(
                                input,
                                &node_bounds,
                                &exec_order,
                                output_dim,
                                input_dim,
                                &relu_name_to_idx,
                                &alpha_state,
                                &mut scratch,
                                engine,
                            )
                            .unwrap_or_else(|_| concrete_bounds.clone());
                        let lower_minus: f32 = bounds_minus
                            .lower
                            .as_slice()
                            .map(|s| s.iter().sum())
                            .unwrap_or(0.0);

                        let diff = lower_plus - lower_minus;
                        for relu_idx in 0..num_relus {
                            for i in 0..alpha_state.alphas[relu_idx].len() {
                                if alpha_state.unstable_mask[relu_idx][i]
                                    && perturbations[relu_idx][i].abs() > 0.5
                                {
                                    avg_grads[relu_idx][i] +=
                                        diff / (2.0 * eps * perturbations[relu_idx][i]);
                                }
                            }
                        }
                    }

                    for (alpha, original) in
                        alpha_state.alphas.iter_mut().zip(original_alphas.iter())
                    {
                        alpha.assign(original);
                    }

                    let num_samples = config.spsa_samples.max(1) as f32;
                    for grad in &mut avg_grads {
                        *grad /= num_samples;
                    }

                    avg_grads
                }
                GradientMethod::FiniteDifferences => {
                    let mut grads = Vec::with_capacity(num_relus);

                    let mut scratch: Vec<Array1<f32>> =
                        gradients.iter().map(|g| Array1::zeros(g.len())).collect();

                    for relu_idx in 0..num_relus {
                        let num_neurons = alpha_state.alphas[relu_idx].len();
                        let mut grad = Array1::<f32>::zeros(num_neurons);

                        for neuron_idx in 0..num_neurons {
                            if !alpha_state.unstable_mask[relu_idx][neuron_idx] {
                                continue;
                            }

                            let orig_alpha = alpha_state.alphas[relu_idx][neuron_idx];

                            alpha_state.alphas[relu_idx][neuron_idx] =
                                (orig_alpha + eps).clamp(0.0, 1.0);
                            scratch.iter_mut().for_each(|g| g.fill(0.0));
                            let bounds_plus = self
                                .dag_alpha_backward_pass_with_engine(
                                    input,
                                    &node_bounds,
                                    &exec_order,
                                    output_dim,
                                    input_dim,
                                    &relu_name_to_idx,
                                    &alpha_state,
                                    &mut scratch,
                                    engine,
                                )
                                .unwrap_or_else(|_| concrete_bounds.clone());
                            let lower_plus: f32 = bounds_plus
                                .lower
                                .as_slice()
                                .map(|s| s.iter().sum())
                                .unwrap_or(0.0);

                            alpha_state.alphas[relu_idx][neuron_idx] =
                                (orig_alpha - eps).clamp(0.0, 1.0);
                            scratch.iter_mut().for_each(|g| g.fill(0.0));
                            let bounds_minus = self
                                .dag_alpha_backward_pass_with_engine(
                                    input,
                                    &node_bounds,
                                    &exec_order,
                                    output_dim,
                                    input_dim,
                                    &relu_name_to_idx,
                                    &alpha_state,
                                    &mut scratch,
                                    engine,
                                )
                                .unwrap_or_else(|_| concrete_bounds.clone());
                            let lower_minus: f32 = bounds_minus
                                .lower
                                .as_slice()
                                .map(|s| s.iter().sum())
                                .unwrap_or(0.0);

                            alpha_state.alphas[relu_idx][neuron_idx] = orig_alpha;
                            grad[neuron_idx] = (lower_plus - lower_minus) / (2.0 * eps);
                        }

                        grads.push(grad);
                    }

                    grads
                }
                GradientMethod::Analytic => {
                    // Local gradients from CROWN backward pass
                    gradients.clone()
                }
                GradientMethod::AnalyticChain => {
                    // True chain-rule gradients using intermediate A matrices.
                    // Run backward pass that stores A matrices at each ReLU node.
                    let mut scratch: Vec<Array1<f32>> =
                        gradients.iter().map(|g| Array1::zeros(g.len())).collect();

                    match self.dag_alpha_backward_pass_with_intermediates(
                        input,
                        &node_bounds,
                        &exec_order,
                        output_dim,
                        input_dim,
                        &relu_name_to_idx,
                        &alpha_state,
                        &mut scratch,
                        engine,
                    ) {
                        Ok((_bounds, intermediate)) => {
                            // Compute chain-rule gradients from stored A matrices
                            let relu_names: Vec<String> =
                                relu_nodes.iter().map(|(name, _)| name.clone()).collect();
                            self.compute_graph_chain_rule_gradients(
                                input,
                                &relu_names,
                                &intermediate,
                            )
                        }
                        Err(e) => {
                            // Fall back to local gradients if intermediate storage failed
                            if iter == 0 {
                                debug!(
                                    "DAG α-CROWN: AnalyticChain failed ({}), using local gradients",
                                    e
                                );
                            }
                            gradients.clone()
                        }
                    }
                }
            };

            // Update alpha values using numerical gradient (gradient ascent to maximize lower bound).
            let adam_params = config.adam_params(lr, iter + 1);
            for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                let neg_grad = grad.mapv(|v| -v);
                match config.optimizer {
                    Optimizer::Adam => {
                        alpha_state.update_adam(relu_idx, &neg_grad, &adam_params);
                    }
                    Optimizer::Sgd => {
                        let momentum = if config.use_momentum {
                            config.momentum
                        } else {
                            0.0
                        };
                        alpha_state.update(relu_idx, &neg_grad, lr, momentum);
                    }
                }
            }

            // Learning rate decay
            lr *= config.lr_decay;

            if iter % 5 == 0 {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    let mut alpha_min = f32::INFINITY;
                    let mut alpha_max = f32::NEG_INFINITY;
                    let mut alpha_sum = 0.0f32;
                    let mut alpha_count = 0usize;
                    let mut vel_abs_sum = 0.0f32;
                    let mut vel_abs_max = 0.0f32;
                    let mut vel_count = 0usize;

                    for (relu_idx, alpha) in alpha_state.alphas.iter().enumerate() {
                        let mask = &alpha_state.unstable_mask[relu_idx];
                        let vel = &alpha_state.velocity[relu_idx];
                        for i in 0..alpha.len() {
                            if mask[i] {
                                let a = alpha[i];
                                if a.is_finite() {
                                    alpha_min = alpha_min.min(a);
                                    alpha_max = alpha_max.max(a);
                                    alpha_sum += a;
                                    alpha_count += 1;
                                }
                                let v = vel[i].abs();
                                if v.is_finite() {
                                    vel_abs_sum += v;
                                    vel_abs_max = vel_abs_max.max(v);
                                    vel_count += 1;
                                }
                            }
                        }
                    }

                    let alpha_mean = if alpha_count > 0 {
                        alpha_sum / (alpha_count as f32)
                    } else {
                        f32::NAN
                    };
                    let vel_abs_mean = if vel_count > 0 {
                        vel_abs_sum / (vel_count as f32)
                    } else {
                        f32::NAN
                    };

                    debug!(
                        "DAG α-CROWN iter {}: best_impr={:.3e} alpha_unstable_mean={:.3e} [{:.3e},{:.3e}] vel_abs_mean={:.3e} vel_abs_max={:.3e}",
                        iter,
                        best_lower_sum - prev_best_lower_sum,
                        alpha_mean,
                        alpha_min,
                        alpha_max,
                        vel_abs_mean,
                        vel_abs_max
                    );
                }
                debug!(
                    "DAG α-CROWN iter {}: lower_sum = {:.6}, lr = {:.6}",
                    iter, lower_sum, lr
                );
            }

            prev_best_lower_sum = best_lower_sum;
        }

        // Return element-wise best bounds found across all iterations.
        // If no valid bounds were found, fall back to CROWN.
        let has_valid_bounds =
            best_lower.iter().all(|&v| v.is_finite()) && best_upper.iter().all(|&v| v.is_finite());

        if has_valid_bounds {
            Ok(BoundedTensor::new(best_lower, best_upper).unwrap_or_else(|_| input.clone()))
        } else {
            // Fall back to CROWN if no valid bounds were found
            self.propagate_crown_with_engine(input, engine)
        }
    }

    /// Helper method to run a single DAG backward pass with alpha values and optional engine.
    /// Returns concrete bounds and populates gradients for alpha optimization.
    #[allow(clippy::too_many_arguments)]
    fn dag_alpha_backward_pass_with_engine(
        &self,
        input: &BoundedTensor,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &std::collections::HashMap<String, usize>,
        alpha_state: &AlphaState,
        gradients: &mut [Array1<f32>],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        // Initialize linear bounds per node (tracking accumulated bounds from consumers)
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();

        // Determine output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        // Output node starts with identity bounds
        node_linear_bounds.insert(output_node_name.clone(), LinearBounds::identity(output_dim));

        // Track if we've accumulated bounds to the input
        let mut input_accumulated = false;

        // Backward pass through nodes in reverse order
        for node_name in exec_order.iter().rev() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get this node's accumulated linear bounds
            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    continue;
                }
            };

            // Get pre-activation bounds for this node
            let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        node.inputs[0]
                    ))
                })?
            };

            // Handle different layer types
            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l
                        .propagate_linear_with_engine(&node_lb, engine)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (Linear): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    // Use alpha parameters for ReLU if available
                    if let Some(&relu_idx) = relu_name_to_idx.get(node_name) {
                        if let Some(alpha) = alpha_state.get_alpha(relu_idx) {
                            let (new_lb, grad) =
                                r.propagate_linear_with_alpha(&node_lb, pre_activation, alpha)?;

                            gradients[relu_idx] = grad;

                            self.accumulate_bounds_to_input(
                                &node.inputs[0],
                                new_lb,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                            continue;
                        }
                    }

                    // Fallback: propagate without alpha
                    let new_lb = r
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    // Add propagates the same bounds to both inputs
                    let (lb_a, lb_b) = add.propagate_linear_binary(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "DAG α-CROWN failed at node '{}' (Add): {}",
                            node_name, e
                        ))
                    })?;

                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv2d(c) => {
                    // Conv2d CROWN backward
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        // Cannot do CROWN on Conv2d without proper shape
                        debug!("DAG α-CROWN: Conv2d input shape too small, falling back to CROWN");
                        return self.propagate_crown_with_engine(input, engine);
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "DAG α-CROWN failed at node '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::BatchNorm(bn) => {
                    let new_lb = bn
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (BatchNorm): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::MaxPool2d(mp) => {
                    let new_lb = mp
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (MaxPool2d): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::AveragePool(ap) => {
                    let new_lb = ap
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (AveragePool): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Transpose(t) => {
                    // Clone transpose and set input_shape for proper column permutation
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let new_lb = transpose_with_shape
                        .propagate_linear(&node_lb)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (Transpose): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "DAG α-CROWN failed at node '{}' (Flatten): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::GELU(g) => {
                    let new_lb = g
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (GELU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::LayerNorm(ln) => {
                    let new_lb = ln
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (LayerNorm): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Softmax(s) => {
                    let new_lb = s
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (Softmax): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::MatMul(matmul) => {
                    // MatMul CROWN backward (both inputs)
                    let input_a_bounds = if node.inputs[0] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "Bounds for {} not found",
                                node.inputs[0]
                            ))
                        })?
                    };
                    let input_b_bounds = if node.inputs[1] == "_input" {
                        input
                    } else {
                        node_bounds.get(&node.inputs[1]).ok_or_else(|| {
                            GammaError::InvalidSpec(format!(
                                "Bounds for {} not found",
                                node.inputs[1]
                            ))
                        })?
                    };

                    let (lb_a, lb_b) = matmul
                        .propagate_linear_binary(&node_lb, input_a_bounds, input_b_bounds)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (MatMul): {}",
                                node_name, e
                            ))
                        })?;

                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Concat(concat) => {
                    // Concat: SPLIT linear bounds across N inputs based on their shapes.
                    // Check constant inputs first (CLS token, etc.), then stored shapes, then node_bounds.
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
                                input.shape().to_vec()
                            } else if let Some(shape) = concat.get_input_shape(i) {
                                shape.clone()
                            } else {
                                node_bounds
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
                                "DAG α-CROWN failed at node '{}' (Concat): {}",
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
                        self.accumulate_bounds_to_input(
                            inp_name,
                            lb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                    }
                }
                _ => {
                    // Unsupported layer, fall back to CROWN
                    debug!(
                        "DAG α-CROWN: Unsupported layer {} ({}), falling back to CROWN",
                        node_name,
                        node.layer.layer_type()
                    );
                    return self.propagate_crown(input);
                }
            }
        }

        // Concretize final bounds
        if let Some(input_lb) = node_linear_bounds.get("_input") {
            Ok(input_lb.concretize(input))
        } else if input_accumulated {
            // Input bounds were accumulated somewhere
            Err(GammaError::InvalidSpec(
                "DAG α-CROWN: Input bounds accumulated but not found".to_string(),
            ))
        } else {
            // No path from output to input
            Err(GammaError::InvalidSpec(
                "DAG α-CROWN: No path from output to input".to_string(),
            ))
        }
    }

    /// DAG backward pass that stores intermediate A matrices for chain-rule gradient computation.
    ///
    /// This is similar to `dag_alpha_backward_pass_with_engine` but also captures the A matrix
    /// at each ReLU node BEFORE the ReLU is applied, enabling true chain-rule gradients.
    #[allow(clippy::too_many_arguments)]
    fn dag_alpha_backward_pass_with_intermediates(
        &self,
        input: &BoundedTensor,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &std::collections::HashMap<String, usize>,
        alpha_state: &AlphaState,
        gradients: &mut [Array1<f32>],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, GraphAlphaCrownIntermediate)> {
        let mut intermediate = GraphAlphaCrownIntermediate::new();

        // Initialize linear bounds per node (tracking accumulated bounds from consumers)
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();

        // Determine output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        // Output node starts with identity bounds
        node_linear_bounds.insert(output_node_name.clone(), LinearBounds::identity(output_dim));

        // Track if we've accumulated bounds to the input
        let mut input_accumulated = false;

        // Backward pass through nodes in reverse order
        for node_name in exec_order.iter().rev() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get this node's accumulated linear bounds
            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    continue;
                }
            };

            // Get pre-activation bounds for this node
            let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                input
            } else {
                node_bounds.get(&node.inputs[0]).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        node.inputs[0]
                    ))
                })?
            };

            // Handle different layer types
            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l
                        .propagate_linear_with_engine(&node_lb, engine)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (Linear): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    // Store A matrix BEFORE this ReLU is applied (for chain-rule gradients)
                    intermediate
                        .a_at_relu
                        .insert(node_name.clone(), node_lb.lower_a.clone());

                    // Store pre-ReLU bounds
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

                    // Use alpha parameters for ReLU if available
                    if let Some(&relu_idx) = relu_name_to_idx.get(node_name) {
                        if let Some(alpha) = alpha_state.get_alpha(relu_idx) {
                            let (new_lb, grad) =
                                r.propagate_linear_with_alpha(&node_lb, pre_activation, alpha)?;

                            gradients[relu_idx] = grad;

                            self.accumulate_bounds_to_input(
                                &node.inputs[0],
                                new_lb,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                            continue;
                        }
                    }

                    // Fallback: propagate without alpha
                    let new_lb = r
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    // Add propagates the same bounds to both inputs
                    let (lb_a, lb_b) = add.propagate_linear_binary(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "DAG α-CROWN failed at node '{}' (Add): {}",
                            node_name, e
                        ))
                    })?;

                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    self.accumulate_bounds_to_input(
                        &node.inputs[1],
                        lb_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv2d(c) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        // Cannot do CROWN on Conv2d without proper shape
                        let bounds = self.propagate_crown_with_engine(input, engine)?;
                        return Ok((bounds, intermediate));
                    };
                    let mut conv_with_shape = c.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "DAG α-CROWN failed at node '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::BatchNorm(bn) => {
                    let new_lb = bn
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (BatchNorm): {}",
                                node_name, e
                            ))
                        })?;
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Transpose(t) => {
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let new_lb = transpose_with_shape
                        .propagate_linear(&node_lb)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "DAG α-CROWN failed at node '{}' (Transpose): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "DAG α-CROWN failed at node '{}' (Flatten): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        Cow::Borrowed(_) => node_lb,
                        Cow::Owned(lb) => lb,
                    };
                    self.accumulate_bounds_to_input(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
                _ => {
                    // Unsupported layer, return what we have so far
                    let bounds = self.propagate_crown_with_engine(input, engine)?;
                    return Ok((bounds, intermediate));
                }
            }
        }

        // Store final bounds and concretize
        if let Some(input_lb) = node_linear_bounds.get("_input") {
            intermediate.final_bounds = input_lb.clone();
            Ok((input_lb.concretize(input), intermediate))
        } else if input_accumulated {
            Err(GammaError::InvalidSpec(
                "DAG α-CROWN: Input bounds accumulated but not found".to_string(),
            ))
        } else {
            Err(GammaError::InvalidSpec(
                "DAG α-CROWN: No path from output to input".to_string(),
            ))
        }
    }

    /// Compute chain-rule gradients for GraphNetwork DAG α-CROWN.
    ///
    /// For each unstable neuron i in ReLU node k:
    /// ∂(output_lower_sum)/∂α_k[i] = Σ_j A_to_relu[j,i] × input_contribution[i]
    ///
    /// Where:
    /// - A_to_relu[j,i] is the coefficient from output j to neuron i (before ReLU k)
    /// - input_contribution captures how the neuron value affects downstream computation
    ///
    /// This properly chains gradients through all downstream layers in the DAG.
    fn compute_graph_chain_rule_gradients(
        &self,
        _input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
    ) -> Vec<Array1<f32>> {
        let mut gradients: Vec<Array1<f32>> = Vec::with_capacity(relu_nodes.len());

        for relu_name in relu_nodes {
            // Get A matrix at this ReLU (before ReLU applied)
            let a_at_relu = match intermediate.get_a_at_relu(relu_name) {
                Some(a) => a,
                None => {
                    // No intermediate stored for this ReLU, return zeros
                    gradients.push(Array1::zeros(1));
                    continue;
                }
            };

            // Get pre-ReLU bounds
            let (pre_lower, pre_upper) = match intermediate.get_pre_relu_bounds(relu_name) {
                Some(b) => b,
                None => {
                    gradients.push(Array1::zeros(a_at_relu.ncols()));
                    continue;
                }
            };

            let n_neurons = pre_lower.len();
            let num_outputs = a_at_relu.nrows();
            let mut grad = Array1::<f32>::zeros(n_neurons);

            // For each neuron in this ReLU layer
            for i in 0..n_neurons {
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Only unstable neurons (l < 0 < u) have non-zero gradient
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Compute gradient contribution from all output dimensions
                // For lower relaxation y >= α*x with x ∈ [l, u] where l < 0 < u:
                // - Contribution to lower bound = A[j,i] * α * min(x) = A[j,i] * α * l
                // - Gradient ∂bound/∂α = A[j,i] * l
                // Note: l < 0 for unstable neurons, so gradient is typically negative
                // when A[j,i] > 0, meaning increasing α decreases the lower bound.
                let mut grad_i = 0.0f32;

                for j in 0..num_outputs {
                    let a_ji = a_at_relu[[j, i]];

                    // When A >= 0, lower relaxation uses y >= α*x
                    // The binding point is x = l (lower bound), not u
                    // because we minimize α*x over [l,u] with α >= 0 and l < 0
                    if a_ji > 0.0 {
                        // Lower relaxation active: y >= α*x
                        // Contribution to lower bound: A[j,i] * α * l
                        // Gradient w.r.t. α: A[j,i] * l
                        grad_i += a_ji * l;
                    }
                    // When A < 0, upper relaxation y <= (u/(u-l))*(x-l) is used
                    // This doesn't depend on α, so gradient is 0
                }

                grad[i] = grad_i;
            }

            gradients.push(grad);
        }

        gradients
    }

    /// Try to convert this graph network to a sequential Network for SDP-CROWN.
    ///
    /// SDP-CROWN requires a sequential network of Linear/ReLU layers. This method
    /// checks if the graph can be converted and returns `Some(Network)` if so.
    ///
    /// Returns `None` if:
    /// - The graph is not sequential (has branches or binary operations)
    /// - The graph contains layers other than Linear or ReLU
    pub fn try_to_sequential_network(&self) -> Option<Network> {
        let exec_order = self.topological_sort().ok()?;

        // Check if graph is sequential
        if !self.is_sequential_graph(&exec_order) {
            return None;
        }

        // Try to extract layers in order
        let mut network = Network::new();

        for node_name in &exec_order {
            let node = self.nodes.get(node_name)?;

            // Only allow Linear and ReLU layers for SDP-CROWN
            match &node.layer {
                Layer::Linear(l) => {
                    network.add_layer(Layer::Linear(l.clone()));
                }
                Layer::ReLU(r) => {
                    network.add_layer(Layer::ReLU(r.clone()));
                }
                // All other layers are not supported for SDP-CROWN
                _ => return None,
            }
        }

        Some(network)
    }

    /// Collect IBP bounds at each node in the graph.
    pub(crate) fn collect_node_bounds(
        &self,
        input: &BoundedTensor,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        let exec_order = self.topological_sort()?;
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        for node_name in &exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let output_bounds = match &node.layer {
                Layer::Where(w) => {
                    if w.has_embedded_constants() {
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        w.propagate_ibp_with_condition(cond)?
                    } else {
                        if node.inputs.len() < 3 {
                            return Err(GammaError::InvalidSpec(format!(
                                "Where node {} requires 3 inputs",
                                node_name
                            )));
                        }
                        let cond = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                        let x = self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                        let y = self.get_bounds_ref(&node.inputs[2], input, &bounds_cache)?;
                        w.propagate_ibp_ternary(cond, x, y)?
                    }
                }
                _ if matches!(&node.layer, Layer::Concat(_)) => {
                    // Concat: N-ary operation (2+ inputs)
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Concat node {} requires at least 2 inputs, got {}",
                            node_name,
                            node.inputs.len()
                        )));
                    }

                    if let Layer::Concat(concat) = &node.layer {
                        let input_bounds: Vec<&BoundedTensor> = node
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(i, inp_name)| {
                                concat.get_constant_input(i).map(Ok).unwrap_or_else(|| {
                                    self.get_bounds_ref(inp_name, input, &bounds_cache)
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        concat.propagate_ibp_nary(&input_bounds)?
                    } else {
                        unreachable!()
                    }
                }
                _ if node.layer.is_binary() => {
                    if node.inputs.len() < 2 {
                        return Err(GammaError::InvalidSpec(format!(
                            "Binary node {} requires 2 inputs",
                            node_name
                        )));
                    }
                    match &node.layer {
                        Layer::MatMul(matmul) if matmul.transpose_b => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            if let Some(tighter) = self.try_attention_matmul_bounds_zonotope(
                                node,
                                input,
                                &bounds_cache,
                            )? {
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(input_a, input_b)?
                            }
                        }
                        Layer::MulBinary(_) => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            // Try zonotope tightening for SwiGLU pattern (up * silu(gate))
                            if let Some(tighter) =
                                self.try_ffn_swiglu_bounds_zonotope(node, input, &bounds_cache)?
                            {
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(input_a, input_b)?
                            }
                        }
                        _ => {
                            let input_a =
                                self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                            let input_b =
                                self.get_bounds_ref(&node.inputs[1], input, &bounds_cache)?;
                            node.layer.propagate_ibp_binary(input_a, input_b)?
                        }
                    }
                }
                _ => {
                    if node.inputs.is_empty() {
                        return Err(GammaError::InvalidSpec(format!(
                            "Node {} has no inputs",
                            node_name
                        )));
                    }
                    let node_input = self.get_bounds_ref(&node.inputs[0], input, &bounds_cache)?;
                    node.layer.propagate_ibp(node_input)?
                }
            };

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        Ok(bounds_cache)
    }

    /// Collect CROWN-IBP bounds at each node in the graph.
    ///
    /// This computes tighter bounds than pure IBP by running backward CROWN from
    /// each node to the network input, then intersecting with IBP bounds.
    ///
    /// Algorithm:
    /// 1. Run IBP forward to get initial bounds at all nodes
    /// 2. For each intermediate node (in topological order):
    ///    a. Run backward CROWN from that node to input
    ///    b. Intersect CROWN bounds with IBP bounds
    ///    c. Store tightened bounds for use in subsequent CROWN passes
    ///
    /// This is O(N^2) where N is the number of nodes, but produces significantly
    /// tighter bounds that improve ReLU relaxation quality.
    pub fn collect_crown_ibp_bounds_dag(
        &self,
        input: &BoundedTensor,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        let exec_order = self.topological_sort()?;

        // Step 1: Collect IBP bounds at all nodes
        let ibp_bounds = self.collect_node_bounds(input)?;

        // Step 2: For each node, try to tighten with CROWN
        let mut crown_ibp_bounds: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        for node_name in &exec_order {
            let ibp_bound = match ibp_bounds.get(node_name) {
                Some(b) => b,
                None => continue,
            };

            // Try to compute CROWN bounds for this node
            match self.propagate_crown_to_node(input, node_name, &crown_ibp_bounds, &ibp_bounds) {
                Ok(crown_bound) => {
                    // Only intersect if shapes match
                    if crown_bound.shape() == ibp_bound.shape() {
                        let tightened = ibp_bound.intersection(&crown_bound);
                        crown_ibp_bounds.insert(node_name.clone(), tightened);
                    } else {
                        debug!(
                            "CROWN-IBP DAG: {} shape mismatch IBP={:?} vs CROWN={:?}, using IBP",
                            node_name,
                            ibp_bound.shape(),
                            crown_bound.shape()
                        );
                        crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                    }
                }
                Err(e) => {
                    // Fall back to IBP if CROWN fails
                    debug!("CROWN-IBP DAG: {} failed ({}), using IBP", node_name, e);
                    crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                }
            }
        }

        Ok(crown_ibp_bounds)
    }

    /// Run backward CROWN from a target node to the network input.
    ///
    /// Returns CROWN bounds at the target node by:
    /// 1. Finding all nodes on paths from input to target
    /// 2. Running backward CROWN through this subgraph
    /// 3. Concretizing at the input
    fn propagate_crown_to_node(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<BoundedTensor> {
        // Get nodes that can reach target_node (in reverse topological order)
        let relevant_nodes = self.get_ancestors(target_node)?;

        if relevant_nodes.is_empty() {
            // Target is directly connected to input
            return Ok(input.clone());
        }

        // Get target node's IBP bounds for dimension info
        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Target node {} not in IBP bounds", target_node))
        })?;
        let target_dim = target_bounds.len();
        let target_shape = target_bounds.shape().to_vec();

        // Initialize linear bounds: identity at target node
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();
        node_linear_bounds.insert(target_node.to_string(), LinearBounds::identity(target_dim));

        let mut input_accumulated = false;

        // Propagate backward through relevant nodes (in reverse order)
        for node_name in relevant_nodes.iter().rev() {
            let node = match self.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };

            // Get this node's accumulated linear bounds
            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => continue, // Node has no consumers in this subgraph
            };

            // Get pre-activation bounds (prefer CROWN-IBP if available)
            let pre_activation_name = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                "_input"
            } else {
                &node.inputs[0]
            };
            let pre_activation = if pre_activation_name == "_input" {
                input
            } else {
                crown_ibp_bounds
                    .get(pre_activation_name)
                    .or_else(|| ibp_bounds.get(pre_activation_name))
                    .ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Pre-activation bounds for {} not found",
                            pre_activation_name
                        ))
                    })?
            };

            // Propagate linear bounds backward based on layer type
            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN-IBP failed at node '{}' (Linear): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    let new_lb = r
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN-IBP failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    // IMPORTANT: split bias so constants are not double-counted when we
                    // accumulate bounds from both branches.
                    let (lb_a, lb_b) = add.propagate_linear_binary(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN-IBP failed at node '{}' (Add): {}",
                            node_name, e
                        ))
                    })?;

                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                    if node.inputs.len() > 1 {
                        Self::accumulate_crown_ibp_bounds(
                            &node.inputs[1],
                            lb_b,
                            &mut node_linear_bounds,
                            &mut input_accumulated,
                        );
                    }
                }
                Layer::Conv2d(conv) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        // Input too small for Conv2d CROWN, skip
                        continue;
                    };
                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN-IBP failed at node '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
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
                                "CROWN-IBP failed at node '{}' (BatchNorm): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN-IBP failed at node '{}' (Flatten): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Transpose(t) => {
                    // Clone transpose and set input_shape for proper column permutation
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let new_lb = transpose_with_shape
                        .propagate_linear(&node_lb)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN-IBP failed at node '{}' (Transpose): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Concat(concat) => {
                    // Concat: SPLIT linear bounds across N inputs based on their shapes.
                    // Each input gets the portion of coefficients corresponding to its
                    // contribution to the concatenated output.
                    let input_shapes: Vec<Vec<usize>> = node
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(i, inp_name)| {
                            if inp_name == "_input" {
                                input.shape().to_vec()
                            } else if let Some(shape) = concat.get_input_shape(i) {
                                shape.clone()
                            } else {
                                // Try to get shape from IBP bounds
                                crown_ibp_bounds
                                    .get(inp_name)
                                    .or_else(|| ibp_bounds.get(inp_name))
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
                                "CROWN-IBP failed at node '{}' (Concat): {}",
                                node_name, e
                            ))
                        })?;

                    // Accumulate split bounds to each input
                    for (inp_name, lb) in node.inputs.iter().zip(bounds_vec.into_iter()) {
                        Self::accumulate_crown_ibp_bounds(
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
                                "CROWN-IBP failed at node '{}' (ReduceSum): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
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
                                "CROWN-IBP failed at node '{}' (ReduceMean): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Slice(sl) => {
                    let new_lb = sl
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN-IBP failed at node '{}' (Slice): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Sigmoid(sig) => {
                    let new_lb = sig
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN-IBP failed at node '{}' (Sigmoid): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::AddConstant(ac) => {
                    let new_lb = ac.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN-IBP failed at node '{}' (AddConstant): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                // For other layers (Div, MulBinary, etc.), propagate identity (conservative)
                _ => {
                    for input_name in &node.inputs {
                        Self::accumulate_crown_ibp_bounds(
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
        if input_accumulated {
            let final_lb = node_linear_bounds
                .remove("_input")
                .ok_or_else(|| GammaError::InvalidSpec("No linear bounds at input".to_string()))?;
            let bounds = final_lb.concretize(input);
            bounds.reshape(&target_shape)
        } else {
            // No backward pass reached input - fall back to IBP
            Ok(target_bounds.clone())
        }
    }

    /// Get all ancestor nodes of a target (nodes that can reach target).
    /// Returns nodes in topological order (dependencies before dependents).
    fn get_ancestors(&self, target: &str) -> Result<Vec<String>> {
        // BFS/DFS backward from target to find all reachable nodes
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut to_visit: Vec<String> = vec![target.to_string()];

        while let Some(node_name) = to_visit.pop() {
            if visited.contains(&node_name) || node_name == "_input" {
                continue;
            }
            visited.insert(node_name.clone());

            // Add inputs to visit
            if let Some(node) = self.nodes.get(&node_name) {
                for input in &node.inputs {
                    if !visited.contains(input) && input != "_input" {
                        to_visit.push(input.clone());
                    }
                }
            }
        }

        // Return in topological order
        let exec_order = self.topological_sort()?;
        let ordered: Vec<String> = exec_order
            .into_iter()
            .filter(|n| visited.contains(n))
            .collect();

        Ok(ordered)
    }

    /// Accumulate linear bounds to a node during backward CROWN-IBP pass.
    fn accumulate_crown_ibp_bounds(
        input_name: &str,
        new_bounds: LinearBounds,
        node_linear_bounds: &mut std::collections::HashMap<String, LinearBounds>,
        input_accumulated: &mut bool,
    ) {
        if input_name == "_input" {
            if *input_accumulated {
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
        } else if let Some(existing) = node_linear_bounds.get_mut(input_name) {
            existing.lower_a = &existing.lower_a + &new_bounds.lower_a;
            existing.lower_b = &existing.lower_b + &new_bounds.lower_b;
            existing.upper_a = &existing.upper_a + &new_bounds.upper_a;
            existing.upper_b = &existing.upper_b + &new_bounds.upper_b;
        } else {
            node_linear_bounds.insert(input_name.to_string(), new_bounds);
        }
    }

    /// Collect α-CROWN bounds for DAG models with gradient-based optimization.
    ///
    /// This is the key technique from α,β-CROWN that provides tighter bounds than CROWN-IBP:
    /// 1. Initialize alpha state for each ReLU node
    /// 2. Run iterative optimization with SPSA gradients
    /// 3. Return optimized intermediate bounds
    ///
    /// Returns a map of node names to their optimized bounds.
    pub fn collect_alpha_crown_bounds_dag(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        let exec_order = self.topological_sort()?;

        // Step 1: Collect bounds at all nodes
        // When fix_interm_bounds=true (default): use IBP bounds (O(N) - fast)
        // When fix_interm_bounds=false: use CROWN-IBP bounds (O(N²) - slow but tighter)
        //
        // This matches α,β-CROWN's fix_interm_bounds option which defaults to True.
        // The IBP bounds are sufficient for determining ReLU stability and are
        // much faster for deep networks (e.g., ResNet-4b: <5s vs ~80s).
        let ibp_bounds = if config.fix_interm_bounds {
            info!(
                "Using IBP bounds for intermediates (fix_interm_bounds=true, O(N) initialization)"
            );
            self.collect_node_bounds(input)?
        } else {
            info!(
                "Using CROWN-IBP bounds for intermediates (fix_interm_bounds=false, O(N²) initialization)"
            );
            self.collect_crown_ibp_bounds_dag(input)?
        };

        // Step 2: Initialize alpha state for all ReLU nodes
        let mut alpha_state = GraphAlphaState::new();
        let relu_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::ReLU(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        for relu_name in &relu_nodes {
            let node = self.nodes.get(relu_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!("ReLU node not found: {}", relu_name))
            })?;
            let pre_act_name = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                "_input"
            } else {
                &node.inputs[0]
            };
            let pre_act = if pre_act_name == "_input" {
                input
            } else {
                ibp_bounds.get(pre_act_name).unwrap_or(input)
            };
            alpha_state.add_relu_node(relu_name, pre_act);
        }

        let num_unstable = alpha_state.num_unstable();
        if num_unstable == 0 {
            debug!("GraphNetwork α-CROWN: No unstable neurons, using IBP bounds");
            return Ok(ibp_bounds);
        }

        debug!(
            "GraphNetwork α-CROWN: Starting optimization with {} unstable neurons across {} ReLU nodes, {} iterations",
            num_unstable,
            relu_nodes.len(),
            config.iterations
        );

        // Step 3: Optimization loop
        // KEY FIX: Optimize OUTPUT lower bound, not sum of all intermediate bounds!
        // This matches α,β-CROWN behavior: we maximize the output specification bound.
        let output_node = if self.output_node.is_empty() {
            exec_order.last().cloned().unwrap_or_default()
        } else {
            self.output_node.clone()
        };

        let mut best_bounds: std::collections::HashMap<String, BoundedTensor> = ibp_bounds.clone();
        let mut best_output_lower = f32::NEG_INFINITY;
        let mut lr = config.learning_rate;
        let eps = 1e-3; // Perturbation magnitude for SPSA

        // Sparse optimization: track which alphas are "active" (being optimized)
        // After first iteration, keep only top sparse_ratio fraction by gradient magnitude
        let mut sparse_mask: Option<std::collections::HashMap<String, Array1<bool>>> = None;
        let use_sparse = config.sparse_ratio < 1.0 && config.sparse_ratio > 0.0;

        for iter in 0..config.iterations {
            // Compute CROWN bounds at OUTPUT node only (for efficiency)
            // We only need output bounds for the objective during optimization
            let output_bounds = self.propagate_crown_to_node_with_alpha(
                input,
                &output_node,
                &std::collections::HashMap::new(), // Don't need intermediate CROWN bounds
                &ibp_bounds,
                &alpha_state,
            )?;

            // Compute objective: sum of OUTPUT lower bounds (higher is better)
            // This is what α,β-CROWN optimizes - the verification objective
            let output_lower: f32 = output_bounds
                .lower
                .as_slice()
                .map(|s| s.iter().sum::<f32>())
                .unwrap_or(0.0);

            // Update best if improved
            if output_lower > best_output_lower {
                best_output_lower = output_lower;
                debug!(
                    "GraphNetwork α-CROWN: iter {} improved output lower to {:.4}",
                    iter, output_lower
                );
            }

            // Skip gradient update on last iteration
            if iter == config.iterations - 1 {
                break;
            }

            // Compute gradients using SPSA - targeting output objective
            // Pass sparse_mask to only perturb active alphas (reduces SPSA variance)
            let gradients = self.compute_spsa_gradients_dag_for_output_sparse(
                input,
                &ibp_bounds,
                &alpha_state,
                &output_node,
                eps,
                config.spsa_samples,
                sparse_mask.as_ref(),
            )?;

            // After first iteration, select top alphas by gradient magnitude
            if iter == 0 && use_sparse {
                sparse_mask = Some(Self::select_top_alphas(&gradients, config.sparse_ratio));
                let active_count: usize = sparse_mask
                    .as_ref()
                    .map(|m| m.values().map(|v| v.iter().filter(|&&b| b).count()).sum())
                    .unwrap_or(0);
                debug!(
                    "GraphNetwork α-CROWN: Sparse mode enabled, optimizing top {} alphas ({}% of {})",
                    active_count,
                    (config.sparse_ratio * 100.0) as usize,
                    num_unstable
                );
            }

            // Update alpha values with gradient ascent (maximize output lower bound)
            // Only update alphas in sparse_mask (or all if no mask)
            let adam_params = config.adam_params(lr, iter + 1);
            for relu_name in &relu_nodes {
                if let Some(grad) = gradients.get(relu_name) {
                    let mask = sparse_mask.as_ref().and_then(|m| m.get(relu_name));
                    // Negate because we want to maximize, but update() does gradient descent
                    let neg_grad = if let Some(mask_arr) = mask {
                        // Zero out gradients for inactive alphas
                        let masked: Array1<f32> = grad
                            .iter()
                            .zip(mask_arr.iter())
                            .map(|(&g, &active)| if active { -g } else { 0.0 })
                            .collect();
                        masked
                    } else {
                        -grad
                    };
                    match config.optimizer {
                        Optimizer::Adam => {
                            alpha_state.update_adam(relu_name, &neg_grad, &adam_params);
                        }
                        Optimizer::Sgd => {
                            alpha_state.update(relu_name, &neg_grad, lr, config.momentum);
                        }
                    }
                }
            }

            // Learning rate decay
            lr *= config.lr_decay;
        }

        // When fix_interm_bounds=true, skip the expensive O(N²) post-optimization
        // CROWN computation for all intermediate nodes. Just use IBP bounds.
        // This matches α,β-CROWN's default behavior.
        if config.fix_interm_bounds {
            // Return IBP bounds directly (already tightened with output during optimization)
            debug!(
                "GraphNetwork α-CROWN: Using fixed IBP intermediate bounds (skipping O(N²) CROWN)"
            );
            best_bounds = ibp_bounds;
        } else {
            // After optimization, compute full intermediate bounds with optimized alphas
            let current_bounds =
                self.collect_crown_bounds_with_alpha(input, &ibp_bounds, &alpha_state)?;

            // Intersect with IBP for soundness
            for (name, ibp_bound) in &ibp_bounds {
                if let Some(crown_bound) = current_bounds.get(name) {
                    if crown_bound.shape() == ibp_bound.shape() {
                        best_bounds.insert(name.clone(), ibp_bound.intersection(crown_bound));
                    }
                }
            }
        }

        debug!(
            "GraphNetwork α-CROWN: Finished optimization, final output_lower={:.4}",
            best_output_lower
        );

        Ok(best_bounds)
    }

    /// Compute CROWN bounds for all nodes using explicit alpha values.
    fn collect_crown_bounds_with_alpha(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        let exec_order = self.topological_sort()?;
        let mut crown_bounds: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // For each node, run backward CROWN with alpha values
        for node_name in &exec_order {
            match self.propagate_crown_to_node_with_alpha(
                input,
                node_name,
                &crown_bounds,
                ibp_bounds,
                alpha_state,
            ) {
                Ok(bounds) => {
                    crown_bounds.insert(node_name.clone(), bounds);
                }
                Err(_) => {
                    // Fall back to IBP
                    if let Some(ibp) = ibp_bounds.get(node_name) {
                        crown_bounds.insert(node_name.clone(), ibp.clone());
                    }
                }
            }
        }

        Ok(crown_bounds)
    }

    /// Run backward CROWN from a target node using explicit alpha values.
    fn propagate_crown_to_node_with_alpha(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
    ) -> Result<BoundedTensor> {
        // Get nodes that can reach target_node
        let relevant_nodes = self.get_ancestors(target_node)?;

        if relevant_nodes.is_empty() {
            return Ok(input.clone());
        }

        // Get target node's IBP bounds for dimension info
        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Target node {} not in IBP bounds", target_node))
        })?;
        let target_dim = target_bounds.len();
        let target_shape = target_bounds.shape().to_vec();

        // Initialize linear bounds: identity at target node
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();
        node_linear_bounds.insert(target_node.to_string(), LinearBounds::identity(target_dim));

        let mut input_accumulated = false;

        // Propagate backward through relevant nodes
        for node_name in relevant_nodes.iter().rev() {
            let node = match self.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };

            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => continue,
            };

            // Get pre-activation bounds
            let pre_activation_name = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                "_input"
            } else {
                &node.inputs[0]
            };
            let pre_activation = if pre_activation_name == "_input" {
                input
            } else {
                crown_bounds
                    .get(pre_activation_name)
                    .or_else(|| ibp_bounds.get(pre_activation_name))
                    .ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Pre-activation bounds for {} not found",
                            pre_activation_name
                        ))
                    })?
            };

            // Propagate based on layer type
            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "α-CROWN failed at '{}' (Linear): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    // Use explicit alpha if available, otherwise default
                    let new_lb = if let Some(alpha) = alpha_state.get_alpha(node_name) {
                        let (bounds, _grad) =
                            r.propagate_linear_with_alpha(&node_lb, pre_activation, alpha)?;
                        bounds
                    } else {
                        r.propagate_linear_with_bounds(&node_lb, pre_activation)?
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(add) => {
                    // IMPORTANT: split bias so constants are not double-counted when we
                    // accumulate bounds from both branches.
                    let (lb_a, lb_b) = add.propagate_linear_binary(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "α-CROWN failed at '{}' (Add): {}",
                            node_name, e
                        ))
                    })?;

                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        lb_a,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                    if node.inputs.len() > 1 {
                        Self::accumulate_crown_ibp_bounds(
                            &node.inputs[1],
                            lb_b,
                            &mut node_linear_bounds,
                            &mut input_accumulated,
                        );
                    }
                }
                Layer::Conv2d(conv) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        continue;
                    };
                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "α-CROWN failed at '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
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
                                "α-CROWN failed at '{}' (BatchNorm): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "α-CROWN failed at '{}' (Flatten): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::AveragePool(ap) => {
                    let new_lb = ap
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "α-CROWN failed at '{}' (AveragePool): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::MaxPool2d(_) => {
                    // MaxPool is not differentiable, fall back to identity propagation
                    for input_name in &node.inputs {
                        Self::accumulate_crown_ibp_bounds(
                            input_name,
                            node_lb.clone(),
                            &mut node_linear_bounds,
                            &mut input_accumulated,
                        );
                    }
                }
                Layer::Transpose(t) => {
                    // Clone transpose and set input_shape for proper column permutation
                    let input_shape = pre_activation.shape().to_vec();
                    let mut transpose_with_shape = t.clone();
                    transpose_with_shape.set_input_shape(input_shape);
                    let new_lb = transpose_with_shape
                        .propagate_linear(&node_lb)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "α-CROWN failed at '{}' (Transpose): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Concat(concat) => {
                    // Concat: SPLIT linear bounds across N inputs based on their shapes.
                    let input_shapes: Vec<Vec<usize>> = node
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(i, inp_name)| {
                            // First check if this is a constant input (CLS token, etc.)
                            // Constant inputs have known shapes stored in the Concat layer
                            if let Some(constant_tensor) = concat.get_constant_input(i) {
                                return constant_tensor.shape().to_vec();
                            }
                            // Then check stored shapes
                            if let Some(shape) = concat.get_input_shape(i) {
                                if !shape.is_empty() {
                                    return shape.clone();
                                }
                            }
                            // Fall back to network input or bounds
                            if inp_name == "_input" {
                                input.shape().to_vec()
                            } else {
                                // Try to get shape from IBP bounds first (more reliable for intermediate nodes)
                                ibp_bounds
                                    .get(inp_name)
                                    .or_else(|| crown_bounds.get(inp_name))
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
                                "α-CROWN failed at '{}' (Concat): {}",
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
                        Self::accumulate_crown_ibp_bounds(
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
                                "α-CROWN failed at '{}' (ReduceSum): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
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
                                "α-CROWN failed at '{}' (ReduceMean): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_ibp_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                _ => {
                    // For other layers, propagate identity
                    for input_name in &node.inputs {
                        Self::accumulate_crown_ibp_bounds(
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
        if input_accumulated {
            let final_lb = node_linear_bounds
                .remove("_input")
                .ok_or_else(|| GammaError::InvalidSpec("No linear bounds at input".to_string()))?;
            let bounds = final_lb.concretize(input);
            bounds.reshape(&target_shape)
        } else {
            Ok(target_bounds.clone())
        }
    }

    /// Compute SPSA gradients for DAG α-CROWN with optional sparse mask.
    ///
    /// When sparse_mask is provided, only perturb alphas where mask[name][i] is true.
    /// This reduces SPSA variance by focusing perturbations on influential alphas.
    #[allow(clippy::too_many_arguments)]
    fn compute_spsa_gradients_dag_for_output_sparse(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        output_node: &str,
        eps: f32,
        num_samples: usize,
        sparse_mask: Option<&std::collections::HashMap<String, Array1<bool>>>,
    ) -> Result<std::collections::HashMap<String, Array1<f32>>> {
        use rand::Rng;
        use rayon::prelude::*;

        // Step 1: Pre-generate all perturbations for all samples
        let mut rng = rand::rng();
        let mut all_perturbations: Vec<std::collections::HashMap<String, Array1<f32>>> =
            Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let mut perturbations: std::collections::HashMap<String, Array1<f32>> =
                std::collections::HashMap::new();
            for (name, alpha) in &alpha_state.alphas {
                let unstable_mask = alpha_state.unstable_mask.get(name);
                let active_mask = sparse_mask.and_then(|m| m.get(name));

                let pert = Array1::from_iter((0..alpha.len()).map(|i| {
                    let is_unstable = unstable_mask.map(|m| m[i]).unwrap_or(false);
                    // If sparse_mask is provided, only perturb if also active
                    let is_active = active_mask.map(|m| m[i]).unwrap_or(true);
                    if is_unstable && is_active {
                        if rng.random_bool(0.5) {
                            1.0
                        } else {
                            -1.0
                        }
                    } else {
                        0.0
                    }
                }));
                perturbations.insert(name.clone(), pert);
            }
            all_perturbations.push(perturbations);
        }

        // Step 2: Create all perturbed alpha states FLATTENED
        let original_alphas = &alpha_state.alphas;
        let mut all_tasks: Vec<(usize, bool, GraphAlphaState)> =
            Vec::with_capacity(num_samples * 2);

        for (sample_idx, perturbations) in all_perturbations.iter().enumerate() {
            // +eps perturbation
            let mut alpha_plus = alpha_state.clone();
            for (name, pert) in perturbations {
                if let (Some(orig), Some(plus_alpha)) =
                    (original_alphas.get(name), alpha_plus.alphas.get_mut(name))
                {
                    for i in 0..orig.len() {
                        plus_alpha[i] = (orig[i] + eps * pert[i]).clamp(0.0, 1.0);
                    }
                }
            }
            all_tasks.push((sample_idx, true, alpha_plus));

            // -eps perturbation
            let mut alpha_minus = alpha_state.clone();
            for (name, pert) in perturbations {
                if let (Some(orig), Some(minus_alpha)) =
                    (original_alphas.get(name), alpha_minus.alphas.get_mut(name))
                {
                    for i in 0..orig.len() {
                        minus_alpha[i] = (orig[i] - eps * pert[i]).clamp(0.0, 1.0);
                    }
                }
            }
            all_tasks.push((sample_idx, false, alpha_minus));
        }

        // Step 3: Compute all CROWN bounds in PARALLEL using Rayon
        let empty_map: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        let all_results: Vec<(usize, bool, f32)> = all_tasks
            .par_iter()
            .map(|(sample_idx, is_plus, perturbed_alpha)| {
                let lower: f32 = self
                    .propagate_crown_to_node_with_alpha(
                        input,
                        output_node,
                        &empty_map,
                        ibp_bounds,
                        perturbed_alpha,
                    )
                    .ok()
                    .and_then(|b| b.lower.as_slice().map(|s| s.iter().sum::<f32>()))
                    .unwrap_or(0.0);
                (*sample_idx, *is_plus, lower)
            })
            .collect();

        // Step 4: Reconstruct (lower_plus, lower_minus) pairs
        let mut sample_results: Vec<(f32, f32)> = vec![(0.0, 0.0); num_samples];
        for (sample_idx, is_plus, lower) in all_results {
            if is_plus {
                sample_results[sample_idx].0 = lower;
            } else {
                sample_results[sample_idx].1 = lower;
            }
        }

        // Step 5: Aggregate gradients
        let mut avg_grads: std::collections::HashMap<String, Array1<f32>> =
            std::collections::HashMap::new();
        for (name, alpha) in &alpha_state.alphas {
            avg_grads.insert(name.clone(), Array1::zeros(alpha.len()));
        }

        for (sample_idx, (lower_plus, lower_minus)) in sample_results.iter().enumerate() {
            let perturbations = &all_perturbations[sample_idx];
            let diff = lower_plus - lower_minus;

            // SPSA gradient estimate: g_i = (f+ - f-) / (2 * eps * Δ_i)
            for (name, pert) in perturbations {
                if let Some(grad) = avg_grads.get_mut(name) {
                    for i in 0..grad.len() {
                        if pert[i].abs() > 0.5 {
                            grad[i] += diff / (2.0 * eps * pert[i]);
                        }
                    }
                }
            }
        }

        // Average the gradients
        let num_samples_f32 = num_samples as f32;
        for grad in avg_grads.values_mut() {
            *grad /= num_samples_f32;
        }

        Ok(avg_grads)
    }

    /// Select top `ratio` fraction of alphas by gradient magnitude.
    ///
    /// Returns a mask where true indicates the alpha should be optimized.
    fn select_top_alphas(
        gradients: &std::collections::HashMap<String, Array1<f32>>,
        ratio: f32,
    ) -> std::collections::HashMap<String, Array1<bool>> {
        // Collect all (gradient_magnitude, name, idx) tuples
        let mut all_grads: Vec<(f32, &str, usize)> = Vec::new();
        for (name, grad) in gradients {
            for (i, &g) in grad.iter().enumerate() {
                all_grads.push((g.abs(), name.as_str(), i));
            }
        }

        // Sort by magnitude (descending)
        all_grads.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Select top `ratio` fraction
        let keep_count = ((all_grads.len() as f32 * ratio).ceil() as usize).max(1);

        // Build mask
        let mut mask: std::collections::HashMap<String, Array1<bool>> =
            std::collections::HashMap::new();
        for (name, grad) in gradients {
            mask.insert(name.clone(), Array1::from_elem(grad.len(), false));
        }

        for (_, name, idx) in all_grads.iter().take(keep_count) {
            if let Some(m) = mask.get_mut(*name) {
                m[*idx] = true;
            }
        }

        mask
    }

    /// Get bounds for a node reference, either from cache or network input.
    ///
    /// Returns a reference to avoid cloning BoundedTensor.
    #[inline]
    fn get_bounds_ref<'a>(
        &self,
        name: &str,
        input: &'a BoundedTensor,
        cache: &'a std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<&'a BoundedTensor> {
        if name == "_input" {
            Ok(input)
        } else {
            cache.get(name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "Bounds for node {} not yet computed (dependency order error)",
                    name
                ))
            })
        }
    }

    /// Try to compute tighter bounds for Q@K^T using zonotope correlation tracking.
    ///
    /// This handles both MHA (Q,K directly from Linear) and GQA (K through reshape/tile ops):
    /// - MHA: input -> q_proj -> Q, input -> k_proj -> K
    /// - GQA: input -> q_proj -> Q, input -> k_proj -> k_reshape -> k_tile -> k_reshape -> K
    ///
    /// For GQA, zonotope propagation through reshape/tile preserves correlations because
    /// the tiled K heads share the same error symbols as the original projection.
    fn try_attention_matmul_bounds_zonotope(
        &self,
        matmul_node: &GraphNode,
        input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        if matmul_node.inputs.len() < 2 {
            return Ok(None);
        }

        let matmul = match &matmul_node.layer {
            Layer::MatMul(m) => m,
            _ => return Ok(None),
        };
        if !matmul.transpose_b {
            return Ok(None);
        }

        let (q_node_name, k_node_name) = (&matmul_node.inputs[0], &matmul_node.inputs[1]);

        // Trace back Q and K to find their base Linear projections and the operations in between
        let q_path = self.trace_zonotope_path(q_node_name);
        let k_path = self.trace_zonotope_path(k_node_name);

        // Find the Linear layer in each path (should be the first element after reverse)
        let (q_linear_name, q_ops) = match q_path.split_last() {
            Some((linear_name, ops)) => (linear_name.clone(), ops.to_vec()),
            None => return Ok(None),
        };
        let (k_linear_name, k_ops) = match k_path.split_last() {
            Some((linear_name, ops)) => (linear_name.clone(), ops.to_vec()),
            None => return Ok(None),
        };

        let q_linear_node = match self.nodes.get(&q_linear_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let k_linear_node = match self.nodes.get(&k_linear_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        let (q_linear, k_linear) = match (&q_linear_node.layer, &k_linear_node.layer) {
            (Layer::Linear(q), Layer::Linear(k)) => (q, k),
            _ => return Ok(None),
        };

        // Check that Q and K projections share the same base input
        let q_base = q_linear_node.inputs.first().ok_or_else(|| {
            GammaError::InvalidSpec(format!("Linear node {} has no inputs", q_linear_node.name))
        })?;
        let k_base = k_linear_node.inputs.first().ok_or_else(|| {
            GammaError::InvalidSpec(format!("Linear node {} has no inputs", k_linear_node.name))
        })?;

        if q_base != k_base {
            return Ok(None);
        }

        // Use the LayerNorm output directly (q_base) without propagating through LayerNorm.
        // Empirical testing shows that the LayerNorm affine approximation error overwhelms
        // any benefit from preserved correlations - creating fresh zonotopes from LayerNorm
        // output gives 10^6-10^9x tighter Q@K^T bounds than propagating through LayerNorm.
        //
        // Evidence (Qwen3-0.6B block-wise verification):
        // - Layer0 with LN propagation: Q@K^T width = 2.440e3
        // - Layers 1-27 without LN prop: Q@K^T width = 1e-6 to 1e-3
        let actual_base_name = q_base.clone();
        let layernorm_node: Option<&LayerNormLayer> = None;

        // Get base bounds and create zonotope
        let base_bounds = self.get_bounds_ref(&actual_base_name, input, bounds_cache)?;
        if base_bounds.shape().len() != 2 {
            return Ok(None);
        }
        let base_width = base_bounds.max_width();

        // When input bounds are large (from cumulative propagation), use a single
        // shared error term to prevent overflow in matmul_transposed. With per-position
        // error terms (n=seq_len), the O(n²) cross terms can overflow. A single error
        // term gives tighter bounds than IBP while avoiding overflow.
        //
        // Threshold: base_width > 1.0 suggests cumulative bound growth from previous layers.
        // For fresh input (epsilon ~ 0.001), base_width < 1.0 and we use per-position tracking.
        //
        // Spectral norm scaling: The Q and K Linear projections amplify zonotope coefficients
        // by their spectral norms. For Q@K^T (quadratic in coefficients), cross-terms explode
        // as O(spec_q * spec_k). We normalize by max spectral norm to keep coefficients ~1
        // after Linear, preventing overflow while preserving soundness.
        let max_spectral = q_linear.spectral_norm().max(k_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };
        let (base_z, needs_rescale) = if zonotope_scale > 1.0 {
            // Large bounds or large spectral norm: use single error term with normalization
            let normalized_bounds = BoundedTensor::new(
                base_bounds.lower.mapv(|v| v / zonotope_scale),
                base_bounds.upper.mapv(|v| v / zonotope_scale),
            )?;
            // Reshape to 2D for matmul compatibility
            let flat_shape = vec![base_bounds.shape().iter().product()];
            let flat_lower = normalized_bounds
                .lower
                .into_shape_with_order(IxDyn(&flat_shape))
                .map_err(|e| GammaError::InvalidSpec(format!("reshape failed: {}", e)))?;
            let flat_upper = normalized_bounds
                .upper
                .into_shape_with_order(IxDyn(&flat_shape))
                .map_err(|e| GammaError::InvalidSpec(format!("reshape failed: {}", e)))?;
            let flat_bounds = BoundedTensor::new(flat_lower, flat_upper)?;

            // Create single-error-term zonotope then reshape to 2D
            let z_flat = ZonotopeTensor::from_bounded_tensor(&flat_bounds);
            let z_2d = z_flat.reshape(base_bounds.shape())?;
            (z_2d, true)
        } else {
            // Small bounds: use per-position error terms for tighter correlation tracking
            let z = ZonotopeTensor::from_bounded_tensor_per_position_2d(base_bounds)?;
            (z, false)
        };
        let _layernorm_node = layernorm_node; // Suppress unused warning

        debug!(
            "Zonotope Q@K^T tightening: base={} base_width={:.3e} max_spectral={:.3e} scale={:.3e} n_err={}",
            actual_base_name, base_width, max_spectral, zonotope_scale, base_z.n_error_terms
        );

        // Apply Q projection
        let q_z = base_z.linear(&q_linear.weight, q_linear.bias.as_ref())?;

        // Apply K projection then propagate through reshape/tile operations
        let mut k_z = base_z.linear(&k_linear.weight, k_linear.bias.as_ref())?;

        // Apply operations in order (k_ops is in forward order: reshape1, tile, reshape2)
        for op_name in k_ops.iter().rev() {
            let op_node = match self.nodes.get(op_name) {
                Some(n) => n,
                None => return Ok(None),
            };

            k_z = match &op_node.layer {
                Layer::Reshape(reshape) => {
                    let output_shape = reshape.compute_output_shape(&k_z.element_shape)?;
                    k_z.reshape(&output_shape)?
                }
                Layer::Tile(tile) => {
                    let ndim = k_z.element_shape.len();
                    let axis = if tile.axis < 0 {
                        (ndim as i32 + tile.axis) as usize
                    } else {
                        tile.axis as usize
                    };
                    k_z.tile(axis, tile.reps)?
                }
                Layer::Transpose(transpose) => {
                    let ndim = transpose.axes.len();
                    if ndim >= 2
                        && transpose.axes[ndim - 2] == ndim - 1
                        && transpose.axes[ndim - 1] == ndim - 2
                    {
                        k_z.transpose_last_two()?
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None), // Unsupported operation in K path
            };
        }

        // Apply Q operations if any
        let mut q_z_final = q_z;
        for op_name in q_ops.iter().rev() {
            let op_node = match self.nodes.get(op_name) {
                Some(n) => n,
                None => return Ok(None),
            };

            q_z_final = match &op_node.layer {
                Layer::Reshape(reshape) => {
                    let output_shape = reshape.compute_output_shape(&q_z_final.element_shape)?;
                    q_z_final.reshape(&output_shape)?
                }
                Layer::Tile(tile) => {
                    let ndim = q_z_final.element_shape.len();
                    let axis = if tile.axis < 0 {
                        (ndim as i32 + tile.axis) as usize
                    } else {
                        tile.axis as usize
                    };
                    q_z_final.tile(axis, tile.reps)?
                }
                Layer::Transpose(transpose) => {
                    let ndim = transpose.axes.len();
                    if ndim >= 2
                        && transpose.axes[ndim - 2] == ndim - 1
                        && transpose.axes[ndim - 1] == ndim - 2
                    {
                        q_z_final.transpose_last_two()?
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            };
        }

        // Check that Q and K zonotopes have compatible shapes for matmul_transposed
        // Q: (seq_q, dim_q), K: (seq_k, dim_k) where dim_q == dim_k
        if q_z_final.element_shape.len() != 2 || k_z.element_shape.len() != 2 {
            return Ok(None);
        }
        if q_z_final.element_shape[1] != k_z.element_shape[1] {
            return Ok(None);
        }

        let mut out = q_z_final.matmul_transposed(&k_z)?;
        if let Some(scale) = matmul.scale {
            let scale_tensor = ndarray::ArrayD::from_elem(out.element_shape.clone(), scale);
            out = out.mul_constant(&scale_tensor)?;
        }

        let result = out.to_bounded_tensor();

        // Scale back the result by zonotope_scale² (matmul is quadratic in input scale)
        // If we normalized input x by scale s, then:
        //   Q_norm = Q / s, K_norm = K / s
        //   (Q_norm @ K_norm^T) = (Q @ K^T) / s²
        // So we multiply by s² to recover the correct scale.
        // Use f64 intermediates to avoid inf * 0 = NaN when scale² overflows f32.
        // IMPORTANT: Check for NaN/Inf BEFORE creating BoundedTensor to avoid panic.
        let result = if needs_rescale {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            let scaled_lower = result.lower.mapv(|v| (v as f64 * scale_sq) as f32);
            let scaled_upper = result.upper.mapv(|v| (v as f64 * scale_sq) as f32);

            // Check for NaN/Inf before creating BoundedTensor
            let has_bad_values = scaled_lower
                .iter()
                .chain(scaled_upper.iter())
                .any(|v| v.is_nan() || v.is_infinite());

            if has_bad_values {
                // Zonotope rescale produced overflow - fall back to IBP
                debug!(
                    "Zonotope Q@K^T rescale overflow: scale²={:.3e}, falling back to IBP",
                    scale_sq
                );
                return Ok(None);
            }

            BoundedTensor::new(scaled_lower, scaled_upper)?
        } else {
            // Check original result for NaN/Inf
            let has_bad_values = result
                .lower
                .iter()
                .chain(result.upper.iter())
                .any(|v| v.is_nan() || v.is_infinite());

            if has_bad_values {
                // Zonotope produced overflow - fall back to IBP
                debug!("Zonotope Q@K^T overflow, falling back to IBP");
                return Ok(None);
            }
            result
        };

        let result_width = result.max_width();

        debug!(
            "Zonotope Q@K^T output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        Ok(Some(result))
    }

    /// Trace back from a node through zonotope-preserving operations to find the source Linear.
    ///
    /// Returns the path of node names from the given node back to the Linear layer (inclusive).
    /// Operations that preserve zonotope form: Reshape, Tile, Transpose, Linear.
    fn trace_zonotope_path(&self, start_node: &str) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = start_node.to_string();

        while let Some(node) = self.nodes.get(&current) {
            path.push(current.clone());

            match &node.layer {
                Layer::Linear(_) => break, // Found the base Linear
                Layer::Reshape(_) | Layer::Tile(_) | Layer::Transpose(_) => {
                    // Continue tracing back
                    if let Some(input) = node.inputs.first() {
                        current = input.clone();
                    } else {
                        break;
                    }
                }
                _ => break, // Non-zonotope-preserving operation
            }
        }

        path
    }

    /// Try to apply zonotope tightening for SwiGLU FFN in block-wise mode.
    ///
    /// SwiGLU: output = up * silu(gate), where both up and gate come from the same base (ffn_norm).
    /// By using zonotopes, we can track correlations through the shared error symbols and
    /// get tighter bounds than IBP which treats them as independent.
    ///
    /// # Pattern
    /// ```text
    /// ffn_norm -> ffn_up (Linear) -------> up
    ///          -> ffn_gate (Linear) -> silu -> gate
    /// MulBinary(up, gate) -> swiglu
    /// ```
    fn try_ffn_swiglu_bounds_zonotope_block(
        &self,
        mul_node: &GraphNode,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        use gamma_tensor::ZonotopeTensor;

        // Must be MulBinary with 2 inputs
        if mul_node.inputs.len() < 2 {
            return Ok(None);
        }

        let input_a_name = &mul_node.inputs[0];
        let input_b_name = &mul_node.inputs[1];

        // Identify up and gate branches
        // Pattern: MulBinary(up, silu(gate)) or MulBinary(silu(gate), up)
        let (up_name, silu_name) = {
            let node_a = self.nodes.get(input_a_name);
            let node_b = self.nodes.get(input_b_name);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    // Check if one is GELU (SiLU) and trace back
                    let a_is_silu = matches!(&a.layer, Layer::GELU(_));
                    let b_is_silu = matches!(&b.layer, Layer::GELU(_));

                    if a_is_silu && !b_is_silu {
                        (input_b_name.clone(), input_a_name.clone())
                    } else if b_is_silu && !a_is_silu {
                        (input_a_name.clone(), input_b_name.clone())
                    } else {
                        // Neither or both are SiLU - not standard SwiGLU pattern
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        };

        // Get the SiLU node and trace back to gate Linear
        let silu_node = match self.nodes.get(&silu_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        if silu_node.inputs.is_empty() {
            return Ok(None);
        }
        let gate_name = &silu_node.inputs[0];

        // Gate should be a Linear layer
        let gate_node = match self.nodes.get(gate_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let gate_linear = match &gate_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Up should be a Linear layer (or trace back to one)
        let up_node = match self.nodes.get(&up_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let up_linear = match &up_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Check that gate and up share the same input (ffn_norm output)
        let gate_base = gate_node.inputs.first().ok_or_else(|| {
            GammaError::InvalidSpec(format!("Gate node {} has no inputs", gate_name))
        })?;
        let up_base = up_node
            .inputs
            .first()
            .ok_or_else(|| GammaError::InvalidSpec(format!("Up node {} has no inputs", up_name)))?;

        if gate_base != up_base {
            // Different bases - can't exploit correlation
            debug!(
                "SwiGLU zonotope: gate_base='{}' != up_base='{}', skipping",
                gate_base, up_base
            );
            return Ok(None);
        }

        // Get base bounds (should be ffn_norm output)
        let base_bounds = self.get_bounds_for_block(gate_base, block_input, bounds_cache)?;
        if base_bounds.shape().len() != 2 {
            // Only 2D supported for now
            return Ok(None);
        }

        let base_width = base_bounds.max_width();

        // Normalize bounds to prevent overflow in quadratic cross-terms
        // The zonotope multiplication (up * silu(gate)) is quadratic, so large
        // coefficients cause cross-term explosion.
        //
        // Key insight: Weight spectral norms amplify the input bounds. For late
        // transformer blocks with σ ≈ 500, normalizing only by input width still
        // leaves coefficients ~500 after the Linear, causing ~250,000 cross-terms.
        //
        // By including max(σ_up, σ_gate) in the scale, we keep post-Linear
        // coefficients ~1, preventing the quadratic explosion.
        let max_spectral = gate_linear.spectral_norm().max(up_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };
        let normalized_bounds = if zonotope_scale > 1.0 {
            match BoundedTensor::new(
                base_bounds.lower.mapv(|v| v / zonotope_scale),
                base_bounds.upper.mapv(|v| v / zonotope_scale),
            ) {
                Ok(b) => b,
                Err(_) => return Ok(None),
            }
        } else {
            base_bounds.clone()
        };

        // Create zonotope from normalized bounds with per-position error symbols
        let base_z = match ZonotopeTensor::from_bounded_tensor_per_position_2d(&normalized_bounds) {
            Ok(z) => z,
            Err(_) => return Ok(None),
        };

        debug!(
            "SwiGLU zonotope: base='{}' base_width={:.3e} max_spectral={:.1} scale={:.3e} n_err={}",
            gate_base, base_width, max_spectral, zonotope_scale, base_z.n_error_terms
        );

        // Apply gate Linear projection
        let gate_z = match base_z.linear(&gate_linear.weight, gate_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(e) => {
                debug!("SwiGLU zonotope: gate linear failed: {}", e);
                return Ok(None);
            }
        };

        // Apply SiLU to gate
        let silu_z = match gate_z.silu_affine() {
            Ok(z) => z,
            Err(e) => {
                debug!("SwiGLU zonotope: silu_affine failed: {}", e);
                return Ok(None);
            }
        };

        // Apply up Linear projection
        let up_z = match base_z.linear(&up_linear.weight, up_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(e) => {
                debug!("SwiGLU zonotope: up linear failed: {}", e);
                return Ok(None);
            }
        };

        // Multiply: up * silu(gate)
        let swiglu_z = match up_z.mul_elementwise(&silu_z) {
            Ok(z) => z,
            Err(e) => {
                debug!("SwiGLU zonotope: mul_elementwise failed: {}", e);
                return Ok(None);
            }
        };

        let result = swiglu_z.to_bounded_tensor();

        // Scale back by zonotope_scale² (multiplication is quadratic in scale)
        // If we normalized input x by scale s, then:
        //   up_norm = up / s, gate_norm = gate / s
        //   (up_norm * silu(gate_norm)) is roughly (up * silu(gate)) / s²
        // So we multiply by s² to recover the correct scale.
        let result = if zonotope_scale > 1.0 {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            match BoundedTensor::new(
                result.lower.mapv(|v| (v as f64 * scale_sq) as f32),
                result.upper.mapv(|v| (v as f64 * scale_sq) as f32),
            ) {
                Ok(b) => b,
                Err(_) => return Ok(None),
            }
        } else {
            result
        };

        let result_width = result.max_width();

        debug!(
            "SwiGLU zonotope output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        // Validate output: if zonotope produced NaN/Inf, fall back to IBP
        let has_bad_values = result
            .lower
            .iter()
            .chain(result.upper.iter())
            .any(|v| v.is_nan() || v.is_infinite());

        if has_bad_values {
            debug!("SwiGLU zonotope: output has NaN/Inf, falling back to IBP");
            return Ok(None);
        }

        Ok(Some(result))
    }

    /// Try to compute tighter bounds for SwiGLU (up * silu(gate)) using zonotope.
    ///
    /// Non-block version for full network propagation. Uses single-error zonotope
    /// for large bounds to prevent overflow.
    fn try_ffn_swiglu_bounds_zonotope(
        &self,
        mul_node: &GraphNode,
        input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        use gamma_tensor::ZonotopeTensor;

        // Must be MulBinary with 2 inputs
        if mul_node.inputs.len() < 2 {
            return Ok(None);
        }

        let input_a_name = &mul_node.inputs[0];
        let input_b_name = &mul_node.inputs[1];

        // Identify up and gate branches
        // Pattern: MulBinary(up, silu(gate)) or MulBinary(silu(gate), up)
        let (up_name, silu_name) = {
            let node_a = self.nodes.get(input_a_name);
            let node_b = self.nodes.get(input_b_name);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    let a_is_silu = matches!(&a.layer, Layer::GELU(_));
                    let b_is_silu = matches!(&b.layer, Layer::GELU(_));

                    if a_is_silu && !b_is_silu {
                        (input_b_name.clone(), input_a_name.clone())
                    } else if b_is_silu && !a_is_silu {
                        (input_a_name.clone(), input_b_name.clone())
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        };

        // Get the SiLU node and trace back to gate Linear
        let silu_node = match self.nodes.get(&silu_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        if silu_node.inputs.is_empty() {
            return Ok(None);
        }
        let gate_name = &silu_node.inputs[0];

        let gate_node = match self.nodes.get(gate_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let gate_linear = match &gate_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        let up_node = match self.nodes.get(&up_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let up_linear = match &up_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Check that gate and up share the same input
        let gate_base = gate_node.inputs.first().ok_or_else(|| {
            GammaError::InvalidSpec(format!("Gate node {} has no inputs", gate_name))
        })?;
        let up_base = up_node
            .inputs
            .first()
            .ok_or_else(|| GammaError::InvalidSpec(format!("Up node {} has no inputs", up_name)))?;

        if gate_base != up_base {
            return Ok(None);
        }

        // Get base bounds
        let base_bounds = self.get_bounds_ref(gate_base, input, bounds_cache)?;
        if base_bounds.shape().len() != 2 {
            return Ok(None);
        }

        let base_width = base_bounds.max_width();

        // Always use per-position error terms for correlation tracking.
        //
        // Normalize bounds to prevent overflow in quadratic cross-terms during
        // the zonotope multiplication (up * silu(gate)).
        //
        // Key insight: Weight spectral norms amplify the input bounds. For late
        // transformer blocks with σ ≈ 500, normalizing only by input width still
        // leaves coefficients ~500 after the Linear, causing ~250,000 cross-terms.
        //
        // By including max(σ_up, σ_gate) in the scale, we keep post-Linear
        // coefficients ~1, preventing the quadratic explosion.
        let max_spectral = gate_linear.spectral_norm().max(up_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };
        let normalized_bounds = if zonotope_scale > 1.0 {
            BoundedTensor::new(
                base_bounds.lower.mapv(|v| v / zonotope_scale),
                base_bounds.upper.mapv(|v| v / zonotope_scale),
            )?
        } else {
            base_bounds.clone()
        };
        // Per-position error terms preserve correlations through up/gate projections
        // This gives tighter bounds than single error term which loses correlation
        let base_z = ZonotopeTensor::from_bounded_tensor_per_position_2d(&normalized_bounds)?;

        debug!(
            "SwiGLU zonotope (full): base='{}' base_width={:.3e} max_spectral={:.1} scale={:.3e} n_err={}",
            gate_base, base_width, max_spectral, zonotope_scale, base_z.n_error_terms
        );

        // Apply gate Linear projection
        let gate_z = match base_z.linear(&gate_linear.weight, gate_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(_) => return Ok(None),
        };

        // Apply SiLU to gate
        let silu_z = match gate_z.silu_affine() {
            Ok(z) => z,
            Err(_) => return Ok(None),
        };

        // Apply up Linear projection
        let up_z = match base_z.linear(&up_linear.weight, up_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(_) => return Ok(None),
        };

        // Multiply: up * silu(gate)
        let swiglu_z = match up_z.mul_elementwise(&silu_z) {
            Ok(z) => z,
            Err(_) => return Ok(None),
        };

        let result = swiglu_z.to_bounded_tensor();

        // Scale back by zonotope_scale² (multiplication is quadratic in scale)
        let result = if zonotope_scale > 1.0 {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            BoundedTensor::new(
                result.lower.mapv(|v| (v as f64 * scale_sq) as f32),
                result.upper.mapv(|v| (v as f64 * scale_sq) as f32),
            )?
        } else {
            result
        };

        let result_width = result.max_width();

        debug!(
            "SwiGLU zonotope (full) output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        // Validate output
        let has_bad_values = result
            .lower
            .iter()
            .chain(result.upper.iter())
            .any(|v| v.is_nan() || v.is_infinite());

        if has_bad_values {
            return Ok(None);
        }

        Ok(Some(result))
    }

    /// Try to apply zonotope tightening for the full FFN (including down projection).
    ///
    /// When processing the ffn_down Linear node, if its input is a SwiGLU pattern,
    /// we can propagate zonotopes through the entire FFN for tighter bounds:
    /// ffn_norm -> up + (gate -> silu) -> mul -> down
    ///
    /// This extends the SwiGLU zonotope tightening to include the down projection,
    /// which previously fell back to IBP and amplified bounds ~16x per block.
    fn try_ffn_down_zonotope_block(
        &self,
        linear_node: &GraphNode,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        use gamma_tensor::ZonotopeTensor;

        // This node must be Linear
        let down_linear = match &linear_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Get input node (should be MulBinary for SwiGLU)
        if linear_node.inputs.is_empty() {
            return Ok(None);
        }
        let mul_name = &linear_node.inputs[0];

        // Check if input is MulBinary
        let mul_node = match self.nodes.get(mul_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        if !matches!(&mul_node.layer, Layer::MulBinary(_)) {
            return Ok(None);
        }

        // Now trace back through SwiGLU pattern: MulBinary(up, silu(gate))
        if mul_node.inputs.len() < 2 {
            return Ok(None);
        }

        let input_a_name = &mul_node.inputs[0];
        let input_b_name = &mul_node.inputs[1];

        // Identify up and gate branches
        let (up_name, silu_name) = {
            let node_a = self.nodes.get(input_a_name);
            let node_b = self.nodes.get(input_b_name);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    let a_is_silu = matches!(&a.layer, Layer::GELU(_));
                    let b_is_silu = matches!(&b.layer, Layer::GELU(_));

                    if a_is_silu && !b_is_silu {
                        (input_b_name.clone(), input_a_name.clone())
                    } else if b_is_silu && !a_is_silu {
                        (input_a_name.clone(), input_b_name.clone())
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        };

        // Get the SiLU node and trace back to gate Linear
        let silu_node = match self.nodes.get(&silu_name) {
            Some(n) => n,
            None => return Ok(None),
        };

        if silu_node.inputs.is_empty() {
            return Ok(None);
        }
        let gate_name = &silu_node.inputs[0];

        // Gate should be a Linear layer
        let gate_node = match self.nodes.get(gate_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let gate_linear = match &gate_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Up should be a Linear layer
        let up_node = match self.nodes.get(&up_name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let up_linear = match &up_node.layer {
            Layer::Linear(l) => l,
            _ => return Ok(None),
        };

        // Check that gate and up share the same input (ffn_norm output)
        let gate_base = gate_node.inputs.first().ok_or_else(|| {
            GammaError::InvalidSpec(format!("Gate node {} has no inputs", gate_name))
        })?;
        let up_base = up_node
            .inputs
            .first()
            .ok_or_else(|| GammaError::InvalidSpec(format!("Up node {} has no inputs", up_name)))?;

        if gate_base != up_base {
            debug!(
                "FFN down zonotope: gate_base='{}' != up_base='{}', skipping",
                gate_base, up_base
            );
            return Ok(None);
        }

        // Get base bounds (ffn_norm output)
        let base_bounds = self.get_bounds_for_block(gate_base, block_input, bounds_cache)?;
        if base_bounds.shape().len() != 2 {
            return Ok(None);
        }

        let base_width = base_bounds.max_width();

        // Normalize bounds to prevent overflow in quadratic cross-terms.
        // Include spectral norms of all three weight matrices (up, gate, down)
        // to prevent coefficient explosion in the multiplication.
        let max_spectral = gate_linear
            .spectral_norm()
            .max(up_linear.spectral_norm())
            .max(down_linear.spectral_norm());
        let zonotope_scale = if base_width > 1.0 || max_spectral > 1.0 {
            (base_width / 2.0).max(1.0) * max_spectral.max(1.0)
        } else {
            1.0
        };
        let normalized_bounds = if zonotope_scale > 1.0 {
            match BoundedTensor::new(
                base_bounds.lower.mapv(|v| v / zonotope_scale),
                base_bounds.upper.mapv(|v| v / zonotope_scale),
            ) {
                Ok(b) => b,
                Err(_) => return Ok(None),
            }
        } else {
            base_bounds.clone()
        };

        // Create zonotope from normalized bounds
        let base_z = match ZonotopeTensor::from_bounded_tensor_per_position_2d(&normalized_bounds) {
            Ok(z) => z,
            Err(_) => return Ok(None),
        };

        debug!(
            "FFN down zonotope: base='{}' base_width={:.3e} max_spectral={:.1} scale={:.3e} n_err={}",
            gate_base, base_width, max_spectral, zonotope_scale, base_z.n_error_terms
        );

        // Apply gate Linear projection
        let gate_z = match base_z.linear(&gate_linear.weight, gate_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(e) => {
                debug!("FFN down zonotope: gate linear failed: {}", e);
                return Ok(None);
            }
        };

        // Apply SiLU to gate
        let silu_z = match gate_z.silu_affine() {
            Ok(z) => z,
            Err(e) => {
                debug!("FFN down zonotope: silu_affine failed: {}", e);
                return Ok(None);
            }
        };

        // Apply up Linear projection
        let up_z = match base_z.linear(&up_linear.weight, up_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(e) => {
                debug!("FFN down zonotope: up linear failed: {}", e);
                return Ok(None);
            }
        };

        // Multiply: up * silu(gate)
        let swiglu_z = match up_z.mul_elementwise(&silu_z) {
            Ok(z) => z,
            Err(e) => {
                debug!("FFN down zonotope: mul_elementwise failed: {}", e);
                return Ok(None);
            }
        };

        // Apply down Linear projection (the key extension!)
        let down_z = match swiglu_z.linear(&down_linear.weight, down_linear.bias.as_ref()) {
            Ok(z) => z,
            Err(e) => {
                debug!("FFN down zonotope: down linear failed: {}", e);
                return Ok(None);
            }
        };

        let result = down_z.to_bounded_tensor();

        // Scale back by zonotope_scale² (multiplication is quadratic in scale)
        let result = if zonotope_scale > 1.0 {
            let scale_sq = (zonotope_scale as f64) * (zonotope_scale as f64);
            match BoundedTensor::new(
                result.lower.mapv(|v| (v as f64 * scale_sq) as f32),
                result.upper.mapv(|v| (v as f64 * scale_sq) as f32),
            ) {
                Ok(b) => b,
                Err(_) => return Ok(None),
            }
        } else {
            result
        };

        let result_width = result.max_width();

        debug!(
            "FFN down zonotope output: width={:.3e} scale²={:.3e} shape={:?}",
            result_width,
            zonotope_scale * zonotope_scale,
            result.shape()
        );

        // Validate output
        let has_bad_values = result
            .lower
            .iter()
            .chain(result.upper.iter())
            .any(|v| v.is_nan() || v.is_infinite());

        if has_bad_values {
            debug!("FFN down zonotope: output has NaN/Inf, falling back to IBP");
            return Ok(None);
        }

        Ok(Some(result))
    }

    /// Convert a sequential Network to a GraphNetwork.
    ///
    /// Creates a linear chain of nodes: input -> layer0 -> layer1 -> ... -> output
    pub fn from_sequential(network: &Network) -> Self {
        let mut graph = GraphNetwork::new();

        for (i, layer) in network.layers.iter().enumerate() {
            let name = format!("layer_{}", i);
            let input_name = if i == 0 {
                "_input".to_string()
            } else {
                format!("layer_{}", i - 1)
            };

            let node = GraphNode::new(name.clone(), layer.clone(), vec![input_name]);
            graph.add_node(node);

            if i == network.layers.len() - 1 {
                graph.set_output(name);
            }
        }

        graph
    }

    /// Propagate CROWN bounds through the graph, treating each position independently.
    ///
    /// For N-D inputs where the last dimension is the feature dimension, this runs
    /// CROWN separately on each position (flattened batch dimensions) and combines
    /// the results. This is useful for transformer MLPs which operate independently
    /// on each position.
    ///
    /// # Arguments
    /// * `input` - Bounded tensor of shape [...batch_dims..., hidden_dim]
    ///
    /// # Returns
    /// * Bounded tensor of shape [...batch_dims..., output_dim]
    ///
    /// # Algorithm
    /// 1. Flatten input from [...batch, hidden] to [num_positions, hidden]
    /// 2. For each position, extract \[hidden\] slice and run CROWN
    /// 3. Stack outputs and reshape to [...batch, output_dim]
    #[inline]
    pub fn propagate_crown_per_position(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        // For 1-D input, just use regular CROWN
        if ndim == 1 {
            return self.propagate_crown(input);
        }

        // Extract batch dimensions and hidden dimension
        let hidden_dim = shape[ndim - 1];
        let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
        let num_positions: usize = batch_shape.iter().product();

        debug!(
            "Per-position CROWN: {} positions x {} hidden, batch shape {:?}",
            num_positions, hidden_dim, batch_shape
        );

        // Flatten input to [num_positions, hidden_dim]
        // Make arrays contiguous first to avoid reshape failures due to memory layout
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
        let first_output = self.propagate_crown(&first_input)?;
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
                    GammaError::shape_mismatch(
                        vec![output_dim],
                        first_output.lower.shape().to_vec(),
                    )
                })?;
            let first_out_upper = first_output
                .upper
                .clone()
                .into_shape_with_order((output_dim,))
                .map_err(|_| {
                    GammaError::shape_mismatch(
                        vec![output_dim],
                        first_output.upper.shape().to_vec(),
                    )
                })?;
            out_lower.row_mut(0).assign(&first_out_lower);
            out_upper.row_mut(0).assign(&first_out_upper);
        }

        // Process remaining positions
        for pos in 1..num_positions {
            let pos_lower = flat_lower.row(pos).to_owned().into_dyn();
            let pos_upper = flat_upper.row(pos).to_owned().into_dyn();
            let pos_input = BoundedTensor::new(pos_lower, pos_upper)?;

            let pos_output = self.propagate_crown(&pos_input)?;

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
}

impl Default for GraphNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing attention-like graph patterns.
///
/// Provides convenient methods for building common attention patterns
/// without manually creating each node.
#[derive(Debug)]
pub struct AttentionGraphBuilder {
    graph: GraphNetwork,
    node_counter: usize,
}

impl AttentionGraphBuilder {
    /// Create a new attention graph builder.
    pub fn new() -> Self {
        Self {
            graph: GraphNetwork::new(),
            node_counter: 0,
        }
    }

    /// Generate a unique node name.
    fn next_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.node_counter);
        self.node_counter += 1;
        name
    }

    /// Add a linear projection from network input.
    pub fn add_projection(
        &mut self,
        name: impl Into<String>,
        weight: Array2<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<String> {
        let name = name.into();
        let layer = Layer::Linear(LinearLayer::new(weight, bias)?);
        self.graph.add_node(GraphNode::from_input(&name, layer));
        Ok(name)
    }

    /// Add a linear projection from a specific node.
    pub fn add_projection_from(
        &mut self,
        name: impl Into<String>,
        input: &str,
        weight: Array2<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<String> {
        let name = name.into();
        let layer = Layer::Linear(LinearLayer::new(weight, bias)?);
        self.graph
            .add_node(GraphNode::new(&name, layer, vec![input.to_string()]));
        Ok(name)
    }

    /// Add a bounded matrix multiplication (Q @ K^T pattern).
    pub fn add_matmul(
        &mut self,
        input_a: &str,
        input_b: &str,
        transpose_b: bool,
        scale: Option<f32>,
    ) -> String {
        let name = self.next_name("matmul");
        let layer = Layer::MatMul(MatMulLayer::new(transpose_b, scale));
        self.graph
            .add_node(GraphNode::binary(&name, layer, input_a, input_b));
        name
    }

    /// Add a softmax operation.
    pub fn add_softmax(&mut self, input: &str, axis: i32) -> String {
        let name = self.next_name("softmax");
        let layer = Layer::Softmax(SoftmaxLayer::new(axis));
        self.graph
            .add_node(GraphNode::new(&name, layer, vec![input.to_string()]));
        name
    }

    /// Add element-wise addition (residual connection).
    pub fn add_residual(&mut self, input_a: &str, input_b: &str) -> String {
        let name = self.next_name("add");
        let layer = Layer::Add(AddLayer);
        self.graph
            .add_node(GraphNode::binary(&name, layer, input_a, input_b));
        name
    }

    /// Add a ReLU activation.
    pub fn add_relu(&mut self, input: &str) -> String {
        let name = self.next_name("relu");
        let layer = Layer::ReLU(ReLULayer);
        self.graph
            .add_node(GraphNode::new(&name, layer, vec![input.to_string()]));
        name
    }

    /// Add a GELU activation.
    pub fn add_gelu(&mut self, input: &str) -> String {
        let name = self.next_name("gelu");
        let layer = Layer::GELU(GELULayer::default());
        self.graph
            .add_node(GraphNode::new(&name, layer, vec![input.to_string()]));
        name
    }

    /// Add a LayerNorm operation.
    pub fn add_layer_norm(
        &mut self,
        input: &str,
        gamma: Array1<f32>,
        beta: Array1<f32>,
        eps: f32,
    ) -> String {
        let name = self.next_name("layernorm");
        let layer = Layer::LayerNorm(LayerNormLayer::new(gamma, beta, eps));
        self.graph
            .add_node(GraphNode::new(&name, layer, vec![input.to_string()]));
        name
    }

    /// Set the output node and return the built graph.
    pub fn build(mut self, output: &str) -> GraphNetwork {
        self.graph.set_output(output);
        self.graph
    }

    /// Get reference to the graph being built.
    pub fn graph(&self) -> &GraphNetwork {
        &self.graph
    }
}

impl Default for AttentionGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// IBP propagation for ReLU activation.
pub fn relu_ibp(input: &BoundedTensor) -> BoundedTensor {
    BoundedTensor {
        lower: input.lower.mapv(|v| v.max(0.0)),
        upper: input.upper.mapv(|v| v.max(0.0)),
    }
}

/// CROWN linear relaxation for ReLU.
///
/// For x in [l, u]:
/// - If l >= 0: ReLU(x) = x (pass-through)
/// - If u <= 0: ReLU(x) = 0 (zero)
/// - If l < 0 < u: Use linear relaxation
pub fn relu_crown_relaxation(lower: f32, upper: f32) -> (f32, f32, f32, f32) {
    // Returns (lower_slope, lower_intercept, upper_slope, upper_intercept)
    if lower >= 0.0 {
        // Positive region: identity
        (1.0, 0.0, 1.0, 0.0)
    } else if upper <= 0.0 {
        // Negative region: zero
        (0.0, 0.0, 0.0, 0.0)
    } else {
        // Crossing region: linear relaxation
        let upper_slope = upper / (upper - lower);
        let upper_intercept = -lower * upper_slope;

        // For lower bound, α-CROWN optimizes this; default to zero or identity
        // depending on which gives tighter bound
        let lower_slope = if upper > -lower { 1.0 } else { 0.0 };
        let lower_intercept = 0.0;

        (lower_slope, lower_intercept, upper_slope, upper_intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, array};

    // ==================== broadcast_shapes tests ====================

    #[test]
    fn test_broadcast_shapes_identical() {
        assert_eq!(
            broadcast_shapes(&[2, 3, 4], &[2, 3, 4]),
            Some(vec![2, 3, 4])
        );
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        // Scalar broadcasts to any shape
        assert_eq!(broadcast_shapes(&[], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[3, 4], &[]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_ones() {
        // Broadcasting with 1 dimensions
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[3, 1], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[1, 1], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_different_ndim() {
        // Different number of dimensions
        assert_eq!(broadcast_shapes(&[4], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[4]), Some(vec![2, 3, 4]));
        assert_eq!(
            broadcast_shapes(&[1, 3, 1], &[2, 1, 4]),
            Some(vec![2, 3, 4])
        );
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // Incompatible shapes
        assert_eq!(broadcast_shapes(&[3], &[4]), None);
        assert_eq!(broadcast_shapes(&[2, 3], &[4, 3]), None);
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[2, 5, 4]), None);
    }

    #[test]
    fn test_broadcast_shapes_batch_broadcast_pattern() {
        // Common batch broadcasting pattern
        assert_eq!(
            broadcast_shapes(&[32, 1, 64], &[1, 8, 64]),
            Some(vec![32, 8, 64])
        );
        // Another common pattern: outer product style
        assert_eq!(
            broadcast_shapes(&[32, 8, 1], &[1, 1, 64]),
            Some(vec![32, 8, 64])
        );
    }

    // ==================== relu_ibp tests ====================

    #[test]
    fn test_relu_ibp_positive() {
        // All positive: output = input
        let input = BoundedTensor::new(
            array![1.0, 2.0, 3.0].into_dyn(),
            array![2.0, 3.0, 4.0].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input);
        assert_eq!(output.lower, array![1.0, 2.0, 3.0].into_dyn());
        assert_eq!(output.upper, array![2.0, 3.0, 4.0].into_dyn());
    }

    #[test]
    fn test_relu_ibp_negative() {
        // All negative: output = 0
        let input = BoundedTensor::new(
            array![-3.0, -2.0, -1.0].into_dyn(),
            array![-2.0, -1.0, -0.5].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input);
        assert_eq!(output.lower, array![0.0, 0.0, 0.0].into_dyn());
        assert_eq!(output.upper, array![0.0, 0.0, 0.0].into_dyn());
    }

    #[test]
    fn test_relu_ibp_crossing() {
        // Crossing zero: lower clamped to 0, upper stays
        let input = BoundedTensor::new(
            array![-1.0, -2.0, 0.0].into_dyn(),
            array![1.0, 3.0, 2.0].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input);
        assert_eq!(output.lower, array![0.0, 0.0, 0.0].into_dyn());
        assert_eq!(output.upper, array![1.0, 3.0, 2.0].into_dyn());
    }

    #[test]
    fn test_relu_ibp_mixed() {
        // Mixed: some positive, some negative, some crossing
        let input = BoundedTensor::new(
            array![1.0, -3.0, -1.0].into_dyn(),
            array![2.0, -1.0, 1.0].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input);
        assert_eq!(output.lower, array![1.0, 0.0, 0.0].into_dyn());
        assert_eq!(output.upper, array![2.0, 0.0, 1.0].into_dyn());
    }

    // ==================== relu_crown_relaxation tests ====================

    #[test]
    fn test_relu_crown_positive_region() {
        // l >= 0: identity (slope=1, intercept=0)
        let (ls, li, us, ui) = relu_crown_relaxation(0.5, 2.0);
        assert_eq!((ls, li, us, ui), (1.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_relu_crown_negative_region() {
        // u <= 0: zero (slope=0, intercept=0)
        let (ls, li, us, ui) = relu_crown_relaxation(-2.0, -0.5);
        assert_eq!((ls, li, us, ui), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_relu_crown_crossing_upper_dominant() {
        // l < 0 < u, u > |l|: lower slope = 1
        let (ls, li, us, ui) = relu_crown_relaxation(-1.0, 3.0);
        // upper_slope = u / (u - l) = 3 / 4 = 0.75
        // upper_intercept = -l * upper_slope = 1 * 0.75 = 0.75
        // lower_slope = 1 (since u > -l)
        assert_eq!(ls, 1.0);
        assert_eq!(li, 0.0);
        assert!((us - 0.75).abs() < 1e-6);
        assert!((ui - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_relu_crown_crossing_lower_dominant() {
        // l < 0 < u, |l| > u: lower slope = 0
        let (ls, li, us, ui) = relu_crown_relaxation(-3.0, 1.0);
        // upper_slope = u / (u - l) = 1 / 4 = 0.25
        // upper_intercept = -l * upper_slope = 3 * 0.25 = 0.75
        // lower_slope = 0 (since u < -l)
        assert_eq!(ls, 0.0);
        assert_eq!(li, 0.0);
        assert!((us - 0.25).abs() < 1e-6);
        assert!((ui - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_relu_crown_boundary_zero_lower() {
        // l = 0: positive region
        let (ls, li, us, ui) = relu_crown_relaxation(0.0, 1.0);
        assert_eq!((ls, li, us, ui), (1.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_relu_crown_boundary_zero_upper() {
        // u = 0: negative region
        let (ls, li, us, ui) = relu_crown_relaxation(-1.0, 0.0);
        assert_eq!((ls, li, us, ui), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_propagate_crown_to_node_add_splits_bias() {
        // Regression test: in backward CROWN-to-node, Add must split the bias term;
        // otherwise downstream constants get double-counted across both branches.

        // Two-branch linear -> add -> linear(with bias) -> output
        let wa = arr2(&[[1.0, 2.0], [-3.0, 0.5]]);
        let ba = arr1(&[0.1, -0.2]);
        let wb = arr2(&[[0.3, -0.7], [1.2, -1.0]]);
        let bb = arr1(&[0.0, 0.05]);
        let wout = arr2(&[[2.0, -1.0]]);
        let bout = arr1(&[0.7]);

        let lin_a = LinearLayer::new(wa.clone(), Some(ba.clone())).unwrap();
        let lin_b = LinearLayer::new(wb.clone(), Some(bb.clone())).unwrap();
        let lin_out = LinearLayer::new(wout.clone(), Some(bout.clone())).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "lin_a",
            Layer::Linear(lin_a),
            vec!["_input".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "lin_b",
            Layer::Linear(lin_b),
            vec!["_input".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "lin_a",
            "lin_b",
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(lin_out),
            vec!["add".to_string()],
        ));
        graph.set_output("out");

        let input =
            BoundedTensor::new(array![-1.0, -2.0].into_dyn(), array![3.0, 4.0].into_dyn()).unwrap();

        let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
        let crown = graph
            .propagate_crown_to_node(
                &input,
                "out",
                &std::collections::HashMap::new(),
                &ibp_bounds,
            )
            .unwrap();

        // Expected exact bounds since the full graph is linear.
        let wsum = &wa + &wb; // 2x2
        let bsum = &ba + &bb; // 2
        let combined_w = wout.dot(&wsum); // 1x2
        let combined_bias = wout.row(0).dot(&bsum) + bout[0];

        let l = [-1.0_f32, -2.0_f32];
        let u = [3.0_f32, 4.0_f32];
        let w0 = combined_w[[0, 0]];
        let w1 = combined_w[[0, 1]];

        let mut expected_lower = combined_bias;
        let mut expected_upper = combined_bias;
        for (w, (li, ui)) in [(w0, (l[0], u[0])), (w1, (l[1], u[1]))] {
            if w >= 0.0 {
                expected_lower += w * li;
                expected_upper += w * ui;
            } else {
                expected_lower += w * ui;
                expected_upper += w * li;
            }
        }

        let got_lower = crown.lower[[0]];
        let got_upper = crown.upper[[0]];
        assert!((got_lower - expected_lower).abs() < 1e-4);
        assert!((got_upper - expected_upper).abs() < 1e-4);
    }

    // ==================== Network tests ====================

    #[test]
    fn test_network_new_empty() {
        let net = Network::new();
        assert_eq!(net.layers.len(), 0);
        assert_eq!(net.num_layers(), 0);
    }

    #[test]
    fn test_network_add_layer() {
        let mut net = Network::new();
        net.add_layer(Layer::ReLU(ReLULayer));
        assert_eq!(net.layers.len(), 1);
        assert_eq!(net.num_layers(), 1);
    }

    #[test]
    fn test_network_set_layernorm_forward_mode() {
        let mut net = Network::new();
        net.add_layer(Layer::ReLU(ReLULayer));

        // No LayerNorm layers
        let count = net.set_layernorm_forward_mode(true);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_network_propagate_ibp_empty() {
        let net = Network::new();
        let input =
            BoundedTensor::new(array![1.0, 2.0].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();
        let output = net.propagate_ibp(&input).unwrap();
        // Empty network returns input unchanged
        assert_eq!(output.lower, input.lower);
        assert_eq!(output.upper, input.upper);
    }

    #[test]
    fn test_network_propagate_ibp_relu() {
        let mut net = Network::new();
        net.add_layer(Layer::ReLU(ReLULayer));

        let input =
            BoundedTensor::new(array![-1.0, 1.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();
        let output = net.propagate_ibp(&input).unwrap();
        assert_eq!(output.lower, array![0.0, 1.0].into_dyn());
        assert_eq!(output.upper, array![1.0, 2.0].into_dyn());
    }

    #[test]
    fn test_network_collect_ibp_bounds_empty() {
        let net = Network::new();
        let input = BoundedTensor::new(array![1.0].into_dyn(), array![2.0].into_dyn()).unwrap();
        let bounds = net.collect_ibp_bounds(&input).unwrap();
        assert_eq!(bounds.len(), 0);
    }

    #[test]
    fn test_network_collect_ibp_bounds_single() {
        let mut net = Network::new();
        net.add_layer(Layer::ReLU(ReLULayer));

        let input =
            BoundedTensor::new(array![-1.0, 2.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
        let bounds = net.collect_ibp_bounds(&input).unwrap();
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].lower, array![0.0, 2.0].into_dyn());
        assert_eq!(bounds[0].upper, array![1.0, 3.0].into_dyn());
    }

    #[test]
    fn test_network_collect_ibp_bounds_chain() {
        let mut net = Network::new();
        net.add_layer(Layer::ReLU(ReLULayer));
        net.add_layer(Layer::ReLU(ReLULayer));

        let input =
            BoundedTensor::new(array![-1.0, 2.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
        let bounds = net.collect_ibp_bounds(&input).unwrap();
        assert_eq!(bounds.len(), 2);
        // First ReLU
        assert_eq!(bounds[0].lower, array![0.0, 2.0].into_dyn());
        assert_eq!(bounds[0].upper, array![1.0, 3.0].into_dyn());
        // Second ReLU (no change since already non-negative lower)
        assert_eq!(bounds[1].lower, array![0.0, 2.0].into_dyn());
        assert_eq!(bounds[1].upper, array![1.0, 3.0].into_dyn());
    }

    // ==================== GraphNetwork tests ====================

    #[test]
    fn test_graphnetwork_new() {
        let graph = GraphNetwork::new();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.node_names().len(), 0);
        assert_eq!(graph.get_output_name(), "");
    }

    #[test]
    fn test_graphnetwork_add_node() {
        let mut graph = GraphNetwork::new();
        let node = GraphNode::from_input("input", Layer::ReLU(ReLULayer));
        graph.add_node(node);

        assert_eq!(graph.num_nodes(), 1);
        assert!(graph.get_node("input").is_some());
    }

    #[test]
    fn test_graphnetwork_set_output() {
        let mut graph = GraphNetwork::new();
        graph.set_output("output");
        assert_eq!(graph.get_output_name(), "output");
    }

    #[test]
    fn test_graphnetwork_topological_sort_linear() {
        // Linear chain: A -> B -> C
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("A", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "B",
            Layer::ReLU(ReLULayer),
            vec!["A".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "C",
            Layer::ReLU(ReLULayer),
            vec!["B".to_string()],
        ));
        graph.set_output("C");

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_graphnetwork_topological_sort_diamond() {
        // Diamond: A -> B, A -> C, B -> D, C -> D
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("A", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "B",
            Layer::ReLU(ReLULayer),
            vec!["A".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "C",
            Layer::ReLU(ReLULayer),
            vec!["A".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "D",
            Layer::ReLU(ReLULayer),
            vec!["B".to_string(), "C".to_string()],
        ));
        graph.set_output("D");

        let sorted = graph.topological_sort().unwrap();
        // A must come first, D must come last, B and C can be in either order
        assert_eq!(sorted[0], "A");
        assert_eq!(sorted[3], "D");
        assert!(sorted[1] == "B" || sorted[1] == "C");
        assert!(sorted[2] == "B" || sorted[2] == "C");
        assert_ne!(sorted[1], sorted[2]);
    }

    #[test]
    fn test_graphnetwork_propagate_ibp_single_node() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");

        let input =
            BoundedTensor::new(array![-1.0, 1.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();
        let output = graph.propagate_ibp(&input).unwrap();
        assert_eq!(output.lower, array![0.0, 1.0].into_dyn());
        assert_eq!(output.upper, array![1.0, 2.0].into_dyn());
    }

    #[test]
    fn test_graphnetwork_propagate_ibp_chain() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["relu1".to_string()],
        ));
        graph.set_output("relu2");

        let input =
            BoundedTensor::new(array![-2.0, 1.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
        let output = graph.propagate_ibp(&input).unwrap();
        // After two ReLUs, result should be same as one ReLU on this input
        assert_eq!(output.lower, array![0.0, 1.0].into_dyn());
        assert_eq!(output.upper, array![1.0, 3.0].into_dyn());
    }

    // ==================== GraphNode tests ====================

    #[test]
    fn test_graphnode_from_input() {
        let node = GraphNode::from_input("input_node", Layer::ReLU(ReLULayer));
        assert_eq!(node.name, "input_node");
        // from_input sets "_input" as the input source
        assert_eq!(node.inputs, vec!["_input"]);
    }

    #[test]
    fn test_graphnode_new_with_inputs() {
        let node = GraphNode::new(
            "compute",
            Layer::ReLU(ReLULayer),
            vec!["a".to_string(), "b".to_string()],
        );
        assert_eq!(node.name, "compute");
        assert_eq!(node.inputs, vec!["a", "b"]);
    }

    // ==================== AttentionGraphBuilder tests ====================

    #[test]
    fn test_attention_graph_builder_new() {
        let builder = AttentionGraphBuilder::new();
        assert_eq!(builder.graph().num_nodes(), 0);
    }

    #[test]
    fn test_attention_graph_builder_add_relu() {
        let mut builder = AttentionGraphBuilder::new();
        // First add an input node
        builder
            .graph
            .add_node(GraphNode::from_input("input", Layer::ReLU(ReLULayer)));

        let output_name = builder.add_relu("input");
        assert!(output_name.contains("relu"));
        assert!(builder.graph().get_node(&output_name).is_some());
    }

    #[test]
    fn test_attention_graph_builder_add_gelu() {
        let mut builder = AttentionGraphBuilder::new();
        builder
            .graph
            .add_node(GraphNode::from_input("input", Layer::ReLU(ReLULayer)));

        let output_name = builder.add_gelu("input");
        assert!(output_name.contains("gelu"));
        assert!(builder.graph().get_node(&output_name).is_some());
    }

    #[test]
    fn test_attention_graph_builder_add_softmax() {
        let mut builder = AttentionGraphBuilder::new();
        builder
            .graph
            .add_node(GraphNode::from_input("input", Layer::ReLU(ReLULayer)));

        let output_name = builder.add_softmax("input", -1);
        assert!(output_name.contains("softmax"));
        assert!(builder.graph().get_node(&output_name).is_some());
    }

    #[test]
    fn test_attention_graph_builder_add_residual() {
        let mut builder = AttentionGraphBuilder::new();
        builder
            .graph
            .add_node(GraphNode::from_input("a", Layer::ReLU(ReLULayer)));
        builder
            .graph
            .add_node(GraphNode::from_input("b", Layer::ReLU(ReLULayer)));

        let output_name = builder.add_residual("a", "b");
        assert!(output_name.contains("add"));
        assert!(builder.graph().get_node(&output_name).is_some());
    }

    #[test]
    fn test_attention_graph_builder_build() {
        let mut builder = AttentionGraphBuilder::new();
        builder
            .graph
            .add_node(GraphNode::from_input("input", Layer::ReLU(ReLULayer)));
        let relu_out = builder.add_relu("input");

        let graph = builder.build(&relu_out);
        assert_eq!(graph.get_output_name(), relu_out);
    }
}
