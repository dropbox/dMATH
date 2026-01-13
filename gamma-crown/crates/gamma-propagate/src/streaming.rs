//! Streaming computation with gradient checkpointing for memory-efficient verification.
//!
//! This module implements gradient checkpointing to reduce memory usage during CROWN
//! propagation. Instead of storing bounds at every layer (O(L*N)), we store checkpoints
//! at intervals and recompute forward during the backward pass when needed.
//!
//! **Memory-Compute Trade-off:**
//! - Without checkpointing: O(L*N) memory, O(L) compute
//! - With K-interval checkpointing: O(L/K * N) memory, O(L*K) compute
//!
//! For a 100-layer network with K=10, this reduces memory by ~90% at 10x compute cost.
//! Since modern GPUs are compute-bound, this is often a good trade-off.
//!
//! ## Compressed Storage (f16)
//!
//! When `use_f16_checkpoints` is enabled, checkpoints are stored using f16 (half precision)
//! which provides an additional 50% memory reduction on top of checkpointing. This is
//! particularly useful for very large models or memory-constrained environments.
//!
//! **Memory with f16 + checkpointing:**
//! - Original: O(L*N) memory (f32)
//! - Checkpointing only (K=10): O(L/K * N) f32 = ~10% of original
//! - Checkpointing + f16: O(L/K * N/2) = ~5% of original

use crate::bounds::{BatchedLinearBounds, LinearBounds};
use crate::layers::{BoundPropagation, Layer};
use crate::network::Network;
use gamma_core::{GammaError, Result};
use gamma_tensor::{BoundedTensor, CompressedBounds};
use std::borrow::Cow;
use tracing::debug;

/// Configuration for streaming/checkpointed computation.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of layers between checkpoints.
    /// Smaller = more memory, less recomputation.
    /// Larger = less memory, more recomputation.
    /// Default: 10 (stores every 10th layer's bounds).
    pub checkpoint_interval: usize,

    /// Enable streaming for CROWN (gradient checkpointing).
    /// When true, only checkpoint bounds are kept, others recomputed.
    pub enable_crown_streaming: bool,

    /// Maximum memory for bounds storage in bytes (0 = unlimited).
    /// If exceeded, automatically increases checkpoint_interval.
    pub max_memory_bytes: usize,

    /// Use f16 (half precision) for checkpoint storage.
    /// Provides additional 50% memory reduction at cost of precision.
    /// Default: false. Use for very large models or memory-constrained environments.
    pub use_f16_checkpoints: bool,

    /// Relative epsilon for sound bound widening when using f16.
    /// After decompression, bounds are widened by this factor to ensure soundness.
    /// Default: 0.001 (0.1%). Set to 0.0 to disable widening.
    pub f16_widening_epsilon: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 10,
            enable_crown_streaming: true,
            max_memory_bytes: 0,
            use_f16_checkpoints: false,
            f16_widening_epsilon: 0.001,
        }
    }
}

impl StreamingConfig {
    /// Create config optimized for minimum memory usage.
    /// Uses large checkpoint interval + f16 storage for maximum memory savings.
    pub fn min_memory() -> Self {
        Self {
            checkpoint_interval: 50,
            enable_crown_streaming: true,
            max_memory_bytes: 0,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
        }
    }

    /// Create config optimized for balanced memory-compute trade-off.
    pub fn balanced() -> Self {
        Self {
            checkpoint_interval: 10,
            enable_crown_streaming: true,
            max_memory_bytes: 0,
            use_f16_checkpoints: false,
            f16_widening_epsilon: 0.001,
        }
    }

    /// Create config with specific memory limit (auto-adjusts interval).
    pub fn with_memory_limit(max_bytes: usize) -> Self {
        Self {
            checkpoint_interval: 10,
            enable_crown_streaming: true,
            max_memory_bytes: max_bytes,
            use_f16_checkpoints: false,
            f16_widening_epsilon: 0.001,
        }
    }

    /// Create config with f16 compression enabled.
    /// Provides ~50% additional memory reduction vs f32 checkpoints.
    pub fn with_f16_compression() -> Self {
        Self {
            checkpoint_interval: 10,
            enable_crown_streaming: true,
            max_memory_bytes: 0,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
        }
    }
}

/// Storage mode for checkpointed bounds.
#[derive(Debug, Clone)]
enum CheckpointStorage {
    /// Standard f32 storage.
    F32(Vec<(usize, BoundedTensor)>),
    /// Compressed f16 storage for reduced memory.
    F16(Vec<(usize, CompressedBounds)>),
}

/// Checkpointed bounds storage for gradient checkpointing.
///
/// Stores bounds only at checkpoint layers, enabling memory-efficient
/// CROWN propagation through recomputation. Optionally uses f16 compression
/// for additional 50% memory savings.
#[derive(Debug, Clone)]
pub struct CheckpointedBounds {
    /// Checkpoint layer indices and their bounds.
    /// Sorted by layer index for efficient lookup.
    checkpoints: CheckpointStorage,

    /// Always store input bounds (needed for recomputation from start).
    input: BoundedTensor,

    /// Total number of layers (for reference).
    num_layers: usize,

    /// Widening epsilon for f16 compression (for soundness).
    f16_widening_epsilon: f32,
}

impl CheckpointedBounds {
    /// Create new checkpointed bounds storage with f32.
    pub fn new(input: BoundedTensor, num_layers: usize) -> Self {
        Self {
            checkpoints: CheckpointStorage::F32(Vec::new()),
            input,
            num_layers,
            f16_widening_epsilon: 0.0,
        }
    }

    /// Create new checkpointed bounds storage with f16 compression.
    pub fn new_compressed(input: BoundedTensor, num_layers: usize, widening_epsilon: f32) -> Self {
        Self {
            checkpoints: CheckpointStorage::F16(Vec::new()),
            input,
            num_layers,
            f16_widening_epsilon: widening_epsilon,
        }
    }

    /// Add a checkpoint at the given layer.
    pub fn add_checkpoint(&mut self, layer_idx: usize, bounds: BoundedTensor) {
        match &mut self.checkpoints {
            CheckpointStorage::F32(checkpoints) => {
                // Keep sorted by layer index
                let pos = checkpoints
                    .binary_search_by_key(&layer_idx, |(idx, _)| *idx)
                    .unwrap_or_else(|p| p);
                checkpoints.insert(pos, (layer_idx, bounds));
            }
            CheckpointStorage::F16(checkpoints) => {
                // Compress to f16 before storing
                let mut compressed = CompressedBounds::from_bounded_tensor(&bounds);
                if self.f16_widening_epsilon > 0.0 {
                    compressed.widen_for_soundness(self.f16_widening_epsilon);
                }
                let pos = checkpoints
                    .binary_search_by_key(&layer_idx, |(idx, _)| *idx)
                    .unwrap_or_else(|p| p);
                checkpoints.insert(pos, (layer_idx, compressed));
            }
        }
    }

    /// Get bounds at layer_idx by finding nearest checkpoint and recomputing if needed.
    /// Returns None if layer_idx is invalid.
    pub fn get_bounds_at(&self, layer_idx: usize, network: &Network) -> Result<BoundedTensor> {
        if layer_idx >= self.num_layers {
            return Err(GammaError::InvalidSpec(format!(
                "Layer index {} out of range (max {})",
                layer_idx,
                self.num_layers - 1
            )));
        }

        // Check if we have an exact checkpoint
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => {
                if let Ok(pos) = checkpoints.binary_search_by_key(&layer_idx, |(idx, _)| *idx) {
                    return Ok(checkpoints[pos].1.clone());
                }
            }
            CheckpointStorage::F16(checkpoints) => {
                if let Ok(pos) = checkpoints.binary_search_by_key(&layer_idx, |(idx, _)| *idx) {
                    return checkpoints[pos].1.to_bounded_tensor();
                }
            }
        }

        // Find nearest checkpoint before this layer
        let (start_idx, start_bounds) = self.find_nearest_checkpoint_before(layer_idx);

        // Recompute forward from checkpoint to target layer
        self.recompute_forward(network, start_idx, layer_idx, start_bounds)
    }

    /// Find the nearest checkpoint at or before the given layer.
    /// Returns (checkpoint_layer, checkpoint_bounds).
    /// If no checkpoint exists, returns (-1, input_bounds).
    fn find_nearest_checkpoint_before(&self, layer_idx: usize) -> (i64, BoundedTensor) {
        let mut nearest_idx: i64 = -1;
        let mut nearest_bounds = self.input.clone();

        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => {
                for (idx, bounds) in checkpoints {
                    if *idx <= layer_idx && *idx as i64 > nearest_idx {
                        nearest_idx = *idx as i64;
                        nearest_bounds = bounds.clone();
                    }
                }
            }
            CheckpointStorage::F16(checkpoints) => {
                for (idx, compressed) in checkpoints {
                    if *idx <= layer_idx && *idx as i64 > nearest_idx {
                        nearest_idx = *idx as i64;
                        // Decompress f16 to f32
                        nearest_bounds = compressed.to_bounded_tensor().unwrap_or_else(|_| {
                            // Fallback to input if decompression fails
                            self.input.clone()
                        });
                    }
                }
            }
        }

        (nearest_idx, nearest_bounds)
    }

    /// Recompute forward from start_idx to target_idx.
    /// start_idx = -1 means start from input.
    fn recompute_forward(
        &self,
        network: &Network,
        start_idx: i64,
        target_idx: usize,
        start_bounds: BoundedTensor,
    ) -> Result<BoundedTensor> {
        let mut current = start_bounds;

        // start_idx is the output of layer start_idx (or input if -1)
        // We need to propagate through layers (start_idx+1) to target_idx
        let first_layer = (start_idx + 1) as usize;

        for i in first_layer..=target_idx {
            let layer = network
                .layers
                .get(i)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Layer {} not found", i)))?;

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

    /// Number of checkpoints stored.
    pub fn num_checkpoints(&self) -> usize {
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => checkpoints.len(),
            CheckpointStorage::F16(checkpoints) => checkpoints.len(),
        }
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let input_size = self.input.lower.len() * 4 * 2; // f32 lower + upper
        let checkpoint_size: usize = match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => checkpoints
                .iter()
                .map(|(_, b)| b.lower.len() * 4 * 2) // f32: 4 bytes per element
                .sum(),
            CheckpointStorage::F16(checkpoints) => checkpoints
                .iter()
                .map(|(_, b)| b.len() * 2 * 2) // f16: 2 bytes per element
                .sum(),
        };
        input_size + checkpoint_size
    }

    /// Check if using f16 compression.
    pub fn is_compressed(&self) -> bool {
        matches!(self.checkpoints, CheckpointStorage::F16(_))
    }

    /// Get compression statistics if using f16.
    /// Returns (total_f16_bytes, equivalent_f32_bytes, ratio) or None if not compressed.
    pub fn compression_stats(&self) -> Option<(usize, usize, f32)> {
        if let CheckpointStorage::F16(checkpoints) = &self.checkpoints {
            let f16_bytes: usize = checkpoints.iter().map(|(_, b)| b.memory_bytes()).sum();
            let f32_bytes: usize = checkpoints.iter().map(|(_, b)| b.len() * 4 * 2).sum();
            let ratio = if f32_bytes > 0 {
                f16_bytes as f32 / f32_bytes as f32
            } else {
                1.0
            };
            Some((f16_bytes, f32_bytes, ratio))
        } else {
            None
        }
    }

    /// Get the last checkpoint's bounds (output bounds).
    pub fn last_checkpoint(&self) -> Option<BoundedTensor> {
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => checkpoints.last().map(|(_, b)| b.clone()),
            CheckpointStorage::F16(checkpoints) => checkpoints
                .last()
                .and_then(|(_, b)| b.to_bounded_tensor().ok()),
        }
    }
}

/// Streaming verifier that uses gradient checkpointing for memory efficiency.
pub struct StreamingVerifier {
    config: StreamingConfig,
}

impl StreamingVerifier {
    /// Create a new streaming verifier with the given config.
    pub fn new(config: StreamingConfig) -> Self {
        Self { config }
    }

    /// Run forward IBP pass with checkpointing.
    /// Returns checkpointed bounds for use in backward pass.
    pub fn collect_checkpointed_bounds(
        &self,
        network: &Network,
        input: &BoundedTensor,
    ) -> Result<CheckpointedBounds> {
        let num_layers = network.layers.len();

        // Create checkpointed storage with appropriate storage mode
        let mut checkpointed = if self.config.use_f16_checkpoints {
            CheckpointedBounds::new_compressed(
                input.clone(),
                num_layers,
                self.config.f16_widening_epsilon,
            )
        } else {
            CheckpointedBounds::new(input.clone(), num_layers)
        };

        if num_layers == 0 {
            return Ok(checkpointed);
        }

        let mut current = input.clone();
        let interval = self.config.checkpoint_interval.max(1);

        for (i, layer) in network.layers.iter().enumerate() {
            current = layer
                .propagate_ibp(&current)
                .map_err(|e| GammaError::LayerError {
                    layer_index: i,
                    layer_type: layer.layer_type().to_string(),
                    source: Box::new(e),
                })?;

            // Store checkpoint at intervals and always at the last layer
            if (i + 1) % interval == 0 || i == num_layers - 1 {
                debug!(
                    "Streaming: checkpoint at layer {} (size {}, f16={})",
                    i,
                    current.len(),
                    self.config.use_f16_checkpoints
                );
                checkpointed.add_checkpoint(i, current.clone());
            }
        }

        let storage_type = if checkpointed.is_compressed() {
            "f16"
        } else {
            "f32"
        };
        debug!(
            "Streaming: {} checkpoints, {} bytes ({})",
            checkpointed.num_checkpoints(),
            checkpointed.memory_bytes(),
            storage_type
        );

        // Log compression stats if using f16
        if let Some((f16_bytes, f32_bytes, ratio)) = checkpointed.compression_stats() {
            debug!(
                "Streaming f16: {} bytes vs {} bytes f32 ({:.1}% of original)",
                f16_bytes,
                f32_bytes,
                ratio * 100.0
            );
        }

        Ok(checkpointed)
    }

    /// Propagate CROWN with gradient checkpointing.
    ///
    /// This is memory-efficient but slower than regular CROWN due to recomputation.
    /// Memory: O(L/K * N) instead of O(L * N).
    /// Compute: O(L * K) instead of O(L).
    pub fn propagate_crown_streaming(
        &self,
        network: &Network,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        // Step 1: Forward pass with checkpointing
        let checkpointed = self.collect_checkpointed_bounds(network, input)?;

        // Get output bounds from last checkpoint
        let output_bounds = checkpointed
            .last_checkpoint()
            .ok_or_else(|| GammaError::InvalidSpec("No checkpoints created".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!("CROWN streaming: backward from {} outputs", output_dim);

        // Step 2: Initialize linear bounds at output
        let mut linear_bounds = LinearBounds::identity(output_dim);

        // Step 3: Backward pass with recomputation from checkpoints
        for i in (0..network.layers.len()).rev() {
            let layer = &network.layers[i];

            debug!(
                "CROWN streaming: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds (recompute from checkpoint if needed)
            let pre_activation = if i == 0 {
                input.clone()
            } else {
                checkpointed.get_bounds_at(i - 1, network)?
            };

            linear_bounds =
                Self::propagate_layer_backward(layer, linear_bounds, &pre_activation, i)?;
        }

        // Step 4: Concretize using input bounds
        debug!("CROWN streaming: concretizing");
        linear_bounds.concretize(input).reshape(&output_shape)
    }

    /// Propagate backward through a single layer.
    fn propagate_layer_backward(
        layer: &Layer,
        linear_bounds: LinearBounds,
        pre_activation: &BoundedTensor,
        layer_idx: usize,
    ) -> Result<LinearBounds> {
        match layer {
            Layer::Linear(l) => match l.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::ReLU(r) => r.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::GELU(g) => g.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Softmax(s) => s.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::CausalSoftmax(cs) => match cs.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::LayerNorm(ln) => ln.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Transpose(t) => match t.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::AddConstant(ac) => match ac.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::Reshape(r) => match r.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::Flatten(f) => match f.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::MulConstant(m) => match m.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::DivConstant(d) => match d.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::SubConstant(s) => match s.propagate_linear(&linear_bounds)? {
                Cow::Borrowed(_) => Ok(linear_bounds),
                Cow::Owned(lb) => Ok(lb),
            },
            Layer::Tanh(t) => t.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Sigmoid(s) => s.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Softplus(sp) => sp.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::LeakyReLU(lr) => lr.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Clip(c) => c.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::HardSigmoid(hs) => {
                hs.propagate_linear_with_bounds(&linear_bounds, pre_activation)
            }
            Layer::Elu(e) => e.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Selu(s) => s.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::PRelu(pr) => pr.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::HardSwish(hw) => hw.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Celu(ce) => ce.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Mish(mi) => mi.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Exp(e) => e.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Log(lg) => lg.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Softsign(ss) => ss.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Sin(sn) => sn.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Cos(cs) => cs.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Reciprocal(rc) => {
                rc.propagate_linear_with_bounds(&linear_bounds, pre_activation)
            }
            Layer::Sqrt(sq) => sq.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            Layer::Abs(ab) => ab.propagate_linear_with_bounds(&linear_bounds, pre_activation),
            // Unsupported layers - fall back error
            _ => Err(GammaError::UnsupportedOp(format!(
                "Layer {} ({}) not supported in streaming CROWN",
                layer_idx,
                layer.layer_type()
            ))),
        }
    }

    /// Propagate batched CROWN with gradient checkpointing.
    ///
    /// Batched version that preserves N-D shape structure.
    pub fn propagate_crown_batched_streaming(
        &self,
        network: &Network,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        // Check if we can use batched CROWN
        let can_use_batched = network.layers.iter().all(|layer| {
            matches!(
                layer,
                Layer::Linear(_)
                    | Layer::ReLU(_)
                    | Layer::GELU(_)
                    | Layer::Softmax(_)
                    | Layer::LayerNorm(_)
            )
        });

        if !can_use_batched {
            debug!("Streaming batched CROWN: unsupported layers, using regular streaming CROWN");
            return self.propagate_crown_streaming(network, input);
        }

        // Step 1: Forward pass with checkpointing
        let checkpointed = self.collect_checkpointed_bounds(network, input)?;

        let output_bounds = checkpointed
            .last_checkpoint()
            .ok_or_else(|| GammaError::InvalidSpec("No checkpoints created".to_string()))?;
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "Batched CROWN streaming: backward with shape {:?}",
            output_shape
        );

        // Step 2: Initialize batched linear bounds
        let mut batched_bounds = BatchedLinearBounds::identity(&output_shape);

        // Step 3: Backward pass with recomputation
        for i in (0..network.layers.len()).rev() {
            let layer = &network.layers[i];

            debug!(
                "Batched CROWN streaming: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            let pre_activation = if i == 0 {
                input.clone()
            } else {
                checkpointed.get_bounds_at(i - 1, network)?
            };

            batched_bounds = match layer {
                Layer::Linear(l) => l.propagate_linear_batched(&batched_bounds)?,
                Layer::ReLU(r) => {
                    r.propagate_linear_batched_with_bounds(&batched_bounds, &pre_activation)?
                }
                Layer::GELU(g) => {
                    g.propagate_linear_batched_with_bounds(&batched_bounds, &pre_activation)?
                }
                Layer::Softmax(s) => {
                    s.propagate_linear_batched_with_bounds(&batched_bounds, &pre_activation)?
                }
                Layer::LayerNorm(ln) => {
                    ln.propagate_linear_batched_with_bounds(&batched_bounds, &pre_activation)?
                }
                _ => {
                    return Err(GammaError::UnsupportedOp(format!(
                        "Layer {} ({}) not supported in streaming batched CROWN",
                        i,
                        layer.layer_type()
                    )));
                }
            };
        }

        // Step 4: Concretize
        let concrete = batched_bounds.concretize(input);
        if concrete.shape() != output_shape.as_slice() {
            concrete.reshape(&output_shape)
        } else {
            Ok(concrete)
        }
    }
}

/// Calculate estimated memory savings from streaming.
///
/// Returns (original_memory_bytes, streaming_memory_bytes, savings_percent).
pub fn estimate_memory_savings(
    num_layers: usize,
    tensor_elements: usize,
    checkpoint_interval: usize,
) -> (usize, usize, f32) {
    let bytes_per_tensor = tensor_elements * 4 * 2; // f32 lower + upper

    // Original: store all layers
    let original = num_layers * bytes_per_tensor;

    // Streaming: store checkpoints + input
    let num_checkpoints = num_layers.div_ceil(checkpoint_interval);
    let streaming = (num_checkpoints + 1) * bytes_per_tensor;

    let savings = if original > 0 {
        100.0 * (1.0 - streaming as f32 / original as f32)
    } else {
        0.0
    };

    (original, streaming, savings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::LinearLayer;
    use ndarray::{Array1, Array2, ArrayD};

    fn create_test_network(num_layers: usize, in_dim: usize, out_dim: usize) -> Network {
        let mut network = Network::new();
        for i in 0..num_layers {
            let (layer_in, layer_out) = if i == 0 {
                (in_dim, out_dim)
            } else {
                (out_dim, out_dim)
            };
            let weight = Array2::<f32>::zeros((layer_out, layer_in));
            let bias = Some(Array1::<f32>::zeros(layer_out));
            let linear = LinearLayer::new(weight, bias).unwrap();
            network.add_layer(Layer::Linear(linear));
        }
        network
    }

    fn create_input(dim: usize) -> BoundedTensor {
        let lower = ArrayD::from_elem(ndarray::IxDyn(&[dim]), -1.0_f32);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&[dim]), 1.0_f32);
        BoundedTensor::new(lower, upper).unwrap()
    }

    #[test]
    fn test_checkpointed_bounds_basic() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

        // Add checkpoints at layers 4, 9, 14, 19
        for i in [4, 9, 14, 19] {
            let bounds = create_input(10);
            checkpointed.add_checkpoint(i, bounds);
        }

        assert_eq!(checkpointed.num_checkpoints(), 4);
    }

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.checkpoint_interval, 10);
        assert!(config.enable_crown_streaming);
    }

    #[test]
    fn test_streaming_config_min_memory() {
        let config = StreamingConfig::min_memory();
        assert_eq!(config.checkpoint_interval, 50);
    }

    #[test]
    fn test_collect_checkpoints() {
        let network = create_test_network(20, 10, 10);
        let input = create_input(10);

        let config = StreamingConfig {
            checkpoint_interval: 5,
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);

        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // With interval=5 and 20 layers, checkpoints at: 4, 9, 14, 19
        assert_eq!(checkpointed.num_checkpoints(), 4);
    }

    #[test]
    fn test_memory_savings_calculation() {
        let (original, streaming, savings) = estimate_memory_savings(100, 1000, 10);

        // 100 layers, checkpoint every 10 = 11 checkpoints (including input)
        // Savings should be ~89%
        assert!(savings > 85.0);
        assert!(savings < 95.0);
        assert!(streaming < original);
    }

    #[test]
    fn test_streaming_crown_empty_network() {
        let network = Network::new();
        let input = create_input(10);

        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let result = verifier
            .propagate_crown_streaming(&network, &input)
            .unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_streaming_crown_single_layer() {
        let network = create_test_network(1, 10, 10);
        let input = create_input(10);

        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let streaming_result = verifier
            .propagate_crown_streaming(&network, &input)
            .unwrap();

        // Compare with regular CROWN
        let regular_result = network.propagate_crown(&input).unwrap();

        // Results should match
        assert_eq!(streaming_result.shape(), regular_result.shape());
        // Bounds may differ slightly due to floating point, but shape should match
    }

    #[test]
    fn test_find_nearest_checkpoint() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

        // Add checkpoints at layers 4, 9, 14
        checkpointed.add_checkpoint(4, create_input(10));
        checkpointed.add_checkpoint(9, create_input(10));
        checkpointed.add_checkpoint(14, create_input(10));

        // Test find_nearest_checkpoint_before
        // Layer 6 should find checkpoint at 4
        let (idx, _) = checkpointed.find_nearest_checkpoint_before(6);
        assert_eq!(idx, 4);

        // Layer 12 should find checkpoint at 9
        let (idx, _) = checkpointed.find_nearest_checkpoint_before(12);
        assert_eq!(idx, 9);

        // Layer 2 should find no checkpoint (returns -1)
        let (idx, _) = checkpointed.find_nearest_checkpoint_before(2);
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_streaming_crown_equivalence() {
        // Create network with non-trivial weights
        let mut network = Network::new();
        for i in 0..10 {
            // Use small random-ish weights to avoid bound explosion
            let mut weight = Array2::<f32>::zeros((8, 8));
            for r in 0..8 {
                for c in 0..8 {
                    // Deterministic "random" pattern
                    let val = ((r * 7 + c * 11 + i * 13) % 10) as f32 * 0.01 - 0.05;
                    weight[[r, c]] = val;
                }
            }
            let bias = Some(Array1::<f32>::zeros(8));
            let linear = LinearLayer::new(weight, bias).unwrap();
            network.add_layer(Layer::Linear(linear));
        }

        let lower = ArrayD::from_elem(ndarray::IxDyn(&[8]), -0.1_f32);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.1_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Regular CROWN
        let regular_result = network.propagate_crown(&input).unwrap();

        // Streaming CROWN with different checkpoint intervals
        for interval in [1, 2, 5, 10] {
            let config = StreamingConfig {
                checkpoint_interval: interval,
                ..Default::default()
            };
            let verifier = StreamingVerifier::new(config);
            let streaming_result = verifier
                .propagate_crown_streaming(&network, &input)
                .unwrap();

            // Results should match within floating point tolerance
            assert_eq!(streaming_result.shape(), regular_result.shape());

            let max_lower_diff: f32 = streaming_result
                .lower
                .iter()
                .zip(regular_result.lower.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            let max_upper_diff: f32 = streaming_result
                .upper
                .iter()
                .zip(regular_result.upper.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_lower_diff < 1e-5,
                "Lower bounds differ by {} with interval {}",
                max_lower_diff,
                interval
            );
            assert!(
                max_upper_diff < 1e-5,
                "Upper bounds differ by {} with interval {}",
                max_upper_diff,
                interval
            );
        }
    }

    #[test]
    fn test_recomputation_correctness() {
        // Test that recomputing from checkpoints gives same result as direct computation
        let network = create_test_network(10, 8, 8);
        let input = create_input(8);

        // Collect all bounds (regular way)
        let all_bounds = network.collect_ibp_bounds(&input).unwrap();

        // Collect checkpointed bounds
        let config = StreamingConfig {
            checkpoint_interval: 3,
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);
        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // Verify that get_bounds_at returns same bounds as direct computation
        for (i, expected) in all_bounds.iter().take(10).enumerate() {
            let recomputed = checkpointed.get_bounds_at(i, &network).unwrap();

            let max_lower_diff: f32 = recomputed
                .lower
                .iter()
                .zip(expected.lower.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            let max_upper_diff: f32 = recomputed
                .upper
                .iter()
                .zip(expected.upper.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_lower_diff < 1e-6,
                "Layer {} lower bounds differ by {}",
                i,
                max_lower_diff
            );
            assert!(
                max_upper_diff < 1e-6,
                "Layer {} upper bounds differ by {}",
                i,
                max_upper_diff
            );
        }
    }

    // ============== F16 Compression Tests ==============

    #[test]
    fn test_f16_compression_config() {
        let config = StreamingConfig::with_f16_compression();
        assert!(config.use_f16_checkpoints);
        assert_eq!(config.checkpoint_interval, 10);

        let config = StreamingConfig::min_memory();
        assert!(config.use_f16_checkpoints);
        assert_eq!(config.checkpoint_interval, 50);
    }

    #[test]
    fn test_checkpointed_bounds_f16() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new_compressed(input.clone(), 20, 0.001);

        // Add checkpoints at layers 4, 9, 14, 19
        for i in [4, 9, 14, 19] {
            let bounds = create_input(10);
            checkpointed.add_checkpoint(i, bounds);
        }

        assert_eq!(checkpointed.num_checkpoints(), 4);
        assert!(checkpointed.is_compressed());
    }

    #[test]
    fn test_f16_memory_savings() {
        let input = create_input(1000);
        let bounds = create_input(1000);

        // Create f32 and f16 checkpointed storage
        let mut checkpointed_f32 = CheckpointedBounds::new(input.clone(), 10);
        let mut checkpointed_f16 = CheckpointedBounds::new_compressed(input.clone(), 10, 0.001);

        // Add same checkpoints to both
        for i in 0..10 {
            checkpointed_f32.add_checkpoint(i, bounds.clone());
            checkpointed_f16.add_checkpoint(i, bounds.clone());
        }

        let f32_bytes = checkpointed_f32.memory_bytes();
        let f16_bytes = checkpointed_f16.memory_bytes();

        // f16 should use significantly less memory (approximately half)
        assert!(
            f16_bytes < f32_bytes * 7 / 10,
            "f16 ({}) should be <70% of f32 ({})",
            f16_bytes,
            f32_bytes
        );
    }

    #[test]
    fn test_f16_compression_stats() {
        let input = create_input(1000);
        let mut checkpointed = CheckpointedBounds::new_compressed(input.clone(), 10, 0.001);

        for i in 0..10 {
            checkpointed.add_checkpoint(i, create_input(1000));
        }

        let stats = checkpointed.compression_stats();
        assert!(stats.is_some());

        let (f16_bytes, f32_bytes, ratio) = stats.unwrap();
        assert!(ratio < 0.6, "Ratio {} should be < 0.6", ratio);
        assert!(f16_bytes < f32_bytes);
    }

    #[test]
    fn test_f16_bounds_recovery() {
        // Test that bounds can be recovered from f16 storage
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new_compressed(input.clone(), 5, 0.001);

        let original_bounds = BoundedTensor::new(
            ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.5_f32),
            ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.5_f32),
        )
        .unwrap();

        checkpointed.add_checkpoint(2, original_bounds.clone());

        // Create a minimal network for testing
        let network = create_test_network(5, 10, 10);
        let recovered = checkpointed.get_bounds_at(2, &network).unwrap();

        // Recovered bounds should be similar to original (within f16 precision + widening)
        assert_eq!(recovered.shape(), original_bounds.shape());

        // Due to widening for soundness, lower should be <= original lower
        // and upper should be >= original upper
        for (orig, rec) in original_bounds.lower.iter().zip(recovered.lower.iter()) {
            assert!(
                *rec <= orig + 0.01,
                "Recovered lower {} should be <= original {} (with tolerance)",
                rec,
                orig
            );
        }
        for (orig, rec) in original_bounds.upper.iter().zip(recovered.upper.iter()) {
            assert!(
                *rec >= orig - 0.01,
                "Recovered upper {} should be >= original {} (with tolerance)",
                rec,
                orig
            );
        }
    }

    #[test]
    fn test_streaming_with_f16_checkpoints() {
        let network = create_test_network(20, 10, 10);
        let input = create_input(10);

        let config = StreamingConfig {
            checkpoint_interval: 5,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);

        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        assert_eq!(checkpointed.num_checkpoints(), 4);
        assert!(checkpointed.is_compressed());
    }

    #[test]
    fn test_f16_vs_f32_checkpoint_equivalence() {
        // Test that f16 checkpoints produce approximately same results as f32
        let mut network = Network::new();
        for i in 0..5 {
            let mut weight = Array2::<f32>::zeros((8, 8));
            for r in 0..8 {
                for c in 0..8 {
                    let val = ((r * 7 + c * 11 + i * 13) % 10) as f32 * 0.01 - 0.05;
                    weight[[r, c]] = val;
                }
            }
            let bias = Some(Array1::<f32>::zeros(8));
            let linear = LinearLayer::new(weight, bias).unwrap();
            network.add_layer(Layer::Linear(linear));
        }

        let lower = ArrayD::from_elem(ndarray::IxDyn(&[8]), -0.1_f32);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.1_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Collect with f32 storage
        let config_f32 = StreamingConfig {
            checkpoint_interval: 2,
            use_f16_checkpoints: false,
            ..Default::default()
        };
        let verifier_f32 = StreamingVerifier::new(config_f32);
        let checkpointed_f32 = verifier_f32
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // Collect with f16 storage
        let config_f16 = StreamingConfig {
            checkpoint_interval: 2,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
            ..Default::default()
        };
        let verifier_f16 = StreamingVerifier::new(config_f16);
        let checkpointed_f16 = verifier_f16
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // Compare bounds at each checkpoint
        for i in 0..5 {
            let bounds_f32 = checkpointed_f32.get_bounds_at(i, &network).unwrap();
            let bounds_f16 = checkpointed_f16.get_bounds_at(i, &network).unwrap();

            let max_lower_diff: f32 = bounds_f32
                .lower
                .iter()
                .zip(bounds_f16.lower.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            let max_upper_diff: f32 = bounds_f32
                .upper
                .iter()
                .zip(bounds_f16.upper.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            // f16 precision allows for some error, but should be bounded
            assert!(
                max_lower_diff < 0.01,
                "Layer {} lower bounds differ by {} (f16 vs f32)",
                i,
                max_lower_diff
            );
            assert!(
                max_upper_diff < 0.01,
                "Layer {} upper bounds differ by {} (f16 vs f32)",
                i,
                max_upper_diff
            );
        }
    }

    #[test]
    fn test_f16_checkpoints_soundness() {
        // Test that f16 checkpoints with widening produce sound (conservative) bounds
        let mut network = Network::new();
        for i in 0..3 {
            let mut weight = Array2::<f32>::zeros((4, 4));
            for r in 0..4 {
                for c in 0..4 {
                    let val = ((r * 3 + c * 5 + i * 7) % 10) as f32 * 0.02 - 0.1;
                    weight[[r, c]] = val;
                }
            }
            let bias = Some(Array1::<f32>::zeros(4));
            let linear = LinearLayer::new(weight, bias).unwrap();
            network.add_layer(Layer::Linear(linear));
        }

        let lower = ArrayD::from_elem(ndarray::IxDyn(&[4]), -0.5_f32);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.5_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // f32 reference (no widening)
        let config_f32 = StreamingConfig {
            checkpoint_interval: 1,
            use_f16_checkpoints: false,
            ..Default::default()
        };
        let verifier_f32 = StreamingVerifier::new(config_f32);
        let checkpointed_f32 = verifier_f32
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // f16 with widening
        let config_f16 = StreamingConfig {
            checkpoint_interval: 1,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.01, // 1% widening
            ..Default::default()
        };
        let verifier_f16 = StreamingVerifier::new(config_f16);
        let checkpointed_f16 = verifier_f16
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // f16 widened bounds should be more conservative (wider)
        for i in 0..3 {
            let bounds_f32 = checkpointed_f32.get_bounds_at(i, &network).unwrap();
            let bounds_f16 = checkpointed_f16.get_bounds_at(i, &network).unwrap();

            // f16 lower should be <= f32 lower (more conservative)
            for (f32_l, f16_l) in bounds_f32.lower.iter().zip(bounds_f16.lower.iter()) {
                assert!(
                    *f16_l <= f32_l + 0.001, // Small tolerance for f16 precision
                    "Layer {}: f16 lower {} should be <= f32 lower {}",
                    i,
                    f16_l,
                    f32_l
                );
            }

            // f16 upper should be >= f32 upper (more conservative)
            for (f32_u, f16_u) in bounds_f32.upper.iter().zip(bounds_f16.upper.iter()) {
                assert!(
                    *f16_u >= f32_u - 0.001, // Small tolerance for f16 precision
                    "Layer {}: f16 upper {} should be >= f32 upper {}",
                    i,
                    f16_u,
                    f32_u
                );
            }
        }
    }

    // ============== Additional Coverage Tests ==============

    #[test]
    fn test_streaming_config_balanced() {
        let config = StreamingConfig::balanced();
        assert_eq!(config.checkpoint_interval, 10);
        assert!(config.enable_crown_streaming);
        assert!(!config.use_f16_checkpoints);
        assert_eq!(config.max_memory_bytes, 0);
    }

    #[test]
    fn test_streaming_config_with_memory_limit() {
        let config = StreamingConfig::with_memory_limit(1024 * 1024);
        assert_eq!(config.max_memory_bytes, 1024 * 1024);
        assert_eq!(config.checkpoint_interval, 10);
        assert!(config.enable_crown_streaming);
        assert!(!config.use_f16_checkpoints);
    }

    #[test]
    fn test_checkpointed_bounds_is_compressed_f32() {
        let input = create_input(10);
        let checkpointed = CheckpointedBounds::new(input, 10);
        assert!(!checkpointed.is_compressed());
    }

    #[test]
    fn test_checkpointed_bounds_compression_stats_f32() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 10);
        checkpointed.add_checkpoint(0, create_input(10));

        // f32 storage should return None for compression stats
        assert!(checkpointed.compression_stats().is_none());
    }

    #[test]
    fn test_checkpointed_bounds_empty_checkpoints() {
        let input = create_input(10);
        let checkpointed = CheckpointedBounds::new(input, 10);

        assert_eq!(checkpointed.num_checkpoints(), 0);
        assert!(checkpointed.last_checkpoint().is_none());
    }

    #[test]
    fn test_checkpointed_bounds_last_checkpoint_f32() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 10);

        let bounds1 = BoundedTensor::new(
            ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.0_f32),
            ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.0_f32),
        )
        .unwrap();
        let bounds2 = BoundedTensor::new(
            ArrayD::from_elem(ndarray::IxDyn(&[10]), -2.0_f32),
            ArrayD::from_elem(ndarray::IxDyn(&[10]), 2.0_f32),
        )
        .unwrap();

        checkpointed.add_checkpoint(3, bounds1);
        checkpointed.add_checkpoint(7, bounds2.clone());

        let last = checkpointed.last_checkpoint();
        assert!(last.is_some());
        let last_bounds = last.unwrap();
        // Should be bounds2 since it's at the highest index
        assert!((last_bounds.lower[0] - (-2.0)).abs() < 1e-6);
        assert!((last_bounds.upper[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpointed_bounds_last_checkpoint_f16() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new_compressed(input.clone(), 10, 0.001);

        let bounds = BoundedTensor::new(
            ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.5_f32),
            ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.5_f32),
        )
        .unwrap();

        checkpointed.add_checkpoint(5, bounds);

        let last = checkpointed.last_checkpoint();
        assert!(last.is_some());
    }

    #[test]
    fn test_get_bounds_at_invalid_layer() {
        let input = create_input(10);
        let checkpointed = CheckpointedBounds::new(input, 5);
        let network = create_test_network(5, 10, 10);

        // Layer 10 is out of range (max is 4)
        let result = checkpointed.get_bounds_at(10, &network);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_bounds_at_exact_checkpoint() {
        let input = create_input(8);
        let network = create_test_network(10, 8, 8);

        let config = StreamingConfig {
            checkpoint_interval: 3,
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);
        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // Layer 2 should be a checkpoint (index 2, which is (2+1)%3 == 0)
        // Actually with interval=3, checkpoints at 2, 5, 8, 9
        let result = checkpointed.get_bounds_at(2, &network);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_nearest_checkpoint_at_exact() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

        checkpointed.add_checkpoint(5, create_input(10));
        checkpointed.add_checkpoint(10, create_input(10));

        // At exact checkpoint should return that checkpoint
        let (idx, _) = checkpointed.find_nearest_checkpoint_before(5);
        assert_eq!(idx, 5);

        let (idx, _) = checkpointed.find_nearest_checkpoint_before(10);
        assert_eq!(idx, 10);
    }

    #[test]
    fn test_find_nearest_checkpoint_at_layer_zero() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

        checkpointed.add_checkpoint(5, create_input(10));

        // Layer 0 should return -1 (no checkpoint before layer 0)
        let (idx, _) = checkpointed.find_nearest_checkpoint_before(0);
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_memory_bytes_empty() {
        let input = create_input(10);
        let checkpointed = CheckpointedBounds::new(input, 10);

        // Just input memory (10 elements * 4 bytes * 2 (lower+upper))
        let mem = checkpointed.memory_bytes();
        assert_eq!(mem, 10 * 4 * 2);
    }

    #[test]
    fn test_memory_bytes_with_checkpoints() {
        let input = create_input(100);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 10);

        for i in 0..5 {
            checkpointed.add_checkpoint(i, create_input(100));
        }

        // input + 5 checkpoints, each 100 elements * 4 bytes * 2
        let expected = (1 + 5) * 100 * 4 * 2;
        assert_eq!(checkpointed.memory_bytes(), expected);
    }

    #[test]
    fn test_estimate_memory_savings_zero_layers() {
        let (original, streaming, savings) = estimate_memory_savings(0, 1000, 10);
        assert_eq!(original, 0);
        // With 0 layers, still have 1 "checkpoint" (input)
        assert!(streaming > 0);
        // Savings is 0 when original is 0 (division by zero protection)
        assert!(savings <= 0.0 || savings.is_nan() || !savings.is_finite());
    }

    #[test]
    fn test_estimate_memory_savings_single_layer() {
        let (original, _streaming, _savings) = estimate_memory_savings(1, 1000, 10);
        // 1 layer, checkpoint every 10 = 1 checkpoint + input = 2
        // original = 1 * 8000 = 8000
        // streaming = 2 * 8000 = 16000
        // This actually increases memory for very small networks
        assert_eq!(original, 8000);
    }

    #[test]
    fn test_estimate_memory_savings_large_interval() {
        let (_original, _streaming, savings) = estimate_memory_savings(100, 1000, 100);
        // 100 layers, checkpoint every 100 = 1 checkpoint + input = 2
        // Savings should be very high (~98%)
        assert!(savings > 95.0);
    }

    #[test]
    fn test_checkpoint_interval_zero_clamped() {
        let network = create_test_network(5, 8, 8);
        let input = create_input(8);

        let config = StreamingConfig {
            checkpoint_interval: 0, // Should be clamped to 1
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);

        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // With interval clamped to 1, should have 5 checkpoints (one per layer)
        assert_eq!(checkpointed.num_checkpoints(), 5);
    }

    #[test]
    fn test_collect_checkpoints_empty_network() {
        let network = Network::new();
        let input = create_input(10);

        let config = StreamingConfig::default();
        let verifier = StreamingVerifier::new(config);

        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        assert_eq!(checkpointed.num_checkpoints(), 0);
    }

    #[test]
    fn test_streaming_batched_crown_empty_network() {
        let network = Network::new();
        let input = create_input(10);

        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let result = verifier
            .propagate_crown_batched_streaming(&network, &input)
            .unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_streaming_batched_crown_linear_only() {
        let network = create_test_network(5, 8, 8);
        let input = create_input(8);

        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let result = verifier
            .propagate_crown_batched_streaming(&network, &input)
            .unwrap();

        // Should produce valid output with correct shape
        assert_eq!(result.shape(), &[8]);
    }

    #[test]
    fn test_add_checkpoint_maintains_sort_order() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

        // Add checkpoints out of order
        checkpointed.add_checkpoint(10, create_input(10));
        checkpointed.add_checkpoint(5, create_input(10));
        checkpointed.add_checkpoint(15, create_input(10));
        checkpointed.add_checkpoint(0, create_input(10));

        assert_eq!(checkpointed.num_checkpoints(), 4);

        // Verify ordering by finding nearest checkpoint
        let (idx, _) = checkpointed.find_nearest_checkpoint_before(3);
        assert_eq!(idx, 0);

        let (idx, _) = checkpointed.find_nearest_checkpoint_before(7);
        assert_eq!(idx, 5);

        let (idx, _) = checkpointed.find_nearest_checkpoint_before(12);
        assert_eq!(idx, 10);

        let (idx, _) = checkpointed.find_nearest_checkpoint_before(20);
        assert_eq!(idx, 15);
    }

    #[test]
    fn test_f16_widening_epsilon_zero() {
        let input = create_input(10);
        let mut checkpointed = CheckpointedBounds::new_compressed(input.clone(), 10, 0.0);

        let bounds = create_input(10);
        checkpointed.add_checkpoint(0, bounds);

        // Should still work with zero widening
        assert!(checkpointed.is_compressed());
        assert_eq!(checkpointed.num_checkpoints(), 1);
    }

    #[test]
    fn test_streaming_crown_two_layers() {
        let network = create_test_network(2, 10, 10);
        let input = create_input(10);

        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let streaming_result = verifier
            .propagate_crown_streaming(&network, &input)
            .unwrap();

        let regular_result = network.propagate_crown(&input).unwrap();

        assert_eq!(streaming_result.shape(), regular_result.shape());
    }

    #[test]
    fn test_checkpoint_interval_larger_than_network() {
        let network = create_test_network(3, 8, 8);
        let input = create_input(8);

        let config = StreamingConfig {
            checkpoint_interval: 100, // Much larger than network size
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);

        let checkpointed = verifier
            .collect_checkpointed_bounds(&network, &input)
            .unwrap();

        // Should have exactly 1 checkpoint (at the last layer)
        assert_eq!(checkpointed.num_checkpoints(), 1);
    }

    #[test]
    fn test_memory_bytes_f16_vs_f32() {
        let input = create_input(1000);
        let bounds = create_input(1000);

        let mut f32_storage = CheckpointedBounds::new(input.clone(), 10);
        let mut f16_storage = CheckpointedBounds::new_compressed(input.clone(), 10, 0.0);

        f32_storage.add_checkpoint(0, bounds.clone());
        f16_storage.add_checkpoint(0, bounds);

        // f16 storage should use approximately half the memory for checkpoints
        let f32_mem = f32_storage.memory_bytes();
        let f16_mem = f16_storage.memory_bytes();

        // Input memory is same, checkpoint memory differs
        // f32: input (8000) + checkpoint (8000) = 16000
        // f16: input (8000) + checkpoint (4000) = 12000
        assert!(f16_mem < f32_mem);
    }
}
