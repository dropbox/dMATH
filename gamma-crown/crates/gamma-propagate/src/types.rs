//! Result types and configuration for bound propagation.
//!
//! This module contains the data structures used to configure propagation
//! and report verification results.

use gamma_core::{GammaError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for bound propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationConfig {
    /// Which propagation method to use.
    pub method: PropagationMethod,
    /// Maximum iterations for optimization-based methods.
    pub max_iterations: usize,
    /// Convergence threshold.
    pub tolerance: f32,
    /// Whether to use GPU acceleration (future).
    pub use_gpu: bool,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            method: PropagationMethod::AlphaCrown,
            max_iterations: 100,
            tolerance: 1e-4,
            use_gpu: false,
        }
    }
}

/// Available bound propagation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationMethod {
    /// Interval Bound Propagation: fastest, loosest.
    Ibp,
    /// CROWN: linear relaxation.
    Crown,
    /// α-CROWN: optimized linear relaxation.
    AlphaCrown,
    /// SDP-CROWN: tighter LiRPA for ℓ2 input sets (Linear/ReLU only for now).
    SdpCrown,
    /// β-CROWN: branch and bound.
    BetaCrown,
}

/// Information about bounds at a specific node in the graph.
///
/// Used for layer-by-layer verification to track bound growth through the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeBoundsInfo {
    /// Node name in the graph.
    pub name: String,
    /// Layer type (e.g., "Linear", "MatMul", "GELU").
    pub layer_type: String,
    /// Input bound width (maximum width across all input elements).
    pub input_width: f32,
    /// Output bound width (maximum width across all output elements).
    pub output_width: f32,
    /// Sensitivity = output_width / input_width.
    pub sensitivity: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Minimum bound value across all outputs.
    pub min_bound: f32,
    /// Maximum bound value across all outputs.
    pub max_bound: f32,
    /// Whether bounds have saturated (near f32::MAX).
    pub saturated: bool,
    /// Whether any bounds are NaN.
    pub has_nan: bool,
    /// Whether any bounds are infinite.
    pub has_infinite: bool,
}

impl NodeBoundsInfo {
    /// Check if this node's bounds have degraded (saturated, NaN, or infinite).
    pub fn has_degraded(&self) -> bool {
        self.saturated || self.has_nan || self.has_infinite
    }

    /// Get the status string for this node.
    pub fn status(&self) -> &'static str {
        if self.has_nan {
            "NAN"
        } else if self.has_infinite {
            "INF"
        } else if self.saturated {
            "SATURATED"
        } else if self.sensitivity > 100.0 {
            "HIGH"
        } else if self.sensitivity > 10.0 {
            "MODERATE"
        } else if self.sensitivity < 1.0 {
            "STABLE"
        } else {
            "OK"
        }
    }
}

/// Result of layer-by-layer verification through a GraphNetwork.
#[derive(Debug, Clone)]
pub struct LayerByLayerResult {
    /// Per-node bounds information.
    pub nodes: Vec<NodeBoundsInfo>,
    /// Input epsilon.
    pub input_epsilon: f32,
    /// Final output bound width.
    pub final_width: f32,
    /// Index of first node where bounds degraded (if any).
    pub degraded_at_node: Option<usize>,
    /// Total nodes processed.
    pub total_nodes: usize,
}

impl LayerByLayerResult {
    /// Generate a summary table of the verification results.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Layer-by-Layer Verification".to_string());
        lines.push("===========================".to_string());
        lines.push(format!(
            "{:<40} | {:>10} | {:>10} | {:>8} | Status",
            "Node", "In Width", "Out Width", "Sens."
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------",
            "", "", "", ""
        ));

        for node in &self.nodes {
            let marker = if node.has_degraded() { " <<<" } else { "" };
            lines.push(format!(
                "{:<40} | {:>10.3e} | {:>10.3e} | {:>8.2} | {}{}",
                truncate_name(&node.name, 40),
                node.input_width,
                node.output_width,
                node.sensitivity,
                node.status(),
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Input epsilon: {:.2e} → Final width: {:.2e}",
            self.input_epsilon, self.final_width
        ));
        lines.push(format!("Total nodes: {}", self.total_nodes));

        if let Some(idx) = self.degraded_at_node {
            if let Some(node) = self.nodes.get(idx) {
                lines.push(format!(
                    "WARNING: Bounds degraded at node {} ({})",
                    node.name,
                    node.status()
                ));
            }
        }

        lines.join("\n")
    }
}

/// Information about bounds for a single transformer block.
///
/// Used for block-wise verification to analyze each block independently
/// without bound explosion from propagation through the entire network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBoundsInfo {
    /// Block index (0-based).
    pub block_index: usize,
    /// Block name prefix (e.g., "layer0").
    pub block_name: String,
    /// Per-node information for this block.
    pub nodes: Vec<NodeBoundsInfo>,
    /// Input bound width at start of block (after reset).
    pub input_width: f32,
    /// Output bound width at end of block.
    pub output_width: f32,
    /// Overall sensitivity = output_width / input_width.
    pub sensitivity: f32,
    /// Q@K^T attention bound width (if zonotope applied).
    pub qk_matmul_width: Option<f32>,
    /// SwiGLU FFN bound width (if zonotope applied).
    pub swiglu_width: Option<f32>,
    /// Whether this block saturated or had NaN/inf.
    pub degraded: bool,
}

impl BlockBoundsInfo {
    /// Get status string for this block.
    pub fn status(&self) -> &'static str {
        if self.degraded {
            "DEGRADED"
        } else if self.sensitivity > 1e6 {
            "HIGH"
        } else if self.sensitivity > 1e3 {
            "MODERATE"
        } else {
            "OK"
        }
    }
}

/// Result of block-wise verification.
///
/// Each transformer block is verified independently with fresh bounds reset
/// at each block boundary. This prevents bound explosion from propagating
/// through the entire network and gives meaningful per-block sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockWiseResult {
    /// Per-block information.
    pub blocks: Vec<BlockBoundsInfo>,
    /// Epsilon used for each block's input perturbation.
    pub block_epsilon: f32,
    /// Total number of blocks.
    pub total_blocks: usize,
    /// Maximum sensitivity across all blocks.
    pub max_sensitivity: f32,
    /// Number of blocks that degraded.
    pub degraded_blocks: usize,
}

/// Checkpoint for resumable block-wise verification.
///
/// Allows long-running verification to be interrupted and resumed without
/// losing progress. Serialized to JSON for human readability and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheckpoint {
    /// Checkpoint format version for compatibility.
    pub version: u32,

    /// Model file path (for validation).
    pub model_path: std::path::PathBuf,

    /// SHA256 hash of model file (first 64KB for fast hashing).
    pub model_hash: String,

    /// Input epsilon used for verification.
    pub epsilon: f32,

    /// Verification method (ibp, crown, alpha, beta).
    pub method: String,

    /// Compute backend (cpu, wgpu, mlx).
    pub backend: String,

    /// Timestamp when verification started (ISO 8601).
    pub start_time: String,

    /// Timestamp of this checkpoint (ISO 8601).
    pub checkpoint_time: String,

    /// Total elapsed time in milliseconds (excluding pauses).
    pub elapsed_ms: u64,

    /// Completed blocks with full results.
    pub completed_blocks: Vec<BlockBoundsInfo>,

    /// Maximum sensitivity across completed blocks.
    pub max_sensitivity: f32,

    /// Number of degraded blocks so far.
    pub degraded_blocks: usize,

    /// Total number of blocks in the model.
    pub total_blocks: usize,

    /// Next block index to process (resume point).
    pub next_block_index: usize,
}

impl VerificationCheckpoint {
    /// Current checkpoint format version.
    pub const VERSION: u32 = 1;

    /// Create a new checkpoint at the start of verification.
    pub fn new(
        model_path: std::path::PathBuf,
        model_hash: String,
        epsilon: f32,
        method: &str,
        backend: &str,
        total_blocks: usize,
    ) -> Self {
        let now = chrono_lite_now();
        Self {
            version: Self::VERSION,
            model_path,
            model_hash,
            epsilon,
            method: method.to_string(),
            backend: backend.to_string(),
            start_time: now.clone(),
            checkpoint_time: now,
            elapsed_ms: 0,
            completed_blocks: Vec::new(),
            max_sensitivity: 0.0,
            degraded_blocks: 0,
            total_blocks,
            next_block_index: 0,
        }
    }

    /// Update checkpoint after completing a block.
    pub fn update(&mut self, block: BlockBoundsInfo, elapsed_ms: u64) {
        if block.sensitivity > self.max_sensitivity {
            self.max_sensitivity = block.sensitivity;
        }
        if block.degraded {
            self.degraded_blocks += 1;
        }
        self.next_block_index = block.block_index + 1;
        self.completed_blocks.push(block);
        self.elapsed_ms = elapsed_ms;
        self.checkpoint_time = chrono_lite_now();
    }

    /// Save checkpoint to file atomically.
    ///
    /// Uses write-to-temp-then-rename pattern to ensure the checkpoint file
    /// is never corrupted, even if a crash occurs during write. This is critical
    /// for multi-hour verification runs.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self).map_err(|e| {
            GammaError::InvalidSpec(format!("Failed to serialize checkpoint: {}", e))
        })?;

        // Create temp file in same directory (ensures same filesystem for atomic rename)
        let temp_path = path.with_extension("json.tmp");

        // Write to temp file
        let mut file = std::fs::File::create(&temp_path).map_err(|e| {
            GammaError::InvalidSpec(format!("Failed to create temp checkpoint file: {}", e))
        })?;

        file.write_all(json.as_bytes()).map_err(|e| {
            GammaError::InvalidSpec(format!("Failed to write temp checkpoint: {}", e))
        })?;

        // Sync to disk before rename (ensures data is durable)
        file.sync_all().map_err(|e| {
            GammaError::InvalidSpec(format!("Failed to sync checkpoint to disk: {}", e))
        })?;

        // Drop file handle before rename
        drop(file);

        // Atomic rename (POSIX guarantees atomicity on same filesystem)
        std::fs::rename(&temp_path, path)
            .map_err(|e| GammaError::InvalidSpec(format!("Failed to rename checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from file.
    ///
    /// Also cleans up any stale temp files from interrupted saves.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        // Clean up stale temp file if it exists (from interrupted save)
        let temp_path = path.with_extension("json.tmp");
        if temp_path.exists() {
            // Best effort cleanup - ignore errors
            let _ = std::fs::remove_file(&temp_path);
        }

        let json = std::fs::read_to_string(path)
            .map_err(|e| GammaError::InvalidSpec(format!("Failed to read checkpoint: {}", e)))?;
        let checkpoint: Self = serde_json::from_str(&json)
            .map_err(|e| GammaError::InvalidSpec(format!("Failed to parse checkpoint: {}", e)))?;

        if checkpoint.version != Self::VERSION {
            return Err(GammaError::InvalidSpec(format!(
                "Checkpoint version mismatch: expected {}, found {}",
                Self::VERSION,
                checkpoint.version
            )));
        }

        Ok(checkpoint)
    }

    /// Validate checkpoint matches current verification config.
    pub fn validate(
        &self,
        model_path: &std::path::Path,
        model_hash: &str,
        epsilon: f32,
        method: &str,
        backend: &str,
    ) -> Result<()> {
        if self.model_path != model_path {
            return Err(GammaError::InvalidSpec(format!(
                "Checkpoint model path mismatch: checkpoint has {:?}, current is {:?}",
                self.model_path, model_path
            )));
        }
        if self.model_hash != model_hash {
            return Err(GammaError::InvalidSpec(
                "Checkpoint model hash mismatch: model file has changed since checkpoint"
                    .to_string(),
            ));
        }
        if (self.epsilon - epsilon).abs() > 1e-9 {
            return Err(GammaError::InvalidSpec(format!(
                "Checkpoint epsilon mismatch: checkpoint has {}, current is {}",
                self.epsilon, epsilon
            )));
        }
        if self.method != method {
            return Err(GammaError::InvalidSpec(format!(
                "Checkpoint method mismatch: checkpoint has {}, current is {}",
                self.method, method
            )));
        }
        if self.backend != backend {
            return Err(GammaError::InvalidSpec(format!(
                "Checkpoint backend mismatch: checkpoint has {}, current is {}",
                self.backend, backend
            )));
        }
        Ok(())
    }

    /// Check if verification is complete.
    pub fn is_complete(&self) -> bool {
        self.next_block_index >= self.total_blocks
    }

    /// Build final result from completed checkpoint.
    pub fn into_result(self) -> BlockWiseResult {
        BlockWiseResult {
            blocks: self.completed_blocks,
            block_epsilon: self.epsilon,
            total_blocks: self.total_blocks,
            max_sensitivity: self.max_sensitivity,
            degraded_blocks: self.degraded_blocks,
        }
    }
}

/// Compute SHA256 hash of first 64KB of file (for fast model identification).
pub fn compute_model_hash(path: &std::path::Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| GammaError::InvalidSpec(format!("Failed to open model for hashing: {}", e)))?;

    let mut buffer = vec![0u8; 65536]; // 64KB
    let bytes_read = file
        .read(&mut buffer)
        .map_err(|e| GammaError::InvalidSpec(format!("Failed to read model for hashing: {}", e)))?;
    buffer.truncate(bytes_read);

    // Simple hash: sum bytes with position weighting
    let mut hash: u64 = 0;
    for (i, &b) in buffer.iter().enumerate() {
        hash = hash.wrapping_add((b as u64).wrapping_mul((i as u64).wrapping_add(1)));
    }

    Ok(format!("{:016x}", hash))
}

/// Simple ISO 8601 timestamp without chrono dependency.
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to UTC components (simplified, not accounting for leap seconds)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since 1970-01-01
    let mut year = 1970i32;
    let mut remaining_days = days as i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days_in_month in &month_days {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }
    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Progress information reported during block-wise verification.
#[derive(Debug, Clone)]
pub struct BlockProgress {
    /// Current block index being processed (0-based).
    pub block_index: usize,
    /// Total number of blocks to process.
    pub total_blocks: usize,
    /// Name of the current block.
    pub block_name: String,
    /// Elapsed time since verification started.
    pub elapsed: std::time::Duration,
    /// Current max sensitivity seen so far.
    pub current_max_sensitivity: f32,
    /// Number of degraded blocks so far.
    pub degraded_so_far: usize,
}

impl BlockProgress {
    /// Progress as a fraction (0.0 to 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total_blocks == 0 {
            1.0
        } else {
            // The callback is invoked after the block completes.
            (self.block_index + 1) as f32 / self.total_blocks as f32
        }
    }

    /// Estimated time remaining based on current progress.
    pub fn estimated_remaining(&self) -> std::time::Duration {
        let fraction = self.fraction();
        if fraction > 0.0 && fraction < 1.0 {
            let elapsed_secs = self.elapsed.as_secs_f64();
            let total_estimated = elapsed_secs / fraction as f64;
            let remaining = total_estimated - elapsed_secs;
            std::time::Duration::from_secs_f64(remaining.max(0.0))
        } else {
            std::time::Duration::ZERO
        }
    }

    /// Estimated time remaining based on average block time so far.
    pub fn eta(&self) -> std::time::Duration {
        let completed = self.block_index + 1;
        if completed == 0 || completed >= self.total_blocks {
            return std::time::Duration::ZERO;
        }
        let avg_per_block = self.elapsed.as_secs_f64() / completed as f64;
        let remaining = self.total_blocks - completed;
        std::time::Duration::from_secs_f64(avg_per_block * remaining as f64)
    }
}

/// Progress information reported during layer-by-layer verification within a block.
#[derive(Debug, Clone)]
pub struct LayerProgress {
    /// Current node index being processed (0-based).
    pub node_index: usize,
    /// Total number of nodes to process.
    pub total_nodes: usize,
    /// Name of the current node.
    pub node_name: String,
    /// Layer type of the current node.
    pub layer_type: String,
    /// Elapsed time since verification started.
    pub elapsed: std::time::Duration,
    /// Current max sensitivity seen so far.
    pub current_max_sensitivity: f32,
    /// Number of degraded nodes so far.
    pub degraded_so_far: usize,
}

impl LayerProgress {
    /// Progress as a fraction complete (0.0 to 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total_nodes == 0 {
            1.0
        } else {
            (self.node_index + 1) as f32 / self.total_nodes as f32
        }
    }

    /// Estimated time remaining based on current progress.
    pub fn estimated_remaining(&self) -> std::time::Duration {
        let fraction = self.fraction();
        if fraction > 0.0 && fraction < 1.0 {
            let elapsed_secs = self.elapsed.as_secs_f64();
            let total_estimated = elapsed_secs / fraction as f64;
            let remaining = total_estimated - elapsed_secs;
            std::time::Duration::from_secs_f64(remaining.max(0.0))
        } else {
            std::time::Duration::ZERO
        }
    }
}

impl BlockWiseResult {
    /// Generate a summary table of block-wise verification results.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Block-wise Verification (zonotope reset per block)".to_string());
        lines.push("=================================================".to_string());
        lines.push(format!(
            "{:<15} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | Status",
            "Block", "In Width", "Out Width", "Sens.", "Q@K^T", "SwiGLU"
        ));
        lines.push(format!(
            "{:-<15}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+--------",
            "", "", "", "", "", ""
        ));

        for block in &self.blocks {
            let qk_str = match block.qk_matmul_width {
                Some(w) => format!("{:.3e}", w),
                None => "-".to_string(),
            };
            let swiglu_str = match block.swiglu_width {
                Some(w) => format!("{:.3e}", w),
                None => "-".to_string(),
            };
            let marker = if block.degraded { " <<<" } else { "" };
            lines.push(format!(
                "{:<15} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10} | {:>10} | {}{}",
                block.block_name,
                block.input_width,
                block.output_width,
                block.sensitivity,
                qk_str,
                swiglu_str,
                block.status(),
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Block epsilon: {:.2e} | Max sensitivity: {:.3e}",
            self.block_epsilon, self.max_sensitivity
        ));
        lines.push(format!(
            "Total blocks: {} | Degraded: {}",
            self.total_blocks, self.degraded_blocks
        ));

        lines.join("\n")
    }

    /// Minimum sensitivity across all blocks.
    pub fn min_sensitivity(&self) -> f32 {
        self.blocks
            .iter()
            .map(|b| b.sensitivity)
            .fold(f32::INFINITY, f32::min)
    }

    /// Median sensitivity across all blocks.
    pub fn median_sensitivity(&self) -> f32 {
        if self.blocks.is_empty() {
            return f32::NAN;
        }
        let mut sensitivities: Vec<f32> = self.blocks.iter().map(|b| b.sensitivity).collect();
        sensitivities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sensitivities.len() / 2;
        if sensitivities.len() % 2 == 0 {
            (sensitivities[mid - 1] + sensitivities[mid]) / 2.0
        } else {
            sensitivities[mid]
        }
    }

    /// Get the worst (highest sensitivity) k blocks, sorted descending.
    /// Returns (block_index, block_name, sensitivity, output_width).
    pub fn worst_k_blocks(&self, k: usize) -> Vec<(usize, String, f32, f32)> {
        let mut indexed: Vec<_> = self
            .blocks
            .iter()
            .map(|b| {
                (
                    b.block_index,
                    b.block_name.clone(),
                    b.sensitivity,
                    b.output_width,
                )
            })
            .collect();
        // Sort by sensitivity descending (highest first)
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Sensitivity range (max / min).
    pub fn sensitivity_range(&self) -> f32 {
        let min = self.min_sensitivity();
        if min <= 0.0 || !min.is_finite() {
            return f32::INFINITY;
        }
        self.max_sensitivity / min
    }
}

/// Truncate a name to fit in a given width.
pub fn truncate_name(name: &str, width: usize) -> String {
    if name.len() <= width {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - width + 3..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ===========================================
    // PropagationConfig tests
    // ===========================================

    #[test]
    fn test_propagation_config_default() {
        let config = PropagationConfig::default();
        assert_eq!(config.method, PropagationMethod::AlphaCrown);
        assert_eq!(config.max_iterations, 100);
        assert!((config.tolerance - 1e-4).abs() < 1e-8);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_propagation_config_custom() {
        let config = PropagationConfig {
            method: PropagationMethod::Ibp,
            max_iterations: 50,
            tolerance: 1e-6,
            use_gpu: true,
        };
        assert_eq!(config.method, PropagationMethod::Ibp);
        assert_eq!(config.max_iterations, 50);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_propagation_config_clone() {
        let config = PropagationConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.method, config.method);
        assert_eq!(cloned.max_iterations, config.max_iterations);
    }

    #[test]
    fn test_propagation_config_serialization() {
        let config = PropagationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PropagationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.method, config.method);
        assert_eq!(deserialized.max_iterations, config.max_iterations);
    }

    // ===========================================
    // PropagationMethod tests
    // ===========================================

    #[test]
    fn test_propagation_method_equality() {
        assert_eq!(PropagationMethod::Ibp, PropagationMethod::Ibp);
        assert_ne!(PropagationMethod::Ibp, PropagationMethod::Crown);
        assert_ne!(PropagationMethod::AlphaCrown, PropagationMethod::BetaCrown);
    }

    #[test]
    fn test_propagation_method_all_variants() {
        let methods = [
            PropagationMethod::Ibp,
            PropagationMethod::Crown,
            PropagationMethod::AlphaCrown,
            PropagationMethod::SdpCrown,
            PropagationMethod::BetaCrown,
        ];
        // All methods should be distinct
        for (i, m1) in methods.iter().enumerate() {
            for (j, m2) in methods.iter().enumerate() {
                if i == j {
                    assert_eq!(m1, m2);
                } else {
                    assert_ne!(m1, m2);
                }
            }
        }
    }

    #[test]
    fn test_propagation_method_serialization() {
        let method = PropagationMethod::SdpCrown;
        let json = serde_json::to_string(&method).unwrap();
        let deserialized: PropagationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, method);
    }

    // ===========================================
    // NodeBoundsInfo tests
    // ===========================================

    fn make_node_info(
        name: &str,
        sensitivity: f32,
        saturated: bool,
        has_nan: bool,
        has_infinite: bool,
    ) -> NodeBoundsInfo {
        NodeBoundsInfo {
            name: name.to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.1,
            output_width: sensitivity * 0.1,
            sensitivity,
            output_shape: vec![10],
            min_bound: -1.0,
            max_bound: 1.0,
            saturated,
            has_nan,
            has_infinite,
        }
    }

    #[test]
    fn test_node_bounds_info_has_degraded_false() {
        let node = make_node_info("test", 5.0, false, false, false);
        assert!(!node.has_degraded());
    }

    #[test]
    fn test_node_bounds_info_has_degraded_saturated() {
        let node = make_node_info("test", 5.0, true, false, false);
        assert!(node.has_degraded());
    }

    #[test]
    fn test_node_bounds_info_has_degraded_nan() {
        let node = make_node_info("test", 5.0, false, true, false);
        assert!(node.has_degraded());
    }

    #[test]
    fn test_node_bounds_info_has_degraded_infinite() {
        let node = make_node_info("test", 5.0, false, false, true);
        assert!(node.has_degraded());
    }

    #[test]
    fn test_node_bounds_info_status_nan() {
        let node = make_node_info("test", 5.0, false, true, false);
        assert_eq!(node.status(), "NAN");
    }

    #[test]
    fn test_node_bounds_info_status_inf() {
        let node = make_node_info("test", 5.0, false, false, true);
        assert_eq!(node.status(), "INF");
    }

    #[test]
    fn test_node_bounds_info_status_saturated() {
        let node = make_node_info("test", 5.0, true, false, false);
        assert_eq!(node.status(), "SATURATED");
    }

    #[test]
    fn test_node_bounds_info_status_high() {
        let node = make_node_info("test", 150.0, false, false, false);
        assert_eq!(node.status(), "HIGH");
    }

    #[test]
    fn test_node_bounds_info_status_moderate() {
        let node = make_node_info("test", 50.0, false, false, false);
        assert_eq!(node.status(), "MODERATE");
    }

    #[test]
    fn test_node_bounds_info_status_stable() {
        let node = make_node_info("test", 0.5, false, false, false);
        assert_eq!(node.status(), "STABLE");
    }

    #[test]
    fn test_node_bounds_info_status_ok() {
        let node = make_node_info("test", 5.0, false, false, false);
        assert_eq!(node.status(), "OK");
    }

    // ===========================================
    // LayerByLayerResult tests
    // ===========================================

    #[test]
    fn test_layer_by_layer_result_summary_empty() {
        let result = LayerByLayerResult {
            nodes: vec![],
            input_epsilon: 0.01,
            final_width: 0.0,
            degraded_at_node: None,
            total_nodes: 0,
        };
        let summary = result.summary();
        assert!(summary.contains("Layer-by-Layer Verification"));
        assert!(summary.contains("Input epsilon:"));
        assert!(summary.contains("Total nodes: 0"));
    }

    #[test]
    fn test_layer_by_layer_result_summary_with_nodes() {
        let result = LayerByLayerResult {
            nodes: vec![make_node_info("layer0", 2.5, false, false, false)],
            input_epsilon: 0.01,
            final_width: 0.025,
            degraded_at_node: None,
            total_nodes: 1,
        };
        let summary = result.summary();
        assert!(summary.contains("layer0"));
        assert!(summary.contains("OK"));
    }

    #[test]
    fn test_layer_by_layer_result_summary_with_degraded() {
        let result = LayerByLayerResult {
            nodes: vec![make_node_info("layer0", 2.5, true, false, false)],
            input_epsilon: 0.01,
            final_width: 0.025,
            degraded_at_node: Some(0),
            total_nodes: 1,
        };
        let summary = result.summary();
        assert!(summary.contains("WARNING"));
        assert!(summary.contains("SATURATED"));
    }

    // ===========================================
    // BlockBoundsInfo tests
    // ===========================================

    fn make_block_info(sensitivity: f32, degraded: bool) -> BlockBoundsInfo {
        BlockBoundsInfo {
            block_index: 0,
            block_name: "block0".to_string(),
            nodes: vec![],
            input_width: 0.1,
            output_width: sensitivity * 0.1,
            sensitivity,
            qk_matmul_width: None,
            swiglu_width: None,
            degraded,
        }
    }

    #[test]
    fn test_block_bounds_info_status_degraded() {
        let block = make_block_info(100.0, true);
        assert_eq!(block.status(), "DEGRADED");
    }

    #[test]
    fn test_block_bounds_info_status_high() {
        let block = make_block_info(1e7, false);
        assert_eq!(block.status(), "HIGH");
    }

    #[test]
    fn test_block_bounds_info_status_moderate() {
        let block = make_block_info(1e4, false);
        assert_eq!(block.status(), "MODERATE");
    }

    #[test]
    fn test_block_bounds_info_status_ok() {
        let block = make_block_info(100.0, false);
        assert_eq!(block.status(), "OK");
    }

    // ===========================================
    // BlockWiseResult tests
    // ===========================================

    fn make_block_wise_result(sensitivities: &[f32]) -> BlockWiseResult {
        let blocks: Vec<BlockBoundsInfo> = sensitivities
            .iter()
            .enumerate()
            .map(|(i, &s)| BlockBoundsInfo {
                block_index: i,
                block_name: format!("block{}", i),
                nodes: vec![],
                input_width: 0.1,
                output_width: s * 0.1,
                sensitivity: s,
                qk_matmul_width: None,
                swiglu_width: None,
                degraded: false,
            })
            .collect();
        let max_sensitivity = sensitivities.iter().cloned().fold(0.0f32, f32::max);
        BlockWiseResult {
            blocks,
            block_epsilon: 0.01,
            total_blocks: sensitivities.len(),
            max_sensitivity,
            degraded_blocks: 0,
        }
    }

    #[test]
    fn test_block_wise_result_min_sensitivity() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0, 3.0]);
        assert!((result.min_sensitivity() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_wise_result_min_sensitivity_empty() {
        let result = make_block_wise_result(&[]);
        assert!(result.min_sensitivity().is_infinite());
    }

    #[test]
    fn test_block_wise_result_median_sensitivity_odd() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0]);
        // Sorted: [5.0, 10.0, 20.0], median = 10.0
        assert!((result.median_sensitivity() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_wise_result_median_sensitivity_even() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0, 15.0]);
        // Sorted: [5.0, 10.0, 15.0, 20.0], median = (10.0 + 15.0) / 2 = 12.5
        assert!((result.median_sensitivity() - 12.5).abs() < 1e-6);
    }

    #[test]
    fn test_block_wise_result_median_sensitivity_empty() {
        let result = make_block_wise_result(&[]);
        assert!(result.median_sensitivity().is_nan());
    }

    #[test]
    fn test_block_wise_result_worst_k_blocks() {
        let result = make_block_wise_result(&[10.0, 50.0, 20.0, 100.0, 5.0]);
        let worst = result.worst_k_blocks(3);
        assert_eq!(worst.len(), 3);
        // Should be sorted by sensitivity descending
        assert_eq!(worst[0].0, 3); // block3, sensitivity 100.0
        assert_eq!(worst[1].0, 1); // block1, sensitivity 50.0
        assert_eq!(worst[2].0, 2); // block2, sensitivity 20.0
    }

    #[test]
    fn test_block_wise_result_worst_k_blocks_larger_than_available() {
        let result = make_block_wise_result(&[10.0, 5.0]);
        let worst = result.worst_k_blocks(10);
        assert_eq!(worst.len(), 2);
    }

    #[test]
    fn test_block_wise_result_sensitivity_range() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0]);
        // max=20, min=5, range=4
        assert!((result.sensitivity_range() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_wise_result_sensitivity_range_zero_min() {
        // Edge case: min sensitivity is 0
        let mut result = make_block_wise_result(&[0.0, 5.0, 20.0]);
        result.max_sensitivity = 20.0;
        assert!(result.sensitivity_range().is_infinite());
    }

    #[test]
    fn test_block_wise_result_summary() {
        let result = make_block_wise_result(&[10.0, 5.0]);
        let summary = result.summary();
        assert!(summary.contains("Block-wise Verification"));
        assert!(summary.contains("block0"));
        assert!(summary.contains("block1"));
        assert!(summary.contains("Total blocks: 2"));
    }

    // ===========================================
    // BlockProgress tests
    // ===========================================

    #[test]
    fn test_block_progress_fraction_normal() {
        let progress = BlockProgress {
            block_index: 2, // Completed blocks 0, 1, 2 (3 total)
            total_blocks: 10,
            block_name: "block2".to_string(),
            elapsed: Duration::from_secs(30),
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        // After completing block 2, we have 3/10 complete
        assert!((progress.fraction() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_block_progress_fraction_zero_total() {
        let progress = BlockProgress {
            block_index: 0,
            total_blocks: 0,
            block_name: "".to_string(),
            elapsed: Duration::ZERO,
            current_max_sensitivity: 0.0,
            degraded_so_far: 0,
        };
        assert!((progress.fraction() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_progress_eta() {
        let progress = BlockProgress {
            block_index: 4, // 5 blocks done
            total_blocks: 10,
            block_name: "block4".to_string(),
            elapsed: Duration::from_secs(50), // 10 sec per block
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        // 5 remaining blocks * 10 sec = 50 sec
        let eta = progress.eta();
        assert!((eta.as_secs_f64() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_block_progress_eta_complete() {
        let progress = BlockProgress {
            block_index: 9,
            total_blocks: 10,
            block_name: "block9".to_string(),
            elapsed: Duration::from_secs(100),
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        assert_eq!(progress.eta(), Duration::ZERO);
    }

    #[test]
    fn test_block_progress_estimated_remaining() {
        let progress = BlockProgress {
            block_index: 4, // 5 blocks done = 50%
            total_blocks: 10,
            block_name: "block4".to_string(),
            elapsed: Duration::from_secs(50),
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        let remaining = progress.estimated_remaining();
        // 50% done in 50 sec, so 50 sec remaining
        assert!((remaining.as_secs_f64() - 50.0).abs() < 0.1);
    }

    // ===========================================
    // LayerProgress tests
    // ===========================================

    #[test]
    fn test_layer_progress_fraction() {
        let progress = LayerProgress {
            node_index: 24, // 25 nodes done
            total_nodes: 100,
            node_name: "layer24".to_string(),
            layer_type: "Linear".to_string(),
            elapsed: Duration::from_secs(25),
            current_max_sensitivity: 3.0,
            degraded_so_far: 0,
        };
        assert!((progress.fraction() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_layer_progress_fraction_zero_total() {
        let progress = LayerProgress {
            node_index: 0,
            total_nodes: 0,
            node_name: "".to_string(),
            layer_type: "".to_string(),
            elapsed: Duration::ZERO,
            current_max_sensitivity: 0.0,
            degraded_so_far: 0,
        };
        assert!((progress.fraction() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_progress_estimated_remaining() {
        let progress = LayerProgress {
            node_index: 49, // 50 nodes done = 50%
            total_nodes: 100,
            node_name: "layer49".to_string(),
            layer_type: "ReLU".to_string(),
            elapsed: Duration::from_secs(50),
            current_max_sensitivity: 2.0,
            degraded_so_far: 0,
        };
        let remaining = progress.estimated_remaining();
        assert!((remaining.as_secs_f64() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_layer_progress_estimated_remaining_complete() {
        let progress = LayerProgress {
            node_index: 99,
            total_nodes: 100,
            node_name: "layer99".to_string(),
            layer_type: "Output".to_string(),
            elapsed: Duration::from_secs(100),
            current_max_sensitivity: 2.0,
            degraded_so_far: 0,
        };
        assert_eq!(progress.estimated_remaining(), Duration::ZERO);
    }

    // ===========================================
    // VerificationCheckpoint tests
    // ===========================================

    #[test]
    fn test_verification_checkpoint_new() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );
        assert_eq!(checkpoint.version, VerificationCheckpoint::VERSION);
        assert_eq!(checkpoint.epsilon, 0.01);
        assert_eq!(checkpoint.method, "alpha");
        assert_eq!(checkpoint.backend, "cpu");
        assert_eq!(checkpoint.total_blocks, 10);
        assert_eq!(checkpoint.next_block_index, 0);
        assert!(checkpoint.completed_blocks.is_empty());
    }

    #[test]
    fn test_verification_checkpoint_update() {
        let mut checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let block = make_block_info(50.0, false);
        checkpoint.update(block, 1000);

        assert_eq!(checkpoint.completed_blocks.len(), 1);
        assert_eq!(checkpoint.next_block_index, 1);
        assert!((checkpoint.max_sensitivity - 50.0).abs() < 1e-6);
        assert_eq!(checkpoint.degraded_blocks, 0);
    }

    #[test]
    fn test_verification_checkpoint_update_degraded() {
        let mut checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let block = make_block_info(50.0, true);
        checkpoint.update(block, 1000);

        assert_eq!(checkpoint.degraded_blocks, 1);
    }

    #[test]
    fn test_verification_checkpoint_is_complete() {
        let mut checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            2,
        );

        assert!(!checkpoint.is_complete());

        checkpoint.next_block_index = 2;
        assert!(checkpoint.is_complete());
    }

    #[test]
    fn test_verification_checkpoint_into_result() {
        let mut checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            2,
        );

        checkpoint
            .completed_blocks
            .push(make_block_info(10.0, false));
        checkpoint
            .completed_blocks
            .push(make_block_info(20.0, false));
        checkpoint.max_sensitivity = 20.0;

        let result = checkpoint.into_result();
        assert_eq!(result.blocks.len(), 2);
        assert_eq!(result.total_blocks, 2);
        assert!((result.max_sensitivity - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_verification_checkpoint_validate_success() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        assert!(checkpoint
            .validate(
                std::path::Path::new("/model.onnx"),
                "abc123",
                0.01,
                "alpha",
                "cpu"
            )
            .is_ok());
    }

    #[test]
    fn test_verification_checkpoint_validate_path_mismatch() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let result = checkpoint.validate(
            std::path::Path::new("/other.onnx"),
            "abc123",
            0.01,
            "alpha",
            "cpu",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model path"));
    }

    #[test]
    fn test_verification_checkpoint_validate_hash_mismatch() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let result = checkpoint.validate(
            std::path::Path::new("/model.onnx"),
            "different",
            0.01,
            "alpha",
            "cpu",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hash"));
    }

    #[test]
    fn test_verification_checkpoint_validate_epsilon_mismatch() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let result = checkpoint.validate(
            std::path::Path::new("/model.onnx"),
            "abc123",
            0.02, // Different epsilon
            "alpha",
            "cpu",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("epsilon"));
    }

    #[test]
    fn test_verification_checkpoint_validate_method_mismatch() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let result = checkpoint.validate(
            std::path::Path::new("/model.onnx"),
            "abc123",
            0.01,
            "ibp", // Different method
            "cpu",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("method"));
    }

    #[test]
    fn test_verification_checkpoint_validate_backend_mismatch() {
        let checkpoint = VerificationCheckpoint::new(
            std::path::PathBuf::from("/model.onnx"),
            "abc123".to_string(),
            0.01,
            "alpha",
            "cpu",
            10,
        );

        let result = checkpoint.validate(
            std::path::Path::new("/model.onnx"),
            "abc123",
            0.01,
            "alpha",
            "wgpu", // Different backend
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("backend"));
    }

    // ===========================================
    // Helper function tests
    // ===========================================

    #[test]
    fn test_truncate_name_short() {
        let name = "layer0";
        assert_eq!(truncate_name(name, 20), "layer0");
    }

    #[test]
    fn test_truncate_name_exact() {
        let name = "exactly_ten";
        assert_eq!(truncate_name(name, 11), "exactly_ten");
    }

    #[test]
    fn test_truncate_name_long() {
        let name = "very_long_layer_name_that_needs_truncation";
        let truncated = truncate_name(name, 20);
        assert_eq!(truncated.len(), 20);
        assert!(truncated.starts_with("..."));
    }

    #[test]
    fn test_is_leap_year_common() {
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(2021));
        assert!(!is_leap_year(1999));
    }

    #[test]
    fn test_is_leap_year_divisible_by_4() {
        assert!(is_leap_year(2024));
        assert!(is_leap_year(2020));
        assert!(is_leap_year(2016));
    }

    #[test]
    fn test_is_leap_year_divisible_by_100_not_400() {
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2100));
        assert!(!is_leap_year(2200));
    }

    #[test]
    fn test_is_leap_year_divisible_by_400() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(1600));
        assert!(is_leap_year(2400));
    }

    #[test]
    fn test_chrono_lite_now_format() {
        let timestamp = chrono_lite_now();
        // Should be ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        assert!(timestamp.contains('T'));
        assert!(timestamp.ends_with('Z'));
        assert_eq!(timestamp.len(), 20);
        // Parse to verify structure
        let parts: Vec<&str> = timestamp[..10].split('-').collect();
        assert_eq!(parts.len(), 3);
        let year: i32 = parts[0].parse().unwrap();
        let month: i32 = parts[1].parse().unwrap();
        let day: i32 = parts[2].parse().unwrap();
        assert!(year >= 2020);
        assert!((1..=12).contains(&month));
        assert!((1..=31).contains(&day));
    }

    #[test]
    fn test_compute_model_hash_nonexistent() {
        let result = compute_model_hash(std::path::Path::new("/nonexistent/file.onnx"));
        assert!(result.is_err());
    }
}
