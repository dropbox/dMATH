//! Lean5 GPU and Parallel Acceleration
//!
//! Provides accelerated batch operations:
//! - GPU WHNF reduction (for very large batches)
//! - CPU-parallel type checking via Rayon
//! - CPU-parallel WHNF reduction
//! - CPU-parallel definitional equality checking
//!
//! # Architecture
//!
//! This crate provides two acceleration strategies:
//!
//! 1. **GPU acceleration** (via wgpu): Best for very large batches (>1000 expressions)
//!    of simple operations. Currently supports WHNF reduction identification.
//!
//! 2. **CPU parallelism** (via Rayon): Best for type checking and complex operations.
//!    Speedup depends on core count and batch size. Run `cargo bench -p lean5-gpu` for measurements.
//!
//! # When to use GPU vs CPU parallelism
//!
//! - Use GPU (`GpuAccelerator::batch_whnf`) for:
//!   - Very large batches (>1000 expressions)
//!   - When expressions are already in WHNF (fast identification)
//!
//! - Use CPU parallelism (`ParallelBatch`) for:
//!   - Type checking (complex control flow)
//!   - Medium-sized batches (4-1000 expressions)
//!   - Operations requiring environment lookups
//!
//! # Example
//!
//! ```ignore
//! use lean5_gpu::{GpuAccelerator, ParallelBatch, ParallelConfig};
//! use lean5_kernel::{Environment, Expr, Level};
//!
//! // CPU-parallel type checking (recommended for most use cases)
//! let env = Environment::new();
//! let batch = ParallelBatch::new(&env);
//! let results = batch.batch_infer_type(&exprs);
//!
//! // GPU acceleration (for very large batches)
//! #[tokio::main]
//! async fn main() {
//!     let mut gpu = GpuAccelerator::new().await.expect("GPU init failed");
//!     let results = gpu.batch_whnf(&env, &large_batch).await;
//! }
//! ```

pub mod arena;
pub mod batch;
pub mod parallel;
pub mod shaders;

// Re-export key types
pub use arena::{GpuExpr, GpuExprArena, GpuLevel};
pub use batch::BatchWhnf;
pub use parallel::{
    parallel_infer_type, parallel_is_def_eq, parallel_whnf, ParallelBatch, ParallelConfig,
};
pub use shaders::{GpuConstDef, GpuPipelines, WhnfUniforms};

use lean5_kernel::{Environment, Expr, TypeError};

/// GPU accelerator for batch operations
///
/// This is the main entry point for GPU-accelerated theorem proving operations.
/// It manages GPU resources and provides high-level batch APIs.
pub struct GpuAccelerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Option<GpuPipelines>,
}

impl GpuAccelerator {
    /// Create a new GPU accelerator
    ///
    /// Returns an error if no GPU adapter is available.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|_| GpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        // Initialize pipelines lazily (can fail on shader compilation)
        let pipelines = None;

        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    /// Initialize compute pipelines
    ///
    /// Call this before using batch operations. Returns error if shader compilation fails.
    pub fn init_pipelines(&mut self) -> Result<(), GpuError> {
        if self.pipelines.is_none() {
            self.pipelines = Some(GpuPipelines::new(&self.device)?);
        }
        Ok(())
    }

    /// Get device reference (for advanced usage)
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference (for advanced usage)
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get pipelines reference (for advanced usage)
    pub fn pipelines(&self) -> Option<&GpuPipelines> {
        self.pipelines.as_ref()
    }

    /// Batch WHNF reduction
    ///
    /// Reduces multiple expressions to weak head normal form in parallel.
    /// Falls back to CPU for complex reductions (beta/delta/iota with substitution).
    ///
    /// For small batches (< 16 expressions), uses CPU directly to avoid GPU overhead.
    pub async fn batch_whnf(
        &mut self,
        env: &Environment,
        exprs: &[Expr],
    ) -> Result<Vec<Expr>, GpuError> {
        // Ensure pipelines are initialized
        self.init_pipelines()?;

        let pipelines = self.pipelines.as_ref().unwrap();
        let batch = BatchWhnf::new(&self.device, &self.queue, pipelines);
        batch.batch_whnf(env, exprs).await
    }

    /// Batch type check multiple expressions
    ///
    /// Type checks multiple expressions in parallel using CPU (Rayon).
    /// GPU is not suitable for type checking due to complex control flow
    /// and environment lookups. CPU parallelism provides better performance.
    pub fn batch_type_check(
        &self,
        env: &Environment,
        exprs: &[Expr],
    ) -> Vec<Result<Expr, TypeError>> {
        ParallelBatch::new(env).batch_infer_type(exprs)
    }

    /// Batch definitional equality check
    ///
    /// Checks if pairs of expressions are definitionally equal using CPU (Rayon).
    pub fn batch_is_def_eq(&self, env: &Environment, pairs: &[(Expr, Expr)]) -> Vec<bool> {
        ParallelBatch::new(env).batch_is_def_eq(pairs)
    }
}

/// GPU acceleration errors
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// No GPU adapter found on the system
    #[error("No GPU adapter found")]
    NoAdapter,

    /// Failed to request GPU device
    #[error("Failed to request device: {0}")]
    DeviceRequest(String),

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Buffer mapping error
    #[error("Buffer mapping failed: {0}")]
    BufferMapping(#[from] wgpu::BufferAsyncError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use lean5_kernel::Level;

    // Note: These tests require a GPU and are marked ignore by default.
    // Run with: cargo test --package lean5-gpu -- --ignored

    #[tokio::test]
    #[ignore = "requires GPU"]
    async fn test_gpu_accelerator_creation() {
        let gpu = GpuAccelerator::new().await;
        assert!(gpu.is_ok(), "GPU accelerator should be created");
    }

    #[tokio::test]
    #[ignore = "requires GPU"]
    async fn test_gpu_pipeline_init() {
        let mut gpu = GpuAccelerator::new().await.expect("GPU init");
        let result = gpu.init_pipelines();
        assert!(
            result.is_ok(),
            "Pipelines should compile: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    #[ignore = "requires GPU"]
    async fn test_batch_whnf_simple() {
        let mut gpu = GpuAccelerator::new().await.expect("GPU init");
        let env = Environment::new();

        // Test with expressions already in WHNF
        let exprs = vec![
            Expr::Sort(Level::Zero),
            Expr::Sort(Level::succ(Level::Zero)),
            Expr::BVar(0),
            Expr::BVar(1),
        ];

        // Small batch uses CPU directly
        let results = gpu
            .batch_whnf(&env, &exprs)
            .await
            .expect("WHNF should succeed");
        assert_eq!(results.len(), exprs.len());

        // Results should be unchanged (already in WHNF)
        assert_eq!(results[0], exprs[0]);
        assert_eq!(results[1], exprs[1]);
        assert_eq!(results[2], exprs[2]);
        assert_eq!(results[3], exprs[3]);
    }

    #[test]
    fn test_gpu_expr_arena_roundtrip() {
        // Test arena without GPU
        let mut arena = GpuExprArena::new();

        let expr = Expr::lam(
            lean5_kernel::BinderInfo::Default,
            Expr::Sort(Level::Zero),
            Expr::BVar(0),
        );

        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).expect("Should deserialize");

        assert_eq!(expr, recovered);
    }
}
