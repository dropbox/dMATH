//! GPU-accelerated bound propagation using wgpu.
//!
//! This module provides GPU acceleration for core bound propagation operations
//! using wgpu (WebGPU) for cross-platform compute shader support.
//!
//! ## Supported Backends
//!
//! - **Metal** (macOS/iOS) - Primary target for Apple Silicon
//! - **Vulkan** (Linux/Windows/Android)
//! - **DX12** (Windows)
//!
//! ## Usage
//!
//! ```ignore
//! use gamma_gpu::WgpuDevice;
//!
//! let device = WgpuDevice::new()?;
//! let result = device.linear_ibp(&input, &weight, bias)?;
//! ```

use crate::AcceleratedBoundPropagation;
use gamma_core::{GammaError, GemmEngine, Result};
use gamma_propagate::GraphNetwork;
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::borrow::Cow;
use tracing::{debug, info};
// DeviceExt is available but we use queue.write_buffer() instead for buffer pooling

/// GPU device for accelerated bound propagation via wgpu.
///
/// This struct manages the wgpu device, queue, and compute pipelines for
/// running bound propagation operations on the GPU.
///
/// ## Buffer Reuse
///
/// The device maintains a buffer pool to avoid per-call allocation overhead.
/// Buffers are reused when their size is sufficient for the current operation.
/// This significantly reduces the overhead for repeated operations with similar
/// sizes (common in neural network verification).
///
/// ## Chained Operations
///
/// For attention computation, the device supports chained operations that keep
/// intermediate results on the GPU, avoiding costly host roundtrips:
/// - `attention_ibp()`: Q @ K^T → scale → softmax → probs @ V in a single call
pub struct WgpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Linear IBP pipeline
    linear_ibp_pipeline: wgpu::ComputePipeline,
    linear_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // MatMul IBP pipeline
    matmul_ibp_pipeline: wgpu::ComputePipeline,
    matmul_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // Softmax IBP pipeline (two passes: reduce + apply)
    softmax_reduce_pipeline: wgpu::ComputePipeline,
    softmax_reduce_bind_group_layout: wgpu::BindGroupLayout,
    softmax_apply_pipeline: wgpu::ComputePipeline,
    softmax_apply_bind_group_layout: wgpu::BindGroupLayout,
    // Transpose IBP pipeline (for fused attention)
    transpose_ibp_pipeline: wgpu::ComputePipeline,
    transpose_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // Scale IBP pipeline (for fused attention)
    scale_ibp_pipeline: wgpu::ComputePipeline,
    scale_ibp_bind_group_layout: wgpu::BindGroupLayout,
    // GEMM pipeline (CROWN linear backward)
    gemm_f32_pipeline: wgpu::ComputePipeline,
    gemm_f32_bind_group_layout: wgpu::BindGroupLayout,
    /// Buffer pool for reuse across calls
    /// Uses Mutex for Sync, allowing use in rayon parallel contexts
    buffer_pool: std::sync::Mutex<BufferPool>,
}

/// Pool of reusable GPU buffers.
///
/// Buffers are grown as needed but never shrunk, avoiding repeated allocations
/// for operations with similar or smaller sizes.
#[derive(Default)]
struct BufferPool {
    // Linear IBP buffers
    /// Params uniform buffer for linear IBP (fixed size)
    linear_params_buffer: Option<wgpu::Buffer>,
    /// Input lower bounds storage buffer
    input_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Input upper bounds storage buffer
    input_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Weight positive parts storage buffer
    weight_pos_buffer: Option<(wgpu::Buffer, usize)>,
    /// Weight negative parts storage buffer
    weight_neg_buffer: Option<(wgpu::Buffer, usize)>,
    /// Bias storage buffer
    bias_buffer: Option<(wgpu::Buffer, usize)>,
    /// Output lower bounds storage buffer (shared with matmul)
    output_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Output upper bounds storage buffer (shared with matmul)
    output_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for output lower bounds readback (shared with matmul)
    staging_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for output upper bounds readback (shared with matmul)
    staging_upper_buffer: Option<(wgpu::Buffer, usize)>,

    // MatMul IBP buffers
    /// Params uniform buffer for matmul IBP (fixed size)
    matmul_params_buffer: Option<wgpu::Buffer>,
    /// A lower bounds storage buffer
    a_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// A upper bounds storage buffer
    a_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// B lower bounds storage buffer
    b_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// B upper bounds storage buffer
    b_upper_buffer: Option<(wgpu::Buffer, usize)>,

    // Softmax IBP buffers
    /// Params uniform buffer for softmax IBP (fixed size)
    softmax_params_buffer: Option<wgpu::Buffer>,
    /// Softmax intermediate: exp_lower
    softmax_exp_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: exp_upper
    softmax_exp_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: sum_exp_lower per row
    softmax_sum_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: sum_exp_upper per row
    softmax_sum_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: max_upper per row
    softmax_max_buffer: Option<(wgpu::Buffer, usize)>,

    // Fused attention buffers - keep intermediate results on GPU
    /// Params uniform buffer for transpose IBP (fixed size)
    transpose_params_buffer: Option<wgpu::Buffer>,
    /// Params uniform buffer for scale IBP (fixed size)
    scale_params_buffer: Option<wgpu::Buffer>,
    /// K transposed buffer (intermediate for attention)
    k_transposed_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// K transposed upper buffer (intermediate for attention)
    k_transposed_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// QK scores buffer (intermediate for attention)
    qk_scores_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// QK scores upper buffer (intermediate for attention)
    qk_scores_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Attention probs buffer (intermediate for attention)
    attn_probs_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Attention probs upper buffer (intermediate for attention)
    attn_probs_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Second matmul params buffer (for probs@V in fused attention)
    matmul_pv_params_buffer: Option<wgpu::Buffer>,

    // GEMM buffers (CROWN linear backward)
    /// Params uniform buffer for GEMM (fixed size)
    gemm_params_buffer: Option<wgpu::Buffer>,
    /// A storage buffer
    gemm_a_buffer: Option<(wgpu::Buffer, usize)>,
    /// B storage buffer
    gemm_b_buffer: Option<(wgpu::Buffer, usize)>,
    /// Output storage buffer
    gemm_out_buffer: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for output readback
    gemm_staging_buffer: Option<(wgpu::Buffer, usize)>,
}

/// Parameters for the linear IBP shader, passed via uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearIbpParams {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    _padding: u32,
}

/// Parameters for the matmul IBP shader, passed via uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulIbpParams {
    batch_size: u32,
    m: u32, // rows of A
    k: u32, // cols of A = rows of B
    n: u32, // cols of B
}

/// Parameters for the GEMM shader, passed via uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

/// Parameters for the softmax IBP shader, passed via uniform buffer.
/// Softmax is computed along the last dimension (row-wise for 2D).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxIbpParams {
    num_rows: u32, // Total number of rows (batch * leading dims)
    row_size: u32, // Size of softmax dimension (last axis)
    _padding: [u32; 2],
}

/// Parameters for the transpose IBP shader, passed via uniform buffer.
/// Transposes the last two dimensions of a tensor.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeIbpParams {
    batch_size: u32, // Product of all dims except last two
    rows: u32,       // Second-to-last dimension (before transpose)
    cols: u32,       // Last dimension (before transpose)
    _padding: u32,
}

/// Parameters for the scale IBP shader, passed via uniform buffer.
/// Element-wise multiplication by a scalar.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleIbpParams {
    total_elements: u32,
    scale: f32,
    _padding: [u32; 2],
}

impl WgpuDevice {
    /// Create a new GPU device.
    ///
    /// This initializes wgpu, selects the best available GPU backend,
    /// and compiles the compute shaders.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        // Create wgpu instance with all backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter (GPU selection)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GammaError::NotSupported("No GPU adapter found".to_string()))?;

        info!(
            "wgpu adapter: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gamma-gpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GammaError::NotSupported(format!("Failed to create device: {}", e)))?;

        // Create compute pipelines and bind group layouts
        let (linear_ibp_pipeline, linear_ibp_bind_group_layout) =
            Self::create_linear_ibp_pipeline(&device);
        let (matmul_ibp_pipeline, matmul_ibp_bind_group_layout) =
            Self::create_matmul_ibp_pipeline(&device);
        let (softmax_reduce_pipeline, softmax_reduce_bind_group_layout) =
            Self::create_softmax_reduce_pipeline(&device);
        let (softmax_apply_pipeline, softmax_apply_bind_group_layout) =
            Self::create_softmax_apply_pipeline(&device);
        let (transpose_ibp_pipeline, transpose_ibp_bind_group_layout) =
            Self::create_transpose_ibp_pipeline(&device);
        let (scale_ibp_pipeline, scale_ibp_bind_group_layout) =
            Self::create_scale_ibp_pipeline(&device);
        let (gemm_f32_pipeline, gemm_f32_bind_group_layout) =
            Self::create_gemm_f32_pipeline(&device);

        Ok(Self {
            device,
            queue,
            linear_ibp_pipeline,
            linear_ibp_bind_group_layout,
            matmul_ibp_pipeline,
            matmul_ibp_bind_group_layout,
            softmax_reduce_pipeline,
            softmax_reduce_bind_group_layout,
            softmax_apply_pipeline,
            softmax_apply_bind_group_layout,
            transpose_ibp_pipeline,
            transpose_ibp_bind_group_layout,
            scale_ibp_pipeline,
            scale_ibp_bind_group_layout,
            gemm_f32_pipeline,
            gemm_f32_bind_group_layout,
            buffer_pool: std::sync::Mutex::new(BufferPool::default()),
        })
    }

    /// Get information about the GPU device.
    pub fn info(&self) -> String {
        format!(
            "wgpu device (backend: {:?})",
            wgpu::Backends::all() // We don't store adapter info, just report all backends
        )
    }

    fn create_linear_ibp_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("linear_ibp_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LINEAR_IBP_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("linear_ibp_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Weight matrix (positive parts)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Weight matrix (negative parts)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bias (optional, zeros if not provided)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("linear_ibp_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("linear_ibp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_matmul_ibp_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_ibp_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MATMUL_IBP_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_ibp_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // A lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // A upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // B lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // B upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_ibp_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_ibp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_softmax_reduce_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_reduce_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFTMAX_REDUCE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("softmax_reduce_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: exp_lower
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: exp_upper
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: sum_exp_lower per row
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: sum_exp_upper per row
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: max_upper per row (for shift)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("softmax_reduce_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_reduce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_softmax_apply_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_apply_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFTMAX_APPLY_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("softmax_apply_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input: exp_lower
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input: exp_upper
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input: sum_exp_lower per row
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input: sum_exp_upper per row
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: softmax lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output: softmax upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("softmax_apply_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_apply_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_transpose_ibp_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transpose_ibp_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TRANSPOSE_IBP_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transpose_ibp_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transpose_ibp_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transpose_ibp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_scale_ibp_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scale_ibp_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCALE_IBP_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scale_ibp_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output lower bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output upper bounds
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scale_ibp_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scale_ibp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_gemm_f32_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_f32_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEMM_F32_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_f32_bind_group_layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // A
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // B
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Out
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_f32_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm_f32_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    /// Get or create a storage buffer, reusing from pool if possible.
    fn get_or_create_storage_buffer(
        &self,
        pool_slot: &mut Option<(wgpu::Buffer, usize)>,
        required_size: usize,
        label: &str,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Check if existing buffer is large enough
        if let Some((ref buffer, size)) = pool_slot {
            if *size >= required_size {
                return buffer.clone();
            }
        }

        // Create new buffer with 20% growth factor to avoid repeated resizing
        let new_size = (required_size as f64 * 1.2).ceil() as usize;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (new_size * std::mem::size_of::<f32>()) as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        *pool_slot = Some((buffer.clone(), new_size));
        buffer
    }

    pub(crate) fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        if m == 0 || k == 0 || n == 0 {
            return Ok(vec![]);
        }
        if a.len() != m * k {
            return Err(GammaError::shape_mismatch(vec![m, k], vec![a.len()]));
        }
        if b.len() != k * n {
            return Err(GammaError::shape_mismatch(vec![k, n], vec![b.len()]));
        }

        let out_elems = m * n;
        let params = GemmParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _padding: 0,
        };

        let mut pool = self.buffer_pool.lock().unwrap();

        if pool.gemm_params_buffer.is_none() {
            pool.gemm_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gemm_params_buffer"),
                size: std::mem::size_of::<GemmParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool.gemm_params_buffer.as_ref().unwrap().clone();

        let a_buffer = self.get_or_create_storage_buffer(
            &mut pool.gemm_a_buffer,
            a.len(),
            "gemm_a_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let b_buffer = self.get_or_create_storage_buffer(
            &mut pool.gemm_b_buffer,
            b.len(),
            "gemm_b_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let out_buffer = self.get_or_create_storage_buffer(
            &mut pool.gemm_out_buffer,
            out_elems,
            "gemm_out_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = self.get_or_create_storage_buffer(
            &mut pool.gemm_staging_buffer,
            out_elems,
            "gemm_staging_buffer",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        drop(pool);

        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue
            .write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm_f32_bind_group"),
            layout: &self.gemm_f32_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm_f32_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm_f32_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.gemm_f32_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // 2D dispatch over (n, m)
            let wg_x = 16u32;
            let wg_y = 16u32;
            compute_pass.dispatch_workgroups(
                (n as u32).div_ceil(wg_x),
                (m as u32).div_ceil(wg_y),
                1,
            );
        }

        let out_bytes = (out_elems * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(pollster::block_on(Self::read_buffer(
            &self.device,
            &staging,
            out_elems,
        )))
    }

    /// Execute linear IBP on the GPU with buffer reuse.
    fn execute_linear_ibp(
        &self,
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

        // Get input data
        let input_lower = input.lower.as_slice().unwrap();
        let input_upper = input.upper.as_slice().unwrap();

        // Bias (zeros if not provided)
        let bias_data: Vec<f32> = bias
            .map(|b| b.as_slice().unwrap().to_vec())
            .unwrap_or_else(|| vec![0.0; out_features]);

        // Create params
        let params = LinearIbpParams {
            batch_size: batch_size as u32,
            in_features: in_features as u32,
            out_features: out_features as u32,
            _padding: 0,
        };

        let input_size = batch_size * in_features;
        let weight_size = in_features * out_features;
        let output_size = batch_size * out_features;

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().unwrap();

        // Params buffer (fixed size, create once)
        if pool.linear_params_buffer.is_none() {
            pool.linear_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("linear_params_buffer"),
                size: std::mem::size_of::<LinearIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool.linear_params_buffer.as_ref().unwrap().clone();

        // Get or resize storage buffers
        let input_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_lower_buffer,
            input_size,
            "input_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let input_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_upper_buffer,
            input_size,
            "input_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let weight_pos_buffer = self.get_or_create_storage_buffer(
            &mut pool.weight_pos_buffer,
            weight_size,
            "weight_pos_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let weight_neg_buffer = self.get_or_create_storage_buffer(
            &mut pool.weight_neg_buffer,
            weight_size,
            "weight_neg_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let bias_buffer = self.get_or_create_storage_buffer(
            &mut pool.bias_buffer,
            out_features,
            "bias_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            output_size,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            output_size,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers for readback
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            output_size,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            output_size,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Drop pool borrow before write operations
        drop(pool);

        // Write data to buffers using queue.write_buffer (avoids buffer creation)
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&input_lower_buffer, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&input_upper_buffer, 0, bytemuck::cast_slice(input_upper));
        self.queue
            .write_buffer(&weight_pos_buffer, 0, bytemuck::cast_slice(&weight_pos));
        self.queue
            .write_buffer(&weight_neg_buffer, 0, bytemuck::cast_slice(&weight_neg));
        self.queue
            .write_buffer(&bias_buffer, 0, bytemuck::cast_slice(&bias_data));

        // Create bind group (must be created each call since buffers may have changed)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("linear_ibp_bind_group"),
            layout: &self.linear_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weight_pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weight_neg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("linear_ibp_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("linear_ibp_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.linear_ibp_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: one thread per (batch, output_feature) pair
            // Workgroup size is 64 threads (defined in shader)
            let workgroup_count = ((batch_size * out_features) as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let output_buffer_size = (output_size * std::mem::size_of::<f32>()) as u64;

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &output_lower_buffer,
            0,
            &staging_lower,
            0,
            output_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &output_upper_buffer,
            0,
            &staging_upper,
            0,
            output_buffer_size,
        );

        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let result_lower =
            pollster::block_on(Self::read_buffer(&self.device, &staging_lower, output_size));
        let result_upper =
            pollster::block_on(Self::read_buffer(&self.device, &staging_upper, output_size));

        // Reshape to output shape [..., out_features]
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(out_features);

        let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
            .map_err(|_| GammaError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
            .map_err(|_| GammaError::shape_mismatch(out_shape, vec![output_size]))?;

        BoundedTensor::new(lower, upper)
    }

    /// Execute matmul IBP on the GPU with buffer reuse.
    #[allow(clippy::too_many_arguments)]
    fn execute_matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
        batch_dims: &[usize],
    ) -> Result<BoundedTensor> {
        // Get input data
        let a_lower_data = input_a.lower.as_slice().unwrap();
        let a_upper_data = input_a.upper.as_slice().unwrap();
        let b_lower_data = input_b.lower.as_slice().unwrap();
        let b_upper_data = input_b.upper.as_slice().unwrap();

        // Create params
        let params = MatmulIbpParams {
            batch_size: batch_size as u32,
            m: m as u32,
            k: k as u32,
            n: n as u32,
        };

        let a_size = batch_size * m * k;
        let b_size = batch_size * k * n;
        let output_size = batch_size * m * n;

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().unwrap();

        // Matmul params buffer
        if pool.matmul_params_buffer.is_none() {
            pool.matmul_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_params_buffer"),
                size: std::mem::size_of::<MatmulIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool.matmul_params_buffer.as_ref().unwrap().clone();

        // Get or resize storage buffers
        let a_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_lower_buffer,
            a_size,
            "a_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let a_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_upper_buffer,
            a_size,
            "a_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let b_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_lower_buffer,
            b_size,
            "b_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let b_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_upper_buffer,
            b_size,
            "b_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Reuse output buffers from linear IBP
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            output_size,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            output_size,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers for readback
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            output_size,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            output_size,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Drop pool borrow before write operations
        drop(pool);

        // Write data to buffers
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&a_lower_buffer, 0, bytemuck::cast_slice(a_lower_data));
        self.queue
            .write_buffer(&a_upper_buffer, 0, bytemuck::cast_slice(a_upper_data));
        self.queue
            .write_buffer(&b_lower_buffer, 0, bytemuck::cast_slice(b_lower_data));
        self.queue
            .write_buffer(&b_upper_buffer, 0, bytemuck::cast_slice(b_upper_data));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_ibp_bind_group"),
            layout: &self.matmul_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_ibp_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_ibp_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_ibp_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per output element
            let workgroup_count = (output_size as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let output_buffer_size = (output_size * std::mem::size_of::<f32>()) as u64;

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &output_lower_buffer,
            0,
            &staging_lower,
            0,
            output_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &output_upper_buffer,
            0,
            &staging_upper,
            0,
            output_buffer_size,
        );

        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let result_lower =
            pollster::block_on(Self::read_buffer(&self.device, &staging_lower, output_size));
        let result_upper =
            pollster::block_on(Self::read_buffer(&self.device, &staging_upper, output_size));

        // Build output shape: [...batch_dims, m, n]
        let mut out_shape = batch_dims.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
            .map_err(|_| GammaError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
            .map_err(|_| GammaError::shape_mismatch(out_shape, vec![output_size]))?;

        BoundedTensor::new(lower, upper)
    }

    async fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        result[..count].to_vec()
    }

    /// Execute softmax IBP on the GPU.
    ///
    /// This is a two-pass operation:
    /// 1. Reduce pass: compute exp values and sums per row
    /// 2. Apply pass: compute final softmax bounds using Auto-LiRPA formula
    pub fn softmax_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(GammaError::InvalidSpec(
                "Empty input to softmax".to_string(),
            ));
        }

        // Softmax is along the last dimension
        let row_size = shape[shape.len() - 1];
        let num_rows: usize = shape[..shape.len() - 1].iter().product();
        let num_rows = if num_rows == 0 { 1 } else { num_rows };
        let total_elements = num_rows * row_size;

        debug!(
            "WgpuDevice softmax_ibp: num_rows={}, row_size={}",
            num_rows, row_size
        );

        // Get input data
        let input_lower = input.lower.as_slice().unwrap();
        let input_upper = input.upper.as_slice().unwrap();

        // Create params
        let params = SoftmaxIbpParams {
            num_rows: num_rows as u32,
            row_size: row_size as u32,
            _padding: [0, 0],
        };

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().unwrap();

        // Softmax params buffer
        if pool.softmax_params_buffer.is_none() {
            pool.softmax_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("softmax_params_buffer"),
                size: std::mem::size_of::<SoftmaxIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool.softmax_params_buffer.as_ref().unwrap().clone();

        // Input buffers (reuse from linear IBP)
        let input_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_lower_buffer,
            total_elements,
            "input_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let input_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_upper_buffer,
            total_elements,
            "input_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Intermediate buffers
        let exp_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_lower_buffer,
            total_elements,
            "exp_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let exp_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_upper_buffer,
            total_elements,
            "exp_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let sum_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_lower_buffer,
            num_rows,
            "sum_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let sum_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_upper_buffer,
            num_rows,
            "sum_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let max_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_max_buffer,
            num_rows,
            "max_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Output buffers
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            total_elements,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            total_elements,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            total_elements,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            total_elements,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        drop(pool);

        // Write data to buffers
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&input_lower_buffer, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&input_upper_buffer, 0, bytemuck::cast_slice(input_upper));

        // Create bind group for reduce pass
        let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_reduce_bind_group"),
            layout: &self.softmax_reduce_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: exp_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: exp_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sum_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sum_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: max_buffer.as_entire_binding(),
                },
            ],
        });

        // Create bind group for apply pass
        let apply_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_apply_bind_group"),
            layout: &self.softmax_apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exp_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: exp_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sum_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sum_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax_ibp_encoder"),
            });

        // Pass 1: Reduce (one thread per row)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_reduce_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_reduce_pipeline);
            compute_pass.set_bind_group(0, &reduce_bind_group, &[]);
            let workgroup_count = (num_rows as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 2: Apply (one thread per element)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_apply_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_apply_pipeline);
            compute_pass.set_bind_group(0, &apply_bind_group, &[]);
            let workgroup_count = (total_elements as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let output_buffer_size = (total_elements * std::mem::size_of::<f32>()) as u64;

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &output_lower_buffer,
            0,
            &staging_lower,
            0,
            output_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &output_upper_buffer,
            0,
            &staging_upper,
            0,
            output_buffer_size,
        );

        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let result_lower = pollster::block_on(Self::read_buffer(
            &self.device,
            &staging_lower,
            total_elements,
        ));
        let result_upper = pollster::block_on(Self::read_buffer(
            &self.device,
            &staging_upper,
            total_elements,
        ));

        // Reshape to original shape
        let lower = ArrayD::from_shape_vec(IxDyn(shape), result_lower)
            .map_err(|_| GammaError::shape_mismatch(shape.to_vec(), vec![total_elements]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(shape), result_upper)
            .map_err(|_| GammaError::shape_mismatch(shape.to_vec(), vec![total_elements]))?;

        BoundedTensor::new(lower, upper)
    }

    /// Chained attention IBP: Q @ K^T -> scale -> softmax -> probs @ V
    ///
    /// This method chains all attention operations on the GPU without intermediate
    /// host roundtrips, providing significant speedup compared to separate calls.
    ///
    /// # Arguments
    /// * `q` - Query tensor with shape [batch, heads, seq, dim]
    /// * `k` - Key tensor with shape [batch, heads, seq, dim]
    /// * `v` - Value tensor with shape [batch, heads, seq, dim]
    /// * `scale` - Scaling factor (typically 1.0 / sqrt(dim))
    ///
    /// # Returns
    /// Output tensor with shape [batch, heads, seq, dim]
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
            "WgpuDevice attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T
        // K shape: [batch, heads, seq, dim]
        // K^T shape: [batch, heads, dim, seq]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T
        // Q: [batch, heads, seq, dim] @ K^T: [batch, heads, dim, seq]
        // -> scores: [batch, heads, seq, seq]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Softmax over last dimension (seq)
        let probs = self.softmax_ibp(&scores_scaled)?;

        // Step 5: probs @ V
        // probs: [batch, heads, seq, seq] @ V: [batch, heads, seq, dim]
        // -> output: [batch, heads, seq, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Fused attention IBP with all operations on GPU.
    ///
    /// This method keeps all intermediate results on the GPU, eliminating host
    /// roundtrips between operations. The pipeline is:
    /// 1. Upload Q, K, V to GPU
    /// 2. K^T = transpose(K) on GPU
    /// 3. scores = Q @ K^T on GPU
    /// 4. scores_scaled = scores * scale on GPU
    /// 5. probs = softmax(scores_scaled) on GPU
    /// 6. output = probs @ V on GPU
    /// 7. Read back output
    ///
    /// This provides significant speedup by eliminating 3 intermediate readbacks.
    ///
    /// # Arguments
    /// * `q` - Query tensor with shape [batch, heads, seq, dim]
    /// * `k` - Key tensor with shape [batch, heads, seq, dim]
    /// * `v` - Value tensor with shape [batch, heads, seq, dim]
    /// * `scale` - Scaling factor (typically 1.0 / sqrt(dim))
    ///
    /// # Returns
    /// Output tensor with shape [batch, heads, seq, dim]
    pub fn attention_ibp_fused(
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
        let batch_heads = batch * heads;

        debug!(
            "WgpuDevice attention_ibp_fused: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Buffer sizes
        let qkv_size = batch_heads * seq * dim; // Q, K, V: [batch*heads, seq, dim]
        let kt_size = batch_heads * dim * seq; // K^T: [batch*heads, dim, seq]
        let scores_size = batch_heads * seq * seq; // scores: [batch*heads, seq, seq]
        let output_size = batch_heads * seq * dim; // output: [batch*heads, seq, dim]
        let softmax_rows = batch_heads * seq; // Number of softmax rows

        // Get input data as contiguous slices
        let q_lower_data = q.lower.as_slice().unwrap();
        let q_upper_data = q.upper.as_slice().unwrap();
        let k_lower_data = k.lower.as_slice().unwrap();
        let k_upper_data = k.upper.as_slice().unwrap();
        let v_lower_data = v.lower.as_slice().unwrap();
        let v_upper_data = v.upper.as_slice().unwrap();

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().unwrap();

        // === INPUT BUFFERS ===
        // Q buffers (reuse a_lower/a_upper from matmul)
        let q_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_lower_buffer,
            qkv_size,
            "q_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let q_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_upper_buffer,
            qkv_size,
            "q_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // K buffers (reuse input_lower/input_upper)
        let k_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_lower_buffer,
            qkv_size,
            "k_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let k_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_upper_buffer,
            qkv_size,
            "k_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // V buffers (reuse b_lower/b_upper from matmul)
        let v_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_lower_buffer,
            qkv_size,
            "v_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let v_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_upper_buffer,
            qkv_size,
            "v_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // === INTERMEDIATE BUFFERS ===
        // K^T buffers
        let kt_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.k_transposed_lower_buffer,
            kt_size,
            "kt_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let kt_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.k_transposed_upper_buffer,
            kt_size,
            "kt_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // QK scores buffers
        let qk_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.qk_scores_lower_buffer,
            scores_size,
            "qk_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let qk_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.qk_scores_upper_buffer,
            scores_size,
            "qk_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Scaled scores buffers (reuse softmax exp buffers)
        let scaled_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_lower_buffer,
            scores_size,
            "scaled_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let scaled_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_upper_buffer,
            scores_size,
            "scaled_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Softmax intermediate buffers
        let sum_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_lower_buffer,
            softmax_rows,
            "sum_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let sum_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_upper_buffer,
            softmax_rows,
            "sum_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let max_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_max_buffer,
            softmax_rows,
            "max_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Attention probs buffers
        let probs_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.attn_probs_lower_buffer,
            scores_size,
            "probs_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let probs_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.attn_probs_upper_buffer,
            scores_size,
            "probs_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // === OUTPUT BUFFERS ===
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            output_size,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            output_size,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers for final readback
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            output_size,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            output_size,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // === PARAMS BUFFERS ===
        // Transpose params
        if pool.transpose_params_buffer.is_none() {
            pool.transpose_params_buffer =
                Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("transpose_params_buffer"),
                    size: std::mem::size_of::<TransposeIbpParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        let transpose_params_buffer = pool.transpose_params_buffer.as_ref().unwrap().clone();

        // Matmul params (for Q @ K^T)
        if pool.matmul_params_buffer.is_none() {
            pool.matmul_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_params_buffer"),
                size: std::mem::size_of::<MatmulIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let matmul_qk_params_buffer = pool.matmul_params_buffer.as_ref().unwrap().clone();

        // Scale params
        if pool.scale_params_buffer.is_none() {
            pool.scale_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scale_params_buffer"),
                size: std::mem::size_of::<ScaleIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let scale_params_buffer = pool.scale_params_buffer.as_ref().unwrap().clone();

        // Softmax params
        if pool.softmax_params_buffer.is_none() {
            pool.softmax_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("softmax_params_buffer"),
                size: std::mem::size_of::<SoftmaxIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let softmax_params_buffer = pool.softmax_params_buffer.as_ref().unwrap().clone();

        // Second matmul params (for probs @ V, different from Q @ K^T)
        if pool.matmul_pv_params_buffer.is_none() {
            pool.matmul_pv_params_buffer =
                Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("matmul_pv_params_buffer"),
                    size: std::mem::size_of::<MatmulIbpParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        let matmul_pv_params_buffer = pool.matmul_pv_params_buffer.as_ref().unwrap().clone();

        drop(pool);

        // === WRITE DATA TO BUFFERS ===
        // Input data
        self.queue
            .write_buffer(&q_lower_buffer, 0, bytemuck::cast_slice(q_lower_data));
        self.queue
            .write_buffer(&q_upper_buffer, 0, bytemuck::cast_slice(q_upper_data));
        self.queue
            .write_buffer(&k_lower_buffer, 0, bytemuck::cast_slice(k_lower_data));
        self.queue
            .write_buffer(&k_upper_buffer, 0, bytemuck::cast_slice(k_upper_data));
        self.queue
            .write_buffer(&v_lower_buffer, 0, bytemuck::cast_slice(v_lower_data));
        self.queue
            .write_buffer(&v_upper_buffer, 0, bytemuck::cast_slice(v_upper_data));

        // Params
        let transpose_params = TransposeIbpParams {
            batch_size: batch_heads as u32,
            rows: seq as u32,
            cols: dim as u32,
            _padding: 0,
        };
        self.queue.write_buffer(
            &transpose_params_buffer,
            0,
            bytemuck::cast_slice(&[transpose_params]),
        );

        let matmul_qk_params = MatmulIbpParams {
            batch_size: batch_heads as u32,
            m: seq as u32, // rows of Q
            k: dim as u32, // cols of Q = rows of K^T
            n: seq as u32, // cols of K^T
        };
        self.queue.write_buffer(
            &matmul_qk_params_buffer,
            0,
            bytemuck::cast_slice(&[matmul_qk_params]),
        );

        let scale_params = ScaleIbpParams {
            total_elements: scores_size as u32,
            scale,
            _padding: [0, 0],
        };
        self.queue.write_buffer(
            &scale_params_buffer,
            0,
            bytemuck::cast_slice(&[scale_params]),
        );

        let softmax_params = SoftmaxIbpParams {
            num_rows: softmax_rows as u32,
            row_size: seq as u32,
            _padding: [0, 0],
        };
        self.queue.write_buffer(
            &softmax_params_buffer,
            0,
            bytemuck::cast_slice(&[softmax_params]),
        );

        // Matmul params for probs @ V (different dimensions from Q @ K^T)
        let matmul_pv_params = MatmulIbpParams {
            batch_size: batch_heads as u32,
            m: seq as u32, // rows of probs
            k: seq as u32, // cols of probs = rows of V
            n: dim as u32, // cols of V
        };
        self.queue.write_buffer(
            &matmul_pv_params_buffer,
            0,
            bytemuck::cast_slice(&[matmul_pv_params]),
        );

        // === CREATE BIND GROUPS ===
        // 1. Transpose K -> K^T
        let transpose_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bind_group"),
            layout: &self.transpose_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transpose_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: k_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kt_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: kt_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // 2. Q @ K^T -> scores
        let matmul_qk_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_qk_bind_group"),
            layout: &self.matmul_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matmul_qk_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: q_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kt_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: kt_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: qk_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: qk_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // 3. Scale scores
        let scale_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bind_group"),
            layout: &self.scale_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scale_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qk_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qk_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scaled_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scaled_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // 4. Softmax reduce pass
        let softmax_reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_reduce_bind_group_fused"),
            layout: &self.softmax_reduce_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: softmax_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scaled_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scaled_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: probs_lower_buffer.as_entire_binding(), // Reuse for exp_lower
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: probs_upper_buffer.as_entire_binding(), // Reuse for exp_upper
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sum_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sum_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: max_buffer.as_entire_binding(),
                },
            ],
        });

        // 5. Softmax apply pass (output to scaled buffers, will be input to final matmul)
        // We need separate buffers for apply output since probs_* has exp values from reduce
        let softmax_apply_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_apply_bind_group_fused"),
            layout: &self.softmax_apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: softmax_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: probs_lower_buffer.as_entire_binding(), // exp_lower from reduce
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: probs_upper_buffer.as_entire_binding(), // exp_upper from reduce
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sum_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sum_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: qk_lower_buffer.as_entire_binding(), // Reuse for final probs
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: qk_upper_buffer.as_entire_binding(), // Reuse for final probs
                },
            ],
        });

        // 6. Final matmul: probs @ V -> output
        // Uses separate params buffer (matmul_pv_params_buffer) written earlier
        let matmul_pv_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_pv_bind_group"),
            layout: &self.matmul_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matmul_pv_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qk_lower_buffer.as_entire_binding(), // Final probs (from softmax apply)
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qk_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // === ENCODE ALL OPERATIONS ===
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("attention_ibp_fused_encoder"),
            });

        // Pass 1: Transpose K -> K^T
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transpose_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.transpose_ibp_pipeline);
            compute_pass.set_bind_group(0, &transpose_bind_group, &[]);
            let workgroup_count = (kt_size as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 2: Q @ K^T -> scores
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_qk_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_ibp_pipeline);
            compute_pass.set_bind_group(0, &matmul_qk_bind_group, &[]);
            let workgroup_count = (scores_size as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 3: Scale scores
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.scale_ibp_pipeline);
            compute_pass.set_bind_group(0, &scale_bind_group, &[]);
            let workgroup_count = (scores_size as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 4: Softmax reduce
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_reduce_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_reduce_pipeline);
            compute_pass.set_bind_group(0, &softmax_reduce_bind_group, &[]);
            let workgroup_count = (softmax_rows as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 5: Softmax apply
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_apply_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_apply_pipeline);
            compute_pass.set_bind_group(0, &softmax_apply_bind_group, &[]);
            let workgroup_count = (scores_size as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 6: probs @ V -> output (uses matmul_pv_params_buffer written earlier)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pv_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_ibp_pipeline);
            compute_pass.set_bind_group(0, &matmul_pv_bind_group, &[]);
            let workgroup_count = (output_size as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy results to staging buffers
        let output_buffer_size = (output_size * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(
            &output_lower_buffer,
            0,
            &staging_lower,
            0,
            output_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &output_upper_buffer,
            0,
            &staging_upper,
            0,
            output_buffer_size,
        );

        // Submit all passes in one command buffer
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back final results only
        let result_lower =
            pollster::block_on(Self::read_buffer(&self.device, &staging_lower, output_size));
        let result_upper =
            pollster::block_on(Self::read_buffer(&self.device, &staging_upper, output_size));

        // Build output shape: [batch, heads, seq, dim]
        let out_shape = vec![batch, heads, seq, dim];
        let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
            .map_err(|_| GammaError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
            .map_err(|_| GammaError::shape_mismatch(out_shape, vec![output_size]))?;

        BoundedTensor::new(lower, upper)
    }

    /// Causal attention IBP with GPU acceleration (hybrid: GPU matmul, CPU causal softmax).
    ///
    /// Computes: causal_softmax((Q @ K^T) * scale) @ V
    ///
    /// This is for decoder-only (LLaMA, GPT) and decoder blocks (Whisper decoder).
    /// Position i can only attend to positions j where j <= i.
    ///
    /// # Arguments
    /// * `q` - Query tensor with shape [batch, heads, seq, dim]
    /// * `k` - Key tensor with shape [batch, heads, seq, dim]
    /// * `v` - Value tensor with shape [batch, heads, seq, dim]
    /// * `scale` - Scaling factor (typically 1.0 / sqrt(dim))
    ///
    /// # Returns
    /// Output tensor with shape [batch, heads, seq, dim]
    ///
    /// # Implementation Note
    /// Uses GPU for matmul operations but CPU for causal softmax since the
    /// causal mask requires per-row varying normalization.
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
            "WgpuDevice causal_attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T on CPU (fast transpose operation)
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using GPU matmul
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Causal softmax using CPU
        // This applies the lower-triangular mask: position i attends only to j <= i
        let probs = gamma_transformer::causal_softmax_bounds(&scores_scaled, -1)?;

        // Step 5: probs @ V using GPU matmul
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Cross-attention IBP for encoder-decoder models (e.g., Whisper) using GPU.
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
    ///
    /// Uses GPU for matmul operations, CPU for standard softmax.
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
            "WgpuDevice cross_attention_ibp: batch={}, heads={}, seq_dec={}, seq_enc={}, dim={}, scale={}",
            batch, heads, seq_dec, seq_enc, dim, scale
        );

        // Step 1: Compute K^T on CPU (fast transpose operation)
        // K shape: [batch, heads, seq_enc, dim]
        // K^T shape: [batch, heads, dim, seq_enc]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using GPU matmul
        // Q: [batch, heads, seq_dec, dim] @ K^T: [batch, heads, dim, seq_enc]
        // -> scores: [batch, heads, seq_dec, seq_enc]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Standard softmax (no causal mask)
        // Softmax over last dimension (seq_enc)
        let probs = gamma_transformer::softmax_bounds(&scores_scaled, -1)?;

        // Step 5: probs @ V using GPU matmul
        // probs: [batch, heads, seq_dec, seq_enc] @ V: [batch, heads, seq_enc, dim]
        // -> output: [batch, heads, seq_dec, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }
}

impl AcceleratedBoundPropagation for WgpuDevice {
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
            "WgpuDevice linear_ibp: batch={}, in={}, out={}",
            batch_size, in_features, out_features
        );

        self.execute_linear_ibp(input, weight, bias, batch_size, in_features, out_features)
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

        debug!(
            "WgpuDevice matmul_ibp: batch={}, m={}, k={}, n={}",
            batch_size, m, k, n
        );

        self.execute_matmul_ibp(input_a, input_b, batch_size, m, k, n, batch_dims_a)
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // GPU engines (wgpu) can't safely participate in Rayon parallel CROWN due to internal
        // buffer reuse and readback synchronization. Instead, run per-position CROWN
        // sequentially while accelerating GEMM via the GPU.
        debug!(
            "WgpuDevice crown_per_position_parallel: using GPU-accelerated sequential per-position CROWN"
        );
        crate::crown_per_position_sequential_with_engine(graph, input, Some(self))
    }
}

/// Implementation of GemmEngine for WgpuDevice to enable GPU-accelerated CROWN.
impl GemmEngine for WgpuDevice {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.gemm_f32(m, k, n, a, b)
    }
}

/// WGSL shader for linear layer IBP.
///
/// This shader computes:
/// - lower = W_pos @ x_l + W_neg @ x_u + bias
/// - upper = W_pos @ x_u + W_neg @ x_l + bias
///
/// where W_pos = max(W, 0), W_neg = min(W, 0)
const LINEAR_IBP_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read> weight_pos: array<f32>;
@group(0) @binding(4) var<storage, read> weight_neg: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = params.batch_size * params.out_features;

    if (idx >= total_outputs) {
        return;
    }

    // Compute batch index and output feature index
    let batch_idx = idx / params.out_features;
    let out_idx = idx % params.out_features;

    // Input offset for this batch element
    let input_offset = batch_idx * params.in_features;

    // Weight offset for this output feature (row-major: [out_features, in_features])
    let weight_offset = out_idx * params.in_features;

    // Compute dot products with interval arithmetic
    var low: f32 = 0.0;
    var high: f32 = 0.0;

    for (var i: u32 = 0u; i < params.in_features; i = i + 1u) {
        let xl = input_lower[input_offset + i];
        let xu = input_upper[input_offset + i];
        let wp = weight_pos[weight_offset + i];
        let wn = weight_neg[weight_offset + i];

        // lower = W_pos @ x_l + W_neg @ x_u
        low = low + wp * xl + wn * xu;
        // upper = W_pos @ x_u + W_neg @ x_l
        high = high + wp * xu + wn * xl;
    }

    // Add bias
    low = low + bias[out_idx];
    high = high + bias[out_idx];

    // Write output
    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for GEMM (C = A @ B) used by CROWN linear backward.
///
/// A: [m, k] row-major
/// B: [k, n] row-major
/// C: [m, n] row-major
const GEMM_F32_SHADER: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) {
        return;
    }
    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < params.k; t = t + 1u) {
        let a_idx = row * params.k + t;
        let b_idx = t * params.n + col;
        sum = sum + a[a_idx] * b[b_idx];
    }
    out[row * params.n + col] = sum;
}
"#;

/// WGSL shader for batched matrix multiplication IBP.
///
/// This shader computes [A_l, A_u] @ [B_l, B_u] with interval arithmetic.
/// For each output element, computes 4 products and takes min/max.
const MATMUL_IBP_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    m: u32,           // rows of A
    k: u32,           // cols of A = rows of B
    n: u32,           // cols of B
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper: array<f32>;
@group(0) @binding(3) var<storage, read> b_lower: array<f32>;
@group(0) @binding(4) var<storage, read> b_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let output_matrix_size = params.m * params.n;
    let total_outputs = params.batch_size * output_matrix_size;

    if (idx >= total_outputs) {
        return;
    }

    // Decompose index: idx = batch * (m * n) + i * n + j
    let batch_idx = idx / output_matrix_size;
    let matrix_idx = idx % output_matrix_size;
    let i = matrix_idx / params.n;  // row in output
    let j = matrix_idx % params.n;  // col in output

    // Matrix offsets
    let a_matrix_size = params.m * params.k;
    let b_matrix_size = params.k * params.n;
    let a_offset = batch_idx * a_matrix_size;
    let b_offset = batch_idx * b_matrix_size;

    // Compute dot product with interval arithmetic
    var low: f32 = 0.0;
    var high: f32 = 0.0;

    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        // A[batch, i, kk]
        let a_l = a_lower[a_offset + i * params.k + kk];
        let a_u = a_upper[a_offset + i * params.k + kk];
        // B[batch, kk, j]
        let b_l = b_lower[b_offset + kk * params.n + j];
        let b_u = b_upper[b_offset + kk * params.n + j];

        // Interval multiplication: compute all 4 products, take min/max
        let p1 = a_l * b_l;
        let p2 = a_l * b_u;
        let p3 = a_u * b_l;
        let p4 = a_u * b_u;

        let min_prod = min(min(p1, p2), min(p3, p4));
        let max_prod = max(max(p1, p2), max(p3, p4));

        low = low + min_prod;
        high = high + max_prod;
    }

    // Write output
    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for softmax IBP - Pass 1: Reduction.
///
/// For each row, computes:
/// - max_upper = max of upper bounds (for numerical stability)
/// - exp_lower[i] = exp(input_lower[i] - max_upper)
/// - exp_upper[i] = exp(input_upper[i] - max_upper)
/// - sum_exp_lower = sum of exp_lower
/// - sum_exp_upper = sum of exp_upper
///
/// This pass runs one thread per row to perform the reduction.
const SOFTMAX_REDUCE_SHADER: &str = r#"
struct Params {
    num_rows: u32,
    row_size: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> exp_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> exp_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> sum_exp_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> sum_exp_upper: array<f32>;
@group(0) @binding(7) var<storage, read_write> max_upper_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.num_rows) {
        return;
    }

    let row_offset = row * params.row_size;

    // Pass 1: Find max of upper bounds for numerical stability
    var max_u: f32 = -3.4028235e+38;  // f32::MIN
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        max_u = max(max_u, input_upper[row_offset + i]);
    }
    max_upper_out[row] = max_u;

    // Pass 2: Compute exp(x - max) and sums
    var sum_l: f32 = 0.0;
    var sum_u: f32 = 0.0;
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        let idx = row_offset + i;
        let el = exp(input_lower[idx] - max_u);
        let eu = exp(input_upper[idx] - max_u);
        exp_lower[idx] = el;
        exp_upper[idx] = eu;
        sum_l = sum_l + el;
        sum_u = sum_u + eu;
    }

    sum_exp_lower[row] = sum_l;
    sum_exp_upper[row] = sum_u;
}
"#;

/// WGSL shader for softmax IBP - Pass 2: Apply bounds formula.
///
/// Using the Auto-LiRPA formula:
/// - output_lower[i] = exp_lower[i] / (sum_exp_upper - exp_upper[i] + exp_lower[i] + epsilon)
/// - output_upper[i] = exp_upper[i] / (sum_exp_lower - exp_lower[i] + exp_upper[i] + epsilon)
///
/// This pass runs one thread per element.
const SOFTMAX_APPLY_SHADER: &str = r#"
struct Params {
    num_rows: u32,
    row_size: u32,
    _padding0: u32,
    _padding1: u32,
}

const EPSILON: f32 = 1e-12;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> exp_lower: array<f32>;
@group(0) @binding(2) var<storage, read> exp_upper: array<f32>;
@group(0) @binding(3) var<storage, read> sum_exp_lower: array<f32>;
@group(0) @binding(4) var<storage, read> sum_exp_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.num_rows * params.row_size;

    if (idx >= total_elements) {
        return;
    }

    let row = idx / params.row_size;

    let el = exp_lower[idx];
    let eu = exp_upper[idx];
    let sum_l = sum_exp_lower[row];
    let sum_u = sum_exp_upper[row];

    // Auto-LiRPA softmax bounds formula
    let denom_lower = sum_u - eu + el + EPSILON;
    let denom_upper = sum_l - el + eu + EPSILON;

    output_lower[idx] = el / denom_lower;
    output_upper[idx] = eu / denom_upper;
}
"#;

/// WGSL shader for transpose IBP.
///
/// Transposes the last two dimensions of a bounded tensor.
/// Input: [batch, rows, cols] -> Output: [batch, cols, rows]
const TRANSPOSE_IBP_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    rows: u32,       // Input second-to-last dim (becomes output last dim)
    cols: u32,       // Input last dim (becomes output second-to-last dim)
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let matrix_size = params.rows * params.cols;
    let total_elements = params.batch_size * matrix_size;

    if (idx >= total_elements) {
        return;
    }

    // Decompose output index: idx = batch * (cols * rows) + col * rows + row
    // Output shape is [batch, cols, rows]
    let batch_idx = idx / matrix_size;
    let out_matrix_idx = idx % matrix_size;
    let out_row = out_matrix_idx / params.rows;  // Actually col in input
    let out_col = out_matrix_idx % params.rows;  // Actually row in input

    // Input index: [batch, rows, cols] layout
    let in_row = out_col;
    let in_col = out_row;
    let in_idx = batch_idx * matrix_size + in_row * params.cols + in_col;

    // Transpose just copies with remapped indices
    output_lower[idx] = input_lower[in_idx];
    output_upper[idx] = input_upper[in_idx];
}
"#;

/// WGSL shader for scale IBP.
///
/// Element-wise multiplication by a scalar with interval arithmetic.
/// For positive scale: [l, u] -> [scale*l, scale*u]
/// For negative scale: [l, u] -> [scale*u, scale*l]
const SCALE_IBP_SHADER: &str = r#"
struct Params {
    total_elements: u32,
    scale: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.total_elements) {
        return;
    }

    let il = input_lower[idx];
    let iu = input_upper[idx];
    let s = params.scale;

    // Interval multiplication by scalar
    if (s >= 0.0) {
        output_lower[idx] = s * il;
        output_upper[idx] = s * iu;
    } else {
        output_lower[idx] = s * iu;
        output_upper[idx] = s * il;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::time::Instant;

    #[test]
    fn test_wgpu_device_creation() {
        match WgpuDevice::new() {
            Ok(device) => {
                println!("Device created: {}", device.info());
            }
            Err(e) => {
                println!("No GPU available: {}", e);
            }
        }
    }

    #[test]
    fn test_wgpu_gemm_f32_basic() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };

        let m = 2usize;
        let k = 3usize;
        let n = 4usize;

        // A: 2x3
        let a = vec![
            1.0f32, 2.0, 3.0, //
            -1.0, 0.5, 2.0, //
        ];
        // B: 3x4
        let b = vec![
            0.25f32, -1.0, 2.0, 0.0, //
            1.5, 0.5, -0.5, 1.0, //
            2.0, 1.0, 0.0, -2.0, //
        ];

        let out = device.gemm_f32(m, k, n, &a, &b).unwrap();
        assert_eq!(out.len(), m * n);

        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    sum += a[row * k + t] * b[t * n + col];
                }
                expected[row * n + col] = sum;
            }
        }

        for i in 0..(m * n) {
            assert_relative_eq!(out[i], expected[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_wgpu_crown_per_position_parallel_gpu_accelerated() {
        use gamma_propagate::{GELULayer, GraphNode, Layer, LinearLayer};
        use ndarray::Array2;

        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };

        // Build a small MLP graph: Linear -> GELU -> Linear
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

        // On the wgpu backend, per-position CROWN should use GPU-accelerated GEMM
        // while matching the CPU reference bounds.
        let result = device.crown_per_position_parallel(&graph, &input).unwrap();
        let sequential_result = graph.propagate_crown_per_position(&input).unwrap();

        assert_eq!(result.shape(), &[2, 3, out_features]);
        assert_eq!(result.shape(), sequential_result.shape());

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..out_features {
                    assert_relative_eq!(
                        result.lower[[i, j, k]],
                        sequential_result.lower[[i, j, k]],
                        epsilon = 1e-3
                    );
                    assert_relative_eq!(
                        result.upper[[i, j, k]],
                        sequential_result.upper[[i, j, k]],
                        epsilon = 1e-3
                    );
                }
            }
        }
    }

    /// Benchmark GPU vs Rayon for various input sizes.
    ///
    /// Run with: cargo test -p gamma-gpu benchmark_gpu_vs_rayon -- --nocapture --ignored
    #[test]
    #[ignore] // Ignored by default, run manually for benchmarks
    fn benchmark_gpu_vs_rayon_linear_ibp() {
        use crate::AcceleratedDevice;

        let wgpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };
        let cpu_device = AcceleratedDevice::new();

        println!("\n=== Linear IBP Benchmark: GPU (wgpu) vs CPU (Rayon) ===\n");
        println!(
            "{:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "Input Size", "In Feat", "Out Feat", "GPU (ms)", "Rayon (ms)", "Speedup"
        );
        println!("{}", "-".repeat(72));

        // Test cases: (batch_size, in_features, out_features)
        let test_cases = [
            // Small matrices (GPU overhead dominates)
            (4, 64, 64),
            (4, 128, 128),
            (4, 384, 384),
            // Medium matrices (Whisper encoder layer sizes)
            (4, 384, 1536),
            (4, 1536, 384),
            // Larger batch sizes
            (64, 384, 384),
            (64, 384, 1536),
            (256, 384, 384),
            // Large matrices
            (512, 512, 512),
            (1024, 256, 256),
        ];

        for (batch, in_feat, out_feat) in test_cases {
            // Create input
            let mut lower_data: Vec<f32> = Vec::with_capacity(batch * in_feat);
            let mut upper_data: Vec<f32> = Vec::with_capacity(batch * in_feat);
            for i in 0..batch * in_feat {
                let x = ((i as f32) * 0.001).sin();
                lower_data.push(x - 0.1);
                upper_data.push(x + 0.1);
            }
            let lower = ArrayD::from_shape_vec(IxDyn(&[batch, in_feat]), lower_data).unwrap();
            let upper = ArrayD::from_shape_vec(IxDyn(&[batch, in_feat]), upper_data).unwrap();
            let input = BoundedTensor::new(lower, upper).unwrap();

            // Create weights
            let weight = Array2::from_shape_fn((out_feat, in_feat), |(i, j)| {
                (((i + j) as f32) * 0.01).sin() * 0.1
            });
            let bias = Array1::from_shape_fn(out_feat, |i| (i as f32) * 0.001);

            // Warmup
            let _ = wgpu_device.linear_ibp(&input, &weight, Some(&bias));
            let _ = cpu_device.linear_ibp(&input, &weight, Some(&bias));

            // Benchmark GPU
            let iterations = 10;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = wgpu_device.linear_ibp(&input, &weight, Some(&bias));
            }
            let gpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark Rayon
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = cpu_device.linear_ibp(&input, &weight, Some(&bias));
            }
            let rayon_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = rayon_time / gpu_time;

            println!(
                "{:>12} {:>12} {:>12} {:>12.3} {:>12.3} {:>10.2}x",
                format!("[{},{}]", batch, in_feat),
                in_feat,
                out_feat,
                gpu_time,
                rayon_time,
                speedup
            );
        }

        println!("\nNote: Speedup > 1.0 means GPU is faster than Rayon.\n");
    }

    /// Benchmark processing multiple layers in sequence (simulating neural network IBP).
    ///
    /// Run with: cargo test -p gamma-gpu benchmark_multi_layer --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn benchmark_multi_layer_ibp() {
        use crate::AcceleratedDevice;

        let wgpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };
        let cpu_device = AcceleratedDevice::new();

        println!("\n=== Multi-Layer IBP Benchmark: GPU vs CPU ===");
        println!("Simulating Whisper MLP: 384 → 1536 → 384 with GELU (3 layers)\n");

        println!(
            "{:>12} {:>12} {:>12} {:>12} {:>10}",
            "Batch×Seq", "GPU (ms)", "Rayon (ms)", "Speedup", "Layers/s"
        );
        println!("{}", "-".repeat(62));

        // Test different batch×seq sizes for Whisper
        let test_cases = [
            (4, 4),    // 4 positions
            (4, 16),   // 16 positions
            (4, 64),   // 64 positions
            (4, 128),  // 128 positions
            (4, 256),  // 256 positions
            (1, 1500), // Full Whisper sequence
        ];

        for (batch, seq) in test_cases {
            let hidden = 384;
            let intermediate = 1536;
            let total_positions = batch * seq;

            // Create weights for 3-layer MLP: Linear1 → GELU (skip) → Linear2
            let weight1 = Array2::from_shape_fn((intermediate, hidden), |(i, j)| {
                (((i + j) as f32) * 0.01).sin() * 0.1
            });
            let bias1 = Array1::from_shape_fn(intermediate, |i| (i as f32) * 0.0001);

            let weight2 = Array2::from_shape_fn((hidden, intermediate), |(i, j)| {
                (((i + j) as f32) * 0.01).cos() * 0.1
            });
            let bias2 = Array1::from_shape_fn(hidden, |i| (i as f32) * 0.0001);

            // Create input
            let mut lower_data: Vec<f32> = Vec::with_capacity(total_positions * hidden);
            let mut upper_data: Vec<f32> = Vec::with_capacity(total_positions * hidden);
            for i in 0..total_positions * hidden {
                let x = ((i as f32) * 0.001).sin();
                lower_data.push(x - 0.01);
                upper_data.push(x + 0.01);
            }
            let lower =
                ArrayD::from_shape_vec(IxDyn(&[total_positions, hidden]), lower_data).unwrap();
            let upper =
                ArrayD::from_shape_vec(IxDyn(&[total_positions, hidden]), upper_data).unwrap();
            let input = BoundedTensor::new(lower, upper).unwrap();

            // Warmup
            let out1 = wgpu_device
                .linear_ibp(&input, &weight1, Some(&bias1))
                .unwrap();
            let _ = wgpu_device
                .linear_ibp(&out1, &weight2, Some(&bias2))
                .unwrap();
            let out1_cpu = cpu_device
                .linear_ibp(&input, &weight1, Some(&bias1))
                .unwrap();
            let _ = cpu_device
                .linear_ibp(&out1_cpu, &weight2, Some(&bias2))
                .unwrap();

            // Benchmark GPU (2 linear layers = simulated MLP without GELU)
            let iterations = 5;
            let start = Instant::now();
            for _ in 0..iterations {
                let out1 = wgpu_device
                    .linear_ibp(&input, &weight1, Some(&bias1))
                    .unwrap();
                let _ = wgpu_device
                    .linear_ibp(&out1, &weight2, Some(&bias2))
                    .unwrap();
            }
            let gpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark Rayon
            let start = Instant::now();
            for _ in 0..iterations {
                let out1 = cpu_device
                    .linear_ibp(&input, &weight1, Some(&bias1))
                    .unwrap();
                let _ = cpu_device
                    .linear_ibp(&out1, &weight2, Some(&bias2))
                    .unwrap();
            }
            let rayon_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = rayon_time / gpu_time;
            let layers_per_sec = (2.0 * 1000.0) / gpu_time;

            println!(
                "{:>12} {:>12.3} {:>12.3} {:>12.2}x {:>10.1}",
                format!("{}×{}", batch, seq),
                gpu_time,
                rayon_time,
                speedup,
                layers_per_sec
            );
        }

        println!("\nNote: Speedup > 1.0 means GPU is faster than Rayon.");
        println!("MLP = 2 linear layers (384→1536→384), GELU bounds skipped.\n");
    }

    /// Benchmark GPU vs Rayon for matmul IBP using attention-like shapes.
    ///
    /// This measures end-to-end time for `matmul_ibp`, including GPU readback into host memory.
    ///
    /// Run with:
    ///   cargo test -p gamma-gpu benchmark_gpu_vs_rayon_matmul_ibp --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_gpu_vs_rayon_matmul_ibp() {
        use crate::AcceleratedDevice;

        let wgpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };
        let cpu_device = AcceleratedDevice::new();

        println!("\n=== MatMul IBP Benchmark: GPU (wgpu) vs CPU (Rayon) ===\n");
        println!(
            "{:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
            "Batch", "Heads", "Seq", "Dim", "GPU (ms)", "Rayon (ms)", "Speedup"
        );
        println!("{}", "-".repeat(80));

        // Attention-like Q @ K^T:
        //   A: [batch, heads, seq, dim]
        //   B: [batch, heads, dim, seq]
        //   Out: [batch, heads, seq, seq]
        let test_cases = [
            (2_usize, 2_usize, 64_usize, 64_usize),
            (2, 4, 128, 64),
            (1, 6, 256, 64),
            (1, 6, 512, 64),
        ];

        for (batch, heads, seq, dim) in test_cases {
            let eps = 0.01_f32;

            // Create A bounds
            let a_elems = batch * heads * seq * dim;
            let mut a_lower_data: Vec<f32> = Vec::with_capacity(a_elems);
            let mut a_upper_data: Vec<f32> = Vec::with_capacity(a_elems);
            for i in 0..a_elems {
                let x = ((i as f32) * 0.001).sin();
                a_lower_data.push(x - eps);
                a_upper_data.push(x + eps);
            }
            let a_lower =
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), a_lower_data).unwrap();
            let a_upper =
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), a_upper_data).unwrap();
            let input_a = BoundedTensor::new(a_lower, a_upper).unwrap();

            // Create B bounds (already transposed K^T): [batch, heads, dim, seq]
            let b_elems = batch * heads * dim * seq;
            let mut b_lower_data: Vec<f32> = Vec::with_capacity(b_elems);
            let mut b_upper_data: Vec<f32> = Vec::with_capacity(b_elems);
            for i in 0..b_elems {
                let x = ((i as f32) * 0.001).cos();
                b_lower_data.push(x - eps);
                b_upper_data.push(x + eps);
            }
            let b_lower =
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, dim, seq]), b_lower_data).unwrap();
            let b_upper =
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, dim, seq]), b_upper_data).unwrap();
            let input_b = BoundedTensor::new(b_lower, b_upper).unwrap();

            // Warmup
            let _ = wgpu_device.matmul_ibp(&input_a, &input_b);
            let _ = cpu_device.matmul_ibp(&input_a, &input_b);

            let iterations = if seq >= 512 { 3 } else { 10 };

            // Benchmark GPU
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = wgpu_device.matmul_ibp(&input_a, &input_b);
            }
            let gpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark Rayon
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = cpu_device.matmul_ibp(&input_a, &input_b);
            }
            let rayon_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = rayon_time / gpu_time;

            println!(
                "{:>10} {:>10} {:>10} {:>10} {:>12.3} {:>12.3} {:>10.2}x",
                batch, heads, seq, dim, gpu_time, rayon_time, speedup
            );
        }

        println!("\nNote: Speedup > 1.0 means GPU is faster than Rayon.\n");
    }

    #[test]
    fn test_wgpu_linear_ibp_basic() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };

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
    fn test_wgpu_linear_ibp_matches_cpu() {
        use crate::AcceleratedDevice;

        let wgpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };
        let cpu_device = AcceleratedDevice::new();

        // Create larger test input: shape [8, 64, 128] with random-ish bounds
        let shape = [8, 64, 128];
        let mut lower_data: Vec<f32> = Vec::with_capacity(8 * 64 * 128);
        let mut upper_data: Vec<f32> = Vec::with_capacity(8 * 64 * 128);

        for i in 0..8 * 64 * 128 {
            let x = ((i as f32) * 0.001).sin();
            lower_data.push(x - 0.1);
            upper_data.push(x + 0.1);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&shape), lower_data).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&shape), upper_data).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Weight [256, 128] -> output [8, 64, 256]
        let weight =
            Array2::from_shape_fn((256, 128), |(i, j)| (((i + j) as f32) * 0.01).sin() * 0.1);
        let bias = Array1::from_shape_fn(256, |i| (i as f32) * 0.001);

        let gpu_result = wgpu_device
            .linear_ibp(&input, &weight, Some(&bias))
            .unwrap();
        let cpu_result = cpu_device.linear_ibp(&input, &weight, Some(&bias)).unwrap();

        assert_eq!(gpu_result.shape(), cpu_result.shape());
        assert_eq!(gpu_result.shape(), &[8, 64, 256]);

        // Compare all values
        let gpu_lower = gpu_result.lower.as_slice().unwrap();
        let cpu_lower = cpu_result.lower.as_slice().unwrap();
        let gpu_upper = gpu_result.upper.as_slice().unwrap();
        let cpu_upper = cpu_result.upper.as_slice().unwrap();

        let mut max_diff_lower = 0.0_f32;
        let mut max_diff_upper = 0.0_f32;

        for i in 0..gpu_lower.len() {
            max_diff_lower = max_diff_lower.max((gpu_lower[i] - cpu_lower[i]).abs());
            max_diff_upper = max_diff_upper.max((gpu_upper[i] - cpu_upper[i]).abs());
        }

        println!("Max difference lower: {}", max_diff_lower);
        println!("Max difference upper: {}", max_diff_upper);

        // Allow for small floating-point differences
        assert!(
            max_diff_lower < 1e-3,
            "Lower bounds differ too much: {}",
            max_diff_lower
        );
        assert!(
            max_diff_upper < 1e-3,
            "Upper bounds differ too much: {}",
            max_diff_upper
        );
    }

    #[test]
    fn test_wgpu_matmul_ibp_matches_cpu() {
        use crate::AcceleratedDevice;

        let wgpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };
        let cpu_device = AcceleratedDevice::new();

        // Create test inputs: [batch, m, k] @ [batch, k, n] = [batch, m, n]
        let batch = 4;
        let m = 8;
        let k = 16;
        let n = 12;

        // Input A: [4, 8, 16] with random-ish bounds
        let mut a_lower_data: Vec<f32> = Vec::with_capacity(batch * m * k);
        let mut a_upper_data: Vec<f32> = Vec::with_capacity(batch * m * k);
        for i in 0..batch * m * k {
            let x = ((i as f32) * 0.001).sin();
            a_lower_data.push(x - 0.1);
            a_upper_data.push(x + 0.1);
        }
        let a_lower = ArrayD::from_shape_vec(IxDyn(&[batch, m, k]), a_lower_data).unwrap();
        let a_upper = ArrayD::from_shape_vec(IxDyn(&[batch, m, k]), a_upper_data).unwrap();
        let input_a = BoundedTensor::new(a_lower, a_upper).unwrap();

        // Input B: [4, 16, 12] with random-ish bounds
        let mut b_lower_data: Vec<f32> = Vec::with_capacity(batch * k * n);
        let mut b_upper_data: Vec<f32> = Vec::with_capacity(batch * k * n);
        for i in 0..batch * k * n {
            let x = ((i as f32) * 0.002).cos();
            b_lower_data.push(x - 0.05);
            b_upper_data.push(x + 0.05);
        }
        let b_lower = ArrayD::from_shape_vec(IxDyn(&[batch, k, n]), b_lower_data).unwrap();
        let b_upper = ArrayD::from_shape_vec(IxDyn(&[batch, k, n]), b_upper_data).unwrap();
        let input_b = BoundedTensor::new(b_lower, b_upper).unwrap();

        // Run on GPU and CPU
        let gpu_result = wgpu_device.matmul_ibp(&input_a, &input_b).unwrap();
        let cpu_result = cpu_device.matmul_ibp(&input_a, &input_b).unwrap();

        // Verify shapes match
        assert_eq!(gpu_result.shape(), cpu_result.shape());
        assert_eq!(gpu_result.shape(), &[batch, m, n]);

        // Compare all values
        let gpu_lower = gpu_result.lower.as_slice().unwrap();
        let cpu_lower = cpu_result.lower.as_slice().unwrap();
        let gpu_upper = gpu_result.upper.as_slice().unwrap();
        let cpu_upper = cpu_result.upper.as_slice().unwrap();

        let mut max_diff_lower = 0.0_f32;
        let mut max_diff_upper = 0.0_f32;

        for i in 0..gpu_lower.len() {
            max_diff_lower = max_diff_lower.max((gpu_lower[i] - cpu_lower[i]).abs());
            max_diff_upper = max_diff_upper.max((gpu_upper[i] - cpu_upper[i]).abs());
        }

        println!("MatMul IBP max difference lower: {}", max_diff_lower);
        println!("MatMul IBP max difference upper: {}", max_diff_upper);

        // Allow for small floating-point differences
        assert!(
            max_diff_lower < 1e-3,
            "Lower bounds differ too much: {}",
            max_diff_lower
        );
        assert!(
            max_diff_upper < 1e-3,
            "Upper bounds differ too much: {}",
            max_diff_upper
        );
    }

    #[test]
    fn test_wgpu_softmax_ibp_basic() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };

        // Create test input: shape [2, 4] with bounds
        // Use values that produce reasonable softmax outputs
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![-0.1, 0.0, 0.1, 0.2, -0.2, -0.1, 0.0, 0.1],
        )
        .unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3])
                .unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = device.softmax_ibp(&input).unwrap();

        // Output shape should match input
        assert_eq!(result.shape(), &[2, 4]);

        // Softmax outputs should be in [0, 1]
        for &v in result.lower.iter() {
            assert!(v >= 0.0, "Softmax lower bound should be >= 0, got {}", v);
        }
        for &v in result.upper.iter() {
            assert!(
                v <= 1.0 + 1e-6,
                "Softmax upper bound should be <= 1, got {}",
                v
            );
        }

        // Lower <= upper
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(l <= u, "Invalid bounds: lower {} > upper {}", l, u);
        }

        // Sum of lower bounds should be close to 1 (softmax sums to 1)
        // Note: sum of bounds can be > 1 due to interval arithmetic looseness
        let row0_lower_sum: f32 = (0..4).map(|i| result.lower[[0, i]]).sum();
        let row0_upper_sum: f32 = (0..4).map(|i| result.upper[[0, i]]).sum();
        println!(
            "Row 0 softmax bounds sum: lower={:.4}, upper={:.4}",
            row0_lower_sum, row0_upper_sum
        );

        // Upper bound sum should be > 1 (due to interval looseness), lower can be < 1
        assert!(
            row0_upper_sum >= 0.9,
            "Softmax upper sum too small: {}",
            row0_upper_sum
        );
    }

    #[test]
    fn test_wgpu_softmax_ibp_matches_cpu() {
        use gamma_propagate::{BoundPropagation, SoftmaxLayer};

        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };

        // Create test input: shape [4, 16] with random-ish bounds
        let rows = 4;
        let cols = 16;
        let eps = 0.05_f32;

        let mut lower_data: Vec<f32> = Vec::with_capacity(rows * cols);
        let mut upper_data: Vec<f32> = Vec::with_capacity(rows * cols);
        for i in 0..rows * cols {
            let x = ((i as f32) * 0.1).sin() * 2.0;
            lower_data.push(x - eps);
            upper_data.push(x + eps);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), lower_data).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), upper_data).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        // GPU result
        let gpu_result = device.softmax_ibp(&input).unwrap();

        // CPU result using SoftmaxLayer
        let softmax_layer = SoftmaxLayer::new(-1);
        let cpu_result = softmax_layer.propagate_ibp(&input).unwrap();

        // Compare
        assert_eq!(gpu_result.shape(), cpu_result.shape());

        let gpu_lower = gpu_result.lower.as_slice().unwrap();
        let cpu_lower = cpu_result.lower.as_slice().unwrap();
        let gpu_upper = gpu_result.upper.as_slice().unwrap();
        let cpu_upper = cpu_result.upper.as_slice().unwrap();

        let mut max_diff_lower = 0.0_f32;
        let mut max_diff_upper = 0.0_f32;

        for i in 0..gpu_lower.len() {
            max_diff_lower = max_diff_lower.max((gpu_lower[i] - cpu_lower[i]).abs());
            max_diff_upper = max_diff_upper.max((gpu_upper[i] - cpu_upper[i]).abs());
        }

        println!("Softmax IBP max difference lower: {}", max_diff_lower);
        println!("Softmax IBP max difference upper: {}", max_diff_upper);

        // Allow for floating-point differences
        assert!(
            max_diff_lower < 1e-4,
            "Lower bounds differ too much: {}",
            max_diff_lower
        );
        assert!(
            max_diff_upper < 1e-4,
            "Upper bounds differ too much: {}",
            max_diff_upper
        );
    }

    #[test]
    fn test_wgpu_attention_ibp_basic() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };

        // Create Q, K, V with shape [batch, heads, seq, dim]
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;
        let eps = 0.01_f32;

        // Generate test data for Q
        let total = batch * heads * seq * dim;
        let mut q_lower_data: Vec<f32> = Vec::with_capacity(total);
        let mut q_upper_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            let x = ((i as f32) * 0.1).sin() * 0.5;
            q_lower_data.push(x - eps);
            q_upper_data.push(x + eps);
        }
        let q_lower =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_lower_data).unwrap();
        let q_upper =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_upper_data).unwrap();
        let q = BoundedTensor::new(q_lower, q_upper).unwrap();

        // Generate test data for K
        let mut k_lower_data: Vec<f32> = Vec::with_capacity(total);
        let mut k_upper_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            let x = ((i as f32) * 0.12).cos() * 0.5;
            k_lower_data.push(x - eps);
            k_upper_data.push(x + eps);
        }
        let k_lower =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_lower_data).unwrap();
        let k_upper =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_upper_data).unwrap();
        let k = BoundedTensor::new(k_lower, k_upper).unwrap();

        // Generate test data for V
        let mut v_lower_data: Vec<f32> = Vec::with_capacity(total);
        let mut v_upper_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            let x = ((i as f32) * 0.08).sin() * 0.5;
            v_lower_data.push(x - eps);
            v_upper_data.push(x + eps);
        }
        let v_lower =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_lower_data).unwrap();
        let v_upper =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_upper_data).unwrap();
        let v = BoundedTensor::new(v_lower, v_upper).unwrap();

        // Scale factor = 1 / sqrt(dim)
        let scale = 1.0 / (dim as f32).sqrt();

        // Run attention IBP
        let result = device.attention_ibp(&q, &k, &v, scale).unwrap();

        // Output shape should be [batch, heads, seq, dim]
        assert_eq!(result.shape(), &[batch, heads, seq, dim]);

        // Bounds should be valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(l <= u, "Invalid bounds: lower {} > upper {}", l, u);
        }

        // Check max width is reasonable
        let max_width = result.max_width();
        println!("Attention IBP max width: {:.6}", max_width);

        // With eps=0.01, the output bounds should not explode
        assert!(
            max_width < 10.0,
            "Attention bounds exploded: max_width={}",
            max_width
        );
    }

    /// Benchmark GPU attention IBP vs separate operations.
    ///
    /// Run with: cargo test -p gamma-gpu benchmark_attention_ibp --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_attention_ibp() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };

        println!("\n=== Attention IBP Benchmark: GPU ===\n");
        println!(
            "{:>10} {:>10} {:>10} {:>10} {:>12} {:>12}",
            "Batch", "Heads", "Seq", "Dim", "Time (ms)", "Elements/ms"
        );
        println!("{}", "-".repeat(70));

        // Test cases: (batch, heads, seq, dim)
        let test_cases = [
            (1, 6, 64, 64),
            (1, 6, 128, 64),
            (1, 6, 256, 64),
            (1, 6, 512, 64),
        ];

        for (batch, heads, seq, dim) in test_cases {
            let eps = 0.01_f32;
            let total = batch * heads * seq * dim;

            // Generate test data
            let mut q_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut q_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_upper_data: Vec<f32> = Vec::with_capacity(total);

            for i in 0..total {
                let xq = ((i as f32) * 0.1).sin() * 0.5;
                let xk = ((i as f32) * 0.12).cos() * 0.5;
                let xv = ((i as f32) * 0.08).sin() * 0.5;
                q_lower_data.push(xq - eps);
                q_upper_data.push(xq + eps);
                k_lower_data.push(xk - eps);
                k_upper_data.push(xk + eps);
                v_lower_data.push(xv - eps);
                v_upper_data.push(xv + eps);
            }

            let q = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_upper_data).unwrap(),
            )
            .unwrap();
            let k = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_upper_data).unwrap(),
            )
            .unwrap();
            let v = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_upper_data).unwrap(),
            )
            .unwrap();

            let scale = 1.0 / (dim as f32).sqrt();

            // Warmup
            let _ = device.attention_ibp(&q, &k, &v, scale);

            let iterations = if seq >= 512 { 3 } else { 5 };

            // Benchmark
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = device.attention_ibp(&q, &k, &v, scale);
            }
            let time_ms = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;
            let elements_per_ms = (total as f64) / time_ms;

            println!(
                "{:>10} {:>10} {:>10} {:>10} {:>12.3} {:>12.0}",
                batch, heads, seq, dim, time_ms, elements_per_ms
            );
        }

        println!("\nNote: Time includes Q@K^T, scale, softmax, and probs@V on GPU.\n");
    }

    /// Benchmark: GPU vs CPU (Rayon) attention IBP across different sizes.
    ///
    /// Run with: cargo test -p gamma-gpu benchmark_gpu_vs_cpu_attention_ibp --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_gpu_vs_cpu_attention_ibp() {
        use crate::AcceleratedDevice;

        let gpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };
        let cpu_device = AcceleratedDevice::new();

        println!("\n=== GPU vs CPU Attention IBP Benchmark ===\n");
        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>10}",
            "Batch", "Heads", "Seq", "Dim", "GPU (ms)", "CPU (ms)", "Speedup"
        );
        println!("{}", "-".repeat(78));

        // Test cases: (batch, heads, seq, dim)
        // Start small and increase to find crossover point
        let test_cases = [
            (1, 2, 16, 64),  // Very small
            (1, 2, 32, 64),  // Small
            (1, 4, 64, 64),  // Medium-small
            (1, 6, 128, 64), // Medium (Whisper-like heads)
            (1, 6, 256, 64), // Medium-large
            (1, 6, 384, 64), // Whisper encoder (seq=384 for 3s audio)
            (1, 6, 512, 64), // Large
        ];

        for (batch, heads, seq, dim) in test_cases {
            let eps = 0.01_f32;
            let total = batch * heads * seq * dim;

            // Generate test data
            let mut q_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut q_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_upper_data: Vec<f32> = Vec::with_capacity(total);

            for i in 0..total {
                let xq = ((i as f32) * 0.1).sin() * 0.5;
                let xk = ((i as f32) * 0.12).cos() * 0.5;
                let xv = ((i as f32) * 0.08).sin() * 0.5;
                q_lower_data.push(xq - eps);
                q_upper_data.push(xq + eps);
                k_lower_data.push(xk - eps);
                k_upper_data.push(xk + eps);
                v_lower_data.push(xv - eps);
                v_upper_data.push(xv + eps);
            }

            let q = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_upper_data).unwrap(),
            )
            .unwrap();
            let k = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_upper_data).unwrap(),
            )
            .unwrap();
            let v = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_upper_data).unwrap(),
            )
            .unwrap();

            let scale = 1.0 / (dim as f32).sqrt();

            // Warmup both
            let _ = gpu_device.attention_ibp(&q, &k, &v, scale);
            let _ = cpu_device.attention_ibp(&q, &k, &v, scale);

            let iterations = if seq >= 384 {
                3
            } else if seq >= 128 {
                5
            } else {
                10
            };

            // Benchmark GPU
            let start_gpu = Instant::now();
            for _ in 0..iterations {
                let _ = gpu_device.attention_ibp(&q, &k, &v, scale);
            }
            let gpu_time_ms = start_gpu.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark CPU
            let start_cpu = Instant::now();
            for _ in 0..iterations {
                let _ = cpu_device.attention_ibp(&q, &k, &v, scale);
            }
            let cpu_time_ms = start_cpu.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = cpu_time_ms / gpu_time_ms;
            let speedup_str = format!("{:.2}x", speedup);

            println!(
                "{:>8} {:>8} {:>8} {:>8} {:>12.3} {:>12.3} {:>10}",
                batch, heads, seq, dim, gpu_time_ms, cpu_time_ms, speedup_str
            );
        }

        println!("\nNote: Speedup > 1.0 means GPU is faster. Current GPU implementation");
        println!("has host roundtrips between matmul/softmax operations.\n");
    }

    #[test]
    fn test_wgpu_attention_ibp_fused_basic() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Skip if no GPU
        };

        // Create Q, K, V with shape [batch, heads, seq, dim]
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;
        let eps = 0.01_f32;

        // Generate test data for Q
        let total = batch * heads * seq * dim;
        let mut q_lower_data: Vec<f32> = Vec::with_capacity(total);
        let mut q_upper_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            let x = ((i as f32) * 0.1).sin() * 0.5;
            q_lower_data.push(x - eps);
            q_upper_data.push(x + eps);
        }
        let q_lower =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_lower_data).unwrap();
        let q_upper =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_upper_data).unwrap();
        let q = BoundedTensor::new(q_lower, q_upper).unwrap();

        // Generate test data for K
        let mut k_lower_data: Vec<f32> = Vec::with_capacity(total);
        let mut k_upper_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            let x = ((i as f32) * 0.12).cos() * 0.5;
            k_lower_data.push(x - eps);
            k_upper_data.push(x + eps);
        }
        let k_lower =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_lower_data).unwrap();
        let k_upper =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_upper_data).unwrap();
        let k = BoundedTensor::new(k_lower, k_upper).unwrap();

        // Generate test data for V
        let mut v_lower_data: Vec<f32> = Vec::with_capacity(total);
        let mut v_upper_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            let x = ((i as f32) * 0.08).sin() * 0.5;
            v_lower_data.push(x - eps);
            v_upper_data.push(x + eps);
        }
        let v_lower =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_lower_data).unwrap();
        let v_upper =
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_upper_data).unwrap();
        let v = BoundedTensor::new(v_lower, v_upper).unwrap();

        // Scale factor = 1 / sqrt(dim)
        let scale = 1.0 / (dim as f32).sqrt();

        // Run fused attention IBP
        let fused_result = device.attention_ibp_fused(&q, &k, &v, scale).unwrap();

        // Run non-fused attention IBP for comparison
        let nonfused_result = device.attention_ibp(&q, &k, &v, scale).unwrap();

        // Output shape should be [batch, heads, seq, dim]
        assert_eq!(fused_result.shape(), &[batch, heads, seq, dim]);

        // Bounds should be valid
        for (l, u) in fused_result.lower.iter().zip(fused_result.upper.iter()) {
            assert!(l <= u, "Invalid bounds: lower {} > upper {}", l, u);
        }

        // Fused and non-fused should produce the same results (within floating point tolerance)
        let max_lower_diff = fused_result
            .lower
            .iter()
            .zip(nonfused_result.lower.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, |a, b| a.max(b));
        let max_upper_diff = fused_result
            .upper
            .iter()
            .zip(nonfused_result.upper.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, |a, b| a.max(b));

        println!(
            "Fused vs Non-fused: max_lower_diff={:.6e}, max_upper_diff={:.6e}",
            max_lower_diff, max_upper_diff
        );

        // Allow small numerical differences (1e-4 relative tolerance)
        assert!(
            max_lower_diff < 1e-4,
            "Lower bounds differ too much: {}",
            max_lower_diff
        );
        assert!(
            max_upper_diff < 1e-4,
            "Upper bounds differ too much: {}",
            max_upper_diff
        );

        // Check max width is reasonable
        let max_width = fused_result.max_width();
        println!("Fused Attention IBP max width: {:.6}", max_width);

        // With eps=0.01, the output bounds should not explode
        assert!(
            max_width < 10.0,
            "Attention bounds exploded: max_width={}",
            max_width
        );
    }

    /// Benchmark: Fused vs Non-fused GPU attention IBP.
    ///
    /// Run with: cargo test -p gamma-gpu benchmark_fused_attention_ibp --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_fused_attention_ibp() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };

        println!("\n=== Fused vs Non-Fused GPU Attention IBP Benchmark ===\n");
        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>10}",
            "Batch", "Heads", "Seq", "Dim", "Non-Fused", "Fused", "Speedup"
        );
        println!("{}", "-".repeat(78));

        // Test cases: (batch, heads, seq, dim)
        let test_cases = [
            (1, 2, 16, 64),  // Very small
            (1, 2, 32, 64),  // Small
            (1, 4, 64, 64),  // Medium-small
            (1, 6, 128, 64), // Medium
            (1, 6, 256, 64), // Medium-large
            (1, 6, 384, 64), // Whisper encoder
            (1, 6, 512, 64), // Large
        ];

        for (batch, heads, seq, dim) in test_cases {
            let eps = 0.01_f32;
            let total = batch * heads * seq * dim;

            // Generate test data
            let mut q_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut q_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_upper_data: Vec<f32> = Vec::with_capacity(total);

            for i in 0..total {
                let xq = ((i as f32) * 0.1).sin() * 0.5;
                let xk = ((i as f32) * 0.12).cos() * 0.5;
                let xv = ((i as f32) * 0.08).sin() * 0.5;
                q_lower_data.push(xq - eps);
                q_upper_data.push(xq + eps);
                k_lower_data.push(xk - eps);
                k_upper_data.push(xk + eps);
                v_lower_data.push(xv - eps);
                v_upper_data.push(xv + eps);
            }

            let q = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_upper_data).unwrap(),
            )
            .unwrap();
            let k = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_upper_data).unwrap(),
            )
            .unwrap();
            let v = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_upper_data).unwrap(),
            )
            .unwrap();

            let scale = 1.0 / (dim as f32).sqrt();

            // Warmup both
            let _ = device.attention_ibp(&q, &k, &v, scale);
            let _ = device.attention_ibp_fused(&q, &k, &v, scale);

            let iterations = if seq >= 384 {
                3
            } else if seq >= 128 {
                5
            } else {
                10
            };

            // Benchmark non-fused
            let start_nonfused = Instant::now();
            for _ in 0..iterations {
                let _ = device.attention_ibp(&q, &k, &v, scale);
            }
            let nonfused_ms = start_nonfused.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark fused
            let start_fused = Instant::now();
            for _ in 0..iterations {
                let _ = device.attention_ibp_fused(&q, &k, &v, scale);
            }
            let fused_ms = start_fused.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = nonfused_ms / fused_ms;

            println!(
                "{:>8} {:>8} {:>8} {:>8} {:>12.3} {:>12.3} {:>10.2}x",
                batch, heads, seq, dim, nonfused_ms, fused_ms, speedup
            );
        }

        println!("\nNote: Speedup > 1.0 means fused kernel is faster.");
        println!("Fused eliminates 3 intermediate host roundtrips.\n");
    }

    /// Benchmark: Fused GPU vs CPU attention IBP.
    ///
    /// Run with: cargo test -p gamma-gpu benchmark_fused_gpu_vs_cpu_attention_ibp --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_fused_gpu_vs_cpu_attention_ibp() {
        use crate::AcceleratedDevice;

        let gpu_device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("No GPU available: {}", e);
                return;
            }
        };
        let cpu_device = AcceleratedDevice::new();

        println!("\n=== Fused GPU vs CPU Attention IBP Benchmark ===\n");
        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>10}",
            "Batch", "Heads", "Seq", "Dim", "GPU Fused", "CPU (ms)", "Speedup"
        );
        println!("{}", "-".repeat(78));

        // Test cases: (batch, heads, seq, dim)
        let test_cases = [
            (1, 2, 16, 64),
            (1, 2, 32, 64),
            (1, 4, 64, 64),
            (1, 6, 128, 64),
            (1, 6, 256, 64),
            (1, 6, 384, 64),
            (1, 6, 512, 64),
        ];

        for (batch, heads, seq, dim) in test_cases {
            let eps = 0.01_f32;
            let total = batch * heads * seq * dim;

            // Generate test data
            let mut q_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut q_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut k_upper_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_lower_data: Vec<f32> = Vec::with_capacity(total);
            let mut v_upper_data: Vec<f32> = Vec::with_capacity(total);

            for i in 0..total {
                let xq = ((i as f32) * 0.1).sin() * 0.5;
                let xk = ((i as f32) * 0.12).cos() * 0.5;
                let xv = ((i as f32) * 0.08).sin() * 0.5;
                q_lower_data.push(xq - eps);
                q_upper_data.push(xq + eps);
                k_lower_data.push(xk - eps);
                k_upper_data.push(xk + eps);
                v_lower_data.push(xv - eps);
                v_upper_data.push(xv + eps);
            }

            let q = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), q_upper_data).unwrap(),
            )
            .unwrap();
            let k = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), k_upper_data).unwrap(),
            )
            .unwrap();
            let v = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_lower_data).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq, dim]), v_upper_data).unwrap(),
            )
            .unwrap();

            let scale = 1.0 / (dim as f32).sqrt();

            // Warmup both
            let _ = gpu_device.attention_ibp_fused(&q, &k, &v, scale);
            let _ = cpu_device.attention_ibp(&q, &k, &v, scale);

            let iterations = if seq >= 384 {
                3
            } else if seq >= 128 {
                5
            } else {
                10
            };

            // Benchmark GPU fused
            let start_gpu = Instant::now();
            for _ in 0..iterations {
                let _ = gpu_device.attention_ibp_fused(&q, &k, &v, scale);
            }
            let gpu_ms = start_gpu.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark CPU
            let start_cpu = Instant::now();
            for _ in 0..iterations {
                let _ = cpu_device.attention_ibp(&q, &k, &v, scale);
            }
            let cpu_ms = start_cpu.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = cpu_ms / gpu_ms;

            println!(
                "{:>8} {:>8} {:>8} {:>8} {:>12.3} {:>12.3} {:>10.2}x",
                batch, heads, seq, dim, gpu_ms, cpu_ms, speedup
            );
        }

        println!("\nNote: Speedup > 1.0 means fused GPU is faster than CPU.\n");
    }
}
