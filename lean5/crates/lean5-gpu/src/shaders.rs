//! Compute shader compilation and management
//!
//! This module handles loading, compiling, and managing GPU compute shaders
//! for batch operations on expressions.

use std::borrow::Cow;

use crate::GpuError;

/// Shader source code embedded at compile time
pub mod sources {
    /// WHNF reduction shader
    pub const WHNF: &str = include_str!("../shaders/whnf.wgsl");
}

/// Compiled compute pipelines for all operations
pub struct GpuPipelines {
    /// WHNF reduction pipeline
    pub whnf: wgpu::ComputePipeline,
    /// Bind group layout for expression operations
    pub expr_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for uniforms
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuPipelines {
    /// Create and compile all compute pipelines
    pub fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
        // Create bind group layout for expression operations (group 0)
        let expr_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Expression Bind Group Layout"),
                entries: &[
                    // @binding(0): input_exprs - read-only storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): output_exprs - read-write storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(2): const_defs - read-only storage
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
                    // @binding(3): status - read-write storage
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
                    // @binding(4): expr_indices - read-only storage
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
                    // @binding(5): scratch - read-write storage
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
                ],
            });

        // Create bind group layout for uniforms (group 1)
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WHNF Pipeline Layout"),
            bind_group_layouts: &[&expr_bind_group_layout, &uniform_bind_group_layout],
            immediate_size: 0,
        });

        // Compile WHNF shader
        let whnf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WHNF Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(sources::WHNF)),
        });

        // Create WHNF compute pipeline
        let whnf = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("WHNF Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &whnf_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            whnf,
            expr_bind_group_layout,
            uniform_bind_group_layout,
        })
    }
}

/// Uniform buffer data for WHNF shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WhnfUniforms {
    /// Number of expressions to process
    pub num_exprs: u32,
    /// Total size of expression buffer
    pub expr_buffer_size: u32,
    /// Total size of constant definitions buffer
    pub const_buffer_size: u32,
    /// Padding for alignment
    pub _pad: u32,
}

/// Constant definition entry for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuConstDef {
    /// Index of definition body (-1 if opaque/axiom)
    pub body_idx: i32,
    /// Number of universe parameters
    pub num_params: u32,
    /// Reserved
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Reduction status values (must match shader)
pub mod status {
    /// Expression is in WHNF
    pub const DONE: u32 = 0;
    /// Reduction was performed
    pub const REDUCED: u32 = 1;
    /// Needs CPU fallback for complex reduction
    pub const NEEDS_CPU: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniforms_size() {
        // Ensure uniforms are 16-byte aligned
        assert_eq!(std::mem::size_of::<WhnfUniforms>(), 16);
    }

    #[test]
    fn test_const_def_size() {
        // Ensure const def is 16 bytes
        assert_eq!(std::mem::size_of::<GpuConstDef>(), 16);
    }
}
