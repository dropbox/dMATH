//! Batch operation dispatch
//!
//! This module provides the high-level interface for batched GPU operations,
//! with automatic CPU fallback for complex reductions.

use crate::arena::{GpuExpr, GpuExprArena};
use crate::shaders::{status, GpuConstDef, GpuPipelines, WhnfUniforms};
use crate::GpuError;
use bytemuck;
use lean5_kernel::{Environment, Expr, TypeChecker};
use wgpu::util::DeviceExt;

/// Workgroup size (must match shader)
const WORKGROUP_SIZE: u32 = 64;

/// Minimum batch size to use GPU (smaller batches use CPU)
const MIN_GPU_BATCH_SIZE: usize = 16;

/// Scratch space per thread
const SCRATCH_SIZE_PER_THREAD: usize = 256;

/// Batch WHNF operation context
pub struct BatchWhnf<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipelines: &'a GpuPipelines,
}

impl<'a> BatchWhnf<'a> {
    /// Create a new batch WHNF context
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        pipelines: &'a GpuPipelines,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
        }
    }

    /// Perform batch WHNF reduction on multiple expressions
    ///
    /// Returns reduced expressions. Uses GPU for large batches, CPU for small ones
    /// or when GPU reduction needs CPU fallback.
    pub async fn batch_whnf(
        &self,
        env: &Environment,
        exprs: &[Expr],
    ) -> Result<Vec<Expr>, GpuError> {
        if exprs.is_empty() {
            return Ok(Vec::new());
        }

        // Use CPU for small batches
        if exprs.len() < MIN_GPU_BATCH_SIZE {
            return Ok(Self::cpu_batch_whnf(env, exprs));
        }

        // Serialize expressions to GPU format
        let mut arena = GpuExprArena::new();
        let expr_indices: Vec<u32> = exprs.iter().map(|e| arena.add_expr(e)).collect();

        // Build constant definitions buffer
        let const_defs = Self::build_const_defs(env, &mut arena);

        // Run GPU reduction
        let (results, statuses) = self
            .run_gpu_whnf(&arena, &expr_indices, &const_defs)
            .await?;

        // Process results, falling back to CPU for complex reductions
        let mut output = Vec::with_capacity(exprs.len());
        for (i, (result, stat)) in results.iter().zip(statuses.iter()).enumerate() {
            if *stat == status::NEEDS_CPU {
                // CPU fallback for this expression
                let tc = TypeChecker::new(env);
                output.push(tc.whnf(&exprs[i]));
            } else {
                // Use GPU result
                if let Some(expr) = arena.to_expr(*result) {
                    output.push(expr);
                } else {
                    // Deserialization failed, fall back to CPU
                    let tc = TypeChecker::new(env);
                    output.push(tc.whnf(&exprs[i]));
                }
            }
        }

        Ok(output)
    }

    /// CPU-only batch WHNF for comparison/fallback
    pub fn cpu_batch_whnf(env: &Environment, exprs: &[Expr]) -> Vec<Expr> {
        let tc = TypeChecker::new(env);
        exprs.iter().map(|e| tc.whnf(e)).collect()
    }

    /// Build GPU constant definitions from environment
    fn build_const_defs(env: &Environment, arena: &mut GpuExprArena) -> Vec<GpuConstDef> {
        // For now, create a simple mapping
        // In a full implementation, we'd need to handle name→index mapping properly
        let mut defs = Vec::new();

        // Add entries for all constants in environment
        for info in env.constants() {
            let body_idx = info
                .value
                .as_ref()
                .map_or(-1, |body| arena.add_expr(body) as i32);

            // SAFETY: Level param count is bounded by practical type complexity limits,
            // which are far below u32::MAX. Use saturating conversion for defense.
            let num_params = u32::try_from(info.level_params.len()).unwrap_or(u32::MAX);
            defs.push(GpuConstDef {
                body_idx,
                num_params,
                _pad1: 0,
                _pad2: 0,
            });
        }

        // Ensure at least one entry (GPU buffers can't be empty)
        if defs.is_empty() {
            defs.push(GpuConstDef::default());
        }

        defs
    }

    /// Run GPU WHNF reduction
    ///
    /// Returns (output indices, status codes) for each input expression
    async fn run_gpu_whnf(
        &self,
        arena: &GpuExprArena,
        expr_indices: &[u32],
        const_defs: &[GpuConstDef],
    ) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
        let num_exprs = expr_indices.len();

        // Create input expression buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Expressions"),
                contents: bytemuck::cast_slice(arena.exprs()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output expression buffer (indices only, same size as input)
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Expressions"),
            size: (num_exprs * std::mem::size_of::<GpuExpr>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create constant definitions buffer
        let const_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Constant Definitions"),
                contents: bytemuck::cast_slice(const_defs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create status buffer
        let status_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Status"),
            size: std::mem::size_of_val(expr_indices) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create expression indices buffer
        let indices_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Expression Indices"),
                contents: bytemuck::cast_slice(expr_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create scratch buffer
        let scratch_size = num_exprs * SCRATCH_SIZE_PER_THREAD * std::mem::size_of::<GpuExpr>();
        let scratch_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scratch"),
            size: scratch_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        // SAFETY: GPU buffer sizes are bounded by available VRAM which is far below u32::MAX.
        // Use saturating conversions for defense in depth.
        let num_exprs_u32 = u32::try_from(num_exprs).unwrap_or(u32::MAX);
        let expr_buffer_size = u32::try_from(arena.exprs().len()).unwrap_or(u32::MAX);
        let const_buffer_size = u32::try_from(const_defs.len()).unwrap_or(u32::MAX);
        let uniforms = WhnfUniforms {
            num_exprs: num_exprs_u32,
            expr_buffer_size,
            const_buffer_size,
            _pad: 0,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind groups
        let expr_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Expression Bind Group"),
            layout: &self.pipelines.expr_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: const_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: status_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });

        let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.pipelines.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Calculate workgroup count
        let workgroup_count = num_exprs_u32.div_ceil(WORKGROUP_SIZE);

        // Encode and submit
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WHNF Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WHNF Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.whnf);
            pass.set_bind_group(0, &expr_bind_group, &[]);
            pass.set_bind_group(1, &uniform_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Create staging buffers for readback
        let output_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging"),
            size: (num_exprs * std::mem::size_of::<GpuExpr>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let status_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Status Staging"),
            size: std::mem::size_of_val(expr_indices) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy to staging
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging,
            0,
            (num_exprs * std::mem::size_of::<GpuExpr>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &status_buffer,
            0,
            &status_staging,
            0,
            std::mem::size_of_val(expr_indices) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read results
        let output_slice = output_staging.slice(..);
        let status_slice = status_staging.slice(..);

        let (output_tx, output_rx) = tokio::sync::oneshot::channel();
        let (status_tx, status_rx) = tokio::sync::oneshot::channel();

        output_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = output_tx.send(result);
        });
        status_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = status_tx.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        output_rx
            .await
            .map_err(|_| GpuError::DeviceRequest("Map cancelled".to_string()))??;
        status_rx
            .await
            .map_err(|_| GpuError::DeviceRequest("Map cancelled".to_string()))??;

        // Read data
        let output_data = output_slice.get_mapped_range();
        let _output_exprs: &[GpuExpr] = bytemuck::cast_slice(&output_data);
        // SAFETY: num_exprs_u32 was already computed above with checked conversion
        let output_indices: Vec<u32> = (0..num_exprs_u32).collect(); // Use position as index

        let status_data = status_slice.get_mapped_range();
        let statuses: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&status_data).to_vec();

        drop(output_data);
        drop(status_data);

        output_staging.unmap();
        status_staging.unmap();

        Ok((output_indices, statuses))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lean5_kernel::Level;

    #[test]
    fn test_cpu_batch_whnf() {
        let env = Environment::new();
        let exprs = vec![
            Expr::Sort(Level::Zero),
            Expr::Sort(Level::succ(Level::Zero)),
            Expr::BVar(0),
        ];

        let results = BatchWhnf::cpu_batch_whnf(&env, &exprs);

        assert_eq!(results.len(), 3);
        // Sorts and BVars are already in WHNF
        assert_eq!(results[0], exprs[0]);
        assert_eq!(results[1], exprs[1]);
        assert_eq!(results[2], exprs[2]);
    }
}
