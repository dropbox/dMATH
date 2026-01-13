//! ONNX Runtime backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Execution provider for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU execution (default, always available)
    #[default]
    CPU,
    /// CUDA execution provider (NVIDIA GPUs)
    CUDA,
    /// TensorRT execution provider (NVIDIA inference)
    TensorRT,
    /// OpenVINO execution provider (Intel)
    OpenVINO,
    /// DirectML execution provider (Windows)
    DirectML,
    /// CoreML execution provider (Apple)
    CoreML,
}

impl ExecutionProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionProvider::CPU => "CPUExecutionProvider",
            ExecutionProvider::CUDA => "CUDAExecutionProvider",
            ExecutionProvider::TensorRT => "TensorRTExecutionProvider",
            ExecutionProvider::OpenVINO => "OpenVINOExecutionProvider",
            ExecutionProvider::DirectML => "DmlExecutionProvider",
            ExecutionProvider::CoreML => "CoreMLExecutionProvider",
        }
    }
}

/// Graph optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphOptimizationLevel {
    /// No optimizations
    Disabled,
    /// Basic optimizations (constant folding, etc.)
    Basic,
    /// Extended optimizations (more aggressive)
    #[default]
    Extended,
    /// All optimizations
    All,
}

impl GraphOptimizationLevel {
    pub fn as_ort_level(&self) -> u8 {
        match self {
            GraphOptimizationLevel::Disabled => 0,
            GraphOptimizationLevel::Basic => 1,
            GraphOptimizationLevel::Extended => 2,
            GraphOptimizationLevel::All => 99,
        }
    }
}

/// ONNX Runtime backend configuration
#[derive(Debug, Clone)]
pub struct OnnxRuntimeConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Execution provider
    pub execution_provider: ExecutionProvider,
    /// Graph optimization level
    pub optimization_level: GraphOptimizationLevel,
    /// Enable memory pattern optimization
    pub enable_memory_pattern: bool,
    /// Enable memory arena
    pub enable_mem_arena: bool,
    /// Number of intra-op threads
    pub intra_op_threads: Option<usize>,
    /// Number of inter-op threads
    pub inter_op_threads: Option<usize>,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for OnnxRuntimeConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            execution_provider: ExecutionProvider::CPU,
            optimization_level: GraphOptimizationLevel::Extended,
            enable_memory_pattern: true,
            enable_mem_arena: true,
            intra_op_threads: None,
            inter_op_threads: None,
            model_path: None,
            timeout: Duration::from_secs(300),
            warmup_iterations: 5,
            benchmark_iterations: 100,
        }
    }
}

impl OnnxRuntimeConfig {
    /// Create config for CUDA execution
    pub fn cuda() -> Self {
        Self {
            execution_provider: ExecutionProvider::CUDA,
            ..Default::default()
        }
    }

    /// Create config for TensorRT execution
    pub fn tensorrt() -> Self {
        Self {
            execution_provider: ExecutionProvider::TensorRT,
            optimization_level: GraphOptimizationLevel::All,
            ..Default::default()
        }
    }

    /// Create config for high-throughput inference
    pub fn high_throughput() -> Self {
        Self {
            optimization_level: GraphOptimizationLevel::All,
            enable_memory_pattern: true,
            enable_mem_arena: true,
            benchmark_iterations: 1000,
            ..Default::default()
        }
    }
}
