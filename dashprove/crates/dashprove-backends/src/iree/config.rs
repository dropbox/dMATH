//! IREE backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// IREE target backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IREETarget {
    /// LLVM CPU backend
    #[default]
    LLVMCPU,
    /// Vulkan SPIR-V backend
    VulkanSPIRV,
    /// CUDA backend
    CUDA,
    /// Metal Performance Shaders (Apple)
    MetalSPIRV,
    /// ROCm backend (AMD)
    ROCm,
    /// WebGPU backend
    WebGPU,
    /// VMVX (Virtual Machine Vector Extensions) for reference
    VMVX,
}

impl IREETarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            IREETarget::LLVMCPU => "llvm-cpu",
            IREETarget::VulkanSPIRV => "vulkan-spirv",
            IREETarget::CUDA => "cuda",
            IREETarget::MetalSPIRV => "metal-spirv",
            IREETarget::ROCm => "rocm",
            IREETarget::WebGPU => "webgpu",
            IREETarget::VMVX => "vmvx",
        }
    }
}

/// IREE input format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputFormat {
    /// MLIR StableHLO dialect
    #[default]
    StableHLO,
    /// MLIR TOSA dialect
    TOSA,
    /// MLIR Linalg dialect
    Linalg,
    /// TensorFlow saved model
    TFSavedModel,
    /// TensorFlow Lite flatbuffer
    TFLite,
    /// ONNX format
    ONNX,
}

impl InputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputFormat::StableHLO => "stablehlo",
            InputFormat::TOSA => "tosa",
            InputFormat::Linalg => "linalg",
            InputFormat::TFSavedModel => "tf_saved_model",
            InputFormat::TFLite => "tflite",
            InputFormat::ONNX => "onnx",
        }
    }
}

/// IREE execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Local synchronous execution
    #[default]
    LocalSync,
    /// Local task-parallel execution
    LocalTask,
    /// Async dispatch execution
    AsyncDispatch,
}

impl ExecutionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionMode::LocalSync => "local-sync",
            ExecutionMode::LocalTask => "local-task",
            ExecutionMode::AsyncDispatch => "async-dispatch",
        }
    }
}

/// IREE backend configuration
#[derive(Debug, Clone)]
pub struct IREEConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Target backend
    pub target: IREETarget,
    /// Input format
    pub input_format: InputFormat,
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Enable optimizations
    pub enable_optimization: bool,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Enable tracing
    pub enable_tracing: bool,
}

impl Default for IREEConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            target: IREETarget::LLVMCPU,
            input_format: InputFormat::StableHLO,
            execution_mode: ExecutionMode::LocalSync,
            enable_optimization: true,
            model_path: None,
            timeout: Duration::from_secs(600),
            warmup_iterations: 10,
            benchmark_iterations: 100,
            enable_tracing: false,
        }
    }
}

impl IREEConfig {
    /// Create config for Vulkan target
    pub fn vulkan() -> Self {
        Self {
            target: IREETarget::VulkanSPIRV,
            ..Default::default()
        }
    }

    /// Create config for CUDA target
    pub fn cuda() -> Self {
        Self {
            target: IREETarget::CUDA,
            ..Default::default()
        }
    }

    /// Create config with task parallelism
    pub fn parallel() -> Self {
        Self {
            execution_mode: ExecutionMode::LocalTask,
            ..Default::default()
        }
    }
}
