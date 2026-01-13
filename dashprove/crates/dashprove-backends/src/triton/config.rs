//! Triton backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Triton compilation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompilationMode {
    /// Just-in-time compilation
    #[default]
    JIT,
    /// Ahead-of-time compilation
    AOT,
    /// Interpreted mode (for debugging)
    Interpret,
}

impl CompilationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompilationMode::JIT => "jit",
            CompilationMode::AOT => "aot",
            CompilationMode::Interpret => "interpret",
        }
    }
}

/// Triton target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TritonTarget {
    /// NVIDIA GPU via CUDA
    #[default]
    CUDA,
    /// AMD GPU via ROCm
    ROCm,
    /// CPU backend (experimental)
    CPU,
}

impl TritonTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            TritonTarget::CUDA => "cuda",
            TritonTarget::ROCm => "hip",
            TritonTarget::CPU => "cpu",
        }
    }
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Standard optimization
    #[default]
    O2,
    /// Aggressive optimization
    O3,
}

impl OptimizationLevel {
    pub fn as_int(&self) -> u8 {
        match self {
            OptimizationLevel::O0 => 0,
            OptimizationLevel::O1 => 1,
            OptimizationLevel::O2 => 2,
            OptimizationLevel::O3 => 3,
        }
    }
}

/// Triton backend configuration
#[derive(Debug, Clone)]
pub struct TritonConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Compilation mode
    pub compilation_mode: CompilationMode,
    /// Target architecture
    pub target: TritonTarget,
    /// Optimization level
    pub opt_level: OptimizationLevel,
    /// Number of warps per block
    pub num_warps: usize,
    /// Number of stages for software pipelining
    pub num_stages: usize,
    /// Enable autotune
    pub autotune: bool,
    /// Kernel path override
    pub kernel_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for TritonConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            compilation_mode: CompilationMode::JIT,
            target: TritonTarget::CUDA,
            opt_level: OptimizationLevel::O2,
            num_warps: 4,
            num_stages: 2,
            autotune: false,
            kernel_path: None,
            timeout: Duration::from_secs(300),
            warmup_iterations: 10,
            benchmark_iterations: 100,
        }
    }
}

impl TritonConfig {
    /// Create config with autotuning enabled
    pub fn with_autotune() -> Self {
        Self {
            autotune: true,
            ..Default::default()
        }
    }

    /// Create config for AMD GPUs
    pub fn rocm() -> Self {
        Self {
            target: TritonTarget::ROCm,
            ..Default::default()
        }
    }

    /// Create config with aggressive optimization
    pub fn optimized() -> Self {
        Self {
            opt_level: OptimizationLevel::O3,
            num_warps: 8,
            num_stages: 4,
            ..Default::default()
        }
    }
}
