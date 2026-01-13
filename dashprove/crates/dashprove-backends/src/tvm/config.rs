//! Apache TVM backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// TVM compilation target
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TVMTarget {
    /// LLVM CPU backend
    #[default]
    LLVM,
    /// CUDA GPU backend
    CUDA,
    /// Metal GPU backend (Apple)
    Metal,
    /// Vulkan GPU backend
    Vulkan,
    /// OpenCL backend
    OpenCL,
    /// ROCm backend (AMD)
    ROCm,
    /// ARM CPU backend
    ARM,
    /// WebGPU backend
    WebGPU,
}

impl TVMTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            TVMTarget::LLVM => "llvm",
            TVMTarget::CUDA => "cuda",
            TVMTarget::Metal => "metal",
            TVMTarget::Vulkan => "vulkan",
            TVMTarget::OpenCL => "opencl",
            TVMTarget::ROCm => "rocm",
            TVMTarget::ARM => "llvm -mtriple=aarch64-linux-gnu",
            TVMTarget::WebGPU => "webgpu",
        }
    }
}

/// TVM optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// No optimization (O0)
    None,
    /// Basic optimization (O1)
    Basic,
    /// Standard optimization (O2)
    #[default]
    Standard,
    /// Aggressive optimization (O3)
    Aggressive,
}

impl OptLevel {
    pub fn as_int(&self) -> u8 {
        match self {
            OptLevel::None => 0,
            OptLevel::Basic => 1,
            OptLevel::Standard => 2,
            OptLevel::Aggressive => 3,
        }
    }
}

/// TVM tuning mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuningMode {
    /// No tuning (use default schedules)
    #[default]
    None,
    /// Auto-TVM tuning
    AutoTVM,
    /// Meta Schedule (newer tuner)
    MetaSchedule,
    /// Use pre-tuned logs
    PreTuned,
}

impl TuningMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            TuningMode::None => "none",
            TuningMode::AutoTVM => "autotvm",
            TuningMode::MetaSchedule => "meta_schedule",
            TuningMode::PreTuned => "pre_tuned",
        }
    }
}

/// Apache TVM backend configuration
#[derive(Debug, Clone)]
pub struct TVMConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Compilation target
    pub target: TVMTarget,
    /// Optimization level
    pub opt_level: OptLevel,
    /// Tuning mode
    pub tuning_mode: TuningMode,
    /// Path to tuning logs (for PreTuned mode)
    pub tuning_log_path: Option<PathBuf>,
    /// Number of tuning trials (for AutoTVM/MetaSchedule)
    pub tuning_trials: usize,
    /// Enable relay optimizations
    pub enable_relay_opt: bool,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for TVMConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            target: TVMTarget::LLVM,
            opt_level: OptLevel::Standard,
            tuning_mode: TuningMode::None,
            tuning_log_path: None,
            tuning_trials: 100,
            enable_relay_opt: true,
            model_path: None,
            timeout: Duration::from_secs(600),
            warmup_iterations: 10,
            benchmark_iterations: 100,
        }
    }
}

impl TVMConfig {
    /// Create config for CUDA target
    pub fn cuda() -> Self {
        Self {
            target: TVMTarget::CUDA,
            opt_level: OptLevel::Aggressive,
            ..Default::default()
        }
    }

    /// Create config with auto-tuning
    pub fn with_autotvm() -> Self {
        Self {
            tuning_mode: TuningMode::AutoTVM,
            tuning_trials: 500,
            opt_level: OptLevel::Aggressive,
            ..Default::default()
        }
    }

    /// Create config with meta schedule tuning
    pub fn with_meta_schedule() -> Self {
        Self {
            tuning_mode: TuningMode::MetaSchedule,
            tuning_trials: 1000,
            opt_level: OptLevel::Aggressive,
            ..Default::default()
        }
    }
}
