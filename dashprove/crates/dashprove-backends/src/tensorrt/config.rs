//! TensorRT backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// TensorRT precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full 32-bit floating point
    #[default]
    FP32,
    /// 16-bit floating point (half precision)
    FP16,
    /// 8-bit integer quantization
    INT8,
    /// TensorFloat-32 (Ampere and later)
    TF32,
}

impl PrecisionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            PrecisionMode::FP32 => "FP32",
            PrecisionMode::FP16 => "FP16",
            PrecisionMode::INT8 => "INT8",
            PrecisionMode::TF32 => "TF32",
        }
    }
}

/// TensorRT builder optimization profile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationProfile {
    /// Optimize for latency
    #[default]
    Latency,
    /// Optimize for throughput
    Throughput,
    /// Balanced optimization
    Balanced,
}

impl OptimizationProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            OptimizationProfile::Latency => "latency",
            OptimizationProfile::Throughput => "throughput",
            OptimizationProfile::Balanced => "balanced",
        }
    }
}

/// TensorRT backend configuration
#[derive(Debug, Clone)]
pub struct TensorRTConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Precision mode
    pub precision: PrecisionMode,
    /// Optimization profile
    pub optimization_profile: OptimizationProfile,
    /// Maximum workspace size in bytes
    pub max_workspace_size: usize,
    /// Enable strict types (no implicit precision conversion)
    pub strict_types: bool,
    /// Maximum batch size for optimization
    pub max_batch_size: usize,
    /// Enable sparsity optimizations
    pub enable_sparsity: bool,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            precision: PrecisionMode::FP32,
            optimization_profile: OptimizationProfile::Latency,
            max_workspace_size: 1 << 30, // 1 GB
            strict_types: false,
            max_batch_size: 1,
            enable_sparsity: false,
            model_path: None,
            timeout: Duration::from_secs(600),
            warmup_iterations: 10,
            benchmark_iterations: 100,
        }
    }
}

impl TensorRTConfig {
    /// Create config for FP16 inference
    pub fn fp16() -> Self {
        Self {
            precision: PrecisionMode::FP16,
            ..Default::default()
        }
    }

    /// Create config for INT8 quantized inference
    pub fn int8() -> Self {
        Self {
            precision: PrecisionMode::INT8,
            ..Default::default()
        }
    }

    /// Create config for high-throughput batch inference
    pub fn high_throughput() -> Self {
        Self {
            precision: PrecisionMode::FP16,
            optimization_profile: OptimizationProfile::Throughput,
            max_batch_size: 32,
            benchmark_iterations: 1000,
            ..Default::default()
        }
    }
}
