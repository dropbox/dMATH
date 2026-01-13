//! OpenVINO backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// OpenVINO device target
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceTarget {
    /// CPU inference
    #[default]
    CPU,
    /// Intel integrated GPU
    GPU,
    /// Intel Movidius VPU
    VPU,
    /// Intel FPGA
    FPGA,
    /// Heterogeneous (multi-device)
    HETERO,
    /// Multi-device execution
    MULTI,
    /// Automatic device selection
    AUTO,
}

impl DeviceTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceTarget::CPU => "CPU",
            DeviceTarget::GPU => "GPU",
            DeviceTarget::VPU => "VPU",
            DeviceTarget::FPGA => "FPGA",
            DeviceTarget::HETERO => "HETERO",
            DeviceTarget::MULTI => "MULTI",
            DeviceTarget::AUTO => "AUTO",
        }
    }
}

/// OpenVINO inference precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferencePrecision {
    /// Full 32-bit floating point
    #[default]
    FP32,
    /// 16-bit floating point
    FP16,
    /// BF16 (Brain Float 16)
    BF16,
    /// 8-bit integer
    INT8,
}

impl InferencePrecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            InferencePrecision::FP32 => "FP32",
            InferencePrecision::FP16 => "FP16",
            InferencePrecision::BF16 => "BF16",
            InferencePrecision::INT8 => "INT8",
        }
    }
}

/// OpenVINO performance hint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerformanceHint {
    /// Optimize for latency
    #[default]
    Latency,
    /// Optimize for throughput
    Throughput,
    /// Let OpenVINO decide
    Undefined,
}

impl PerformanceHint {
    pub fn as_str(&self) -> &'static str {
        match self {
            PerformanceHint::Latency => "LATENCY",
            PerformanceHint::Throughput => "THROUGHPUT",
            PerformanceHint::Undefined => "UNDEFINED",
        }
    }
}

/// OpenVINO backend configuration
#[derive(Debug, Clone)]
pub struct OpenVINOConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Target device
    pub device: DeviceTarget,
    /// Inference precision
    pub precision: InferencePrecision,
    /// Performance hint
    pub performance_hint: PerformanceHint,
    /// Number of inference threads (0 = auto)
    pub num_threads: usize,
    /// Number of streams for throughput mode
    pub num_streams: usize,
    /// Enable dynamic shapes
    pub enable_dynamic_shapes: bool,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for OpenVINOConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            device: DeviceTarget::CPU,
            precision: InferencePrecision::FP32,
            performance_hint: PerformanceHint::Latency,
            num_threads: 0,
            num_streams: 1,
            enable_dynamic_shapes: false,
            model_path: None,
            timeout: Duration::from_secs(300),
            warmup_iterations: 5,
            benchmark_iterations: 100,
        }
    }
}

impl OpenVINOConfig {
    /// Create config for GPU inference
    pub fn gpu() -> Self {
        Self {
            device: DeviceTarget::GPU,
            precision: InferencePrecision::FP16,
            ..Default::default()
        }
    }

    /// Create config for high-throughput inference
    pub fn high_throughput() -> Self {
        Self {
            device: DeviceTarget::CPU,
            performance_hint: PerformanceHint::Throughput,
            num_streams: 4,
            benchmark_iterations: 1000,
            ..Default::default()
        }
    }

    /// Create config for INT8 quantized inference
    pub fn int8() -> Self {
        Self {
            precision: InferencePrecision::INT8,
            ..Default::default()
        }
    }
}
