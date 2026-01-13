//! Brevitas backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Weight quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightBitWidth {
    /// Binary weights (-1, +1)
    Binary,
    /// Ternary weights (-1, 0, +1)
    Ternary,
    /// 2-bit weights
    Bits2,
    /// 4-bit weights
    Bits4,
    /// 8-bit weights
    #[default]
    Bits8,
}

impl WeightBitWidth {
    pub fn as_int(&self) -> Option<u8> {
        match self {
            WeightBitWidth::Binary => Some(1),
            WeightBitWidth::Ternary => Some(2),
            WeightBitWidth::Bits2 => Some(2),
            WeightBitWidth::Bits4 => Some(4),
            WeightBitWidth::Bits8 => Some(8),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            WeightBitWidth::Binary => "binary",
            WeightBitWidth::Ternary => "ternary",
            WeightBitWidth::Bits2 => "2bit",
            WeightBitWidth::Bits4 => "4bit",
            WeightBitWidth::Bits8 => "8bit",
        }
    }
}

/// Activation quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationBitWidth {
    /// Binary activations
    Binary,
    /// 4-bit activations
    Bits4,
    /// 8-bit activations
    #[default]
    Bits8,
    /// Full precision (no quantization)
    Full,
}

impl ActivationBitWidth {
    pub fn as_int(&self) -> Option<u8> {
        match self {
            ActivationBitWidth::Binary => Some(1),
            ActivationBitWidth::Bits4 => Some(4),
            ActivationBitWidth::Bits8 => Some(8),
            ActivationBitWidth::Full => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ActivationBitWidth::Binary => "binary",
            ActivationBitWidth::Bits4 => "4bit",
            ActivationBitWidth::Bits8 => "8bit",
            ActivationBitWidth::Full => "full",
        }
    }
}

/// Quantization scaling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// Per-tensor scaling
    #[default]
    PerTensor,
    /// Per-channel scaling
    PerChannel,
    /// Per-group scaling
    PerGroup,
}

impl ScalingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScalingMode::PerTensor => "per_tensor",
            ScalingMode::PerChannel => "per_channel",
            ScalingMode::PerGroup => "per_group",
        }
    }
}

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantMethod {
    /// Symmetric quantization
    #[default]
    Symmetric,
    /// Asymmetric quantization
    Asymmetric,
    /// Power-of-two scaling
    PowerOfTwo,
}

impl QuantMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantMethod::Symmetric => "symmetric",
            QuantMethod::Asymmetric => "asymmetric",
            QuantMethod::PowerOfTwo => "power_of_two",
        }
    }
}

/// Export format for quantized model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    /// PyTorch native
    #[default]
    PyTorch,
    /// ONNX format
    ONNX,
    /// FINN format (for FPGA deployment)
    FINN,
    /// QONNX format
    QONNX,
}

impl ExportFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExportFormat::PyTorch => "pytorch",
            ExportFormat::ONNX => "onnx",
            ExportFormat::FINN => "finn",
            ExportFormat::QONNX => "qonnx",
        }
    }
}

/// Brevitas backend configuration
#[derive(Debug, Clone)]
pub struct BrevitasConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Weight bit width
    pub weight_bit_width: WeightBitWidth,
    /// Activation bit width
    pub activation_bit_width: ActivationBitWidth,
    /// Scaling mode
    pub scaling_mode: ScalingMode,
    /// Quantization method
    pub quant_method: QuantMethod,
    /// Export format
    pub export_format: ExportFormat,
    /// Group size for per-group quantization
    pub group_size: Option<usize>,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of calibration samples
    pub calibration_samples: usize,
}

impl Default for BrevitasConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            weight_bit_width: WeightBitWidth::Bits8,
            activation_bit_width: ActivationBitWidth::Bits8,
            scaling_mode: ScalingMode::PerTensor,
            quant_method: QuantMethod::Symmetric,
            export_format: ExportFormat::PyTorch,
            group_size: None,
            model_path: None,
            timeout: Duration::from_secs(600),
            calibration_samples: 100,
        }
    }
}

impl BrevitasConfig {
    /// Create config for 4-bit quantization
    pub fn int4() -> Self {
        Self {
            weight_bit_width: WeightBitWidth::Bits4,
            activation_bit_width: ActivationBitWidth::Bits8,
            scaling_mode: ScalingMode::PerChannel,
            ..Default::default()
        }
    }

    /// Create config for binary neural network
    pub fn binary() -> Self {
        Self {
            weight_bit_width: WeightBitWidth::Binary,
            activation_bit_width: ActivationBitWidth::Binary,
            quant_method: QuantMethod::Symmetric,
            ..Default::default()
        }
    }

    /// Create config for ternary neural network
    pub fn ternary() -> Self {
        Self {
            weight_bit_width: WeightBitWidth::Ternary,
            activation_bit_width: ActivationBitWidth::Full,
            ..Default::default()
        }
    }

    /// Create config for FINN FPGA export
    pub fn finn() -> Self {
        Self {
            weight_bit_width: WeightBitWidth::Bits4,
            activation_bit_width: ActivationBitWidth::Bits4,
            export_format: ExportFormat::FINN,
            ..Default::default()
        }
    }

    /// Create config for per-group quantization (LLM-style)
    pub fn grouped(group_size: usize) -> Self {
        Self {
            weight_bit_width: WeightBitWidth::Bits4,
            scaling_mode: ScalingMode::PerGroup,
            group_size: Some(group_size),
            ..Default::default()
        }
    }
}
