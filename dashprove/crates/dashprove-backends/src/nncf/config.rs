//! NNCF backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// NNCF compression mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMode {
    /// Quantization only
    #[default]
    Quantization,
    /// Pruning only
    Pruning,
    /// Sparsity only
    Sparsity,
    /// Filter pruning
    FilterPruning,
    /// Combined quantization and pruning
    QuantizationPruning,
}

impl CompressionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionMode::Quantization => "quantization",
            CompressionMode::Pruning => "pruning",
            CompressionMode::Sparsity => "sparsity",
            CompressionMode::FilterPruning => "filter_pruning",
            CompressionMode::QuantizationPruning => "quantization_pruning",
        }
    }
}

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationMode {
    /// Symmetric quantization
    #[default]
    Symmetric,
    /// Asymmetric quantization
    Asymmetric,
}

impl QuantizationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantizationMode::Symmetric => "symmetric",
            QuantizationMode::Asymmetric => "asymmetric",
        }
    }
}

/// Quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitWidth {
    /// 8-bit quantization
    #[default]
    Bits8,
    /// 4-bit quantization
    Bits4,
    /// Mixed precision
    Mixed,
}

impl BitWidth {
    pub fn as_int(&self) -> Option<u8> {
        match self {
            BitWidth::Bits8 => Some(8),
            BitWidth::Bits4 => Some(4),
            BitWidth::Mixed => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            BitWidth::Bits8 => "8",
            BitWidth::Bits4 => "4",
            BitWidth::Mixed => "mixed",
        }
    }
}

/// Pruning schedule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PruningSchedule {
    /// Constant sparsity
    #[default]
    Constant,
    /// Polynomial decay schedule
    Polynomial,
    /// Exponential decay schedule
    Exponential,
}

impl PruningSchedule {
    pub fn as_str(&self) -> &'static str {
        match self {
            PruningSchedule::Constant => "constant",
            PruningSchedule::Polynomial => "polynomial",
            PruningSchedule::Exponential => "exponential",
        }
    }
}

/// NNCF backend configuration
#[derive(Debug, Clone)]
pub struct NNCFConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Compression mode
    pub compression_mode: CompressionMode,
    /// Quantization mode
    pub quantization_mode: QuantizationMode,
    /// Bit width
    pub bit_width: BitWidth,
    /// Target sparsity for pruning (0.0-1.0)
    pub target_sparsity: f64,
    /// Pruning schedule
    pub pruning_schedule: PruningSchedule,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of calibration samples
    pub calibration_samples: usize,
    /// Ignored scopes (layer names to skip)
    pub ignored_scopes: Vec<String>,
}

impl Default for NNCFConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            compression_mode: CompressionMode::Quantization,
            quantization_mode: QuantizationMode::Symmetric,
            bit_width: BitWidth::Bits8,
            target_sparsity: 0.5,
            pruning_schedule: PruningSchedule::Constant,
            model_path: None,
            timeout: Duration::from_secs(600),
            calibration_samples: 100,
            ignored_scopes: vec![],
        }
    }
}

impl NNCFConfig {
    /// Create config for 4-bit quantization
    pub fn int4() -> Self {
        Self {
            bit_width: BitWidth::Bits4,
            quantization_mode: QuantizationMode::Asymmetric,
            ..Default::default()
        }
    }

    /// Create config for pruning
    pub fn pruning(sparsity: f64) -> Self {
        Self {
            compression_mode: CompressionMode::Pruning,
            target_sparsity: sparsity,
            ..Default::default()
        }
    }

    /// Create config for filter pruning
    pub fn filter_pruning(sparsity: f64) -> Self {
        Self {
            compression_mode: CompressionMode::FilterPruning,
            target_sparsity: sparsity,
            ..Default::default()
        }
    }

    /// Create config for combined quantization and pruning
    pub fn combined(sparsity: f64) -> Self {
        Self {
            compression_mode: CompressionMode::QuantizationPruning,
            target_sparsity: sparsity,
            ..Default::default()
        }
    }
}
