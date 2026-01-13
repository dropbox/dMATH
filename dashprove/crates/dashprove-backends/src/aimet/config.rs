//! AIMET backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// AIMET quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantScheme {
    /// Post-training quantization
    #[default]
    PostTraining,
    /// Quantization-aware training
    QuantizationAware,
    /// Cross-layer equalization + PTQ
    CrossLayerEqualization,
    /// Adaptive rounding
    AdaRound,
}

impl QuantScheme {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantScheme::PostTraining => "post_training",
            QuantScheme::QuantizationAware => "qat",
            QuantScheme::CrossLayerEqualization => "cle",
            QuantScheme::AdaRound => "adaround",
        }
    }
}

/// Bit width for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AimetBitWidth {
    /// 8-bit quantization
    #[default]
    W8A8,
    /// 4-bit weights, 8-bit activations
    W4A8,
    /// 4-bit weights and activations
    W4A4,
    /// 16-bit quantization
    W16A16,
}

impl AimetBitWidth {
    pub fn weight_bits(&self) -> u8 {
        match self {
            AimetBitWidth::W8A8 => 8,
            AimetBitWidth::W4A8 => 4,
            AimetBitWidth::W4A4 => 4,
            AimetBitWidth::W16A16 => 16,
        }
    }

    pub fn activation_bits(&self) -> u8 {
        match self {
            AimetBitWidth::W8A8 => 8,
            AimetBitWidth::W4A8 => 8,
            AimetBitWidth::W4A4 => 4,
            AimetBitWidth::W16A16 => 16,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AimetBitWidth::W8A8 => "w8a8",
            AimetBitWidth::W4A8 => "w4a8",
            AimetBitWidth::W4A4 => "w4a4",
            AimetBitWidth::W16A16 => "w16a16",
        }
    }
}

/// Rounding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundingMode {
    /// Round to nearest
    #[default]
    Nearest,
    /// Stochastic rounding
    Stochastic,
}

impl RoundingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            RoundingMode::Nearest => "nearest",
            RoundingMode::Stochastic => "stochastic",
        }
    }
}

/// Compression mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AimetCompressionMode {
    /// Quantization only
    #[default]
    QuantizationOnly,
    /// Spatial SVD compression
    SpatialSVD,
    /// Channel pruning
    ChannelPruning,
    /// Weight SVD
    WeightSVD,
}

impl AimetCompressionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            AimetCompressionMode::QuantizationOnly => "quantization",
            AimetCompressionMode::SpatialSVD => "spatial_svd",
            AimetCompressionMode::ChannelPruning => "channel_pruning",
            AimetCompressionMode::WeightSVD => "weight_svd",
        }
    }
}

/// AIMET backend configuration
#[derive(Debug, Clone)]
pub struct AimetConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Quantization scheme
    pub quant_scheme: QuantScheme,
    /// Bit width configuration
    pub bit_width: AimetBitWidth,
    /// Rounding mode
    pub rounding_mode: RoundingMode,
    /// Compression mode
    pub compression_mode: AimetCompressionMode,
    /// Number of calibration batches
    pub num_batches: usize,
    /// Use per-channel quantization
    pub per_channel: bool,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
}

impl Default for AimetConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            quant_scheme: QuantScheme::PostTraining,
            bit_width: AimetBitWidth::W8A8,
            rounding_mode: RoundingMode::Nearest,
            compression_mode: AimetCompressionMode::QuantizationOnly,
            num_batches: 32,
            per_channel: true,
            model_path: None,
            timeout: Duration::from_secs(600),
        }
    }
}

impl AimetConfig {
    /// Create config with AdaRound optimization
    pub fn adaround() -> Self {
        Self {
            quant_scheme: QuantScheme::AdaRound,
            num_batches: 100,
            ..Default::default()
        }
    }

    /// Create config with cross-layer equalization
    pub fn cle() -> Self {
        Self {
            quant_scheme: QuantScheme::CrossLayerEqualization,
            ..Default::default()
        }
    }

    /// Create config with QAT
    pub fn qat() -> Self {
        Self {
            quant_scheme: QuantScheme::QuantizationAware,
            ..Default::default()
        }
    }

    /// Create config with 4-bit quantization
    pub fn int4() -> Self {
        Self {
            bit_width: AimetBitWidth::W4A8,
            quant_scheme: QuantScheme::AdaRound,
            ..Default::default()
        }
    }

    /// Create config with spatial SVD compression
    pub fn spatial_svd() -> Self {
        Self {
            compression_mode: AimetCompressionMode::SpatialSVD,
            ..Default::default()
        }
    }
}
