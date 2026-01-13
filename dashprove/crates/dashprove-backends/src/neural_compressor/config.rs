//! Intel Neural Compressor backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Quantization approach
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationApproach {
    /// Post-training static quantization
    #[default]
    PostTrainingStatic,
    /// Post-training dynamic quantization
    PostTrainingDynamic,
    /// Quantization-aware training
    QuantizationAwareTraining,
}

impl QuantizationApproach {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantizationApproach::PostTrainingStatic => "post_training_static_quant",
            QuantizationApproach::PostTrainingDynamic => "post_training_dynamic_quant",
            QuantizationApproach::QuantizationAwareTraining => "quant_aware_training",
        }
    }
}

/// Quantization data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantDataType {
    /// INT8 quantization
    #[default]
    INT8,
    /// UINT8 quantization
    UINT8,
    /// FP16 quantization
    FP16,
    /// BF16 quantization
    BF16,
}

impl QuantDataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantDataType::INT8 => "int8",
            QuantDataType::UINT8 => "uint8",
            QuantDataType::FP16 => "fp16",
            QuantDataType::BF16 => "bf16",
        }
    }
}

/// Quantization calibration method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CalibrationMethod {
    /// Min-max calibration
    #[default]
    MinMax,
    /// Entropy calibration
    Entropy,
    /// Percentile calibration
    Percentile,
}

impl CalibrationMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            CalibrationMethod::MinMax => "minmax",
            CalibrationMethod::Entropy => "kl",
            CalibrationMethod::Percentile => "percentile",
        }
    }
}

/// Tuning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuningStrategy {
    /// Basic tuning
    #[default]
    Basic,
    /// Bayesian optimization
    Bayesian,
    /// Exhaustive search
    Exhaustive,
    /// Random search
    Random,
    /// MSE-based tuning
    MSE,
}

impl TuningStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            TuningStrategy::Basic => "basic",
            TuningStrategy::Bayesian => "bayesian",
            TuningStrategy::Exhaustive => "exhaustive",
            TuningStrategy::Random => "random",
            TuningStrategy::MSE => "mse",
        }
    }
}

/// Intel Neural Compressor backend configuration
#[derive(Debug, Clone)]
pub struct NeuralCompressorConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Quantization approach
    pub approach: QuantizationApproach,
    /// Quantization data type
    pub quant_dtype: QuantDataType,
    /// Calibration method
    pub calibration: CalibrationMethod,
    /// Tuning strategy
    pub tuning_strategy: TuningStrategy,
    /// Target accuracy loss (0.0-1.0)
    pub accuracy_criterion: f64,
    /// Maximum number of tuning trials
    pub max_trials: usize,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of calibration samples
    pub calibration_samples: usize,
    /// Enable pruning
    pub enable_pruning: bool,
    /// Pruning sparsity target (0.0-1.0)
    pub pruning_sparsity: f64,
}

impl Default for NeuralCompressorConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            approach: QuantizationApproach::PostTrainingStatic,
            quant_dtype: QuantDataType::INT8,
            calibration: CalibrationMethod::MinMax,
            tuning_strategy: TuningStrategy::Basic,
            accuracy_criterion: 0.01,
            max_trials: 100,
            model_path: None,
            timeout: Duration::from_secs(600),
            calibration_samples: 100,
            enable_pruning: false,
            pruning_sparsity: 0.5,
        }
    }
}

impl NeuralCompressorConfig {
    /// Create config for dynamic quantization
    pub fn dynamic() -> Self {
        Self {
            approach: QuantizationApproach::PostTrainingDynamic,
            ..Default::default()
        }
    }

    /// Create config for quantization-aware training
    pub fn qat() -> Self {
        Self {
            approach: QuantizationApproach::QuantizationAwareTraining,
            tuning_strategy: TuningStrategy::MSE,
            ..Default::default()
        }
    }

    /// Create config with pruning enabled
    pub fn with_pruning(sparsity: f64) -> Self {
        Self {
            enable_pruning: true,
            pruning_sparsity: sparsity,
            ..Default::default()
        }
    }
}
