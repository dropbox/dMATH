//! RobustBench backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Dataset for RobustBench evaluation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RobustBenchDataset {
    /// CIFAR-10 dataset (10 classes, 32x32 images)
    #[default]
    Cifar10,
    /// CIFAR-100 dataset (100 classes, 32x32 images)
    Cifar100,
    /// ImageNet dataset (1000 classes, various sizes)
    ImageNet,
}

impl RobustBenchDataset {
    /// Get the RobustBench dataset name
    pub fn dataset_name(&self) -> &'static str {
        match self {
            RobustBenchDataset::Cifar10 => "cifar10",
            RobustBenchDataset::Cifar100 => "cifar100",
            RobustBenchDataset::ImageNet => "imagenet",
        }
    }
}

/// Threat model for RobustBench evaluation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ThreatModel {
    /// Linf perturbations (most common)
    #[default]
    Linf,
    /// L2 perturbations
    L2,
    /// Common corruptions (blur, noise, etc.)
    Corruptions,
}

impl ThreatModel {
    /// Get the RobustBench threat model name
    pub fn model_name(&self) -> &'static str {
        match self {
            ThreatModel::Linf => "Linf",
            ThreatModel::L2 => "L2",
            ThreatModel::Corruptions => "corruptions",
        }
    }

    /// Get default epsilon for this threat model
    pub fn default_epsilon(&self) -> f64 {
        match self {
            ThreatModel::Linf => 8.0 / 255.0, // 8/255 for Linf
            ThreatModel::L2 => 0.5,           // 0.5 for L2
            ThreatModel::Corruptions => 0.0,  // N/A for corruptions
        }
    }
}

/// RobustBench backend configuration
#[derive(Debug, Clone)]
pub struct RobustBenchConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Dataset to evaluate on
    pub dataset: RobustBenchDataset,
    /// Threat model
    pub threat_model: ThreatModel,
    /// Epsilon bound for adversarial perturbation
    pub epsilon: f64,
    /// Number of samples to evaluate
    pub num_samples: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model name from RobustBench (e.g., "Carmon2019Unlabeled")
    pub model_name: Option<String>,
    /// Custom model path (if not using RobustBench model)
    pub model_path: Option<PathBuf>,
    /// Use AutoAttack ensemble (stronger but slower)
    pub use_autoattack: bool,
    /// Batch size for evaluation
    pub batch_size: usize,
}

impl Default for RobustBenchConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            dataset: RobustBenchDataset::Cifar10,
            threat_model: ThreatModel::Linf,
            epsilon: 8.0 / 255.0,
            num_samples: 1000,
            timeout: Duration::from_secs(600),
            model_name: None,
            model_path: None,
            use_autoattack: true,
            batch_size: 100,
        }
    }
}

impl RobustBenchConfig {
    /// Create config for CIFAR-10 Linf evaluation
    pub fn cifar10_linf() -> Self {
        Self {
            dataset: RobustBenchDataset::Cifar10,
            threat_model: ThreatModel::Linf,
            epsilon: 8.0 / 255.0,
            ..Default::default()
        }
    }

    /// Create config for CIFAR-10 L2 evaluation
    pub fn cifar10_l2() -> Self {
        Self {
            dataset: RobustBenchDataset::Cifar10,
            threat_model: ThreatModel::L2,
            epsilon: 0.5,
            ..Default::default()
        }
    }

    /// Create config for CIFAR-100 evaluation
    pub fn cifar100_linf() -> Self {
        Self {
            dataset: RobustBenchDataset::Cifar100,
            threat_model: ThreatModel::Linf,
            epsilon: 8.0 / 255.0,
            ..Default::default()
        }
    }

    /// Create config for ImageNet evaluation
    pub fn imagenet_linf() -> Self {
        Self {
            dataset: RobustBenchDataset::ImageNet,
            threat_model: ThreatModel::Linf,
            epsilon: 4.0 / 255.0,
            batch_size: 32,
            ..Default::default()
        }
    }

    /// Create config for corruption robustness evaluation
    pub fn corruptions() -> Self {
        Self {
            dataset: RobustBenchDataset::Cifar10,
            threat_model: ThreatModel::Corruptions,
            epsilon: 0.0,
            use_autoattack: false,
            ..Default::default()
        }
    }
}
