//! Foolbox backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Attack type for Foolbox adversarial evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum FoolboxAttack {
    /// Fast Gradient Sign Method
    #[default]
    FGSM,
    /// Projected Gradient Descent (L-infinity)
    LinfPGD,
    /// Projected Gradient Descent (L2)
    L2PGD,
    /// Carlini & Wagner L2 attack
    CarliniWagner,
    /// DeepFool
    DeepFool,
    /// Brendel & Bethge attack
    BrendelBethge,
}

impl FoolboxAttack {
    /// Get the Foolbox class name for this attack
    pub fn foolbox_class(&self) -> &'static str {
        match self {
            FoolboxAttack::FGSM => "LinfFastGradientAttack",
            FoolboxAttack::LinfPGD => "LinfPGD",
            FoolboxAttack::L2PGD => "L2PGD",
            FoolboxAttack::CarliniWagner => "L2CarliniWagnerAttack",
            FoolboxAttack::DeepFool => "LinfDeepFoolAttack",
            FoolboxAttack::BrendelBethge => "LinfinityBrendelBethgeAttack",
        }
    }
}

/// Foolbox backend configuration
#[derive(Debug, Clone)]
pub struct FoolboxConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Epsilon bound for adversarial perturbation
    pub epsilon: f64,
    /// Attack type to use
    pub attack_type: FoolboxAttack,
    /// Number of attack steps (for iterative attacks)
    pub steps: usize,
    /// Number of samples to evaluate
    pub num_samples: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override (if not in USL spec)
    pub model_path: Option<PathBuf>,
}

impl Default for FoolboxConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            epsilon: 0.3,
            attack_type: FoolboxAttack::FGSM,
            steps: 40,
            num_samples: 100,
            timeout: Duration::from_secs(300),
            model_path: None,
        }
    }
}

impl FoolboxConfig {
    /// Create config with PGD attack
    pub fn pgd() -> Self {
        Self {
            attack_type: FoolboxAttack::LinfPGD,
            epsilon: 0.031,
            steps: 40,
            ..Default::default()
        }
    }

    /// Create config with DeepFool attack
    pub fn deepfool() -> Self {
        Self {
            attack_type: FoolboxAttack::DeepFool,
            steps: 50,
            ..Default::default()
        }
    }
}
