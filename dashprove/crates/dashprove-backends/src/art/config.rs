//! ART backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Attack type for ART adversarial evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum AttackType {
    /// Fast Gradient Sign Method (fast, less accurate)
    #[default]
    FGSM,
    /// Projected Gradient Descent (iterative, more accurate)
    PGD,
    /// Carlini & Wagner L2 attack (optimization-based)
    CW,
    /// DeepFool (minimal perturbation)
    DeepFool,
    /// Auto-PGD (adaptive step size)
    AutoPGD,
}

impl AttackType {
    /// Get the ART class name for this attack
    pub fn art_class(&self) -> &'static str {
        match self {
            AttackType::FGSM => "FastGradientMethod",
            AttackType::PGD => "ProjectedGradientDescent",
            AttackType::CW => "CarliniL2Method",
            AttackType::DeepFool => "DeepFool",
            AttackType::AutoPGD => "AutoProjectedGradientDescent",
        }
    }

    /// Get the ART module for this attack
    pub fn art_module(&self) -> &'static str {
        match self {
            AttackType::FGSM => "art.attacks.evasion",
            AttackType::PGD => "art.attacks.evasion",
            AttackType::CW => "art.attacks.evasion",
            AttackType::DeepFool => "art.attacks.evasion",
            AttackType::AutoPGD => "art.attacks.evasion",
        }
    }
}

/// ART backend configuration
#[derive(Debug, Clone)]
pub struct ArtConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Epsilon bound for adversarial perturbation (default: 0.3 for FGSM)
    pub epsilon: f64,
    /// Attack type to use
    pub attack_type: AttackType,
    /// Number of attack iterations (for iterative attacks like PGD)
    pub max_iter: usize,
    /// Number of samples to evaluate
    pub num_samples: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override (if not in USL spec)
    pub model_path: Option<PathBuf>,
    /// Norm type for perturbation bounds (inf, 2, 1)
    pub norm: String,
}

impl Default for ArtConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            epsilon: 0.3,
            attack_type: AttackType::FGSM,
            max_iter: 40,
            num_samples: 100,
            timeout: Duration::from_secs(300),
            model_path: None,
            norm: "inf".to_string(),
        }
    }
}

impl ArtConfig {
    /// Create config with PGD attack (more thorough)
    pub fn pgd() -> Self {
        Self {
            attack_type: AttackType::PGD,
            epsilon: 0.031,
            max_iter: 40,
            ..Default::default()
        }
    }

    /// Create config with AutoPGD (adaptive)
    pub fn auto_pgd() -> Self {
        Self {
            attack_type: AttackType::AutoPGD,
            epsilon: 0.031,
            max_iter: 100,
            ..Default::default()
        }
    }
}
