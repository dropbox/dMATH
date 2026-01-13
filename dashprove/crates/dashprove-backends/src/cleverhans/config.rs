//! CleverHans backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Attack type for CleverHans adversarial evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum CleverHansAttack {
    /// Fast Gradient Sign Method
    #[default]
    FGSM,
    /// Basic Iterative Method (PGD variant)
    BIM,
    /// Momentum Iterative Method
    MIM,
    /// SPSA (gradient-free)
    SPSA,
    /// Carlini-Wagner L2 attack
    CarliniWagnerL2,
    /// Elastic Net attack
    ElasticNet,
}

impl CleverHansAttack {
    /// Get the CleverHans function name for this attack
    pub fn function_name(&self) -> &'static str {
        match self {
            CleverHansAttack::FGSM => "fast_gradient_method",
            CleverHansAttack::BIM => "projected_gradient_descent",
            CleverHansAttack::MIM => "momentum_iterative_method",
            CleverHansAttack::SPSA => "spsa",
            CleverHansAttack::CarliniWagnerL2 => "carlini_wagner_l2",
            CleverHansAttack::ElasticNet => "elastic_net_method",
        }
    }

    /// Get the CleverHans module for this attack
    pub fn module(&self) -> &'static str {
        "cleverhans.torch.attacks"
    }
}

/// CleverHans backend configuration
#[derive(Debug, Clone)]
pub struct CleverHansConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Epsilon bound for adversarial perturbation
    pub epsilon: f64,
    /// Attack type to use
    pub attack_type: CleverHansAttack,
    /// Number of attack iterations (for iterative attacks)
    pub nb_iter: usize,
    /// Step size for iterative attacks
    pub eps_iter: f64,
    /// Number of samples to evaluate
    pub num_samples: usize,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override (if not in USL spec)
    pub model_path: Option<PathBuf>,
    /// Norm type for perturbation bounds (inf, 2, 1)
    pub norm: String,
    /// Clip min value for inputs
    pub clip_min: f64,
    /// Clip max value for inputs
    pub clip_max: f64,
}

impl Default for CleverHansConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            epsilon: 0.3,
            attack_type: CleverHansAttack::FGSM,
            nb_iter: 40,
            eps_iter: 0.01,
            num_samples: 100,
            timeout: Duration::from_secs(300),
            model_path: None,
            norm: "inf".to_string(),
            clip_min: 0.0,
            clip_max: 1.0,
        }
    }
}

impl CleverHansConfig {
    /// Create config with BIM (Basic Iterative Method) attack
    pub fn bim() -> Self {
        Self {
            attack_type: CleverHansAttack::BIM,
            epsilon: 0.031,
            nb_iter: 40,
            eps_iter: 0.007,
            ..Default::default()
        }
    }

    /// Create config with MIM (Momentum Iterative Method) attack
    pub fn mim() -> Self {
        Self {
            attack_type: CleverHansAttack::MIM,
            epsilon: 0.031,
            nb_iter: 40,
            eps_iter: 0.007,
            ..Default::default()
        }
    }

    /// Create config with Carlini-Wagner L2 attack
    pub fn carlini_wagner() -> Self {
        Self {
            attack_type: CleverHansAttack::CarliniWagnerL2,
            epsilon: 0.5, // L2 norm is typically larger
            nb_iter: 1000,
            ..Default::default()
        }
    }
}
