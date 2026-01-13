//! DNNV backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Verifier backend for DNNV (the framework supports multiple verifiers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerifierBackend {
    /// Planet verifier
    #[default]
    Planet,
    /// Marabou verifier
    Marabou,
    /// ERAN verifier
    Eran,
    /// MIPVerify
    MIPVerify,
    /// nnenum
    Nnenum,
    /// Neurify
    Neurify,
    /// Reluplex
    Reluplex,
}

impl VerifierBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            VerifierBackend::Planet => "planet",
            VerifierBackend::Marabou => "marabou",
            VerifierBackend::Eran => "eran",
            VerifierBackend::MIPVerify => "mipverify",
            VerifierBackend::Nnenum => "nnenum",
            VerifierBackend::Neurify => "neurify",
            VerifierBackend::Reluplex => "reluplex",
        }
    }
}

/// DNNV backend configuration
#[derive(Debug, Clone)]
pub struct DnnvConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Verifier backend to use
    pub verifier: VerifierBackend,
    /// Epsilon bound for perturbation
    pub epsilon: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Model path override
    pub model_path: Option<PathBuf>,
    /// Property file path
    pub property_path: Option<PathBuf>,
}

impl Default for DnnvConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            verifier: VerifierBackend::Planet,
            epsilon: 0.01,
            timeout: Duration::from_secs(300),
            model_path: None,
            property_path: None,
        }
    }
}

impl DnnvConfig {
    pub fn with_marabou() -> Self {
        Self {
            verifier: VerifierBackend::Marabou,
            ..Default::default()
        }
    }

    pub fn with_eran() -> Self {
        Self {
            verifier: VerifierBackend::Eran,
            ..Default::default()
        }
    }
}
