//! ERAN configuration types
//!
//! Contains domain selection and configuration for the ERAN backend.

use std::path::PathBuf;
use std::time::Duration;

/// ERAN abstract domain
#[derive(Debug, Clone, Copy, Default)]
pub enum EranDomain {
    /// DeepZ - zonotope abstraction
    DeepZ,
    /// DeepPoly - polyhedra abstraction
    #[default]
    DeepPoly,
    /// RefinePoly - refined polyhedra
    RefinePoly,
    /// GPUPoly - GPU-accelerated polyhedra
    GpuPoly,
}

impl EranDomain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DeepZ => "deepzono",
            Self::DeepPoly => "deeppoly",
            Self::RefinePoly => "refinepoly",
            Self::GpuPoly => "gpupoly",
        }
    }
}

/// Configuration for ERAN backend
#[derive(Debug, Clone)]
pub struct EranConfig {
    /// Path to ERAN installation
    pub eran_path: Option<PathBuf>,
    /// Python interpreter
    pub python_path: Option<PathBuf>,
    /// Abstract domain to use
    pub domain: EranDomain,
    /// Epsilon for robustness verification
    pub epsilon: f64,
    /// Timeout for verification
    pub timeout: Duration,
    /// Use GPU acceleration (for GPUPoly)
    pub use_gpu: bool,
}

impl Default for EranConfig {
    fn default() -> Self {
        Self {
            eran_path: None,
            python_path: None,
            domain: EranDomain::default(),
            epsilon: 0.01,
            timeout: Duration::from_secs(300),
            use_gpu: false,
        }
    }
}
