//! Captum backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Attribution method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptumMethod {
    /// Integrated Gradients
    #[default]
    IntegratedGradients,
    /// Saliency maps
    Saliency,
    /// DeepLIFT
    DeepLift,
    /// Gradient SHAP
    GradientShap,
}

impl CaptumMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            CaptumMethod::IntegratedGradients => "integrated_gradients",
            CaptumMethod::Saliency => "saliency",
            CaptumMethod::DeepLift => "deeplift",
            CaptumMethod::GradientShap => "gradient_shap",
        }
    }
}

/// Captum backend configuration
#[derive(Debug, Clone)]
pub struct CaptumConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Attribution method
    pub method: CaptumMethod,
    /// Steps for methods like Integrated Gradients
    pub steps: u32,
    /// Whether to apply noise tunnel smoothing
    pub use_noise_tunnel: bool,
    /// Number of top features to report
    pub top_k: usize,
    /// Attribution magnitude threshold
    pub attribution_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for CaptumConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            method: CaptumMethod::IntegratedGradients,
            steps: 50,
            use_noise_tunnel: false,
            top_k: 5,
            attribution_threshold: 0.01,
            timeout: Duration::from_secs(300),
        }
    }
}

impl CaptumConfig {
    /// Enable noise tunnel smoothing
    pub fn with_noise_tunnel() -> Self {
        Self {
            use_noise_tunnel: true,
            ..Default::default()
        }
    }

    /// Use gradient SHAP method
    pub fn gradient_shap() -> Self {
        Self {
            method: CaptumMethod::GradientShap,
            use_noise_tunnel: true,
            ..Default::default()
        }
    }
}
