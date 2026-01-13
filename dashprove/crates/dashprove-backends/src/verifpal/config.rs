//! Configuration types for Verifpal backend

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Verifpal backend
#[derive(Debug, Clone)]
pub struct VerifpalConfig {
    /// Path to verifpal binary
    pub verifpal_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Analysis type
    pub analysis: VerifpalAnalysis,
    /// Output format
    pub json_output: bool,
}

/// Verifpal analysis type
#[derive(Debug, Clone, Copy, Default)]
pub enum VerifpalAnalysis {
    /// Passive attacker (eavesdropper only)
    Passive,
    /// Active attacker (default - can modify messages)
    #[default]
    Active,
}

impl VerifpalAnalysis {
    /// Get command line argument for this analysis type
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::Passive => "passive",
            Self::Active => "active",
        }
    }
}

impl Default for VerifpalConfig {
    fn default() -> Self {
        Self {
            verifpal_path: None,
            timeout: Duration::from_secs(120),
            analysis: VerifpalAnalysis::default(),
            json_output: true,
        }
    }
}
