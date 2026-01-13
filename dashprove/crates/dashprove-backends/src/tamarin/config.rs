//! Configuration types for Tamarin backend

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Tamarin backend
#[derive(Debug, Clone)]
pub struct TamarinConfig {
    /// Path to tamarin-prover binary
    pub tamarin_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Use auto mode (--prove)
    pub auto_prove: bool,
}

impl Default for TamarinConfig {
    fn default() -> Self {
        Self {
            tamarin_path: None,
            timeout: Duration::from_secs(300),
            auto_prove: true,
        }
    }
}
