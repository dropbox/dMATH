//! Configuration for alpha-beta-CROWN backend

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for alpha-beta-CROWN backend
#[derive(Debug, Clone)]
pub struct AbCrownConfig {
    /// Path to abcrown.py script
    pub abcrown_path: Option<PathBuf>,
    /// Path to Python interpreter (default: python3)
    pub python_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Batch size for verification
    pub batch_size: Option<usize>,
}

impl Default for AbCrownConfig {
    fn default() -> Self {
        Self {
            abcrown_path: None,
            python_path: None,
            timeout: Duration::from_secs(300),
            use_gpu: true,
            batch_size: None,
        }
    }
}
