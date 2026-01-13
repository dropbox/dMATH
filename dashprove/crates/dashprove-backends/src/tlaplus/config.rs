//! Configuration types for TLA+ backend

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for TLA+ backend
#[derive(Debug, Clone)]
pub struct TlaPlusConfig {
    /// Path to TLC executable or tla2tools.jar
    pub tlc_path: Option<PathBuf>,
    /// Path to Java executable (if using tla2tools.jar)
    pub java_path: PathBuf,
    /// Maximum workers for TLC
    pub workers: u32,
    /// Model checking depth limit
    pub depth_limit: Option<u32>,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for TlaPlusConfig {
    fn default() -> Self {
        Self {
            tlc_path: None,
            java_path: PathBuf::from("java"),
            workers: 1,
            depth_limit: Some(100),
            timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}
