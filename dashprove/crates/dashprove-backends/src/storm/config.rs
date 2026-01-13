//! Configuration types for Storm backend

use std::path::PathBuf;
use std::time::Duration;

/// State variable extracted from USL spec
#[derive(Debug, Clone)]
pub struct StateVar {
    /// Variable name
    pub name: String,
    /// Minimum value
    pub min: i32,
    /// Maximum value
    pub max: i32,
    /// Initial value
    pub init: i32,
}

/// Configuration for Storm backend
#[derive(Debug, Clone)]
pub struct StormConfig {
    /// Path to storm binary
    pub storm_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Numerical precision
    pub precision: f64,
}

impl Default for StormConfig {
    fn default() -> Self {
        Self {
            storm_path: None,
            timeout: Duration::from_secs(300),
            precision: 1e-6,
        }
    }
}
