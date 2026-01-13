//! alpha-beta-CROWN backend for neural network verification
//!
//! alpha-beta-CROWN is a GPU-accelerated neural network verifier that uses
//! linear bound propagation with branch-and-bound. It has won multiple
//! VNN-COMP competitions.
//!
//! See: <https://github.com/Verified-Intelligence/alpha-beta-CROWN>
//!
//! ## USL Compilation
//!
//! This backend compiles USL specifications to VNNLIB format:
//! - Input bounds from comparisons: `x0 >= 0 and x0 <= 1` → input constraints
//! - Output constraints: `y0 > threshold` → output assertions
//! - Model paths from strings: `model = "path/to/model.onnx"` → model reference
//!
//! ## Counterexample Parsing
//!
//! alpha-beta-CROWN outputs counterexamples (adversarial inputs) when properties
//! are violated (SAT result). The parser handles multiple output formats:
//! - CSV format: `input: [0.1, 0.2, 0.3]` or `adv_example: [[...]]`
//! - Variable assignments: `X_0 = 0.5`
//! - JSON-like arrays: `[0.1, 0.2, 0.3]`

mod config;
mod detection;
mod model;
mod parsing;
mod vnnlib;

#[cfg(test)]
mod tests;

pub use config::AbCrownConfig;

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// alpha-beta-CROWN neural network verification backend
pub struct AbCrownBackend {
    config: AbCrownConfig,
}

impl Default for AbCrownBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AbCrownBackend {
    /// Create a new alpha-beta-CROWN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AbCrownConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: AbCrownConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for AbCrownBackend {
    fn id(&self) -> BackendId {
        BackendId::AlphaBetaCrown
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect alpha-beta-CROWN
        let (python_path, abcrown_path) = detection::detect_abcrown(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Compile USL spec to VNNLIB property
        let vnnlib_content = vnnlib::generate_vnnlib(spec)?;
        let property_path = temp_dir.path().join("property.vnnlib");
        std::fs::write(&property_path, &vnnlib_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write VNNLIB: {}", e))
        })?;

        // Extract model path from USL spec or use placeholder
        let model_path = if let Some(path) = model::extract_model_path(spec) {
            PathBuf::from(path)
        } else {
            // Placeholder for when USL has neural model references
            temp_dir.path().join("model.onnx")
        };

        let config_path = temp_dir.path().join("config.yaml");

        // Generate config
        let yaml_config = vnnlib::generate_config(
            &self.config,
            model_path.to_str().unwrap(),
            property_path.to_str().unwrap(),
        );
        std::fs::write(&config_path, &yaml_config).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write config: {}", e))
        })?;

        // Build command
        let mut cmd = Command::new(&python_path);
        cmd.arg(&abcrown_path)
            .arg("--config")
            .arg(&config_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Run with timeout
        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("alpha-beta-CROWN stdout: {}", stdout);
                debug!("alpha-beta-CROWN stderr: {}", stderr);

                let status = parsing::parse_output(&stdout, &stderr);

                // Parse counterexample if property was disproven (SAT)
                let counterexample = if matches!(status, VerificationStatus::Disproven) {
                    parsing::parse_counterexample(&stdout)
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::AlphaBetaCrown,
                    status,
                    proof: None,
                    counterexample,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute alpha-beta-CROWN: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_abcrown(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
