//! CleverHans backend for adversarial ML testing
//!
//! CleverHans is a library for benchmarking machine learning systems'
//! vulnerability to adversarial examples, providing:
//! - Adversarial attacks (FGSM, PGD/BIM, MIM, C&W, etc.)
//! - Attack implementations for PyTorch and TensorFlow
//! - Reference implementations from adversarial ML research
//!
//! See: <https://github.com/cleverhans-lab/cleverhans>
//!
//! # USL to CleverHans Compilation
//!
//! This backend compiles USL adversarial robustness properties to CleverHans Python scripts:
//!
//! - Robustness properties: `|x - x0| <= epsilon` → epsilon bound for attack
//! - Model references: `.pt` paths → model loading
//! - Attack type selection based on property constraints
//!
//! # Installation
//!
//! ```bash
//! pip install cleverhans
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::CleverHansConfig;

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::process::Stdio;
use std::time::Instant;
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// CleverHans adversarial robustness backend
pub struct CleverHansBackend {
    config: CleverHansConfig,
}

impl Default for CleverHansBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CleverHansBackend {
    /// Create a new CleverHans backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CleverHansConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: CleverHansConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for CleverHansBackend {
    fn id(&self) -> BackendId {
        BackendId::CleverHans
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::AdversarialRobustness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect CleverHans installation
        let python_path = detection::detect_cleverhans(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for the script
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the CleverHans evaluation script
        let script_content = script::generate_cleverhans_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("cleverhans_eval.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write CleverHans script: {}", e))
        })?;

        debug!("Generated CleverHans script:\n{}", script_content);

        // Run the CleverHans evaluation script
        let mut cmd = Command::new(&python_path);
        cmd.arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("CleverHans stdout: {}", stdout);
                debug!("CleverHans stderr: {}", stderr);

                let (status, counterexample) = script::parse_cleverhans_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::CleverHans,
                    status,
                    proof: None,
                    counterexample,
                    diagnostics: if stderr.is_empty() {
                        vec![]
                    } else {
                        vec![stderr]
                    },
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute CleverHans script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_cleverhans(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
