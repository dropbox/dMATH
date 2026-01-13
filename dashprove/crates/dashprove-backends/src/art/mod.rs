//! ART (Adversarial Robustness Toolbox) backend for adversarial ML testing
//!
//! ART is IBM's comprehensive library for adversarial machine learning, providing:
//! - Adversarial attacks (FGSM, PGD, C&W, DeepFool, etc.)
//! - Adversarial defenses (adversarial training, certified defenses)
//! - Robustness metrics and evaluation
//!
//! See: <https://github.com/Trusted-AI/adversarial-robustness-toolbox>
//!
//! # USL to ART Compilation
//!
//! This backend compiles USL adversarial robustness properties to ART Python scripts:
//!
//! - Robustness properties: `|x - x0| <= epsilon` → epsilon bound for attack
//! - Model references: `.onnx` or `.pt` paths → model loading
//! - Attack type selection based on property constraints
//!
//! # Installation
//!
//! ```bash
//! pip install adversarial-robustness-toolbox
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::ArtConfig;

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

/// ART (Adversarial Robustness Toolbox) backend
pub struct ArtBackend {
    config: ArtConfig,
}

impl Default for ArtBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtBackend {
    /// Create a new ART backend with default configuration
    pub fn new() -> Self {
        Self {
            config: ArtConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: ArtConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for ArtBackend {
    fn id(&self) -> BackendId {
        BackendId::ART
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::AdversarialRobustness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect ART installation
        let python_path = detection::detect_art(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for the script
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the ART evaluation script
        let script_content = script::generate_art_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("art_eval.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write ART script: {}", e))
        })?;

        debug!("Generated ART script:\n{}", script_content);

        // Run the ART evaluation script
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

                debug!("ART stdout: {}", stdout);
                debug!("ART stderr: {}", stderr);

                let (status, counterexample) = script::parse_art_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::ART,
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
                "Failed to execute ART script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_art(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
