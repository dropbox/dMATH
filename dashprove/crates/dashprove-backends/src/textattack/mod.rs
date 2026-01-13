//! TextAttack backend for NLP adversarial robustness testing
//!
//! TextAttack is a Python library for NLP adversarial attacks, providing:
//! - Pre-built attack recipes (TextFooler, BERT-Attack, BAE, etc.)
//! - Support for HuggingFace transformers models
//! - Word-level and character-level perturbations
//! - Semantic similarity constraints
//!
//! See: <https://github.com/QData/TextAttack>
//!
//! # USL to TextAttack Compilation
//!
//! This backend compiles USL NLP robustness properties to TextAttack Python scripts:
//!
//! - Model references: HuggingFace model identifiers
//! - Dataset references: HuggingFace dataset identifiers
//! - Attack recipe selection based on property constraints
//!
//! # Installation
//!
//! ```bash
//! pip install textattack
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::TextAttackConfig;

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

/// TextAttack NLP adversarial robustness backend
pub struct TextAttackBackend {
    config: TextAttackConfig,
}

impl Default for TextAttackBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TextAttackBackend {
    /// Create a new TextAttack backend with default configuration
    pub fn new() -> Self {
        Self {
            config: TextAttackConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: TextAttackConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for TextAttackBackend {
    fn id(&self) -> BackendId {
        BackendId::TextAttack
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::AdversarialRobustness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect TextAttack installation
        let python_path = detection::detect_textattack(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for the script
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the TextAttack evaluation script
        let script_content = script::generate_textattack_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("textattack_eval.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write TextAttack script: {}", e))
        })?;

        debug!("Generated TextAttack script:\n{}", script_content);

        // Run the TextAttack evaluation script
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

                debug!("TextAttack stdout: {}", stdout);
                debug!("TextAttack stderr: {}", stderr);

                let (status, counterexample) = script::parse_textattack_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::TextAttack,
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
                "Failed to execute TextAttack script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_textattack(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
