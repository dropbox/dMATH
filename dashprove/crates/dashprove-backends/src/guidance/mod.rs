//! Guidance structured LLM generation backend
//!
//! Provides constrained and structured generation for LLMs using
//! Microsoft's Guidance library with grammars, regex, and JSON schemas.
//!
//! Installation:
//! ```bash
//! pip install guidance
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::{GenerationMode, GuidanceConfig, ValidationMode};

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

/// Guidance backend implementation
pub struct GuidanceBackend {
    config: GuidanceConfig,
}

impl Default for GuidanceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GuidanceBackend {
    pub fn new() -> Self {
        Self {
            config: GuidanceConfig::default(),
        }
    }

    pub fn with_config(config: GuidanceConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for GuidanceBackend {
    fn id(&self) -> BackendId {
        BackendId::Guidance
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::LLMGuardrails]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let python_path = detection::detect_guidance(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let script_content = script::generate_guidance_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("guidance_verify.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write script: {}", e))
        })?;

        debug!("Generated Guidance script:\n{}", script_content);

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

                debug!("Guidance stdout: {}", stdout);
                debug!("Guidance stderr: {}", stderr);

                let (status, counterexample) = script::parse_guidance_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Guidance,
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
                "Failed to execute: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_guidance(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
