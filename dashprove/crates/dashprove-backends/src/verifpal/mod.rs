//! Verifpal backend for security protocol verification
//!
//! Verifpal is a modern security protocol verification tool designed for
//! accessibility. It uses a simplified modeling language and can verify
//! confidentiality, authentication, freshness, and equivalence properties.
//!
//! See: <https://verifpal.com/>
//!
//! # USL to Verifpal Compilation
//!
//! This backend compiles USL security properties to Verifpal models:
//!
//! - `forall x: T . P(x)` → confidentiality queries for secrets
//! - `not knows(agent, secret)` → `confidentiality? secret`
//! - `authorized(a, r) implies modified(a, r)` → authentication queries
//! - Types are compiled to Verifpal constants
//! - Functions referenced in properties become protocol operations

mod config;
mod detection;
mod model;
mod parsing;

#[cfg(test)]
mod tests;

pub use config::{VerifpalAnalysis, VerifpalConfig};

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

/// Verifpal security protocol verification backend
pub struct VerifpalBackend {
    config: VerifpalConfig,
}

impl Default for VerifpalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifpalBackend {
    /// Create a new Verifpal backend with default configuration
    pub fn new() -> Self {
        Self {
            config: VerifpalConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: VerifpalConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for VerifpalBackend {
    fn id(&self) -> BackendId {
        BackendId::Verifpal
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::SecurityProtocol]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let verifpal_path = detection::detect_verifpal(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate model file
        let model_code = model::generate_verifpal_model(&self.config, spec);
        let model_path = temp_dir.path().join("protocol.vp");
        std::fs::write(&model_path, &model_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write model: {}", e))
        })?;

        let mut cmd = Command::new(&verifpal_path);
        cmd.arg("verify").arg(&model_path);

        if self.config.json_output {
            cmd.arg("--json");
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Verifpal stdout: {}", stdout);
                let status = parsing::parse_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Verifpal,
                    status,
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Verifpal: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_verifpal(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
