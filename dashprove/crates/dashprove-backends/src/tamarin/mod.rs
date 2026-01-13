//! Tamarin backend for security protocol verification
//!
//! Tamarin Prover is a powerful tool for symbolic verification of security protocols.
//! It supports multiset rewriting rules and can prove properties using backward search.
//!
//! See: <https://tamarin-prover.com/>
//!
//! # USL to Tamarin Compilation
//!
//! This backend compiles USL security properties to Tamarin theories:
//!
//! - `forall x: T . P(x)` → `All x. P(x)` in Tamarin lemmas
//! - `not knows(agent, secret)` → secrecy lemma with `not (Ex #j. K(secret) @ j)`
//! - `authorized(a, r) implies modified(a, r)` → correspondence lemma
//! - Types are compiled to Tamarin sorts
//! - Functions become Tamarin function symbols

mod config;
mod detection;
mod parsing;
mod theory;

#[cfg(test)]
mod tests;

pub use config::TamarinConfig;

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

/// Tamarin security protocol verification backend
pub struct TamarinBackend {
    config: TamarinConfig,
}

impl Default for TamarinBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TamarinBackend {
    /// Create a new Tamarin backend with default configuration
    pub fn new() -> Self {
        Self {
            config: TamarinConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TamarinConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for TamarinBackend {
    fn id(&self) -> BackendId {
        BackendId::Tamarin
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::SecurityProtocol]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let tamarin_path = detection::detect_tamarin(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let theory_code = theory::generate_theory(spec);
        let theory_path = temp_dir.path().join("theory.spthy");
        std::fs::write(&theory_path, &theory_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write theory: {}", e))
        })?;

        let mut cmd = Command::new(&tamarin_path);
        cmd.arg("--prove")
            .arg(&theory_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Tamarin stdout: {}", stdout);
                let status = parsing::parse_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::Tamarin,
                    status,
                    proof: None,
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Tamarin: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_tamarin(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
