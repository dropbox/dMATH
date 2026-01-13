//! RobustBench backend for standardized adversarial robustness evaluation
//!
//! RobustBench is a standardized benchmark for adversarial robustness, providing:
//! - Pre-trained robust models (50+ models for CIFAR-10/100, ImageNet)
//! - Standardized evaluation with AutoAttack
//! - Leaderboard tracking state-of-the-art robustness
//!
//! See: <https://github.com/RobustBench/robustbench>
//!
//! # USL to RobustBench Compilation
//!
//! This backend compiles USL robustness properties to RobustBench Python scripts:
//!
//! - Epsilon bounds from USL properties
//! - Dataset selection (CIFAR-10, CIFAR-100, ImageNet)
//! - Threat model selection (Linf, L2, corruptions)
//!
//! # Installation
//!
//! ```bash
//! pip install robustbench
//! pip install autoattack  # Optional but recommended
//! ```

mod config;
mod detection;
mod script;

#[cfg(test)]
mod tests;

pub use config::RobustBenchConfig;

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

/// RobustBench adversarial robustness evaluation backend
pub struct RobustBenchBackend {
    config: RobustBenchConfig,
}

impl Default for RobustBenchBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustBenchBackend {
    /// Create a new RobustBench backend with default configuration
    pub fn new() -> Self {
        Self {
            config: RobustBenchConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: RobustBenchConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for RobustBenchBackend {
    fn id(&self) -> BackendId {
        BackendId::RobustBench
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::AdversarialRobustness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect RobustBench installation
        let python_path = detection::detect_robustbench(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for the script
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate the RobustBench evaluation script
        let script_content = script::generate_robustbench_script(spec, &self.config)?;
        let script_path = temp_dir.path().join("robustbench_eval.py");
        std::fs::write(&script_path, &script_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write RobustBench script: {}", e))
        })?;

        debug!("Generated RobustBench script:\n{}", script_content);

        // Run the RobustBench evaluation script
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

                debug!("RobustBench stdout: {}", stdout);
                debug!("RobustBench stderr: {}", stderr);

                let (status, counterexample) = script::parse_robustbench_output(&stdout, &stderr);

                Ok(BackendResult {
                    backend: BackendId::RobustBench,
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
                "Failed to execute RobustBench script: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_robustbench(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
