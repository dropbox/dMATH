//! SelfCheckGPT backend for hallucination detection
//!
//! This backend integrates with SelfCheckGPT to provide:
//! - Self-consistency based hallucination detection
//! - BERTScore consistency checking
//! - N-gram overlap checking
//! - NLI-based entailment checking

mod config;
mod detection;
pub mod script;
#[cfg(test)]
mod tests;

pub use config::{CheckMethod, SamplingStrategy, SelfCheckGPTConfig};

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// SelfCheckGPT verification backend
pub struct SelfCheckGPTBackend {
    config: SelfCheckGPTConfig,
}

impl SelfCheckGPTBackend {
    /// Create a new SelfCheckGPT backend with default configuration
    pub fn new() -> Self {
        Self {
            config: SelfCheckGPTConfig::default(),
        }
    }

    /// Create a new SelfCheckGPT backend with custom configuration
    pub fn with_config(config: SelfCheckGPTConfig) -> Self {
        Self { config }
    }
}

impl Default for SelfCheckGPTBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for SelfCheckGPTBackend {
    fn id(&self) -> BackendId {
        BackendId::SelfCheckGPT
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::HallucinationDetection]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Check if SelfCheckGPT is available
        let python_path = match &self.config.python_path {
            Some(p) => p.clone(),
            None => detection::detect_selfcheckgpt().ok_or_else(|| {
                BackendError::Unavailable(
                    "SelfCheckGPT not found. Install with: pip install selfcheckgpt".to_string(),
                )
            })?,
        };

        // Generate verification script
        let script = script::generate_selfcheckgpt_script(spec, &self.config)?;

        // Run the script
        let mut child = Command::new(&python_path)
            .arg("-c")
            .arg(&script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to spawn Python: {}", e))
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.shutdown().await;
        }

        let output = tokio::time::timeout(self.config.timeout, child.wait_with_output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Process error: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let (status, counterexample) = script::parse_selfcheckgpt_output(&stdout, &stderr);

        Ok(BackendResult {
            backend: BackendId::SelfCheckGPT,
            status,
            proof: None,
            counterexample,
            diagnostics: if stderr.is_empty() {
                vec![]
            } else {
                vec![stderr.to_string()]
            },
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_selfcheckgpt() {
            Some(_) => {
                if detection::get_selfcheckgpt_version().is_some() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded {
                        reason: "SelfCheckGPT found but version check failed".to_string(),
                    }
                }
            }
            None => HealthStatus::Unavailable {
                reason: "SelfCheckGPT not installed. Install with: pip install selfcheckgpt"
                    .to_string(),
            },
        }
    }
}
