//! Kani Fast backend - Enhanced Kani verification with k-induction, CHC, and portfolio solving
//!
//! This backend wraps the kani-fast library to provide:
//! - Portfolio solving with multiple SAT/SMT solvers in parallel
//! - K-induction for unbounded verification
//! - CHC solving via Z3 Spacer
//! - AI-assisted invariant synthesis
//! - Beautiful counterexamples with natural language explanations
//!
//! Kani Fast dramatically improves on Kani with 10-100x faster verification.

mod config;
mod detection;
mod execution;

#[cfg(test)]
mod tests;

pub use config::KaniFastConfig;

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use config::KaniFastDetection;
use dashprove_usl::typecheck::TypedSpec;
use tracing::{debug, info, warn};

/// Kani Fast verification backend
///
/// Wraps the kani-fast CLI to provide enhanced verification capabilities:
/// - Multiple verification modes (bounded, k-induction, CHC, portfolio)
/// - AI-assisted invariant synthesis
/// - Beautiful counterexample explanations
pub struct KaniFastBackend {
    config: KaniFastConfig,
}

impl KaniFastBackend {
    /// Create a new Kani Fast backend with default configuration
    pub fn new() -> Self {
        Self {
            config: KaniFastConfig::default(),
        }
    }

    /// Create a new Kani Fast backend with custom configuration
    pub fn with_config(config: KaniFastConfig) -> Self {
        Self { config }
    }
}

impl Default for KaniFastBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for KaniFastBackend {
    fn id(&self) -> BackendId {
        BackendId::KaniFast
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = detection::detect_kani_fast(&self.config).await;
        if let KaniFastDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let project_dir = self.config.project_dir.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "KaniFast backend requires project_dir pointing to a Rust crate".to_string(),
            )
        })?;

        let manifest_path = project_dir.join("Cargo.toml");
        if !manifest_path.exists() {
            return Err(BackendError::Unavailable(format!(
                "Cargo.toml not found at {}",
                manifest_path.display()
            )));
        }

        debug!("Running kani-fast on {}", project_dir.display());

        let output = execution::run_kani_fast(&self.config, &detection, &manifest_path).await?;
        Ok(execution::parse_output(&output, spec))
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_kani_fast(&self.config).await {
            KaniFastDetection::Available { cli_path } => {
                info!("kani-fast available at {:?}", cli_path);
                HealthStatus::Healthy
            }
            KaniFastDetection::NotFound(reason) => {
                warn!("KaniFast backend unavailable: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}
