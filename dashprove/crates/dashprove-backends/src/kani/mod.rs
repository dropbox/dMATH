//! Kani backend for Rust contract verification
//!
//! This backend generates Kani proof harnesses from USL contracts and executes
//! them via `cargo kani`. It expects a Rust crate containing the implementation
//! referenced by the contracts.

mod config;
mod detection;
mod execution;
mod parsing;
mod project;

#[cfg(test)]
mod tests;

pub use config::KaniConfig;

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
};
use async_trait::async_trait;
use config::KaniDetection;
use dashprove_usl::typecheck::TypedSpec;
use tracing::{info, warn};

/// Kani verification backend using `cargo kani`
pub struct KaniBackend {
    config: KaniConfig,
}

impl KaniBackend {
    /// Create a new Kani backend with default configuration
    pub fn new() -> Self {
        Self {
            config: KaniConfig::default(),
        }
    }

    /// Create a new Kani backend with custom configuration
    pub fn with_config(config: KaniConfig) -> Self {
        Self { config }
    }

    /// Verify inline Rust code against USL contracts
    ///
    /// This method creates a self-contained project with the provided Rust code
    /// and generates Kani proof harnesses from the USL specification.
    ///
    /// # Arguments
    /// * `code` - Rust source code containing function implementations
    /// * `spec` - Typed USL specification with contracts to verify
    ///
    /// # Returns
    /// * `BackendResult` indicating verification success, failure, or error
    pub async fn verify_code(
        &self,
        code: &str,
        spec: &TypedSpec,
    ) -> Result<BackendResult, BackendError> {
        let detection = detection::detect_kani(&self.config).await;
        if let KaniDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let (_temp_dir, manifest_path) = project::write_inline_project(code, spec).await?;

        let output = execution::run_kani(&self.config, &detection, &manifest_path).await?;
        Ok(parsing::parse_output(&output))
    }
}

impl Default for KaniBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for KaniBackend {
    fn id(&self) -> BackendId {
        BackendId::Kani
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Contract]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = detection::detect_kani(&self.config).await;
        if let KaniDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let project_dir = self.config.project_dir.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Kani backend requires project_dir pointing to a Rust crate".to_string(),
            )
        })?;

        let manifest_path = project_dir.join("Cargo.toml");
        if !manifest_path.exists() {
            return Err(BackendError::Unavailable(format!(
                "Cargo.toml not found at {}",
                manifest_path.display()
            )));
        }

        let package_name = project::read_package_name(&manifest_path)?;
        let crate_ident = package_name.replace('-', "_");

        let (_temp_dir, harness_manifest) =
            project::write_harness_project(spec, &project_dir, &package_name, &crate_ident).await?;

        let output = execution::run_kani(&self.config, &detection, &harness_manifest).await?;
        Ok(parsing::parse_output(&output))
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_kani(&self.config).await {
            KaniDetection::Available { cargo_path } => {
                info!("cargo-kani available at {:?}", cargo_path);
                HealthStatus::Healthy
            }
            KaniDetection::NotFound(reason) => {
                warn!("Kani backend unavailable: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}
