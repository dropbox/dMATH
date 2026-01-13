//! ERAN backend for neural network verification
//!
//! ERAN (ETH Robustness Analyzer for Neural Networks) uses abstract interpretation
//! with multiple domains (DeepZ, DeepPoly, RefinePoly, GPUPoly) to certify
//! neural network robustness.
//!
//! See: <https://github.com/eth-sri/eran>
//!
//! # USL to ERAN Compilation
//!
//! This backend compiles USL neural properties to ERAN command-line arguments:
//!
//! - Robustness properties: `|x - x0| <= epsilon` → `--epsilon` parameter
//! - Input bounds from comparisons → zonotope specification file
//! - Output constraints: Safety/reachability properties
//! - Type fields with "input"/"output" in name → variable dimensions
//!
//! ERAN uses abstract interpretation domains rather than SMT solving, so properties
//! are compiled to epsilon bounds and optional zonotope specifications rather than
//! VNNLIB format.

// =============================================
// Kani Proofs for ERAN Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use std::time::Duration;

    // ---- EranDomain Tests ----

    /// Verify EranDomain::default is DeepPoly
    #[kani::proof]
    fn proof_eran_domain_default_is_deeppoly() {
        let domain = EranDomain::default();
        kani::assert(
            matches!(domain, EranDomain::DeepPoly),
            "Default domain should be DeepPoly",
        );
    }

    /// Verify EranDomain::DeepZ as_str
    #[kani::proof]
    fn proof_eran_domain_deepz_str() {
        let domain = EranDomain::DeepZ;
        kani::assert(domain.as_str() == "deepzono", "DeepZ should be deepzono");
    }

    /// Verify EranDomain::DeepPoly as_str
    #[kani::proof]
    fn proof_eran_domain_deeppoly_str() {
        let domain = EranDomain::DeepPoly;
        kani::assert(domain.as_str() == "deeppoly", "DeepPoly should be deeppoly");
    }

    /// Verify EranDomain::RefinePoly as_str
    #[kani::proof]
    fn proof_eran_domain_refinepoly_str() {
        let domain = EranDomain::RefinePoly;
        kani::assert(
            domain.as_str() == "refinepoly",
            "RefinePoly should be refinepoly",
        );
    }

    /// Verify EranDomain::GpuPoly as_str
    #[kani::proof]
    fn proof_eran_domain_gpupoly_str() {
        let domain = EranDomain::GpuPoly;
        kani::assert(domain.as_str() == "gpupoly", "GpuPoly should be gpupoly");
    }

    // ---- EranConfig Default Tests ----

    /// Verify EranConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_eran_config_default_timeout() {
        let config = EranConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify EranConfig::default eran_path is None
    #[kani::proof]
    fn proof_eran_config_default_eran_path_none() {
        let config = EranConfig::default();
        kani::assert(
            config.eran_path.is_none(),
            "Default eran_path should be None",
        );
    }

    /// Verify EranConfig::default python_path is None
    #[kani::proof]
    fn proof_eran_config_default_python_path_none() {
        let config = EranConfig::default();
        kani::assert(
            config.python_path.is_none(),
            "Default python_path should be None",
        );
    }

    /// Verify EranConfig::default epsilon is 0.01
    #[kani::proof]
    fn proof_eran_config_default_epsilon() {
        let config = EranConfig::default();
        kani::assert(config.epsilon == 0.01, "Default epsilon should be 0.01");
    }

    /// Verify EranConfig::default use_gpu is false
    #[kani::proof]
    fn proof_eran_config_default_use_gpu() {
        let config = EranConfig::default();
        kani::assert(!config.use_gpu, "Default use_gpu should be false");
    }

    /// Verify EranConfig::default domain is DeepPoly
    #[kani::proof]
    fn proof_eran_config_default_domain() {
        let config = EranConfig::default();
        kani::assert(
            matches!(config.domain, EranDomain::DeepPoly),
            "Default domain should be DeepPoly",
        );
    }

    // ---- EranBackend Construction Tests ----

    /// Verify EranBackend::new uses default config
    #[kani::proof]
    fn proof_eran_backend_new_defaults() {
        let backend = EranBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify EranBackend::default equals EranBackend::new
    #[kani::proof]
    fn proof_eran_backend_default_equals_new() {
        let default_backend = EranBackend::default();
        let new_backend = EranBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify EranBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_eran_backend_with_config_timeout() {
        let config = EranConfig {
            eran_path: None,
            python_path: None,
            domain: EranDomain::DeepPoly,
            epsilon: 0.01,
            timeout: Duration::from_secs(600),
            use_gpu: false,
        };
        let backend = EranBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify EranBackend::with_config preserves custom epsilon
    #[kani::proof]
    fn proof_eran_backend_with_config_epsilon() {
        let config = EranConfig {
            eran_path: None,
            python_path: None,
            domain: EranDomain::DeepPoly,
            epsilon: 0.05,
            timeout: Duration::from_secs(300),
            use_gpu: false,
        };
        let backend = EranBackend::with_config(config);
        kani::assert(
            backend.config.epsilon == 0.05,
            "Custom epsilon should be preserved",
        );
    }

    /// Verify EranBackend::with_config preserves use_gpu
    #[kani::proof]
    fn proof_eran_backend_with_config_use_gpu() {
        let config = EranConfig {
            eran_path: None,
            python_path: None,
            domain: EranDomain::GpuPoly,
            epsilon: 0.01,
            timeout: Duration::from_secs(300),
            use_gpu: true,
        };
        let backend = EranBackend::with_config(config);
        kani::assert(backend.config.use_gpu, "Custom use_gpu should be preserved");
    }

    /// Verify EranBackend::with_config preserves domain
    #[kani::proof]
    fn proof_eran_backend_with_config_domain() {
        let config = EranConfig {
            eran_path: None,
            python_path: None,
            domain: EranDomain::DeepZ,
            epsilon: 0.01,
            timeout: Duration::from_secs(300),
            use_gpu: false,
        };
        let backend = EranBackend::with_config(config);
        kani::assert(
            matches!(backend.config.domain, EranDomain::DeepZ),
            "Custom domain should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Eran
    #[kani::proof]
    fn proof_backend_id_is_eran() {
        let backend = EranBackend::new();
        kani::assert(backend.id() == BackendId::Eran, "ID should be Eran");
    }

    /// Verify supports() includes NeuralRobustness
    #[kani::proof]
    fn proof_eran_supports_neural_robustness() {
        let backend = EranBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralRobustness),
            "Should support NeuralRobustness",
        );
    }

    /// Verify supports() returns exactly one property type
    #[kani::proof]
    fn proof_eran_supports_count() {
        let backend = EranBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Should support exactly one property type",
        );
    }

    // ---- Parsing Module Tests ----

    /// Verify parse_output detects 100% certified as Proven
    #[kani::proof]
    fn proof_parse_output_certified_100() {
        let (status, pct) = parsing::parse_output("certified: 100%", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "100% certified should be Proven",
        );
        kani::assert(pct == Some(100.0), "Percentage should be 100");
    }

    /// Verify parse_output detects 0% certified as Disproven
    #[kani::proof]
    fn proof_parse_output_certified_0() {
        let (status, pct) = parsing::parse_output("certified: 0%", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "0% certified should be Disproven",
        );
        kani::assert(pct == Some(0.0), "Percentage should be 0");
    }

    /// Verify parse_output detects partial certification
    #[kani::proof]
    fn proof_parse_output_partial() {
        let (status, pct) = parsing::parse_output("certified: 75%", "");
        kani::assert(
            matches!(status, VerificationStatus::Partial { .. }),
            "75% certified should be Partial",
        );
        kani::assert(pct == Some(75.0), "Percentage should be 75");
    }

    /// Verify parse_output detects timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let (status, _) = parsing::parse_output("timeout", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Timeout should be Unknown",
        );
    }

    /// Verify parse_array_values parses simple array
    #[kani::proof]
    fn proof_parse_array_simple() {
        let result = parsing::parse_array_values("[0.1, 0.2, 0.3]");
        kani::assert(result.is_some(), "Should parse array");
        if let Some(values) = result {
            kani::assert(values.len() == 3, "Should have 3 elements");
        }
    }

    /// Verify parse_array_values parses nested array
    #[kani::proof]
    fn proof_parse_array_nested() {
        let result = parsing::parse_array_values("[[0.1, 0.2]]");
        kani::assert(result.is_some(), "Should parse nested array");
    }

    /// Verify parse_array_values returns None for invalid
    #[kani::proof]
    fn proof_parse_array_invalid() {
        let result = parsing::parse_array_values("not an array");
        kani::assert(result.is_none(), "Should return None for invalid");
    }

    /// Verify normalize_var_name handles input format
    #[kani::proof]
    fn proof_normalize_var_input() {
        let result = parsing::normalize_var_name("input[0]");
        kani::assert(result == "x0", "input[0] should normalize to x0");
    }

    /// Verify normalize_var_name handles output format
    #[kani::proof]
    fn proof_normalize_var_output() {
        let result = parsing::normalize_var_name("output[1]");
        kani::assert(result == "y1", "output[1] should normalize to y1");
    }

    /// Verify normalize_var_name handles x prefix
    #[kani::proof]
    fn proof_normalize_var_x_prefix() {
        let result = parsing::normalize_var_name("x5");
        kani::assert(result == "x5", "x5 should stay as x5");
    }

    /// Verify normalize_var_name handles y prefix
    #[kani::proof]
    fn proof_normalize_var_y_prefix() {
        let result = parsing::normalize_var_name("y3");
        kani::assert(result == "y3", "y3 should stay as y3");
    }

    /// Verify parse_counterexample returns None for certified
    #[kani::proof]
    fn proof_parse_counterexample_certified() {
        let result = parsing::parse_counterexample("certified: 100%");
        kani::assert(result.is_none(), "Should return None for certified");
    }

    /// Verify parse_counterexample returns None for verified
    #[kani::proof]
    fn proof_parse_counterexample_verified() {
        let result = parsing::parse_counterexample("verified: all samples safe");
        kani::assert(result.is_none(), "Should return None for verified");
    }
}

mod config;
mod detection;
mod model;
mod parsing;
#[cfg(test)]
mod tests;
mod usl_analysis;
mod zonotope;

pub use config::{EranConfig, EranDomain};

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// ERAN neural network verification backend
pub struct EranBackend {
    config: EranConfig,
}

impl Default for EranBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EranBackend {
    /// Create a new ERAN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: EranConfig::default(),
        }
    }

    /// Create a new backend with custom configuration
    pub fn with_config(config: EranConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for EranBackend {
    fn id(&self) -> BackendId {
        BackendId::Eran
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::NeuralRobustness]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let (python_path, eran_path) = detection::detect_eran(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Extract epsilon from USL spec or use configured default
        let epsilon = usl_analysis::extract_epsilon_from_spec(spec, self.config.epsilon);

        // Select optimal domain based on property characteristics
        let domain = zonotope::select_domain_for_property(spec, self.config.domain);

        // Create temp directory for zonotope spec if needed
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Extract model path from USL spec or use placeholder
        let model_path = if let Some(path) = model::extract_model_path(spec) {
            PathBuf::from(path)
        } else {
            // Placeholder for when USL has neural model references
            temp_dir.path().join("model.onnx")
        };

        // Build command
        let mut cmd = Command::new(&python_path);
        cmd.current_dir(&eran_path)
            .arg(".")
            .arg("--netname")
            .arg(&model_path)
            .arg("--domain")
            .arg(domain.as_str())
            .arg("--epsilon")
            .arg(epsilon.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add zonotope specification if we have complex input bounds
        if usl_analysis::needs_zonotope_spec(spec) {
            let zonotope_spec = zonotope::generate_zonotope_spec(spec);
            let zonotope_path = temp_dir.path().join("zonotope.csv");
            std::fs::write(&zonotope_path, &zonotope_spec).map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to write zonotope spec: {}", e))
            })?;
            cmd.arg("--zonotope").arg(&zonotope_path);
            debug!("Generated zonotope spec:\n{}", zonotope_spec);
        }

        debug!(
            "Running ERAN with domain={}, epsilon={}, model={:?}",
            domain.as_str(),
            epsilon,
            model_path
        );

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("ERAN stdout: {}", stdout);
                debug!("ERAN stderr: {}", stderr);

                let (status, _pct) = parsing::parse_output(&stdout, &stderr);

                // Parse counterexample if verification failed
                let counterexample = if matches!(
                    status,
                    VerificationStatus::Disproven | VerificationStatus::Partial { .. }
                ) {
                    parsing::parse_counterexample(&stdout)
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Eran,
                    status,
                    proof: None,
                    counterexample,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute ERAN: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match detection::detect_eran(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
