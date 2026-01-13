//! Marabou backend for neural network verification
//!
//! Marabou is an SMT-based neural network verifier that supports:
//! - Feed-forward networks (ReLU, sigmoid, tanh activations)
//! - ONNX model format
//! - VNNLIB property specification format
//!
//! See: <https://github.com/NeuralNetworkVerification/Marabou>
//!
//! # USL to VNNLIB Compilation
//!
//! This backend compiles USL neural properties to VNNLIB format:
//!
//! - Input bounds from comparisons: `x >= 0 and x <= 1` → `(assert (>= X_0 0.0)) (assert (<= X_0 1.0))`
//! - Robustness properties: `|x - x0| <= epsilon` → input perturbation bounds
//! - Output constraints: `output > threshold` → negated for counterexample search
//! - Type fields with "input"/"output" in name → variable declarations
//!
//! # Module Structure
//!
//! - [`config`]: Configuration and Marabou detection
//! - [`vnnlib`]: VNNLIB generation and SMT-LIB 2 compilation
//! - [`parsing`]: Output parsing and counterexample extraction
//! - [`model`]: Model path extraction from USL specs

// =============================================
// Kani Proofs for Marabou Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use std::time::Duration;

    // ---- MarabouConfig Default Tests ----

    /// Verify MarabouConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_marabou_config_default_timeout() {
        let config = MarabouConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify MarabouConfig::default marabou_path is None
    #[kani::proof]
    fn proof_marabou_config_default_path_none() {
        let config = MarabouConfig::default();
        kani::assert(
            config.marabou_path.is_none(),
            "Default marabou_path should be None",
        );
    }

    /// Verify MarabouConfig::default num_threads is None
    #[kani::proof]
    fn proof_marabou_config_default_num_threads_none() {
        let config = MarabouConfig::default();
        kani::assert(
            config.num_threads.is_none(),
            "Default num_threads should be None",
        );
    }

    /// Verify MarabouConfig::default split_and_conquer is false
    #[kani::proof]
    fn proof_marabou_config_default_split_and_conquer() {
        let config = MarabouConfig::default();
        kani::assert(
            !config.split_and_conquer,
            "Default split_and_conquer should be false",
        );
    }

    // ---- MarabouBackend Construction Tests ----

    /// Verify MarabouBackend::new uses default config
    #[kani::proof]
    fn proof_marabou_backend_new_defaults() {
        let backend = MarabouBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify MarabouBackend::default equals MarabouBackend::new
    #[kani::proof]
    fn proof_marabou_backend_default_equals_new() {
        let default_backend = MarabouBackend::default();
        let new_backend = MarabouBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify MarabouBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_marabou_backend_with_config_timeout() {
        let config = MarabouConfig {
            marabou_path: None,
            timeout: Duration::from_secs(600),
            num_threads: None,
            split_and_conquer: false,
        };
        let backend = MarabouBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify MarabouBackend::with_config preserves split_and_conquer
    #[kani::proof]
    fn proof_marabou_backend_with_config_snc() {
        let config = MarabouConfig {
            marabou_path: None,
            timeout: Duration::from_secs(300),
            num_threads: None,
            split_and_conquer: true,
        };
        let backend = MarabouBackend::with_config(config);
        kani::assert(
            backend.config.split_and_conquer,
            "Custom split_and_conquer should be preserved",
        );
    }

    /// Verify MarabouBackend::with_config preserves num_threads
    #[kani::proof]
    fn proof_marabou_backend_with_config_threads() {
        let config = MarabouConfig {
            marabou_path: None,
            timeout: Duration::from_secs(300),
            num_threads: Some(4),
            split_and_conquer: false,
        };
        let backend = MarabouBackend::with_config(config);
        kani::assert(
            backend.config.num_threads == Some(4),
            "Custom num_threads should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Marabou
    #[kani::proof]
    fn proof_backend_id_is_marabou() {
        let backend = MarabouBackend::new();
        kani::assert(backend.id() == BackendId::Marabou, "ID should be Marabou");
    }

    /// Verify supports() includes NeuralRobustness
    #[kani::proof]
    fn proof_marabou_supports_neural_robustness() {
        let backend = MarabouBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralRobustness),
            "Should support NeuralRobustness",
        );
    }

    /// Verify supports() includes NeuralReachability
    #[kani::proof]
    fn proof_marabou_supports_neural_reachability() {
        let backend = MarabouBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::NeuralReachability),
            "Should support NeuralReachability",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_marabou_supports_count() {
        let backend = MarabouBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly two property types",
        );
    }

    // ---- VNNLIB Module Tests ----

    /// Verify to_vnnlib_ident preserves simple alphanumeric names
    #[kani::proof]
    fn proof_vnnlib_ident_simple() {
        let result = vnnlib::to_vnnlib_ident("x0");
        kani::assert(result == "X_0", "x0 should convert to X_0");
    }

    /// Verify to_vnnlib_ident handles output variables
    #[kani::proof]
    fn proof_vnnlib_ident_output() {
        let result = vnnlib::to_vnnlib_ident("y1");
        kani::assert(result == "Y_1", "y1 should convert to Y_1");
    }

    /// Verify extract_index extracts from x0
    #[kani::proof]
    fn proof_extract_index_x0() {
        let result = vnnlib::extract_index("x0");
        kani::assert(result == Some(0), "Should extract 0 from x0");
    }

    /// Verify extract_index extracts from input_5
    #[kani::proof]
    fn proof_extract_index_input5() {
        let result = vnnlib::extract_index("input_5");
        kani::assert(result == Some(5), "Should extract 5 from input_5");
    }

    /// Verify extract_index extracts from output10
    #[kani::proof]
    fn proof_extract_index_output10() {
        let result = vnnlib::extract_index("output10");
        kani::assert(result == Some(10), "Should extract 10 from output10");
    }

    /// Verify extract_index returns None for no digits
    #[kani::proof]
    fn proof_extract_index_no_digits() {
        let result = vnnlib::extract_index("input");
        kani::assert(result.is_none(), "Should return None for no digits");
    }

    // ---- Parsing Module Tests ----

    /// Verify parse_output detects UNSAT as Proven
    #[kani::proof]
    fn proof_parse_output_unsat_proven() {
        let status = parsing::parse_output("Result: unsat", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "UNSAT should be Proven",
        );
    }

    /// Verify parse_output detects SAT as Disproven
    #[kani::proof]
    fn proof_parse_output_sat_disproven() {
        let status = parsing::parse_output("sat\nx0 = 0.5", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "SAT should be Disproven",
        );
    }

    /// Verify parse_output detects TIMEOUT
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let status = parsing::parse_output("TIMEOUT", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "TIMEOUT should be Unknown",
        );
    }

    /// Verify parse_output detects ERROR
    #[kani::proof]
    fn proof_parse_output_error() {
        let status = parsing::parse_output("", "ERROR: something went wrong");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "ERROR should be Unknown",
        );
    }

    /// Verify parse_value parses float
    #[kani::proof]
    fn proof_parse_value_float() {
        let result = parsing::parse_value("0.5");
        kani::assert(result.is_some(), "Should parse 0.5 as float");
    }

    /// Verify parse_value parses integer
    #[kani::proof]
    fn proof_parse_value_int() {
        let result = parsing::parse_value("42");
        kani::assert(result.is_some(), "Should parse 42");
    }

    /// Verify parse_value handles infinity
    #[kani::proof]
    fn proof_parse_value_inf() {
        let result = parsing::parse_value("inf");
        kani::assert(result.is_some(), "Should parse inf");
    }

    /// Verify parse_value handles negative infinity
    #[kani::proof]
    fn proof_parse_value_neg_inf() {
        let result = parsing::parse_value("-inf");
        kani::assert(result.is_some(), "Should parse -inf");
    }

    /// Verify parse_value handles nan
    #[kani::proof]
    fn proof_parse_value_nan() {
        let result = parsing::parse_value("nan");
        kani::assert(result.is_some(), "Should parse nan");
    }

    /// Verify parse_assignment_line parses equals format
    #[kani::proof]
    fn proof_parse_assignment_equals() {
        let result = parsing::parse_assignment_line("x0 = 0.5", "=");
        kani::assert(result.is_some(), "Should parse x0 = 0.5");
        if let Some((name, _)) = result {
            kani::assert(name == "x0", "Variable name should be x0");
        }
    }

    /// Verify parse_assignment_line parses colon format
    #[kani::proof]
    fn proof_parse_assignment_colon() {
        let result = parsing::parse_assignment_line("x0 : 0.5", ":");
        kani::assert(result.is_some(), "Should parse x0 : 0.5");
    }

    /// Verify parse_labeled_variable parses Input format
    #[kani::proof]
    fn proof_parse_labeled_input() {
        let result = parsing::parse_labeled_variable("Input 0 = 0.5");
        kani::assert(result.is_some(), "Should parse Input 0 = 0.5");
        if let Some((name, _)) = result {
            kani::assert(name == "x0", "Should convert to x0");
        }
    }

    /// Verify parse_labeled_variable parses Output format
    #[kani::proof]
    fn proof_parse_labeled_output() {
        let result = parsing::parse_labeled_variable("Output 1 = 0.8");
        kani::assert(result.is_some(), "Should parse Output 1 = 0.8");
        if let Some((name, _)) = result {
            kani::assert(name == "y1", "Should convert to y1");
        }
    }

    /// Verify parse_labeled_variable handles bracket format
    #[kani::proof]
    fn proof_parse_labeled_bracket() {
        let result = parsing::parse_labeled_variable("Input[2] = 0.3");
        kani::assert(result.is_some(), "Should parse Input[2] = 0.3");
        if let Some((name, _)) = result {
            kani::assert(name == "x2", "Should convert to x2");
        }
    }

    // ---- Model Path Tests ----

    /// Verify parse_counterexample returns None for UNSAT
    #[kani::proof]
    fn proof_parse_counterexample_unsat() {
        let result = parsing::parse_counterexample("unsat");
        kani::assert(result.is_none(), "Should return None for unsat");
    }
}

pub mod config;
pub mod model;
pub mod parsing;
pub mod vnnlib;

#[cfg(test)]
mod tests;

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

pub use config::MarabouConfig;

/// Marabou neural network verification backend
pub struct MarabouBackend {
    config: MarabouConfig,
}

impl Default for MarabouBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MarabouBackend {
    /// Create a new Marabou backend with default configuration
    pub fn new() -> Self {
        Self {
            config: MarabouConfig::default(),
        }
    }

    /// Create a new Marabou backend with custom configuration
    pub fn with_config(config: MarabouConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl VerificationBackend for MarabouBackend {
    fn id(&self) -> BackendId {
        BackendId::Marabou
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::NeuralRobustness,
            PropertyType::NeuralReachability,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect Marabou
        let marabou_path = config::detect_marabou(&self.config)
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for files
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate VNNLIB property
        let vnnlib_content = vnnlib::generate_vnnlib(spec)?;
        let vnnlib_path = temp_dir.path().join("property.vnnlib");
        std::fs::write(&vnnlib_path, &vnnlib_content).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write VNNLIB: {}", e))
        })?;

        // Extract model path from USL spec or use placeholder
        let model_path = if let Some(path) = model::extract_model_path(spec) {
            PathBuf::from(path)
        } else {
            // Placeholder for when USL has neural model references
            temp_dir.path().join("model.onnx")
        };

        debug!("Using model path: {:?}", model_path);

        // Build command
        let mut cmd = Command::new(&marabou_path);
        cmd.arg("--input")
            .arg(&model_path)
            .arg("--property")
            .arg(&vnnlib_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(threads) = self.config.num_threads {
            cmd.arg("--num-workers").arg(threads.to_string());
        }

        if self.config.split_and_conquer {
            cmd.arg("--snc");
        }

        // Run with timeout
        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Marabou stdout: {}", stdout);
                debug!("Marabou stderr: {}", stderr);

                let status = parsing::parse_output(&stdout, &stderr);

                // Parse counterexample if SAT (property violated)
                let counterexample = if matches!(status, VerificationStatus::Disproven) {
                    parsing::parse_counterexample(&stdout)
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Marabou,
                    status,
                    proof: None,
                    counterexample,
                    diagnostics: vec![],
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Marabou: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match config::detect_marabou(&self.config).await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}
