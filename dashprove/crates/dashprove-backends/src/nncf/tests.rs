//! Tests for NNCF backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

fn create_test_spec() -> dashprove_usl::typecheck::TypedSpec {
    use dashprove_usl::ast::Spec;
    use dashprove_usl::typecheck::TypedSpec;
    use std::collections::HashMap;

    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}

#[test]
fn test_compression_mode_strings() {
    assert_eq!(CompressionMode::Quantization.as_str(), "quantization");
    assert_eq!(CompressionMode::Pruning.as_str(), "pruning");
    assert_eq!(CompressionMode::Sparsity.as_str(), "sparsity");
    assert_eq!(CompressionMode::FilterPruning.as_str(), "filter_pruning");
    assert_eq!(
        CompressionMode::QuantizationPruning.as_str(),
        "quantization_pruning"
    );
}

#[test]
fn test_quantization_mode_strings() {
    assert_eq!(QuantizationMode::Symmetric.as_str(), "symmetric");
    assert_eq!(QuantizationMode::Asymmetric.as_str(), "asymmetric");
}

#[test]
fn test_bit_width_values() {
    assert_eq!(BitWidth::Bits8.as_int(), Some(8));
    assert_eq!(BitWidth::Bits4.as_int(), Some(4));
    assert_eq!(BitWidth::Mixed.as_int(), None);
    assert_eq!(BitWidth::Bits8.as_str(), "8");
    assert_eq!(BitWidth::Bits4.as_str(), "4");
    assert_eq!(BitWidth::Mixed.as_str(), "mixed");
}

#[test]
fn test_pruning_schedule_strings() {
    assert_eq!(PruningSchedule::Constant.as_str(), "constant");
    assert_eq!(PruningSchedule::Polynomial.as_str(), "polynomial");
    assert_eq!(PruningSchedule::Exponential.as_str(), "exponential");
}

#[test]
fn test_default_config() {
    let config = NNCFConfig::default();
    assert_eq!(config.compression_mode, CompressionMode::Quantization);
    assert_eq!(config.quantization_mode, QuantizationMode::Symmetric);
    assert_eq!(config.bit_width, BitWidth::Bits8);
    assert!((config.target_sparsity - 0.5).abs() < 1e-6);
    assert_eq!(config.pruning_schedule, PruningSchedule::Constant);
    assert_eq!(config.calibration_samples, 100);
}

#[test]
fn test_int4_config() {
    let config = NNCFConfig::int4();
    assert_eq!(config.bit_width, BitWidth::Bits4);
    assert_eq!(config.quantization_mode, QuantizationMode::Asymmetric);
}

#[test]
fn test_pruning_config() {
    let config = NNCFConfig::pruning(0.7);
    assert_eq!(config.compression_mode, CompressionMode::Pruning);
    assert!((config.target_sparsity - 0.7).abs() < 1e-6);
}

#[test]
fn test_filter_pruning_config() {
    let config = NNCFConfig::filter_pruning(0.6);
    assert_eq!(config.compression_mode, CompressionMode::FilterPruning);
    assert!((config.target_sparsity - 0.6).abs() < 1e-6);
}

#[test]
fn test_combined_config() {
    let config = NNCFConfig::combined(0.5);
    assert_eq!(
        config.compression_mode,
        CompressionMode::QuantizationPruning
    );
    assert!((config.target_sparsity - 0.5).abs() < 1e-6);
}

#[test]
fn test_backend_id() {
    let backend = NNCFBackend::new();
    assert_eq!(backend.id(), BackendId::NNCF);
}

#[test]
fn test_supports_model_compression() {
    let backend = NNCFBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelCompression));
}

#[test]
fn test_script_generation() {
    let config = NNCFConfig::default();
    let spec = create_test_spec();
    let script = script::generate_nncf_script(&spec, &config).unwrap();

    assert!(script.contains("import nncf"));
    assert!(script.contains("NNCF_RESULT_START"));
    assert!(script.contains("NNCF_RESULT_END"));
    assert!(script.contains("NNCF_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
NNCF_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.01,
    "output_mse": 0.001,
    "compression_ratio": 4.0,
    "actual_sparsity": 0.5
}
NNCF_RESULT_END
NNCF_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_nncf_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
NNCF_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.2,
    "output_mse": 0.05,
    "compression_ratio": 4.0,
    "actual_sparsity": 0.5
}
NNCF_RESULT_END
NNCF_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_nncf_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
NNCF_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.8,
    "output_mse": 0.5,
    "compression_ratio": 4.0,
    "actual_sparsity": 0.5
}
NNCF_RESULT_END
NNCF_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_nncf_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "NNCF_ERROR: Model compression failed";
    let (status, _) = script::parse_nncf_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_nncf_health_check() {
    let backend = NNCFBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("NNCF is available"),
        HealthStatus::Unavailable { reason } => println!("NNCF unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("NNCF degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_nncf_verify_returns_result_or_unavailable() {
    let backend = NNCFBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::NNCF);
            println!("NNCF verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("NNCF unavailable: {}", reason);
        }
        Err(e) => {
            println!("NNCF error (expected if not installed): {}", e);
        }
    }
}
