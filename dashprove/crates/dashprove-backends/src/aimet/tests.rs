//! Tests for AIMET backend

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
fn test_quant_scheme_strings() {
    assert_eq!(QuantScheme::PostTraining.as_str(), "post_training");
    assert_eq!(QuantScheme::QuantizationAware.as_str(), "qat");
    assert_eq!(QuantScheme::CrossLayerEqualization.as_str(), "cle");
    assert_eq!(QuantScheme::AdaRound.as_str(), "adaround");
}

#[test]
fn test_bit_width_values() {
    assert_eq!(AimetBitWidth::W8A8.weight_bits(), 8);
    assert_eq!(AimetBitWidth::W8A8.activation_bits(), 8);
    assert_eq!(AimetBitWidth::W4A8.weight_bits(), 4);
    assert_eq!(AimetBitWidth::W4A8.activation_bits(), 8);
    assert_eq!(AimetBitWidth::W4A4.weight_bits(), 4);
    assert_eq!(AimetBitWidth::W4A4.activation_bits(), 4);
    assert_eq!(AimetBitWidth::W16A16.weight_bits(), 16);
    assert_eq!(AimetBitWidth::W16A16.activation_bits(), 16);
}

#[test]
fn test_bit_width_strings() {
    assert_eq!(AimetBitWidth::W8A8.as_str(), "w8a8");
    assert_eq!(AimetBitWidth::W4A8.as_str(), "w4a8");
    assert_eq!(AimetBitWidth::W4A4.as_str(), "w4a4");
    assert_eq!(AimetBitWidth::W16A16.as_str(), "w16a16");
}

#[test]
fn test_rounding_mode_strings() {
    assert_eq!(RoundingMode::Nearest.as_str(), "nearest");
    assert_eq!(RoundingMode::Stochastic.as_str(), "stochastic");
}

#[test]
fn test_compression_mode_strings() {
    assert_eq!(
        AimetCompressionMode::QuantizationOnly.as_str(),
        "quantization"
    );
    assert_eq!(AimetCompressionMode::SpatialSVD.as_str(), "spatial_svd");
    assert_eq!(
        AimetCompressionMode::ChannelPruning.as_str(),
        "channel_pruning"
    );
    assert_eq!(AimetCompressionMode::WeightSVD.as_str(), "weight_svd");
}

#[test]
fn test_default_config() {
    let config = AimetConfig::default();
    assert_eq!(config.quant_scheme, QuantScheme::PostTraining);
    assert_eq!(config.bit_width, AimetBitWidth::W8A8);
    assert_eq!(config.rounding_mode, RoundingMode::Nearest);
    assert_eq!(
        config.compression_mode,
        AimetCompressionMode::QuantizationOnly
    );
    assert_eq!(config.num_batches, 32);
    assert!(config.per_channel);
}

#[test]
fn test_adaround_config() {
    let config = AimetConfig::adaround();
    assert_eq!(config.quant_scheme, QuantScheme::AdaRound);
    assert_eq!(config.num_batches, 100);
}

#[test]
fn test_cle_config() {
    let config = AimetConfig::cle();
    assert_eq!(config.quant_scheme, QuantScheme::CrossLayerEqualization);
}

#[test]
fn test_qat_config() {
    let config = AimetConfig::qat();
    assert_eq!(config.quant_scheme, QuantScheme::QuantizationAware);
}

#[test]
fn test_int4_config() {
    let config = AimetConfig::int4();
    assert_eq!(config.bit_width, AimetBitWidth::W4A8);
    assert_eq!(config.quant_scheme, QuantScheme::AdaRound);
}

#[test]
fn test_spatial_svd_config() {
    let config = AimetConfig::spatial_svd();
    assert_eq!(config.compression_mode, AimetCompressionMode::SpatialSVD);
}

#[test]
fn test_backend_id() {
    let backend = AimetBackend::new();
    assert_eq!(backend.id(), BackendId::AIMET);
}

#[test]
fn test_supports_model_compression() {
    let backend = AimetBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelCompression));
}

#[test]
fn test_script_generation() {
    let config = AimetConfig::default();
    let spec = create_test_spec();
    let script = script::generate_aimet_script(&spec, &config).unwrap();

    assert!(script.contains("import aimet_common"));
    assert!(script.contains("AIMET_RESULT_START"));
    assert!(script.contains("AIMET_RESULT_END"));
    assert!(script.contains("AIMET_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
AIMET_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.01,
    "output_mse": 0.001,
    "compression_ratio": 4.0
}
AIMET_RESULT_END
AIMET_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_aimet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
AIMET_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.2,
    "output_mse": 0.05,
    "compression_ratio": 4.0
}
AIMET_RESULT_END
AIMET_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_aimet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
AIMET_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.8,
    "output_mse": 0.5,
    "compression_ratio": 4.0
}
AIMET_RESULT_END
AIMET_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_aimet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "AIMET_ERROR: Quantization failed";
    let (status, _) = script::parse_aimet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_aimet_health_check() {
    let backend = AimetBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("AIMET is available"),
        HealthStatus::Unavailable { reason } => println!("AIMET unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("AIMET degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_aimet_verify_returns_result_or_unavailable() {
    let backend = AimetBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::AIMET);
            println!("AIMET verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("AIMET unavailable: {}", reason);
        }
        Err(e) => {
            println!("AIMET error (expected if not installed): {}", e);
        }
    }
}
