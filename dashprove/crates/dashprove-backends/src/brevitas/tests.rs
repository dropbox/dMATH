//! Tests for Brevitas backend

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
fn test_weight_bit_width_values() {
    assert_eq!(WeightBitWidth::Binary.as_int(), Some(1));
    assert_eq!(WeightBitWidth::Ternary.as_int(), Some(2));
    assert_eq!(WeightBitWidth::Bits2.as_int(), Some(2));
    assert_eq!(WeightBitWidth::Bits4.as_int(), Some(4));
    assert_eq!(WeightBitWidth::Bits8.as_int(), Some(8));
}

#[test]
fn test_weight_bit_width_strings() {
    assert_eq!(WeightBitWidth::Binary.as_str(), "binary");
    assert_eq!(WeightBitWidth::Ternary.as_str(), "ternary");
    assert_eq!(WeightBitWidth::Bits2.as_str(), "2bit");
    assert_eq!(WeightBitWidth::Bits4.as_str(), "4bit");
    assert_eq!(WeightBitWidth::Bits8.as_str(), "8bit");
}

#[test]
fn test_activation_bit_width_values() {
    assert_eq!(ActivationBitWidth::Binary.as_int(), Some(1));
    assert_eq!(ActivationBitWidth::Bits4.as_int(), Some(4));
    assert_eq!(ActivationBitWidth::Bits8.as_int(), Some(8));
    assert_eq!(ActivationBitWidth::Full.as_int(), None);
}

#[test]
fn test_activation_bit_width_strings() {
    assert_eq!(ActivationBitWidth::Binary.as_str(), "binary");
    assert_eq!(ActivationBitWidth::Bits4.as_str(), "4bit");
    assert_eq!(ActivationBitWidth::Bits8.as_str(), "8bit");
    assert_eq!(ActivationBitWidth::Full.as_str(), "full");
}

#[test]
fn test_scaling_mode_strings() {
    assert_eq!(ScalingMode::PerTensor.as_str(), "per_tensor");
    assert_eq!(ScalingMode::PerChannel.as_str(), "per_channel");
    assert_eq!(ScalingMode::PerGroup.as_str(), "per_group");
}

#[test]
fn test_quant_method_strings() {
    assert_eq!(QuantMethod::Symmetric.as_str(), "symmetric");
    assert_eq!(QuantMethod::Asymmetric.as_str(), "asymmetric");
    assert_eq!(QuantMethod::PowerOfTwo.as_str(), "power_of_two");
}

#[test]
fn test_export_format_strings() {
    assert_eq!(ExportFormat::PyTorch.as_str(), "pytorch");
    assert_eq!(ExportFormat::ONNX.as_str(), "onnx");
    assert_eq!(ExportFormat::FINN.as_str(), "finn");
    assert_eq!(ExportFormat::QONNX.as_str(), "qonnx");
}

#[test]
fn test_default_config() {
    let config = BrevitasConfig::default();
    assert_eq!(config.weight_bit_width, WeightBitWidth::Bits8);
    assert_eq!(config.activation_bit_width, ActivationBitWidth::Bits8);
    assert_eq!(config.scaling_mode, ScalingMode::PerTensor);
    assert_eq!(config.quant_method, QuantMethod::Symmetric);
    assert_eq!(config.export_format, ExportFormat::PyTorch);
    assert!(config.group_size.is_none());
    assert_eq!(config.calibration_samples, 100);
}

#[test]
fn test_int4_config() {
    let config = BrevitasConfig::int4();
    assert_eq!(config.weight_bit_width, WeightBitWidth::Bits4);
    assert_eq!(config.activation_bit_width, ActivationBitWidth::Bits8);
    assert_eq!(config.scaling_mode, ScalingMode::PerChannel);
}

#[test]
fn test_binary_config() {
    let config = BrevitasConfig::binary();
    assert_eq!(config.weight_bit_width, WeightBitWidth::Binary);
    assert_eq!(config.activation_bit_width, ActivationBitWidth::Binary);
}

#[test]
fn test_ternary_config() {
    let config = BrevitasConfig::ternary();
    assert_eq!(config.weight_bit_width, WeightBitWidth::Ternary);
    assert_eq!(config.activation_bit_width, ActivationBitWidth::Full);
}

#[test]
fn test_finn_config() {
    let config = BrevitasConfig::finn();
    assert_eq!(config.weight_bit_width, WeightBitWidth::Bits4);
    assert_eq!(config.activation_bit_width, ActivationBitWidth::Bits4);
    assert_eq!(config.export_format, ExportFormat::FINN);
}

#[test]
fn test_grouped_config() {
    let config = BrevitasConfig::grouped(128);
    assert_eq!(config.weight_bit_width, WeightBitWidth::Bits4);
    assert_eq!(config.scaling_mode, ScalingMode::PerGroup);
    assert_eq!(config.group_size, Some(128));
}

#[test]
fn test_backend_id() {
    let backend = BrevitasBackend::new();
    assert_eq!(backend.id(), BackendId::Brevitas);
}

#[test]
fn test_supports_model_compression() {
    let backend = BrevitasBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelCompression));
}

#[test]
fn test_script_generation() {
    let config = BrevitasConfig::default();
    let spec = create_test_spec();
    let script = script::generate_brevitas_script(&spec, &config).unwrap();

    assert!(script.contains("import brevitas"));
    assert!(script.contains("BREVITAS_RESULT_START"));
    assert!(script.contains("BREVITAS_RESULT_END"));
    assert!(script.contains("BREVITAS_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
BREVITAS_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.01,
    "output_mse": 0.001,
    "compression_ratio": 4.0
}
BREVITAS_RESULT_END
BREVITAS_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_brevitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
BREVITAS_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.2,
    "output_mse": 0.05,
    "compression_ratio": 4.0
}
BREVITAS_RESULT_END
BREVITAS_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_brevitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
BREVITAS_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.8,
    "output_mse": 0.5,
    "compression_ratio": 4.0
}
BREVITAS_RESULT_END
BREVITAS_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_brevitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "BREVITAS_ERROR: Quantization failed";
    let (status, _) = script::parse_brevitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_brevitas_health_check() {
    let backend = BrevitasBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Brevitas is available"),
        HealthStatus::Unavailable { reason } => println!("Brevitas unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Brevitas degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_brevitas_verify_returns_result_or_unavailable() {
    let backend = BrevitasBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Brevitas);
            println!("Brevitas verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Brevitas unavailable: {}", reason);
        }
        Err(e) => {
            println!("Brevitas error (expected if not installed): {}", e);
        }
    }
}
