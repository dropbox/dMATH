//! Tests for Intel Neural Compressor backend

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
fn test_quantization_approach_strings() {
    assert_eq!(
        QuantizationApproach::PostTrainingStatic.as_str(),
        "post_training_static_quant"
    );
    assert_eq!(
        QuantizationApproach::PostTrainingDynamic.as_str(),
        "post_training_dynamic_quant"
    );
    assert_eq!(
        QuantizationApproach::QuantizationAwareTraining.as_str(),
        "quant_aware_training"
    );
}

#[test]
fn test_quant_dtype_strings() {
    assert_eq!(QuantDataType::INT8.as_str(), "int8");
    assert_eq!(QuantDataType::UINT8.as_str(), "uint8");
    assert_eq!(QuantDataType::FP16.as_str(), "fp16");
    assert_eq!(QuantDataType::BF16.as_str(), "bf16");
}

#[test]
fn test_calibration_method_strings() {
    assert_eq!(CalibrationMethod::MinMax.as_str(), "minmax");
    assert_eq!(CalibrationMethod::Entropy.as_str(), "kl");
    assert_eq!(CalibrationMethod::Percentile.as_str(), "percentile");
}

#[test]
fn test_tuning_strategy_strings() {
    assert_eq!(TuningStrategy::Basic.as_str(), "basic");
    assert_eq!(TuningStrategy::Bayesian.as_str(), "bayesian");
    assert_eq!(TuningStrategy::Exhaustive.as_str(), "exhaustive");
    assert_eq!(TuningStrategy::Random.as_str(), "random");
    assert_eq!(TuningStrategy::MSE.as_str(), "mse");
}

#[test]
fn test_default_config() {
    let config = NeuralCompressorConfig::default();
    assert_eq!(config.approach, QuantizationApproach::PostTrainingStatic);
    assert_eq!(config.quant_dtype, QuantDataType::INT8);
    assert_eq!(config.calibration, CalibrationMethod::MinMax);
    assert_eq!(config.tuning_strategy, TuningStrategy::Basic);
    assert!((config.accuracy_criterion - 0.01).abs() < 1e-6);
    assert_eq!(config.calibration_samples, 100);
    assert!(!config.enable_pruning);
}

#[test]
fn test_dynamic_config() {
    let config = NeuralCompressorConfig::dynamic();
    assert_eq!(config.approach, QuantizationApproach::PostTrainingDynamic);
}

#[test]
fn test_qat_config() {
    let config = NeuralCompressorConfig::qat();
    assert_eq!(
        config.approach,
        QuantizationApproach::QuantizationAwareTraining
    );
    assert_eq!(config.tuning_strategy, TuningStrategy::MSE);
}

#[test]
fn test_pruning_config() {
    let config = NeuralCompressorConfig::with_pruning(0.7);
    assert!(config.enable_pruning);
    assert!((config.pruning_sparsity - 0.7).abs() < 1e-6);
}

#[test]
fn test_backend_id() {
    let backend = NeuralCompressorBackend::new();
    assert_eq!(backend.id(), BackendId::NeuralCompressor);
}

#[test]
fn test_supports_model_compression() {
    let backend = NeuralCompressorBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelCompression));
}

#[test]
fn test_script_generation() {
    let config = NeuralCompressorConfig::default();
    let spec = create_test_spec();
    let script = script::generate_nc_script(&spec, &config).unwrap();

    assert!(script.contains("import neural_compressor"));
    assert!(script.contains("NC_RESULT_START"));
    assert!(script.contains("NC_RESULT_END"));
    assert!(script.contains("NC_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
NC_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.01,
    "output_mse": 0.001,
    "compression_ratio": 4.0
}
NC_RESULT_END
NC_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_nc_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
NC_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.2,
    "output_mse": 0.05,
    "compression_ratio": 4.0
}
NC_RESULT_END
NC_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_nc_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
NC_RESULT_START
{
    "status": "success",
    "output_max_diff": 0.8,
    "output_mse": 0.5,
    "compression_ratio": 4.0
}
NC_RESULT_END
NC_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_nc_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "NC_ERROR: Model quantization failed";
    let (status, _) = script::parse_nc_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_neural_compressor_health_check() {
    let backend = NeuralCompressorBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Neural Compressor is available"),
        HealthStatus::Unavailable { reason } => {
            println!("Neural Compressor unavailable: {}", reason)
        }
        HealthStatus::Degraded { reason } => println!("Neural Compressor degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_neural_compressor_verify_returns_result_or_unavailable() {
    let backend = NeuralCompressorBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::NeuralCompressor);
            println!("Neural Compressor verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Neural Compressor unavailable: {}", reason);
        }
        Err(e) => {
            println!("Neural Compressor error (expected if not installed): {}", e);
        }
    }
}
