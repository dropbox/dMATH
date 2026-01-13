//! Tests for TensorRT backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_tensorrt_backend_id() {
    let backend = TensorRTBackend::new();
    assert_eq!(backend.id(), BackendId::TensorRT);
}

#[test]
fn test_tensorrt_supports_optimization() {
    let backend = TensorRTBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelOptimization));
}

#[test]
fn test_tensorrt_config_default() {
    let config = TensorRTConfig::default();
    assert_eq!(config.precision, PrecisionMode::FP32);
    assert_eq!(config.optimization_profile, OptimizationProfile::Latency);
    assert_eq!(config.max_batch_size, 1);
    assert!(!config.strict_types);
}

#[test]
fn test_tensorrt_config_fp16() {
    let config = TensorRTConfig::fp16();
    assert_eq!(config.precision, PrecisionMode::FP16);
}

#[test]
fn test_tensorrt_config_int8() {
    let config = TensorRTConfig::int8();
    assert_eq!(config.precision, PrecisionMode::INT8);
}

#[test]
fn test_tensorrt_config_high_throughput() {
    let config = TensorRTConfig::high_throughput();
    assert_eq!(config.precision, PrecisionMode::FP16);
    assert_eq!(config.optimization_profile, OptimizationProfile::Throughput);
    assert_eq!(config.max_batch_size, 32);
}

#[test]
fn test_precision_mode_as_str() {
    assert_eq!(PrecisionMode::FP32.as_str(), "FP32");
    assert_eq!(PrecisionMode::FP16.as_str(), "FP16");
    assert_eq!(PrecisionMode::INT8.as_str(), "INT8");
    assert_eq!(PrecisionMode::TF32.as_str(), "TF32");
}

#[test]
fn test_optimization_profile_as_str() {
    assert_eq!(OptimizationProfile::Latency.as_str(), "latency");
    assert_eq!(OptimizationProfile::Throughput.as_str(), "throughput");
    assert_eq!(OptimizationProfile::Balanced.as_str(), "balanced");
}

#[test]
fn test_generate_tensorrt_script() {
    let config = TensorRTConfig::default();
    let spec = create_test_spec();
    let script = script::generate_tensorrt_script(&spec, &config).unwrap();

    assert!(script.contains("import tensorrt"));
    assert!(script.contains("TRT_RESULT_START"));
    assert!(script.contains("TRT_RESULT_END"));
}

#[test]
fn test_parse_tensorrt_output_verified() {
    let stdout = r#"
TRT_INFO: Using synthetic test engine
TRT_RESULT_START
{
  "model_path": "model.onnx",
  "precision": "FP32",
  "optimization_profile": "latency",
  "max_workspace_bytes": 1073741824,
  "max_batch_size": 1,
  "input_shape": [1, 10],
  "output_shape": [1, 10],
  "warmup_iterations": 10,
  "benchmark_iterations": 100,
  "cuda_available": false,
  "consistent_outputs": true,
  "max_output_diff": 0.0,
  "latency_ms": {
    "mean": 0.3,
    "std": 0.05,
    "p50": 0.3,
    "p95": 0.4,
    "p99": 0.45
  },
  "throughput_ips": 3333.3
}
TRT_RESULT_END
TRT_SUMMARY: Mean latency: 0.300ms
TRT_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_tensorrt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_tensorrt_output_not_verified() {
    let stdout = r#"
TRT_RESULT_START
{
  "model_path": "model.onnx",
  "precision": "FP16",
  "optimization_profile": "throughput",
  "max_workspace_bytes": 1073741824,
  "max_batch_size": 32,
  "input_shape": [32, 10],
  "output_shape": [32, 10],
  "warmup_iterations": 10,
  "benchmark_iterations": 100,
  "cuda_available": true,
  "consistent_outputs": false,
  "max_output_diff": 0.01,
  "latency_ms": {
    "mean": 0.5,
    "std": 0.1,
    "p50": 0.5,
    "p95": 0.7,
    "p99": 0.8
  },
  "throughput_ips": 2000.0
}
TRT_RESULT_END
TRT_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_tensorrt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("max_output_diff"));
}

#[test]
fn test_parse_tensorrt_output_error() {
    let stdout = "TRT_ERROR: Failed to create engine";
    let (status, _) = script::parse_tensorrt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_tensorrt_health_check() {
    let backend = TensorRTBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_tensorrt_verify_returns_result_or_unavailable() {
    let backend = TensorRTBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::TensorRT),
        Err(BackendError::Unavailable(_)) => {}
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

fn create_test_spec() -> dashprove_usl::typecheck::TypedSpec {
    use dashprove_usl::ast::Spec;
    use dashprove_usl::typecheck::TypedSpec;
    use std::collections::HashMap;

    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}
