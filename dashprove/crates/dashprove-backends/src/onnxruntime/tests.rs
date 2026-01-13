//! Tests for ONNX Runtime backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_onnxruntime_backend_id() {
    let backend = OnnxRuntimeBackend::new();
    assert_eq!(backend.id(), BackendId::ONNXRuntime);
}

#[test]
fn test_onnxruntime_supports_optimization() {
    let backend = OnnxRuntimeBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelOptimization));
}

#[test]
fn test_onnxruntime_config_default() {
    let config = OnnxRuntimeConfig::default();
    assert_eq!(config.execution_provider, ExecutionProvider::CPU);
    assert_eq!(config.optimization_level, GraphOptimizationLevel::Extended);
    assert!(config.enable_memory_pattern);
    assert_eq!(config.warmup_iterations, 5);
    assert_eq!(config.benchmark_iterations, 100);
}

#[test]
fn test_onnxruntime_config_cuda() {
    let config = OnnxRuntimeConfig::cuda();
    assert_eq!(config.execution_provider, ExecutionProvider::CUDA);
}

#[test]
fn test_onnxruntime_config_tensorrt() {
    let config = OnnxRuntimeConfig::tensorrt();
    assert_eq!(config.execution_provider, ExecutionProvider::TensorRT);
    assert_eq!(config.optimization_level, GraphOptimizationLevel::All);
}

#[test]
fn test_execution_provider_as_str() {
    assert_eq!(ExecutionProvider::CPU.as_str(), "CPUExecutionProvider");
    assert_eq!(ExecutionProvider::CUDA.as_str(), "CUDAExecutionProvider");
    assert_eq!(
        ExecutionProvider::TensorRT.as_str(),
        "TensorRTExecutionProvider"
    );
    assert_eq!(
        ExecutionProvider::OpenVINO.as_str(),
        "OpenVINOExecutionProvider"
    );
    assert_eq!(ExecutionProvider::DirectML.as_str(), "DmlExecutionProvider");
    assert_eq!(
        ExecutionProvider::CoreML.as_str(),
        "CoreMLExecutionProvider"
    );
}

#[test]
fn test_graph_optimization_level() {
    assert_eq!(GraphOptimizationLevel::Disabled.as_ort_level(), 0);
    assert_eq!(GraphOptimizationLevel::Basic.as_ort_level(), 1);
    assert_eq!(GraphOptimizationLevel::Extended.as_ort_level(), 2);
    assert_eq!(GraphOptimizationLevel::All.as_ort_level(), 99);
}

#[test]
fn test_generate_onnxruntime_script() {
    let config = OnnxRuntimeConfig::default();
    let spec = create_test_spec();
    let script = script::generate_onnxruntime_script(&spec, &config).unwrap();

    assert!(script.contains("import onnxruntime"));
    assert!(script.contains("ORT_RESULT_START"));
    assert!(script.contains("ORT_RESULT_END"));
}

#[test]
fn test_parse_onnxruntime_output_verified() {
    let stdout = r#"
ORT_INFO: Using synthetic test model
ORT_RESULT_START
{
  "model_path": "model.onnx",
  "execution_provider_requested": "CPUExecutionProvider",
  "execution_providers_available": ["CPUExecutionProvider"],
  "optimization_level": 2,
  "input_shape": [1, 10],
  "output_shape": [1, 5],
  "warmup_iterations": 5,
  "benchmark_iterations": 100,
  "consistent_outputs": true,
  "max_output_diff": 0.0,
  "latency_ms": {
    "mean": 0.5,
    "std": 0.1,
    "min": 0.4,
    "max": 0.8,
    "p50": 0.5,
    "p95": 0.7,
    "p99": 0.75
  },
  "throughput_ips": 2000.0
}
ORT_RESULT_END
ORT_SUMMARY: Mean latency: 0.500ms (std: 0.100ms)
ORT_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_onnxruntime_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_onnxruntime_output_not_verified() {
    let stdout = r#"
ORT_RESULT_START
{
  "model_path": "model.onnx",
  "execution_provider_requested": "CPUExecutionProvider",
  "execution_providers_available": ["CPUExecutionProvider"],
  "optimization_level": 2,
  "input_shape": [1, 10],
  "output_shape": [1, 5],
  "warmup_iterations": 5,
  "benchmark_iterations": 100,
  "consistent_outputs": false,
  "max_output_diff": 0.01,
  "latency_ms": {
    "mean": 1.0,
    "std": 0.5,
    "min": 0.5,
    "max": 2.0,
    "p50": 1.0,
    "p95": 1.8,
    "p99": 1.9
  },
  "throughput_ips": 1000.0
}
ORT_RESULT_END
ORT_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_onnxruntime_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("max_output_diff"));
    assert!(cex.witness.contains_key("consistent_outputs"));
}

#[test]
fn test_parse_onnxruntime_output_error() {
    let stdout = "ORT_ERROR: Missing dependency";
    let (status, _) = script::parse_onnxruntime_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_onnxruntime_health_check() {
    let backend = OnnxRuntimeBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_onnxruntime_verify_returns_result_or_unavailable() {
    let backend = OnnxRuntimeBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::ONNXRuntime),
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
