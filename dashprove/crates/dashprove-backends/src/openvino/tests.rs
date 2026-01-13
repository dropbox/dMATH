//! Tests for OpenVINO backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_openvino_backend_id() {
    let backend = OpenVINOBackend::new();
    assert_eq!(backend.id(), BackendId::OpenVINO);
}

#[test]
fn test_openvino_supports_optimization() {
    let backend = OpenVINOBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelOptimization));
}

#[test]
fn test_openvino_config_default() {
    let config = OpenVINOConfig::default();
    assert_eq!(config.device, DeviceTarget::CPU);
    assert_eq!(config.precision, InferencePrecision::FP32);
    assert_eq!(config.performance_hint, PerformanceHint::Latency);
    assert_eq!(config.num_streams, 1);
}

#[test]
fn test_openvino_config_gpu() {
    let config = OpenVINOConfig::gpu();
    assert_eq!(config.device, DeviceTarget::GPU);
    assert_eq!(config.precision, InferencePrecision::FP16);
}

#[test]
fn test_openvino_config_high_throughput() {
    let config = OpenVINOConfig::high_throughput();
    assert_eq!(config.performance_hint, PerformanceHint::Throughput);
    assert_eq!(config.num_streams, 4);
}

#[test]
fn test_openvino_config_int8() {
    let config = OpenVINOConfig::int8();
    assert_eq!(config.precision, InferencePrecision::INT8);
}

#[test]
fn test_device_target_as_str() {
    assert_eq!(DeviceTarget::CPU.as_str(), "CPU");
    assert_eq!(DeviceTarget::GPU.as_str(), "GPU");
    assert_eq!(DeviceTarget::VPU.as_str(), "VPU");
    assert_eq!(DeviceTarget::FPGA.as_str(), "FPGA");
    assert_eq!(DeviceTarget::HETERO.as_str(), "HETERO");
    assert_eq!(DeviceTarget::MULTI.as_str(), "MULTI");
    assert_eq!(DeviceTarget::AUTO.as_str(), "AUTO");
}

#[test]
fn test_inference_precision_as_str() {
    assert_eq!(InferencePrecision::FP32.as_str(), "FP32");
    assert_eq!(InferencePrecision::FP16.as_str(), "FP16");
    assert_eq!(InferencePrecision::BF16.as_str(), "BF16");
    assert_eq!(InferencePrecision::INT8.as_str(), "INT8");
}

#[test]
fn test_performance_hint_as_str() {
    assert_eq!(PerformanceHint::Latency.as_str(), "LATENCY");
    assert_eq!(PerformanceHint::Throughput.as_str(), "THROUGHPUT");
    assert_eq!(PerformanceHint::Undefined.as_str(), "UNDEFINED");
}

#[test]
fn test_generate_openvino_script() {
    let config = OpenVINOConfig::default();
    let spec = create_test_spec();
    let script = script::generate_openvino_script(&spec, &config).unwrap();

    assert!(script.contains("from openvino"));
    assert!(script.contains("OV_RESULT_START"));
    assert!(script.contains("OV_RESULT_END"));
}

#[test]
fn test_parse_openvino_output_verified() {
    let stdout = r#"
OV_INFO: Using synthetic test model
OV_RESULT_START
{
  "model_path": "model.onnx",
  "device_requested": "CPU",
  "device_used": "CPU",
  "available_devices": ["CPU", "GPU"],
  "precision": "FP32",
  "performance_hint": "LATENCY",
  "input_shape": [1, 10],
  "output_shape": [1, 5],
  "warmup_iterations": 5,
  "benchmark_iterations": 100,
  "consistent_outputs": true,
  "max_output_diff": 0.0,
  "latency_ms": {
    "mean": 0.4,
    "std": 0.05,
    "min": 0.3,
    "max": 0.6,
    "p50": 0.4,
    "p95": 0.5,
    "p99": 0.55
  },
  "throughput_ips": 2500.0
}
OV_RESULT_END
OV_SUMMARY: Mean latency: 0.400ms
OV_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_openvino_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_openvino_output_not_verified() {
    let stdout = r#"
OV_RESULT_START
{
  "model_path": "model.onnx",
  "device_requested": "GPU",
  "device_used": "CPU",
  "available_devices": ["CPU"],
  "precision": "FP16",
  "performance_hint": "THROUGHPUT",
  "input_shape": [1, 10],
  "output_shape": [1, 5],
  "warmup_iterations": 5,
  "benchmark_iterations": 100,
  "consistent_outputs": false,
  "max_output_diff": 0.005,
  "latency_ms": {
    "mean": 0.8,
    "std": 0.2,
    "min": 0.5,
    "max": 1.5,
    "p50": 0.8,
    "p95": 1.2,
    "p99": 1.4
  },
  "throughput_ips": 1250.0
}
OV_RESULT_END
OV_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_openvino_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("max_output_diff"));
}

#[test]
fn test_parse_openvino_output_error() {
    let stdout = "OV_ERROR: Failed to compile model";
    let (status, _) = script::parse_openvino_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_openvino_health_check() {
    let backend = OpenVINOBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_openvino_verify_returns_result_or_unavailable() {
    let backend = OpenVINOBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::OpenVINO),
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
