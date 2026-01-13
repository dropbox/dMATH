//! Tests for Apache TVM backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_tvm_backend_id() {
    let backend = TVMBackend::new();
    assert_eq!(backend.id(), BackendId::TVM);
}

#[test]
fn test_tvm_supports_optimization() {
    let backend = TVMBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelOptimization));
}

#[test]
fn test_tvm_config_default() {
    let config = TVMConfig::default();
    assert_eq!(config.target, TVMTarget::LLVM);
    assert_eq!(config.opt_level, OptLevel::Standard);
    assert_eq!(config.tuning_mode, TuningMode::None);
    assert!(config.enable_relay_opt);
}

#[test]
fn test_tvm_config_cuda() {
    let config = TVMConfig::cuda();
    assert_eq!(config.target, TVMTarget::CUDA);
    assert_eq!(config.opt_level, OptLevel::Aggressive);
}

#[test]
fn test_tvm_config_autotvm() {
    let config = TVMConfig::with_autotvm();
    assert_eq!(config.tuning_mode, TuningMode::AutoTVM);
    assert_eq!(config.tuning_trials, 500);
}

#[test]
fn test_tvm_config_meta_schedule() {
    let config = TVMConfig::with_meta_schedule();
    assert_eq!(config.tuning_mode, TuningMode::MetaSchedule);
    assert_eq!(config.tuning_trials, 1000);
}

#[test]
fn test_tvm_target_as_str() {
    assert_eq!(TVMTarget::LLVM.as_str(), "llvm");
    assert_eq!(TVMTarget::CUDA.as_str(), "cuda");
    assert_eq!(TVMTarget::Metal.as_str(), "metal");
    assert_eq!(TVMTarget::Vulkan.as_str(), "vulkan");
    assert_eq!(TVMTarget::OpenCL.as_str(), "opencl");
    assert_eq!(TVMTarget::ROCm.as_str(), "rocm");
    assert_eq!(TVMTarget::WebGPU.as_str(), "webgpu");
}

#[test]
fn test_opt_level_as_int() {
    assert_eq!(OptLevel::None.as_int(), 0);
    assert_eq!(OptLevel::Basic.as_int(), 1);
    assert_eq!(OptLevel::Standard.as_int(), 2);
    assert_eq!(OptLevel::Aggressive.as_int(), 3);
}

#[test]
fn test_tuning_mode_as_str() {
    assert_eq!(TuningMode::None.as_str(), "none");
    assert_eq!(TuningMode::AutoTVM.as_str(), "autotvm");
    assert_eq!(TuningMode::MetaSchedule.as_str(), "meta_schedule");
    assert_eq!(TuningMode::PreTuned.as_str(), "pre_tuned");
}

#[test]
fn test_generate_tvm_script() {
    let config = TVMConfig::default();
    let spec = create_test_spec();
    let script = script::generate_tvm_script(&spec, &config).unwrap();

    assert!(script.contains("import tvm"));
    assert!(script.contains("TVM_RESULT_START"));
    assert!(script.contains("TVM_RESULT_END"));
}

#[test]
fn test_parse_tvm_output_verified() {
    let stdout = r#"
TVM_INFO: Using synthetic test model
TVM_RESULT_START
{
  "model_path": "model.onnx",
  "target_requested": "llvm",
  "target_used": "llvm",
  "opt_level": 2,
  "tuning_mode": "none",
  "tvm_version": "0.14.0",
  "input_shape": [1, 10],
  "output_shape": [1, 5],
  "warmup_iterations": 10,
  "benchmark_iterations": 100,
  "consistent_outputs": true,
  "max_output_diff": 0.0,
  "latency_ms": {
    "mean": 0.2,
    "std": 0.03,
    "min": 0.15,
    "max": 0.35,
    "p50": 0.2,
    "p95": 0.28,
    "p99": 0.32
  },
  "throughput_ips": 5000.0
}
TVM_RESULT_END
TVM_SUMMARY: Mean latency: 0.200ms
TVM_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_tvm_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_tvm_output_not_verified() {
    let stdout = r#"
TVM_RESULT_START
{
  "model_path": "model.onnx",
  "target_requested": "cuda",
  "target_used": "llvm",
  "opt_level": 3,
  "tuning_mode": "autotvm",
  "tvm_version": "0.14.0",
  "input_shape": [1, 10],
  "output_shape": [1, 5],
  "warmup_iterations": 10,
  "benchmark_iterations": 100,
  "consistent_outputs": false,
  "max_output_diff": 0.01,
  "latency_ms": {
    "mean": 0.5,
    "std": 0.1,
    "min": 0.3,
    "max": 0.8,
    "p50": 0.5,
    "p95": 0.7,
    "p99": 0.75
  },
  "throughput_ips": 2000.0
}
TVM_RESULT_END
TVM_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_tvm_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("max_output_diff"));
}

#[test]
fn test_parse_tvm_output_error() {
    let stdout = "TVM_ERROR: Compilation failed";
    let (status, _) = script::parse_tvm_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_tvm_health_check() {
    let backend = TVMBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_tvm_verify_returns_result_or_unavailable() {
    let backend = TVMBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::TVM),
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
