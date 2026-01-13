//! Tests for IREE backend

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
fn test_iree_target_strings() {
    assert_eq!(IREETarget::LLVMCPU.as_str(), "llvm-cpu");
    assert_eq!(IREETarget::VulkanSPIRV.as_str(), "vulkan-spirv");
    assert_eq!(IREETarget::CUDA.as_str(), "cuda");
    assert_eq!(IREETarget::MetalSPIRV.as_str(), "metal-spirv");
    assert_eq!(IREETarget::ROCm.as_str(), "rocm");
    assert_eq!(IREETarget::WebGPU.as_str(), "webgpu");
    assert_eq!(IREETarget::VMVX.as_str(), "vmvx");
}

#[test]
fn test_input_format_strings() {
    assert_eq!(InputFormat::StableHLO.as_str(), "stablehlo");
    assert_eq!(InputFormat::TOSA.as_str(), "tosa");
    assert_eq!(InputFormat::Linalg.as_str(), "linalg");
    assert_eq!(InputFormat::TFSavedModel.as_str(), "tf_saved_model");
    assert_eq!(InputFormat::TFLite.as_str(), "tflite");
    assert_eq!(InputFormat::ONNX.as_str(), "onnx");
}

#[test]
fn test_execution_mode_strings() {
    assert_eq!(ExecutionMode::LocalSync.as_str(), "local-sync");
    assert_eq!(ExecutionMode::LocalTask.as_str(), "local-task");
    assert_eq!(ExecutionMode::AsyncDispatch.as_str(), "async-dispatch");
}

#[test]
fn test_default_config() {
    let config = IREEConfig::default();
    assert_eq!(config.target, IREETarget::LLVMCPU);
    assert_eq!(config.input_format, InputFormat::StableHLO);
    assert_eq!(config.execution_mode, ExecutionMode::LocalSync);
    assert!(config.enable_optimization);
    assert_eq!(config.warmup_iterations, 10);
    assert_eq!(config.benchmark_iterations, 100);
}

#[test]
fn test_vulkan_config() {
    let config = IREEConfig::vulkan();
    assert_eq!(config.target, IREETarget::VulkanSPIRV);
}

#[test]
fn test_cuda_config() {
    let config = IREEConfig::cuda();
    assert_eq!(config.target, IREETarget::CUDA);
}

#[test]
fn test_parallel_config() {
    let config = IREEConfig::parallel();
    assert_eq!(config.execution_mode, ExecutionMode::LocalTask);
}

#[test]
fn test_backend_id() {
    let backend = IREEBackend::new();
    assert_eq!(backend.id(), BackendId::IREE);
}

#[test]
fn test_supports_model_optimization() {
    let backend = IREEBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelOptimization));
}

#[test]
fn test_script_generation() {
    let config = IREEConfig::default();
    let spec = create_test_spec();
    let script = script::generate_iree_script(&spec, &config).unwrap();

    assert!(script.contains("import iree.runtime"));
    assert!(script.contains("import iree.compiler"));
    assert!(script.contains("IREE_RESULT_START"));
    assert!(script.contains("IREE_RESULT_END"));
    assert!(script.contains("IREE_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
IREE_RESULT_START
{
    "consistent_outputs": true,
    "max_output_diff": 0.0,
    "latency_ms": {"mean": 1.5}
}
IREE_RESULT_END
IREE_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_iree_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
IREE_RESULT_START
{
    "consistent_outputs": true,
    "max_output_diff": 1e-4,
    "latency_ms": {"mean": 1.5}
}
IREE_RESULT_END
IREE_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_iree_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
IREE_RESULT_START
{
    "consistent_outputs": false,
    "max_output_diff": 0.5,
    "latency_ms": {"mean": 1.5}
}
IREE_RESULT_END
IREE_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_iree_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "IREE_ERROR: Compilation failed: invalid module";
    let (status, _) = script::parse_iree_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_iree_health_check() {
    let backend = IREEBackend::new();
    let status = backend.health_check().await;
    // IREE may or may not be installed
    match status {
        HealthStatus::Healthy => println!("IREE is available"),
        HealthStatus::Unavailable { reason } => println!("IREE unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("IREE degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_iree_verify_returns_result_or_unavailable() {
    let backend = IREEBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::IREE);
            println!("IREE verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("IREE unavailable: {}", reason);
        }
        Err(e) => {
            println!("IREE error (expected if not installed): {}", e);
        }
    }
}
