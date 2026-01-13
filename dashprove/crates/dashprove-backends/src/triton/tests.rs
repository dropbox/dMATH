//! Tests for Triton backend

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
fn test_compilation_mode_strings() {
    assert_eq!(CompilationMode::JIT.as_str(), "jit");
    assert_eq!(CompilationMode::AOT.as_str(), "aot");
    assert_eq!(CompilationMode::Interpret.as_str(), "interpret");
}

#[test]
fn test_target_strings() {
    assert_eq!(TritonTarget::CUDA.as_str(), "cuda");
    assert_eq!(TritonTarget::ROCm.as_str(), "hip");
    assert_eq!(TritonTarget::CPU.as_str(), "cpu");
}

#[test]
fn test_opt_level_values() {
    assert_eq!(OptimizationLevel::O0.as_int(), 0);
    assert_eq!(OptimizationLevel::O1.as_int(), 1);
    assert_eq!(OptimizationLevel::O2.as_int(), 2);
    assert_eq!(OptimizationLevel::O3.as_int(), 3);
}

#[test]
fn test_default_config() {
    let config = TritonConfig::default();
    assert_eq!(config.compilation_mode, CompilationMode::JIT);
    assert_eq!(config.target, TritonTarget::CUDA);
    assert_eq!(config.opt_level, OptimizationLevel::O2);
    assert_eq!(config.num_warps, 4);
    assert_eq!(config.num_stages, 2);
    assert!(!config.autotune);
}

#[test]
fn test_autotune_config() {
    let config = TritonConfig::with_autotune();
    assert!(config.autotune);
}

#[test]
fn test_rocm_config() {
    let config = TritonConfig::rocm();
    assert_eq!(config.target, TritonTarget::ROCm);
}

#[test]
fn test_optimized_config() {
    let config = TritonConfig::optimized();
    assert_eq!(config.opt_level, OptimizationLevel::O3);
    assert_eq!(config.num_warps, 8);
    assert_eq!(config.num_stages, 4);
}

#[test]
fn test_backend_id() {
    let backend = TritonBackend::new();
    assert_eq!(backend.id(), BackendId::Triton);
}

#[test]
fn test_supports_model_optimization() {
    let backend = TritonBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::ModelOptimization));
}

#[test]
fn test_script_generation() {
    let config = TritonConfig::default();
    let spec = create_test_spec();
    let script = script::generate_triton_script(&spec, &config).unwrap();

    assert!(script.contains("import triton"));
    assert!(script.contains("import triton.language as tl"));
    assert!(script.contains("TRITON_RESULT_START"));
    assert!(script.contains("TRITON_RESULT_END"));
    assert!(script.contains("TRITON_STATUS:"));
    assert!(script.contains("@triton.jit"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
TRITON_RESULT_START
{
    "consistent_outputs": true,
    "max_output_diff": 0.0,
    "latency_ms": {"mean": 0.05}
}
TRITON_RESULT_END
TRITON_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_triton_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
TRITON_RESULT_START
{
    "consistent_outputs": true,
    "max_output_diff": 1e-4,
    "latency_ms": {"mean": 0.05},
    "bandwidth_gbps": 500.0
}
TRITON_RESULT_END
TRITON_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_triton_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
TRITON_RESULT_START
{
    "consistent_outputs": false,
    "max_output_diff": 0.5,
    "latency_ms": {"mean": 0.05}
}
TRITON_RESULT_END
TRITON_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_triton_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "TRITON_ERROR: CUDA out of memory";
    let (status, _) = script::parse_triton_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_triton_health_check() {
    let backend = TritonBackend::new();
    let status = backend.health_check().await;
    // Triton may or may not be installed
    match status {
        HealthStatus::Healthy => println!("Triton is available"),
        HealthStatus::Unavailable { reason } => println!("Triton unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Triton degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_triton_verify_returns_result_or_unavailable() {
    let backend = TritonBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Triton);
            println!("Triton verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Triton unavailable: {}", reason);
        }
        Err(e) => {
            println!("Triton error (expected if not installed): {}", e);
        }
    }
}
