//! Tests for SelfCheckGPT backend

use super::*;
use crate::traits::VerificationStatus;

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
fn test_backend_id() {
    let backend = SelfCheckGPTBackend::new();
    assert_eq!(backend.id(), BackendId::SelfCheckGPT);
}

#[test]
fn test_supports_hallucination_detection() {
    let backend = SelfCheckGPTBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::HallucinationDetection));
}

#[test]
fn test_default_config() {
    let config = SelfCheckGPTConfig::default();
    assert_eq!(config.check_method, CheckMethod::BertScore);
    assert_eq!(config.sampling_strategy, SamplingStrategy::Standard);
    assert_eq!(config.num_samples, 5);
    assert!((config.hallucination_threshold - 0.5).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_nli_config() {
    let config = SelfCheckGPTConfig::nli();
    assert_eq!(config.check_method, CheckMethod::NLI);
    assert!((config.hallucination_threshold - 0.4).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_ensemble_config() {
    let config = SelfCheckGPTConfig::ensemble();
    assert_eq!(config.check_method, CheckMethod::Ensemble);
    assert_eq!(config.num_samples, 10);
    assert!((config.hallucination_threshold - 0.3).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_fast_ngram_config() {
    let config = SelfCheckGPTConfig::fast_ngram();
    assert_eq!(config.check_method, CheckMethod::Ngram);
    assert_eq!(config.num_samples, 3);
    assert!((config.hallucination_threshold - 0.6).abs() < 0.01);
}

#[test]
fn test_strict_config() {
    let config = SelfCheckGPTConfig::strict();
    assert_eq!(config.check_method, CheckMethod::BertScore);
    assert_eq!(config.num_samples, 10);
    assert!((config.hallucination_threshold - 0.2).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.95).abs() < 0.01);
}

#[test]
fn test_check_method_strings() {
    assert_eq!(CheckMethod::BertScore.as_str(), "bertscore");
    assert_eq!(CheckMethod::Ngram.as_str(), "ngram");
    assert_eq!(CheckMethod::NLI.as_str(), "nli");
    assert_eq!(CheckMethod::Prompt.as_str(), "prompt");
    assert_eq!(CheckMethod::Ensemble.as_str(), "ensemble");
}

#[test]
fn test_sampling_strategy_strings() {
    assert_eq!(SamplingStrategy::Standard.as_str(), "standard");
    assert_eq!(SamplingStrategy::Temperature.as_str(), "temperature");
    assert_eq!(SamplingStrategy::TopK.as_str(), "top_k");
    assert_eq!(SamplingStrategy::Nucleus.as_str(), "nucleus");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
SELFCHECKGPT_RESULT_START
{
    "status": "success",
    "check_method": "bertscore",
    "sampling_strategy": "standard",
    "num_samples": 5,
    "hallucination_threshold": 0.5,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.8,
    "errors": [],
    "duration_s": 0.5
}
SELFCHECKGPT_RESULT_END
SELFCHECKGPT_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_selfcheckgpt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
SELFCHECKGPT_RESULT_START
{
    "status": "success",
    "check_method": "nli",
    "sampling_strategy": "standard",
    "num_samples": 5,
    "hallucination_threshold": 0.2,
    "passed": 3,
    "failed": 5,
    "total": 8,
    "pass_rate": 0.375,
    "pass_threshold": 0.9,
    "errors": ["Case 0: hallucination score 0.35 > threshold 0.2"],
    "duration_s": 0.4
}
SELFCHECKGPT_RESULT_END
SELFCHECKGPT_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_selfcheckgpt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
SELFCHECKGPT_RESULT_START
{
    "status": "success",
    "check_method": "ensemble",
    "sampling_strategy": "standard",
    "num_samples": 10,
    "hallucination_threshold": 0.3,
    "passed": 6,
    "failed": 2,
    "total": 8,
    "pass_rate": 0.75,
    "pass_threshold": 0.9,
    "errors": ["Case 3: ensemble hallucination 0.257 > threshold 0.3"],
    "duration_s": 0.45
}
SELFCHECKGPT_RESULT_END
SELFCHECKGPT_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_selfcheckgpt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "SELFCHECKGPT_ERROR: Missing dependencies: No module named 'selfcheckgpt'";
    let (status, _) = script::parse_selfcheckgpt_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = SelfCheckGPTConfig::default();
    let spec = create_test_spec();
    let script = script::generate_selfcheckgpt_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("check_method = \"bertscore\""));
    assert!(script_content.contains("sampling_strategy = \"standard\""));
    assert!(script_content.contains("num_samples = 5"));
}

#[tokio::test]
async fn test_selfcheckgpt_health_check() {
    let backend = SelfCheckGPTBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("selfcheckgpt")
                    || reason.contains("not installed")
                    || reason.contains("SelfCheckGPT")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_selfcheckgpt_verify_returns_result_or_unavailable() {
    let backend = SelfCheckGPTBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::SelfCheckGPT);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
