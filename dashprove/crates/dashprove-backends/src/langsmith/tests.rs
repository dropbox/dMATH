//! Tests for LangSmith backend

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
    let backend = LangSmithBackend::new();
    assert_eq!(backend.id(), BackendId::LangSmith);
}

#[test]
fn test_supports_llm_evaluation() {
    let backend = LangSmithBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMEvaluation));
}

#[test]
fn test_default_config() {
    let config = LangSmithConfig::default();
    assert_eq!(config.tracing_mode, TracingMode::Full);
    assert_eq!(config.evaluation_type, EvaluationType::Custom);
    assert!(config.enable_feedback);
    assert!(!config.enable_comparisons);
    assert!((config.sample_rate - 1.0).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_llm_judge_config() {
    let config = LangSmithConfig::llm_judge();
    assert_eq!(config.evaluation_type, EvaluationType::LLMJudge);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_similarity_config() {
    let config = LangSmithConfig::similarity();
    assert_eq!(config.evaluation_type, EvaluationType::Similarity);
    assert!((config.pass_rate_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_sampled_config() {
    let config = LangSmithConfig::sampled(0.5);
    assert_eq!(config.tracing_mode, TracingMode::Sample);
    assert!((config.sample_rate - 0.5).abs() < 0.01);
}

#[test]
fn test_exact_match_strict_config() {
    let config = LangSmithConfig::exact_match_strict();
    assert_eq!(config.evaluation_type, EvaluationType::ExactMatch);
    assert!((config.pass_rate_threshold - 1.0).abs() < 0.01);
}

#[test]
fn test_tracing_mode_strings() {
    assert_eq!(TracingMode::Full.as_str(), "full");
    assert_eq!(TracingMode::Sample.as_str(), "sample");
    assert_eq!(TracingMode::Debug.as_str(), "debug");
    assert_eq!(TracingMode::ErrorsOnly.as_str(), "errors_only");
}

#[test]
fn test_evaluation_type_strings() {
    assert_eq!(EvaluationType::Custom.as_str(), "custom");
    assert_eq!(EvaluationType::LLMJudge.as_str(), "llm_judge");
    assert_eq!(EvaluationType::Similarity.as_str(), "similarity");
    assert_eq!(EvaluationType::ExactMatch.as_str(), "exact_match");
    assert_eq!(EvaluationType::Regex.as_str(), "regex");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
LANGSMITH_RESULT_START
{
    "status": "success",
    "tracing_mode": "full",
    "evaluation_type": "custom",
    "enable_feedback": true,
    "enable_comparisons": false,
    "sample_rate": 1.0,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.8,
    "errors": [],
    "duration_s": 0.3
}
LANGSMITH_RESULT_END
LANGSMITH_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_langsmith_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
LANGSMITH_RESULT_START
{
    "status": "success",
    "tracing_mode": "full",
    "evaluation_type": "exact_match",
    "enable_feedback": true,
    "enable_comparisons": false,
    "sample_rate": 1.0,
    "passed": 4,
    "failed": 4,
    "total": 8,
    "pass_rate": 0.5,
    "pass_threshold": 1.0,
    "errors": ["Case 3: expected 'true', got 'True'"],
    "duration_s": 0.2
}
LANGSMITH_RESULT_END
LANGSMITH_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_langsmith_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
LANGSMITH_RESULT_START
{
    "status": "success",
    "tracing_mode": "full",
    "evaluation_type": "llm_judge",
    "enable_feedback": true,
    "enable_comparisons": false,
    "sample_rate": 1.0,
    "passed": 6,
    "failed": 2,
    "total": 8,
    "pass_rate": 0.75,
    "pass_threshold": 0.85,
    "errors": ["Case 6: rating 0.75 < threshold 0.8"],
    "duration_s": 0.25
}
LANGSMITH_RESULT_END
LANGSMITH_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_langsmith_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "LANGSMITH_ERROR: Missing dependencies: No module named 'langsmith'";
    let (status, _) = script::parse_langsmith_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = LangSmithConfig::default();
    let spec = create_test_spec();
    let script = script::generate_langsmith_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("tracing_mode = \"full\""));
    assert!(script_content.contains("evaluation_type = \"custom\""));
    assert!(script_content.contains("enable_feedback = True"));
}

#[tokio::test]
async fn test_langsmith_health_check() {
    let backend = LangSmithBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("langsmith")
                    || reason.contains("not installed")
                    || reason.contains("LangSmith")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_langsmith_verify_returns_result_or_unavailable() {
    let backend = LangSmithBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::LangSmith);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
