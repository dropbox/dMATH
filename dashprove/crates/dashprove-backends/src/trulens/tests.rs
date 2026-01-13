//! Tests for TruLens backend

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
    let backend = TruLensBackend::new();
    assert_eq!(backend.id(), BackendId::TruLens);
}

#[test]
fn test_supports_llm_evaluation() {
    let backend = TruLensBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMEvaluation));
}

#[test]
fn test_default_config() {
    let config = TruLensConfig::default();
    assert_eq!(config.feedback_type, FeedbackType::AnswerRelevance);
    assert_eq!(config.provider, FeedbackProvider::Local);
    assert!(config.check_groundedness);
    assert!(config.check_relevance);
    assert!(!config.check_coherence);
    assert!((config.score_threshold - 0.7).abs() < 0.01);
}

#[test]
fn test_rag_eval_config() {
    let config = TruLensConfig::rag_eval();
    assert_eq!(config.feedback_type, FeedbackType::Groundedness);
    assert!(config.check_groundedness);
    assert!(config.check_relevance);
    assert!(config.check_coherence);
    assert!((config.score_threshold - 0.75).abs() < 0.01);
}

#[test]
fn test_answer_quality_config() {
    let config = TruLensConfig::answer_quality();
    assert_eq!(config.feedback_type, FeedbackType::AnswerRelevance);
    assert!(!config.check_groundedness);
    assert!(config.check_relevance);
    assert!(config.check_coherence);
    assert!((config.score_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_strict_config() {
    let config = TruLensConfig::strict();
    assert!(config.check_groundedness);
    assert!(config.check_relevance);
    assert!(config.check_coherence);
    assert!((config.score_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_feedback_type_strings() {
    assert_eq!(FeedbackType::AnswerRelevance.as_str(), "answer_relevance");
    assert_eq!(FeedbackType::ContextRelevance.as_str(), "context_relevance");
    assert_eq!(FeedbackType::Groundedness.as_str(), "groundedness");
    assert_eq!(FeedbackType::Coherence.as_str(), "coherence");
    assert_eq!(FeedbackType::Helpfulness.as_str(), "helpfulness");
    assert_eq!(FeedbackType::Custom.as_str(), "custom");
}

#[test]
fn test_feedback_provider_strings() {
    assert_eq!(FeedbackProvider::OpenAI.as_str(), "openai");
    assert_eq!(FeedbackProvider::HuggingFace.as_str(), "huggingface");
    assert_eq!(FeedbackProvider::Local.as_str(), "local");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
TRULENS_RESULT_START
{
    "status": "success",
    "feedback_type": "answer_relevance",
    "provider": "local",
    "check_groundedness": true,
    "check_relevance": true,
    "check_coherence": false,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "avg_score": 0.78,
    "score_threshold": 0.7,
    "errors": [],
    "duration_s": 0.5
}
TRULENS_RESULT_END
TRULENS_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_trulens_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
TRULENS_RESULT_START
{
    "status": "success",
    "feedback_type": "groundedness",
    "provider": "local",
    "check_groundedness": true,
    "check_relevance": true,
    "check_coherence": true,
    "passed": 3,
    "failed": 5,
    "total": 8,
    "pass_rate": 0.375,
    "avg_score": 0.55,
    "score_threshold": 0.75,
    "errors": ["Case 0: avg score 0.45 < threshold 0.75"],
    "duration_s": 0.4
}
TRULENS_RESULT_END
TRULENS_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_trulens_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
TRULENS_RESULT_START
{
    "status": "success",
    "feedback_type": "answer_relevance",
    "provider": "local",
    "check_groundedness": true,
    "check_relevance": true,
    "check_coherence": false,
    "passed": 5,
    "failed": 3,
    "total": 8,
    "pass_rate": 0.625,
    "avg_score": 0.68,
    "score_threshold": 0.7,
    "errors": ["Case 2: avg score 0.55 < threshold 0.7"],
    "duration_s": 0.3
}
TRULENS_RESULT_END
TRULENS_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_trulens_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "TRULENS_ERROR: Missing dependencies: No module named 'trulens_eval'";
    let (status, _) = script::parse_trulens_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = TruLensConfig::default();
    let spec = create_test_spec();
    let script = script::generate_trulens_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("feedback_type = \"answer_relevance\""));
    assert!(script_content.contains("provider = \"local\""));
    assert!(script_content.contains("check_groundedness = True"));
}

#[tokio::test]
async fn test_trulens_health_check() {
    let backend = TruLensBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("trulens")
                    || reason.contains("not installed")
                    || reason.contains("TruLens")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_trulens_verify_returns_result_or_unavailable() {
    let backend = TruLensBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::TruLens);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
