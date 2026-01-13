//! Tests for DeepEval backend

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
    let backend = DeepEvalBackend::new();
    assert_eq!(backend.id(), BackendId::DeepEval);
}

#[test]
fn test_supports_llm_evaluation() {
    let backend = DeepEvalBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMEvaluation));
    assert!(supported.contains(&PropertyType::HallucinationDetection));
}

#[test]
fn test_default_config() {
    let config = DeepEvalConfig::default();
    assert_eq!(config.metric, DeepEvalMetric::AnswerRelevancy);
    assert_eq!(config.test_case_type, TestCaseType::LLM);
    assert!(!config.strict_mode);
    assert!((config.threshold - 0.7).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_hallucination_config() {
    let config = DeepEvalConfig::hallucination();
    assert_eq!(config.metric, DeepEvalMetric::Hallucination);
    assert!((config.threshold - 0.5).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_safety_config() {
    let config = DeepEvalConfig::safety();
    assert_eq!(config.metric, DeepEvalMetric::Toxicity);
    assert!(config.strict_mode);
    assert!((config.threshold - 0.2).abs() < 0.01);
    assert!((config.pass_rate_threshold - 0.95).abs() < 0.01);
}

#[test]
fn test_faithfulness_strict_config() {
    let config = DeepEvalConfig::faithfulness_strict();
    assert_eq!(config.metric, DeepEvalMetric::Faithfulness);
    assert!(config.strict_mode);
    assert!((config.threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_conversational_config() {
    let config = DeepEvalConfig::conversational();
    assert_eq!(config.test_case_type, TestCaseType::Conversational);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_metric_strings() {
    assert_eq!(DeepEvalMetric::AnswerRelevancy.as_str(), "answer_relevancy");
    assert_eq!(DeepEvalMetric::Faithfulness.as_str(), "faithfulness");
    assert_eq!(
        DeepEvalMetric::ContextualPrecision.as_str(),
        "contextual_precision"
    );
    assert_eq!(
        DeepEvalMetric::ContextualRecall.as_str(),
        "contextual_recall"
    );
    assert_eq!(DeepEvalMetric::Hallucination.as_str(), "hallucination");
    assert_eq!(DeepEvalMetric::Toxicity.as_str(), "toxicity");
    assert_eq!(DeepEvalMetric::Bias.as_str(), "bias");
    assert_eq!(DeepEvalMetric::GEval.as_str(), "g_eval");
}

#[test]
fn test_test_case_type_strings() {
    assert_eq!(TestCaseType::LLM.as_str(), "llm");
    assert_eq!(TestCaseType::Conversational.as_str(), "conversational");
    assert_eq!(TestCaseType::MultiModal.as_str(), "multi_modal");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
DEEPEVAL_RESULT_START
{
    "status": "success",
    "metric": "answer_relevancy",
    "test_case_type": "llm",
    "strict_mode": false,
    "threshold": 0.7,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.8,
    "errors": [],
    "duration_s": 0.3
}
DEEPEVAL_RESULT_END
DEEPEVAL_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_deepeval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
DEEPEVAL_RESULT_START
{
    "status": "success",
    "metric": "hallucination",
    "test_case_type": "llm",
    "strict_mode": false,
    "threshold": 0.5,
    "passed": 4,
    "failed": 4,
    "total": 8,
    "pass_rate": 0.5,
    "pass_threshold": 0.9,
    "errors": ["Case 1: hallucination 0.9 > threshold 0.5"],
    "duration_s": 0.25
}
DEEPEVAL_RESULT_END
DEEPEVAL_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_deepeval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
DEEPEVAL_RESULT_START
{
    "status": "success",
    "metric": "faithfulness",
    "test_case_type": "llm",
    "strict_mode": true,
    "threshold": 0.9,
    "passed": 6,
    "failed": 2,
    "total": 8,
    "pass_rate": 0.75,
    "pass_threshold": 0.95,
    "errors": ["Case 3: faithfulness 0.75 < threshold 0.9"],
    "duration_s": 0.28
}
DEEPEVAL_RESULT_END
DEEPEVAL_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_deepeval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "DEEPEVAL_ERROR: Missing dependencies: No module named 'deepeval'";
    let (status, _) = script::parse_deepeval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = DeepEvalConfig::default();
    let spec = create_test_spec();
    let script = script::generate_deepeval_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("metric = \"answer_relevancy\""));
    assert!(script_content.contains("test_case_type = \"llm\""));
    assert!(script_content.contains("strict_mode = False"));
}

#[tokio::test]
async fn test_deepeval_health_check() {
    let backend = DeepEvalBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("deepeval")
                    || reason.contains("not installed")
                    || reason.contains("DeepEval")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_deepeval_verify_returns_result_or_unavailable() {
    let backend = DeepEvalBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::DeepEval);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
