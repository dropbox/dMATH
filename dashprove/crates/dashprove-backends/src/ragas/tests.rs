//! Tests for Ragas backend

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
    let backend = RagasBackend::new();
    assert_eq!(backend.id(), BackendId::Ragas);
}

#[test]
fn test_supports_llm_evaluation() {
    let backend = RagasBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::LLMEvaluation));
}

#[test]
fn test_default_config() {
    let config = RagasConfig::default();
    assert_eq!(config.metric, RagasMetric::Faithfulness);
    assert_eq!(config.mode, EvaluationMode::Single);
    assert!(config.context_metrics);
    assert!(config.answer_metrics);
    assert_eq!(config.batch_size, 8);
    assert!((config.pass_rate_threshold - 0.75).abs() < 0.01);
}

#[test]
fn test_pipeline_config() {
    let config = RagasConfig::pipeline();
    assert_eq!(config.mode, EvaluationMode::Pipeline);
    assert!(config.context_metrics);
    assert!(config.answer_metrics);
    assert!((config.pass_rate_threshold - 0.8).abs() < 0.01);
}

#[test]
fn test_faithfulness_strict_config() {
    let config = RagasConfig::faithfulness_strict();
    assert_eq!(config.metric, RagasMetric::Faithfulness);
    assert!((config.pass_rate_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_context_only_config() {
    let config = RagasConfig::context_only();
    assert_eq!(config.metric, RagasMetric::ContextPrecision);
    assert_eq!(config.mode, EvaluationMode::Multi);
    assert!(config.context_metrics);
    assert!(!config.answer_metrics);
}

#[test]
fn test_answer_correctness_config() {
    let config = RagasConfig::answer_correctness();
    assert_eq!(config.metric, RagasMetric::AnswerCorrectness);
    assert!((config.pass_rate_threshold - 0.85).abs() < 0.01);
}

#[test]
fn test_metric_strings() {
    assert_eq!(RagasMetric::Faithfulness.as_str(), "faithfulness");
    assert_eq!(RagasMetric::AnswerRelevancy.as_str(), "answer_relevancy");
    assert_eq!(RagasMetric::ContextPrecision.as_str(), "context_precision");
    assert_eq!(RagasMetric::ContextRecall.as_str(), "context_recall");
    assert_eq!(RagasMetric::ContextRelevancy.as_str(), "context_relevancy");
    assert_eq!(
        RagasMetric::AnswerCorrectness.as_str(),
        "answer_correctness"
    );
    assert_eq!(RagasMetric::AnswerSimilarity.as_str(), "answer_similarity");
}

#[test]
fn test_evaluation_mode_strings() {
    assert_eq!(EvaluationMode::Single.as_str(), "single");
    assert_eq!(EvaluationMode::Multi.as_str(), "multi");
    assert_eq!(EvaluationMode::Pipeline.as_str(), "pipeline");
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
RAGAS_RESULT_START
{
    "status": "success",
    "metric": "faithfulness",
    "mode": "single",
    "context_metrics": true,
    "answer_metrics": true,
    "batch_size": 8,
    "passed": 7,
    "failed": 1,
    "total": 8,
    "pass_rate": 0.875,
    "pass_threshold": 0.75,
    "errors": [],
    "duration_s": 0.5
}
RAGAS_RESULT_END
RAGAS_STATUS: VERIFIED
"#;

    let (status, counterexample) = script::parse_ragas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
RAGAS_RESULT_START
{
    "status": "success",
    "metric": "faithfulness",
    "mode": "single",
    "context_metrics": true,
    "answer_metrics": true,
    "batch_size": 8,
    "passed": 3,
    "failed": 5,
    "total": 8,
    "pass_rate": 0.375,
    "pass_threshold": 0.9,
    "errors": ["Case 3: faithfulness score 0.7 < threshold 0.75"],
    "duration_s": 0.4
}
RAGAS_RESULT_END
RAGAS_STATUS: NOT_VERIFIED
"#;

    let (status, counterexample) = script::parse_ragas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());
    let cx = counterexample.unwrap();
    assert!(cx.raw.is_some());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
RAGAS_RESULT_START
{
    "status": "success",
    "metric": "answer_relevancy",
    "mode": "single",
    "context_metrics": true,
    "answer_metrics": true,
    "batch_size": 8,
    "passed": 5,
    "failed": 3,
    "total": 8,
    "pass_rate": 0.625,
    "pass_threshold": 0.8,
    "errors": ["Case 6: relevancy score 0.8 < threshold 0.8"],
    "duration_s": 0.35
}
RAGAS_RESULT_END
RAGAS_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, counterexample) = script::parse_ragas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(counterexample.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "RAGAS_ERROR: Missing dependencies: No module named 'ragas'";
    let (status, _) = script::parse_ragas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_script_generation() {
    let config = RagasConfig::default();
    let spec = create_test_spec();
    let script = script::generate_ragas_script(&spec, &config);
    assert!(script.is_ok());
    let script_content = script.unwrap();
    assert!(script_content.contains("metric = \"faithfulness\""));
    assert!(script_content.contains("mode = \"single\""));
    assert!(script_content.contains("context_metrics = True"));
}

#[tokio::test]
async fn test_ragas_health_check() {
    let backend = RagasBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => (),
        HealthStatus::Unavailable { reason } => {
            assert!(
                reason.contains("ragas")
                    || reason.contains("not installed")
                    || reason.contains("Ragas")
            );
        }
        HealthStatus::Degraded { .. } => (),
    }
}

#[tokio::test]
async fn test_ragas_verify_returns_result_or_unavailable() {
    let backend = RagasBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::Ragas);
        }
        Err(BackendError::Unavailable(_)) => (),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
