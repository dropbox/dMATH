//! Shared helpers for LLM evaluation counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by LLM eval backends
//! (DeepEval, Ragas, TruLens, PromptFoo, LangSmith, FactScore, SelfCheckGPT).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for LLM evaluation backends.
///
/// Expects a JSON object with fields like `pass_rate`, `passed`, `failed`,
/// `total`, `metric`, `pass_threshold` (or `score_threshold`), and `errors`.
/// Also supports `feedback_type` as an alias for `metric`.
pub fn build_llm_eval_counterexample(
    backend: &str,
    result: &Value,
) -> Option<StructuredCounterexample> {
    let pass_rate = result.get("pass_rate").and_then(Value::as_f64)?;
    // Support both pass_threshold and score_threshold (TruLens uses score_threshold)
    let threshold = result
        .get("pass_threshold")
        .or_else(|| result.get("score_threshold"))
        .and_then(Value::as_f64)
        .unwrap_or(0.8);

    // Only create counterexample if pass_rate is below threshold
    if pass_rate >= threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "pass_rate".to_string(),
        CounterexampleValue::Float { value: pass_rate },
    );

    if let Some(passed) = result.get("passed").and_then(Value::as_u64) {
        witness.insert(
            "passed".to_string(),
            CounterexampleValue::Int {
                value: passed as i128,
                type_hint: None,
            },
        );
    }

    if let Some(failed) = result.get("failed").and_then(Value::as_u64) {
        witness.insert(
            "failed".to_string(),
            CounterexampleValue::Int {
                value: failed as i128,
                type_hint: None,
            },
        );
    }

    if let Some(total) = result.get("total").and_then(Value::as_u64) {
        witness.insert(
            "total".to_string(),
            CounterexampleValue::Int {
                value: total as i128,
                type_hint: None,
            },
        );
    }

    // Support metric, feedback_type (TruLens), assertion_type (PromptFoo), evaluation_type (LangSmith),
    // extraction_method (FactScore), check_method (SelfCheckGPT)
    if let Some(metric) = result
        .get("metric")
        .or_else(|| result.get("feedback_type"))
        .or_else(|| result.get("assertion_type"))
        .or_else(|| result.get("evaluation_type"))
        .or_else(|| result.get("extraction_method"))
        .or_else(|| result.get("check_method"))
        .and_then(Value::as_str)
    {
        witness.insert(
            "metric".to_string(),
            CounterexampleValue::String(metric.to_string()),
        );
    }

    // Handle avg_score if present (TruLens specific)
    if let Some(avg_score) = result.get("avg_score").and_then(Value::as_f64) {
        witness.insert(
            "avg_score".to_string(),
            CounterexampleValue::Float { value: avg_score },
        );
    }

    let failed_checks = build_llm_eval_failed_checks(backend, result, pass_rate, threshold);
    let raw = serde_json::to_string(result).ok();

    Some(StructuredCounterexample {
        witness,
        failed_checks,
        playback_test: None,
        trace: vec![],
        raw,
        minimized: false,
    })
}

fn build_llm_eval_failed_checks(
    backend: &str,
    result: &Value,
    pass_rate: f64,
    threshold: f64,
) -> Vec<FailedCheck> {
    // Support metric, feedback_type (TruLens), assertion_type (PromptFoo), evaluation_type (LangSmith),
    // extraction_method (FactScore), check_method (SelfCheckGPT)
    let metric = result
        .get("metric")
        .or_else(|| result.get("feedback_type"))
        .or_else(|| result.get("assertion_type"))
        .or_else(|| result.get("evaluation_type"))
        .or_else(|| result.get("extraction_method"))
        .or_else(|| result.get("check_method"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let passed = result.get("passed").and_then(Value::as_u64).unwrap_or(0);
    let failed = result.get("failed").and_then(Value::as_u64).unwrap_or(0);
    let total = result.get("total").and_then(Value::as_u64).unwrap_or(0);
    let avg_score = result.get("avg_score").and_then(Value::as_f64);

    let mut description = format!("{} evaluation failed for {} metric.", backend, metric);

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{}).",
        pass_rate * 100.0,
        passed,
        total
    ));

    if failed > 0 {
        description.push_str(&format!(" {} test cases failed.", failed));
    }

    description.push_str(&format!(" Required threshold: {:.1}%.", threshold * 100.0));

    let gap = threshold - pass_rate;
    if gap > 0.0 {
        description.push_str(&format!(" Gap: {:.2}% below threshold.", gap * 100.0));
    }

    // Add avg_score detail if present
    if let Some(score) = avg_score {
        description.push_str(&format!(" Average score: {:.4}.", score));
    }

    // Add error summary if available
    if let Some(errors) = result.get("errors").and_then(Value::as_array) {
        let error_count = errors.len();
        if error_count > 0 {
            let sample_errors: Vec<_> = errors.iter().take(3).filter_map(|e| e.as_str()).collect();
            if !sample_errors.is_empty() {
                description.push_str(&format!(
                    " Sample failures: [{}].",
                    sample_errors.join("; ")
                ));
            }
            if error_count > 3 {
                description.push_str(&format!(" ({} more errors omitted.)", error_count - 3));
            }
        }
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_evaluation_failure",
            backend.to_lowercase().replace(' ', "_")
        ),
        description,
        location: None,
        function: None,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_llm_eval_counterexample_pass() {
        let result = json!({
            "pass_rate": 0.9,
            "pass_threshold": 0.8,
            "passed": 9,
            "failed": 1,
            "total": 10,
            "metric": "faithfulness"
        });
        assert!(build_llm_eval_counterexample("DeepEval", &result).is_none());
    }

    #[test]
    fn test_llm_eval_counterexample_fail() {
        let result = json!({
            "pass_rate": 0.6,
            "pass_threshold": 0.8,
            "passed": 6,
            "failed": 4,
            "total": 10,
            "metric": "answer_relevancy",
            "errors": ["Case 0: score 0.5 < threshold 0.8", "Case 1: score 0.4 < threshold 0.8"]
        });
        let cex = build_llm_eval_counterexample("DeepEval", &result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "deepeval_evaluation_failure");
        assert!(cex.failed_checks[0]
            .description
            .contains("answer_relevancy"));
        assert!(cex.failed_checks[0].description.contains("60.0%"));
        assert!(cex.failed_checks[0]
            .description
            .contains("4 test cases failed"));
        assert!(cex.witness.contains_key("pass_rate"));
        assert!(cex.witness.contains_key("failed"));
    }

    #[test]
    fn test_llm_eval_counterexample_ragas() {
        let result = json!({
            "pass_rate": 0.5,
            "pass_threshold": 0.75,
            "passed": 4,
            "failed": 4,
            "total": 8,
            "metric": "faithfulness",
            "errors": ["Case 3: faithfulness score 0.7 < threshold 0.75"]
        });
        let cex = build_llm_eval_counterexample("Ragas", &result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "ragas_evaluation_failure");
        assert!(cex.failed_checks[0].description.contains("faithfulness"));
        assert!(cex.failed_checks[0].description.contains("50.0%"));
    }

    #[test]
    fn test_llm_eval_counterexample_trulens_with_avg_score() {
        let result = json!({
            "pass_rate": 0.7,
            "pass_threshold": 0.8,
            "passed": 7,
            "failed": 3,
            "total": 10,
            "metric": "hallucination",
            "avg_score": 0.65,
            "errors": ["Case 2: hallucination 0.9 > threshold 0.2"]
        });
        let cex = build_llm_eval_counterexample("TruLens", &result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "trulens_evaluation_failure");
        assert!(cex.failed_checks[0].description.contains("0.6500"));
        assert!(cex.witness.contains_key("avg_score"));
    }

    #[test]
    fn test_llm_eval_counterexample_promptfoo() {
        let result = json!({
            "pass_rate": 0.4,
            "pass_threshold": 0.8,
            "passed": 4,
            "failed": 6,
            "total": 10,
            "metric": "contains",
            "errors": [
                "Case 0: expected match",
                "Case 1: score below threshold",
                "Case 2: assertion failed",
                "Case 3: timeout",
                "Case 4: empty response"
            ]
        });
        let cex = build_llm_eval_counterexample("PromptFoo", &result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "promptfoo_evaluation_failure"
        );
        // Should show 3 sample errors and note about omitted ones
        assert!(cex.failed_checks[0].description.contains("Sample failures"));
        assert!(cex.failed_checks[0]
            .description
            .contains("2 more errors omitted"));
    }

    #[test]
    fn test_llm_eval_missing_pass_rate() {
        let result = json!({
            "status": "success",
            "metric": "test"
        });
        assert!(build_llm_eval_counterexample("Test", &result).is_none());
    }
}
