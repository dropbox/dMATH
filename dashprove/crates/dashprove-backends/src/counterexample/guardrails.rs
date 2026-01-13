//! Shared helpers for guardrails counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by guardrails backends
//! (GuardrailsAI, Guidance, NeMo Guardrails).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for GuardrailsAI validation.
///
/// Expects a JSON object with fields like `pass_rate`, `pass_threshold`,
/// `passed`, `failed`, `total`, `guardrail_type`, and optionally `errors`.
pub fn build_guardrails_ai_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let pass_rate = result
        .get("pass_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let threshold = result
        .get("pass_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.8);

    if pass_rate >= threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "pass_rate".to_string(),
        CounterexampleValue::Float { value: pass_rate },
    );

    witness.insert(
        "pass_threshold".to_string(),
        CounterexampleValue::Float { value: threshold },
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

    if let Some(guardrail_type) = result.get("guardrail_type").and_then(Value::as_str) {
        witness.insert(
            "guardrail_type".to_string(),
            CounterexampleValue::String(guardrail_type.to_string()),
        );
    }

    // Extract error messages
    let errors: Vec<String> = result
        .get("errors")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if !errors.is_empty() {
        witness.insert(
            "errors".to_string(),
            CounterexampleValue::String(errors.join("; ")),
        );
    }

    let failed_checks = build_guardrails_ai_failed_checks(result, &errors);
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

fn build_guardrails_ai_failed_checks(result: &Value, errors: &[String]) -> Vec<FailedCheck> {
    let pass_rate = result
        .get("pass_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let threshold = result
        .get("pass_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.8);
    let passed = result.get("passed").and_then(Value::as_u64).unwrap_or(0);
    let failed = result.get("failed").and_then(Value::as_u64).unwrap_or(0);
    let guardrail_type = result
        .get("guardrail_type")
        .and_then(Value::as_str)
        .unwrap_or("schema");
    let strictness = result
        .get("strictness")
        .and_then(Value::as_str)
        .unwrap_or("normal");

    let mut description = format!(
        "GuardrailsAI {} guardrail failed (strictness: {}).",
        guardrail_type, strictness
    );

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{} validations), threshold: {:.1}%.",
        pass_rate * 100.0,
        passed,
        passed + failed,
        threshold * 100.0
    ));

    let gap = threshold - pass_rate;
    if gap > 0.0 {
        description.push_str(&format!(" Gap: {:.1}% below threshold.", gap * 100.0));
    }

    if !errors.is_empty() {
        let display_errors: Vec<&str> = errors.iter().take(3).map(|s| s.as_str()).collect();
        description.push_str(&format!(" Sample errors: [{}]", display_errors.join("; ")));
        if errors.len() > 3 {
            description.push_str(&format!(" and {} more.", errors.len() - 3));
        } else {
            description.push('.');
        }
    }

    vec![FailedCheck {
        check_id: "guardrails_ai_validation_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for Guidance structured generation.
///
/// Expects a JSON object with fields like `pass_rate`, `pass_threshold`,
/// `passed`, `failed`, `total`, `generation_mode`, and optionally `errors`.
pub fn build_guidance_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let pass_rate = result
        .get("pass_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let threshold = result
        .get("pass_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.85);

    if pass_rate >= threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "pass_rate".to_string(),
        CounterexampleValue::Float { value: pass_rate },
    );

    witness.insert(
        "pass_threshold".to_string(),
        CounterexampleValue::Float { value: threshold },
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

    if let Some(gen_mode) = result.get("generation_mode").and_then(Value::as_str) {
        witness.insert(
            "generation_mode".to_string(),
            CounterexampleValue::String(gen_mode.to_string()),
        );
    }

    // Extract error messages
    let errors: Vec<String> = result
        .get("errors")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if !errors.is_empty() {
        witness.insert(
            "errors".to_string(),
            CounterexampleValue::String(errors.join("; ")),
        );
    }

    let failed_checks = build_guidance_failed_checks(result, &errors);
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

fn build_guidance_failed_checks(result: &Value, errors: &[String]) -> Vec<FailedCheck> {
    let pass_rate = result
        .get("pass_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let threshold = result
        .get("pass_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.85);
    let passed = result.get("passed").and_then(Value::as_u64).unwrap_or(0);
    let failed = result.get("failed").and_then(Value::as_u64).unwrap_or(0);
    let gen_mode = result
        .get("generation_mode")
        .and_then(Value::as_str)
        .unwrap_or("constrained");
    let val_mode = result
        .get("validation_mode")
        .and_then(Value::as_str)
        .unwrap_or("strict");

    let mut description = format!(
        "Guidance {} generation failed (validation: {}).",
        gen_mode, val_mode
    );

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{} tests), threshold: {:.1}%.",
        pass_rate * 100.0,
        passed,
        passed + failed,
        threshold * 100.0
    ));

    let gap = threshold - pass_rate;
    if gap > 0.0 {
        description.push_str(&format!(" Gap: {:.1}% below threshold.", gap * 100.0));
    }

    if !errors.is_empty() {
        let display_errors: Vec<&str> = errors.iter().take(3).map(|s| s.as_str()).collect();
        description.push_str(&format!(" Failures: [{}]", display_errors.join("; ")));
        if errors.len() > 3 {
            description.push_str(&format!(" and {} more.", errors.len() - 3));
        } else {
            description.push('.');
        }
    }

    vec![FailedCheck {
        check_id: "guidance_generation_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for NeMo Guardrails.
///
/// Expects a JSON object with fields like `pass_rate`, `pass_threshold`,
/// `passed`, `failed`, `total`, `rail_type`, and optionally `errors`.
pub fn build_nemo_guardrails_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let pass_rate = result
        .get("pass_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let threshold = result
        .get("pass_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.85);

    if pass_rate >= threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "pass_rate".to_string(),
        CounterexampleValue::Float { value: pass_rate },
    );

    witness.insert(
        "pass_threshold".to_string(),
        CounterexampleValue::Float { value: threshold },
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

    if let Some(rail_type) = result.get("rail_type").and_then(Value::as_str) {
        witness.insert(
            "rail_type".to_string(),
            CounterexampleValue::String(rail_type.to_string()),
        );
    }

    // Extract error messages
    let errors: Vec<String> = result
        .get("errors")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if !errors.is_empty() {
        witness.insert(
            "errors".to_string(),
            CounterexampleValue::String(errors.join("; ")),
        );
    }

    let failed_checks = build_nemo_failed_checks(result, &errors);
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

fn build_nemo_failed_checks(result: &Value, errors: &[String]) -> Vec<FailedCheck> {
    let pass_rate = result
        .get("pass_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let threshold = result
        .get("pass_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.85);
    let passed = result.get("passed").and_then(Value::as_u64).unwrap_or(0);
    let failed = result.get("failed").and_then(Value::as_u64).unwrap_or(0);
    let rail_type = result
        .get("rail_type")
        .and_then(Value::as_str)
        .unwrap_or("output");
    let jailbreak = result
        .get("jailbreak_detection")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let topical = result
        .get("topical_rail")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let fact_check = result
        .get("fact_checking")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut description = format!("NeMo Guardrails {} rail failed.", rail_type);

    // Add enabled features
    let mut features = Vec::new();
    if jailbreak {
        features.push("jailbreak_detection");
    }
    if topical {
        features.push("topical_rail");
    }
    if fact_check {
        features.push("fact_checking");
    }
    if !features.is_empty() {
        description.push_str(&format!(" Features: [{}].", features.join(", ")));
    }

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{} tests), threshold: {:.1}%.",
        pass_rate * 100.0,
        passed,
        passed + failed,
        threshold * 100.0
    ));

    let gap = threshold - pass_rate;
    if gap > 0.0 {
        description.push_str(&format!(" Gap: {:.1}% below threshold.", gap * 100.0));
    }

    if !errors.is_empty() {
        let display_errors: Vec<&str> = errors.iter().take(3).map(|s| s.as_str()).collect();
        description.push_str(&format!(" Failures: [{}]", display_errors.join("; ")));
        if errors.len() > 3 {
            description.push_str(&format!(" and {} more.", errors.len() - 3));
        } else {
            description.push('.');
        }
    }

    vec![FailedCheck {
        check_id: "nemo_guardrails_rail_failure".to_string(),
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
    fn test_guardrails_ai_success() {
        let result = json!({
            "pass_rate": 0.9,
            "pass_threshold": 0.8,
            "passed": 9,
            "failed": 1,
            "total": 10
        });
        assert!(build_guardrails_ai_counterexample(&result).is_none());
    }

    #[test]
    fn test_guardrails_ai_failure() {
        let result = json!({
            "pass_rate": 0.5,
            "pass_threshold": 0.8,
            "passed": 5,
            "failed": 5,
            "total": 10,
            "guardrail_type": "schema",
            "strictness": "strict",
            "errors": ["Case 3: validation failed", "Case 5: schema mismatch"]
        });
        let cex = build_guardrails_ai_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "guardrails_ai_validation_failure"
        );
        assert!(cex.failed_checks[0].description.contains("50.0%"));
        assert!(cex.failed_checks[0].description.contains("schema"));
        assert!(cex.failed_checks[0].description.contains("strict"));
        assert!(cex.witness.contains_key("pass_rate"));
        assert!(cex.witness.contains_key("errors"));
    }

    #[test]
    fn test_guidance_success() {
        let result = json!({
            "pass_rate": 0.9,
            "pass_threshold": 0.85,
            "passed": 9,
            "failed": 1
        });
        assert!(build_guidance_counterexample(&result).is_none());
    }

    #[test]
    fn test_guidance_failure() {
        let result = json!({
            "pass_rate": 0.6,
            "pass_threshold": 0.85,
            "passed": 6,
            "failed": 4,
            "generation_mode": "json_schema",
            "validation_mode": "strict",
            "errors": ["Case 2: output doesn't match pattern", "Case 4: invalid JSON"]
        });
        let cex = build_guidance_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "guidance_generation_failure");
        assert!(cex.failed_checks[0].description.contains("60.0%"));
        assert!(cex.failed_checks[0].description.contains("json_schema"));
        assert!(cex.witness.contains_key("generation_mode"));
    }

    #[test]
    fn test_nemo_success() {
        let result = json!({
            "pass_rate": 0.9,
            "pass_threshold": 0.85,
            "passed": 9,
            "failed": 1
        });
        assert!(build_nemo_guardrails_counterexample(&result).is_none());
    }

    #[test]
    fn test_nemo_failure() {
        let result = json!({
            "pass_rate": 0.5,
            "pass_threshold": 0.85,
            "passed": 4,
            "failed": 4,
            "rail_type": "input",
            "jailbreak_detection": true,
            "topical_rail": false,
            "fact_checking": true,
            "errors": ["Input 3: potential jailbreak detected", "Input 5: blocked"]
        });
        let cex = build_nemo_guardrails_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "nemo_guardrails_rail_failure"
        );
        assert!(cex.failed_checks[0].description.contains("50.0%"));
        assert!(cex.failed_checks[0].description.contains("input rail"));
        assert!(cex.failed_checks[0]
            .description
            .contains("jailbreak_detection"));
        assert!(cex.failed_checks[0].description.contains("fact_checking"));
        assert!(cex.witness.contains_key("rail_type"));
    }

    #[test]
    fn test_nemo_no_features() {
        let result = json!({
            "pass_rate": 0.6,
            "pass_threshold": 0.85,
            "passed": 6,
            "failed": 4,
            "rail_type": "output",
            "jailbreak_detection": false,
            "topical_rail": false,
            "fact_checking": false
        });
        let cex = build_nemo_guardrails_counterexample(&result).unwrap();
        // Should not mention features since all are disabled
        assert!(!cex.failed_checks[0].description.contains("Features:"));
    }
}
