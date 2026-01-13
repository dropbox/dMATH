//! Shared helpers for data quality counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by data quality backends
//! (Great Expectations, WhyLogs, Evidently, Deepchecks).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for Great Expectations validation.
///
/// Expects a JSON object with fields like `success_rate`, `expectations_passed`,
/// `expectations_failed`, and optionally `validation_results`.
pub fn build_great_expectations_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let success_rate = result
        .get("success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    if success_rate >= 1.0 {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "success_rate".to_string(),
        CounterexampleValue::Float {
            value: success_rate,
        },
    );

    if let Some(passed) = result.get("expectations_passed").and_then(Value::as_u64) {
        witness.insert(
            "expectations_passed".to_string(),
            CounterexampleValue::Int {
                value: passed as i128,
                type_hint: None,
            },
        );
    }

    if let Some(failed) = result.get("expectations_failed").and_then(Value::as_u64) {
        witness.insert(
            "expectations_failed".to_string(),
            CounterexampleValue::Int {
                value: failed as i128,
                type_hint: None,
            },
        );
    }

    // Extract failed expectation names
    let failed_expectations: Vec<String> = result
        .get("validation_results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter(|r| r.get("success").and_then(Value::as_bool) == Some(false))
                .filter_map(|r| {
                    r.get("expectation")
                        .and_then(Value::as_str)
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();

    if !failed_expectations.is_empty() {
        witness.insert(
            "failed_expectations".to_string(),
            CounterexampleValue::String(failed_expectations.join(", ")),
        );
    }

    let failed_checks = build_great_expectations_failed_checks(result, &failed_expectations);
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

fn build_great_expectations_failed_checks(
    result: &Value,
    failed_expectations: &[String],
) -> Vec<FailedCheck> {
    let success_rate = result
        .get("success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let passed = result
        .get("expectations_passed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let failed = result
        .get("expectations_failed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let validation_level = result
        .get("validation_level")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let mut description = format!(
        "Great Expectations validation failed at {} level.",
        validation_level
    );

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{} expectations).",
        success_rate * 100.0,
        passed,
        passed + failed
    ));

    if !failed_expectations.is_empty() {
        let display_list: Vec<&str> = failed_expectations
            .iter()
            .take(5)
            .map(|s| s.as_str())
            .collect();
        description.push_str(&format!(" Failed: [{}]", display_list.join(", ")));
        if failed_expectations.len() > 5 {
            description.push_str(&format!(" and {} more.", failed_expectations.len() - 5));
        } else {
            description.push('.');
        }
    }

    vec![FailedCheck {
        check_id: "great_expectations_validation_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for WhyLogs constraint validation.
///
/// Expects a JSON object with fields like `success_rate`, `constraints_passed`,
/// `constraints_failed`, and optionally `constraint_results`.
pub fn build_whylogs_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let success_rate = result
        .get("success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    if success_rate >= 1.0 {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "success_rate".to_string(),
        CounterexampleValue::Float {
            value: success_rate,
        },
    );

    if let Some(passed) = result.get("constraints_passed").and_then(Value::as_u64) {
        witness.insert(
            "constraints_passed".to_string(),
            CounterexampleValue::Int {
                value: passed as i128,
                type_hint: None,
            },
        );
    }

    if let Some(failed) = result.get("constraints_failed").and_then(Value::as_u64) {
        witness.insert(
            "constraints_failed".to_string(),
            CounterexampleValue::Int {
                value: failed as i128,
                type_hint: None,
            },
        );
    }

    // Extract failed constraint names
    let failed_constraints: Vec<String> = result
        .get("constraint_results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter(|r| r.get("passed").and_then(Value::as_bool) == Some(false))
                .filter_map(|r| {
                    let feature = r.get("feature").and_then(Value::as_str)?;
                    let constraint = r.get("constraint").and_then(Value::as_str)?;
                    Some(format!("{}:{}", feature, constraint))
                })
                .collect()
        })
        .unwrap_or_default();

    if !failed_constraints.is_empty() {
        witness.insert(
            "failed_constraints".to_string(),
            CounterexampleValue::String(failed_constraints.join(", ")),
        );
    }

    let failed_checks = build_whylogs_failed_checks(result, &failed_constraints);
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

fn build_whylogs_failed_checks(result: &Value, failed_constraints: &[String]) -> Vec<FailedCheck> {
    let success_rate = result
        .get("success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let passed = result
        .get("constraints_passed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let failed = result
        .get("constraints_failed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let profile_type = result
        .get("profile_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let constraint_type = result
        .get("constraint_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let mut description = format!(
        "WhyLogs profiling failed for {} profile with {} constraints.",
        profile_type, constraint_type
    );

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{} constraints).",
        success_rate * 100.0,
        passed,
        passed + failed
    ));

    if !failed_constraints.is_empty() {
        let display_list: Vec<&str> = failed_constraints
            .iter()
            .take(5)
            .map(|s| s.as_str())
            .collect();
        description.push_str(&format!(" Failed: [{}]", display_list.join(", ")));
        if failed_constraints.len() > 5 {
            description.push_str(&format!(" and {} more.", failed_constraints.len() - 5));
        } else {
            description.push('.');
        }
    }

    vec![FailedCheck {
        check_id: "whylogs_constraint_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for Evidently drift detection.
///
/// Expects a JSON object with fields like `drift_detected`, `drift_score`,
/// `drift_threshold`, and optionally `drifted_features`.
pub fn build_evidently_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let drift_detected = result
        .get("drift_detected")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let drift_score = result
        .get("drift_score")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let drift_threshold = result
        .get("drift_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1);

    if !drift_detected && drift_score <= drift_threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "drift_detected".to_string(),
        CounterexampleValue::Bool(drift_detected),
    );

    witness.insert(
        "drift_score".to_string(),
        CounterexampleValue::Float { value: drift_score },
    );

    witness.insert(
        "drift_threshold".to_string(),
        CounterexampleValue::Float {
            value: drift_threshold,
        },
    );

    let drifted_features: Vec<String> = result
        .get("drifted_features")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|f| f.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if !drifted_features.is_empty() {
        witness.insert(
            "drifted_features".to_string(),
            CounterexampleValue::String(drifted_features.join(", ")),
        );
    }

    let failed_checks = build_evidently_failed_checks(result, &drifted_features);
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

fn build_evidently_failed_checks(result: &Value, drifted_features: &[String]) -> Vec<FailedCheck> {
    let drift_score = result
        .get("drift_score")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let drift_threshold = result
        .get("drift_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1);
    let report_type = result
        .get("report_type")
        .and_then(Value::as_str)
        .unwrap_or("data_drift");
    let total_features = result
        .get("total_features")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let mut description = format!("Evidently {} report detected drift.", report_type);

    description.push_str(&format!(
        " Drift score: {:.1}% (threshold: {:.1}%).",
        drift_score * 100.0,
        drift_threshold * 100.0
    ));

    if drift_score > drift_threshold {
        description.push_str(&format!(
            " Exceeds threshold by {:.1}%.",
            (drift_score - drift_threshold) * 100.0
        ));
    }

    if !drifted_features.is_empty() {
        description.push_str(&format!(
            " Drifted features: {}/{} [{}].",
            drifted_features.len(),
            total_features,
            drifted_features
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
        if drifted_features.len() > 5 {
            description.push_str(&format!(" and {} more.", drifted_features.len() - 5));
        }
    }

    vec![FailedCheck {
        check_id: "evidently_drift_detected".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for Deepchecks validation.
///
/// Expects a JSON object with fields like `success_rate`, `checks_passed`,
/// `checks_failed`, and optionally `check_results`.
pub fn build_deepchecks_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let success_rate = result
        .get("success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    if success_rate >= 1.0 {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "success_rate".to_string(),
        CounterexampleValue::Float {
            value: success_rate,
        },
    );

    if let Some(passed) = result.get("checks_passed").and_then(Value::as_u64) {
        witness.insert(
            "checks_passed".to_string(),
            CounterexampleValue::Int {
                value: passed as i128,
                type_hint: None,
            },
        );
    }

    if let Some(failed) = result.get("checks_failed").and_then(Value::as_u64) {
        witness.insert(
            "checks_failed".to_string(),
            CounterexampleValue::Int {
                value: failed as i128,
                type_hint: None,
            },
        );
    }

    // Extract failed check names
    let failed_check_names: Vec<String> = result
        .get("check_results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter(|r| r.get("passed").and_then(Value::as_bool) == Some(false))
                .filter_map(|r| r.get("check").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if !failed_check_names.is_empty() {
        witness.insert(
            "failed_checks".to_string(),
            CounterexampleValue::String(failed_check_names.join(", ")),
        );
    }

    let failed_checks = build_deepchecks_failed_checks(result, &failed_check_names);
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

fn build_deepchecks_failed_checks(
    result: &Value,
    failed_check_names: &[String],
) -> Vec<FailedCheck> {
    let success_rate = result
        .get("success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let passed = result
        .get("checks_passed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let failed = result
        .get("checks_failed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let suite_type = result
        .get("suite_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let task_type = result
        .get("task_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let mut description = format!(
        "Deepchecks {} suite failed for {} task.",
        suite_type, task_type
    );

    description.push_str(&format!(
        " Pass rate: {:.1}% ({}/{} checks).",
        success_rate * 100.0,
        passed,
        passed + failed
    ));

    if !failed_check_names.is_empty() {
        let display_list: Vec<&str> = failed_check_names
            .iter()
            .take(5)
            .map(|s| s.as_str())
            .collect();
        description.push_str(&format!(" Failed: [{}]", display_list.join(", ")));
        if failed_check_names.len() > 5 {
            description.push_str(&format!(" and {} more.", failed_check_names.len() - 5));
        } else {
            description.push('.');
        }
    }

    vec![FailedCheck {
        check_id: "deepchecks_validation_failure".to_string(),
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
    fn test_great_expectations_success() {
        let result = json!({
            "success_rate": 1.0,
            "expectations_passed": 10,
            "expectations_failed": 0
        });
        assert!(build_great_expectations_counterexample(&result).is_none());
    }

    #[test]
    fn test_great_expectations_failure() {
        let result = json!({
            "success_rate": 0.7,
            "expectations_passed": 7,
            "expectations_failed": 3,
            "validation_level": "standard",
            "validation_results": [
                {"expectation": "expect_column_to_exist", "success": true},
                {"expectation": "expect_column_values_to_be_unique", "success": false},
                {"expectation": "expect_column_mean_to_be_between", "success": false}
            ]
        });
        let cex = build_great_expectations_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "great_expectations_validation_failure"
        );
        assert!(cex.failed_checks[0].description.contains("70.0%"));
        assert!(cex.failed_checks[0].description.contains("standard"));
        assert!(cex.witness.contains_key("success_rate"));
        assert!(cex.witness.contains_key("failed_expectations"));
    }

    #[test]
    fn test_whylogs_success() {
        let result = json!({
            "success_rate": 1.0,
            "constraints_passed": 5,
            "constraints_failed": 0
        });
        assert!(build_whylogs_counterexample(&result).is_none());
    }

    #[test]
    fn test_whylogs_failure() {
        let result = json!({
            "success_rate": 0.6,
            "constraints_passed": 3,
            "constraints_failed": 2,
            "profile_type": "standard",
            "constraint_type": "value",
            "constraint_results": [
                {"feature": "value", "constraint": "greater_than_number", "passed": true},
                {"feature": "value", "constraint": "smaller_than_number", "passed": false},
                {"feature": "count", "constraint": "is_in_range", "passed": false}
            ]
        });
        let cex = build_whylogs_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "whylogs_constraint_failure");
        assert!(cex.failed_checks[0].description.contains("60.0%"));
        assert!(cex.failed_checks[0]
            .description
            .contains("standard profile"));
        assert!(cex.witness.contains_key("failed_constraints"));
    }

    #[test]
    fn test_evidently_no_drift() {
        let result = json!({
            "drift_detected": false,
            "drift_score": 0.05,
            "drift_threshold": 0.1
        });
        assert!(build_evidently_counterexample(&result).is_none());
    }

    #[test]
    fn test_evidently_drift_detected() {
        let result = json!({
            "drift_detected": true,
            "drift_score": 0.25,
            "drift_threshold": 0.1,
            "report_type": "data_drift",
            "total_features": 5,
            "drifted_features": ["feature_1", "feature_2"]
        });
        let cex = build_evidently_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "evidently_drift_detected");
        assert!(cex.failed_checks[0].description.contains("25.0%"));
        assert!(cex.failed_checks[0].description.contains("feature_1"));
        assert!(cex.witness.contains_key("drift_score"));
        assert!(cex.witness.contains_key("drifted_features"));
    }

    #[test]
    fn test_deepchecks_success() {
        let result = json!({
            "success_rate": 1.0,
            "checks_passed": 8,
            "checks_failed": 0
        });
        assert!(build_deepchecks_counterexample(&result).is_none());
    }

    #[test]
    fn test_deepchecks_failure() {
        let result = json!({
            "success_rate": 0.75,
            "checks_passed": 6,
            "checks_failed": 2,
            "suite_type": "data_integrity",
            "task_type": "binary",
            "check_results": [
                {"check": "DataDuplicates", "passed": true},
                {"check": "MixedNulls", "passed": false},
                {"check": "FeatureFeatureCorrelation", "passed": false}
            ]
        });
        let cex = build_deepchecks_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "deepchecks_validation_failure"
        );
        assert!(cex.failed_checks[0].description.contains("75.0%"));
        assert!(cex.failed_checks[0].description.contains("data_integrity"));
        assert!(cex.failed_checks[0].description.contains("binary"));
        assert!(cex.witness.contains_key("success_rate"));
        assert!(cex.witness.contains_key("failed_checks"));
    }
}
