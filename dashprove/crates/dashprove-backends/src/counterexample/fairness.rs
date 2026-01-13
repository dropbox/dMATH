//! Shared helpers for fairness/bias counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by fairness backends
//! (Aequitas, AIF360, Fairlearn).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for Aequitas fairness audit.
///
/// Expects a JSON object with fields like `min_disparity_ratio`, `avg_disparity_ratio`,
/// `disparity_tolerance`, `fairness_metric`, and optionally `metrics_by_group`.
pub fn build_aequitas_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let is_fair = result
        .get("is_fair")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if is_fair {
        return None;
    }

    let mut witness = HashMap::new();

    if let Some(min_disp) = result.get("min_disparity_ratio").and_then(Value::as_f64) {
        witness.insert(
            "min_disparity_ratio".to_string(),
            CounterexampleValue::Float { value: min_disp },
        );
    }

    if let Some(avg_disp) = result.get("avg_disparity_ratio").and_then(Value::as_f64) {
        witness.insert(
            "avg_disparity_ratio".to_string(),
            CounterexampleValue::Float { value: avg_disp },
        );
    }

    if let Some(max_disp) = result.get("max_disparity_ratio").and_then(Value::as_f64) {
        witness.insert(
            "max_disparity_ratio".to_string(),
            CounterexampleValue::Float { value: max_disp },
        );
    }

    let tolerance = result
        .get("disparity_tolerance")
        .and_then(Value::as_f64)
        .unwrap_or(0.8);
    witness.insert(
        "disparity_tolerance".to_string(),
        CounterexampleValue::Float { value: tolerance },
    );

    let failed_checks = build_aequitas_failed_checks(result);
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

fn build_aequitas_failed_checks(result: &Value) -> Vec<FailedCheck> {
    let fairness_metric = result
        .get("fairness_metric")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let min_disp = result.get("min_disparity_ratio").and_then(Value::as_f64);
    let avg_disp = result.get("avg_disparity_ratio").and_then(Value::as_f64);
    let tolerance = result
        .get("disparity_tolerance")
        .and_then(Value::as_f64)
        .unwrap_or(0.8);

    let mut description = format!(
        "Aequitas bias audit failed for {} fairness metric.",
        fairness_metric
    );

    if let Some(min) = min_disp {
        description.push_str(&format!(" Minimum disparity ratio: {:.4}.", min));
    }

    if let Some(avg) = avg_disp {
        description.push_str(&format!(" Average disparity ratio: {:.4}.", avg));
    }

    description.push_str(&format!(" Required tolerance: {:.2}.", tolerance));

    if let Some(min) = min_disp {
        if min < tolerance {
            description.push_str(&format!(" Gap: {:.4} below threshold.", tolerance - min));
        }
    }

    // Add group-specific failures if available
    if let Some(metrics_by_group) = result.get("metrics_by_group").and_then(Value::as_object) {
        let failing_groups: Vec<_> = metrics_by_group
            .iter()
            .filter_map(|(group, metrics)| {
                let ppr_disp = metrics.get("ppr_disparity").and_then(Value::as_f64);
                if let Some(disp) = ppr_disp {
                    if disp < tolerance {
                        return Some(format!("{} (ppr_disparity={:.4})", group, disp));
                    }
                }
                None
            })
            .collect();

        if !failing_groups.is_empty() {
            description.push_str(&format!(" Failing groups: {}.", failing_groups.join(", ")));
        }
    }

    vec![FailedCheck {
        check_id: "aequitas_fairness_violation".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for AIF360 bias assessment.
///
/// Expects a JSON object with fields like `primary_metric_value`,
/// `prediction_statistical_parity_diff`, `prediction_disparate_impact`,
/// `average_odds_difference`, `equal_opportunity_difference`, etc.
pub fn build_aif360_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let is_fair = result
        .get("is_fair")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if is_fair {
        return None;
    }

    let mut witness = HashMap::new();

    if let Some(val) = result.get("primary_metric_value").and_then(Value::as_f64) {
        witness.insert(
            "primary_metric_value".to_string(),
            CounterexampleValue::Float { value: val },
        );
    }

    if let Some(spd) = result
        .get("prediction_statistical_parity_diff")
        .and_then(Value::as_f64)
    {
        witness.insert(
            "statistical_parity_diff".to_string(),
            CounterexampleValue::Float { value: spd },
        );
    }

    if let Some(di) = result
        .get("prediction_disparate_impact")
        .and_then(Value::as_f64)
    {
        witness.insert(
            "disparate_impact".to_string(),
            CounterexampleValue::Float { value: di },
        );
    }

    if let Some(aod) = result
        .get("average_odds_difference")
        .and_then(Value::as_f64)
    {
        witness.insert(
            "average_odds_difference".to_string(),
            CounterexampleValue::Float { value: aod },
        );
    }

    if let Some(eod) = result
        .get("equal_opportunity_difference")
        .and_then(Value::as_f64)
    {
        witness.insert(
            "equal_opportunity_difference".to_string(),
            CounterexampleValue::Float { value: eod },
        );
    }

    let failed_checks = build_aif360_failed_checks(result);
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

fn build_aif360_failed_checks(result: &Value) -> Vec<FailedCheck> {
    let bias_metric = result
        .get("bias_metric")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let primary_val = result.get("primary_metric_value").and_then(Value::as_f64);
    let fairness_threshold = result
        .get("fairness_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1);
    let di_threshold = result
        .get("disparate_impact_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.8);
    let spd = result
        .get("prediction_statistical_parity_diff")
        .and_then(Value::as_f64);
    let di = result
        .get("prediction_disparate_impact")
        .and_then(Value::as_f64);
    let aod = result
        .get("average_odds_difference")
        .and_then(Value::as_f64);
    let eod = result
        .get("equal_opportunity_difference")
        .and_then(Value::as_f64);

    let mut description = format!("AIF360 bias assessment failed for {} metric.", bias_metric);

    if let Some(val) = primary_val {
        description.push_str(&format!(" Primary metric value: {:.4}.", val));
    }

    // Add threshold info based on metric type
    if bias_metric == "disparate_impact" {
        description.push_str(&format!(
            " Required disparate impact >= {:.2}.",
            di_threshold
        ));
        if let Some(di_val) = di {
            if di_val < di_threshold {
                description.push_str(&format!(
                    " Gap: {:.4} below threshold.",
                    di_threshold - di_val
                ));
            }
        }
    } else {
        description.push_str(&format!(" Required |metric| <= {:.2}.", fairness_threshold));
        if let Some(val) = primary_val {
            if val.abs() > fairness_threshold {
                description.push_str(&format!(
                    " Exceeds threshold by {:.4}.",
                    val.abs() - fairness_threshold
                ));
            }
        }
    }

    // Add additional metric details
    let mut metrics_detail = Vec::new();
    if let Some(spd_val) = spd {
        metrics_detail.push(format!("SPD={:.4}", spd_val));
    }
    if let Some(di_val) = di {
        metrics_detail.push(format!("DI={:.4}", di_val));
    }
    if let Some(aod_val) = aod {
        metrics_detail.push(format!("AOD={:.4}", aod_val));
    }
    if let Some(eod_val) = eod {
        metrics_detail.push(format!("EOD={:.4}", eod_val));
    }

    if !metrics_detail.is_empty() {
        description.push_str(&format!(" Metrics: [{}].", metrics_detail.join(", ")));
    }

    vec![FailedCheck {
        check_id: "aif360_bias_violation".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for Fairlearn fairness assessment.
///
/// Expects a JSON object with fields like `primary_metric_value`,
/// `demographic_parity_diff`, `equalized_odds_diff`, `fairness_threshold`, etc.
pub fn build_fairlearn_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let is_fair = result
        .get("is_fair")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if is_fair {
        return None;
    }

    let mut witness = HashMap::new();

    let primary_metric = result
        .get("primary_metric_value")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    witness.insert(
        "primary_metric_value".to_string(),
        CounterexampleValue::Float {
            value: primary_metric,
        },
    );

    let threshold = result
        .get("fairness_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1);
    witness.insert(
        "fairness_threshold".to_string(),
        CounterexampleValue::Float { value: threshold },
    );

    if let Some(dp_diff) = result
        .get("demographic_parity_diff")
        .and_then(Value::as_f64)
    {
        witness.insert(
            "demographic_parity_diff".to_string(),
            CounterexampleValue::Float { value: dp_diff },
        );
    }

    if let Some(eq_odds_diff) = result.get("equalized_odds_diff").and_then(Value::as_f64) {
        witness.insert(
            "equalized_odds_diff".to_string(),
            CounterexampleValue::Float {
                value: eq_odds_diff,
            },
        );
    }

    if let Some(accuracy) = result.get("accuracy").and_then(Value::as_f64) {
        witness.insert(
            "model_accuracy".to_string(),
            CounterexampleValue::Float { value: accuracy },
        );
    }

    let failed_checks = build_fairlearn_failed_checks(result);
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

fn build_fairlearn_failed_checks(result: &Value) -> Vec<FailedCheck> {
    let fairness_metric = result
        .get("fairness_metric")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let primary_val = result
        .get("primary_metric_value")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let threshold = result
        .get("fairness_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1);
    let dp_diff = result
        .get("demographic_parity_diff")
        .and_then(Value::as_f64);
    let eq_odds_diff = result.get("equalized_odds_diff").and_then(Value::as_f64);
    let accuracy = result.get("accuracy").and_then(Value::as_f64);

    let mut description = format!(
        "Fairlearn assessment failed for {} metric.",
        fairness_metric
    );

    description.push_str(&format!(
        " Primary metric value: {:.4}, threshold: {:.2}.",
        primary_val, threshold
    ));

    if primary_val.abs() > threshold {
        description.push_str(&format!(
            " Exceeds threshold by {:.4}.",
            primary_val.abs() - threshold
        ));
    }

    // Add additional metric details
    let mut metrics_detail = Vec::new();
    if let Some(dp) = dp_diff {
        metrics_detail.push(format!("demographic_parity_diff={:.4}", dp));
    }
    if let Some(eo) = eq_odds_diff {
        metrics_detail.push(format!("equalized_odds_diff={:.4}", eo));
    }
    if let Some(acc) = accuracy {
        metrics_detail.push(format!("accuracy={:.2}%", acc * 100.0));
    }

    if !metrics_detail.is_empty() {
        description.push_str(&format!(
            " Additional metrics: [{}].",
            metrics_detail.join(", ")
        ));
    }

    vec![FailedCheck {
        check_id: "fairlearn_fairness_violation".to_string(),
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
    fn test_aequitas_counterexample_fair() {
        let result = json!({
            "is_fair": true,
            "min_disparity_ratio": 0.9,
            "disparity_tolerance": 0.8
        });
        assert!(build_aequitas_counterexample(&result).is_none());
    }

    #[test]
    fn test_aequitas_counterexample_unfair() {
        let result = json!({
            "is_fair": false,
            "min_disparity_ratio": 0.6,
            "avg_disparity_ratio": 0.7,
            "disparity_tolerance": 0.8,
            "fairness_metric": "predictive_parity"
        });
        let cex = build_aequitas_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "aequitas_fairness_violation");
        assert!(cex.failed_checks[0]
            .description
            .contains("predictive_parity"));
        assert!(cex.failed_checks[0].description.contains("0.6"));
        assert!(cex.witness.contains_key("min_disparity_ratio"));
    }

    #[test]
    fn test_aif360_counterexample_fair() {
        let result = json!({
            "is_fair": true,
            "primary_metric_value": 0.05
        });
        assert!(build_aif360_counterexample(&result).is_none());
    }

    #[test]
    fn test_aif360_counterexample_unfair() {
        let result = json!({
            "is_fair": false,
            "bias_metric": "statistical_parity_difference",
            "primary_metric_value": 0.25,
            "prediction_statistical_parity_diff": 0.25,
            "prediction_disparate_impact": 0.7,
            "fairness_threshold": 0.1
        });
        let cex = build_aif360_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "aif360_bias_violation");
        assert!(cex.failed_checks[0]
            .description
            .contains("statistical_parity_difference"));
        assert!(cex.witness.contains_key("primary_metric_value"));
        assert!(cex.witness.contains_key("statistical_parity_diff"));
    }

    #[test]
    fn test_aif360_disparate_impact_failure() {
        let result = json!({
            "is_fair": false,
            "bias_metric": "disparate_impact",
            "primary_metric_value": 0.65,
            "prediction_disparate_impact": 0.65,
            "disparate_impact_threshold": 0.8
        });
        let cex = build_aif360_counterexample(&result).unwrap();
        assert!(cex.failed_checks[0]
            .description
            .contains("disparate_impact"));
        assert!(cex.failed_checks[0].description.contains(">= 0.80"));
    }

    #[test]
    fn test_fairlearn_counterexample_fair() {
        let result = json!({
            "is_fair": true,
            "primary_metric_value": 0.05
        });
        assert!(build_fairlearn_counterexample(&result).is_none());
    }

    #[test]
    fn test_fairlearn_counterexample_unfair() {
        let result = json!({
            "is_fair": false,
            "fairness_metric": "demographic_parity",
            "primary_metric_value": 0.25,
            "demographic_parity_diff": 0.25,
            "fairness_threshold": 0.1,
            "accuracy": 0.85
        });
        let cex = build_fairlearn_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "fairlearn_fairness_violation"
        );
        assert!(cex.failed_checks[0]
            .description
            .contains("demographic_parity"));
        assert!(cex.failed_checks[0].description.contains("0.25"));
        assert!(cex.witness.contains_key("primary_metric_value"));
        assert!(cex.witness.contains_key("demographic_parity_diff"));
        assert!(cex.witness.contains_key("model_accuracy"));
    }
}
