//! Shared helpers for interpretability/explainability counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by interpretability backends
//! (SHAP, Captum, LIME, InterpretML, Alibi).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for SHAP explanations.
///
/// Expects a JSON object with fields like `mean_abs_shap`, `max_importance`,
/// `importance_threshold`, `stability_gap`, `top_feature`.
pub fn build_shap_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let mean_abs = result.get("mean_abs_shap").and_then(Value::as_f64)?;
    let threshold = result
        .get("importance_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let stability_gap = result.get("stability_gap").and_then(Value::as_f64);

    let stability_exceeded = stability_gap.is_some_and(|gap| gap > threshold);

    // Only create counterexample if mean_abs is below threshold or stability exceeded
    if mean_abs >= threshold && !stability_exceeded {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "mean_abs_shap".to_string(),
        CounterexampleValue::Float { value: mean_abs },
    );

    if let Some(max_imp) = result.get("max_importance").and_then(Value::as_f64) {
        witness.insert(
            "max_importance".to_string(),
            CounterexampleValue::Float { value: max_imp },
        );
    }

    if let Some(gap) = stability_gap {
        witness.insert(
            "stability_gap".to_string(),
            CounterexampleValue::Float { value: gap },
        );
    }

    if let Some(top_feature) = result.get("top_feature").and_then(Value::as_u64) {
        witness.insert(
            "top_feature".to_string(),
            CounterexampleValue::Int {
                value: top_feature as i128,
                type_hint: None,
            },
        );
    }

    let failed_checks = build_shap_failed_checks(result, mean_abs, threshold, stability_gap);
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

fn build_shap_failed_checks(
    result: &Value,
    mean_abs: f64,
    threshold: f64,
    stability_gap: Option<f64>,
) -> Vec<FailedCheck> {
    let explainer = result
        .get("explainer")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_type = result
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let top_feature = result.get("top_feature").and_then(Value::as_u64);

    let mut description = format!(
        "SHAP explanation verification failed using {} explainer on {} model.",
        explainer, model_type
    );

    description.push_str(&format!(
        " Mean absolute SHAP: {:.4}, threshold: {:.4}.",
        mean_abs, threshold
    ));

    if mean_abs < threshold {
        description.push_str(&format!(
            " Gap: {:.4} below threshold.",
            threshold - mean_abs
        ));
    }

    if let Some(gap) = stability_gap {
        description.push_str(&format!(" Stability gap: {:.4}.", gap));
        if gap > threshold {
            description.push_str(" Explanations are unstable across perturbations.");
        }
    }

    if let Some(feature) = top_feature {
        description.push_str(&format!(" Most important feature: index {}.", feature));
    }

    if let Some(model_score) = result.get("model_score").and_then(Value::as_f64) {
        description.push_str(&format!(" Model accuracy: {:.2}%.", model_score * 100.0));
    }

    vec![FailedCheck {
        check_id: "shap_explanation_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for LIME explanations.
///
/// Expects a JSON object with fields like `fidelity`, `coverage`, `threshold` (or `fidelity_threshold`),
/// `top_features`, `model_type`.
pub fn build_lime_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let fidelity = result.get("fidelity").and_then(Value::as_f64)?;
    let threshold = result
        .get("threshold")
        .or_else(|| result.get("fidelity_threshold"))
        .and_then(Value::as_f64)
        .unwrap_or(0.8);
    let coverage = result
        .get("coverage")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    // Only create counterexample if fidelity or coverage below thresholds
    if fidelity >= threshold && coverage >= 0.6 {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "fidelity".to_string(),
        CounterexampleValue::Float { value: fidelity },
    );

    witness.insert(
        "coverage".to_string(),
        CounterexampleValue::Float { value: coverage },
    );

    if let Some(num_features) = result.get("num_features").and_then(Value::as_u64) {
        witness.insert(
            "num_features".to_string(),
            CounterexampleValue::Int {
                value: num_features as i128,
                type_hint: None,
            },
        );
    }

    let failed_checks = build_lime_failed_checks(result, fidelity, threshold, coverage);
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

fn build_lime_failed_checks(
    result: &Value,
    fidelity: f64,
    threshold: f64,
    coverage: f64,
) -> Vec<FailedCheck> {
    let mode = result
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_type = result
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let kernel = result.get("kernel_width").and_then(Value::as_f64);

    let mut description = format!(
        "LIME explanation verification failed in {} mode for {} model.",
        mode, model_type
    );

    description.push_str(&format!(
        " Local fidelity: {:.4}, threshold: {:.4}.",
        fidelity, threshold
    ));

    if fidelity < threshold {
        description.push_str(&format!(
            " Fidelity gap: {:.4} below threshold.",
            threshold - fidelity
        ));
    }

    description.push_str(&format!(" Coverage: {:.2}%.", coverage * 100.0));
    if coverage < 0.6 {
        description.push_str(&format!(
            " Coverage gap: {:.2}% below 60% minimum.",
            (0.6 - coverage) * 100.0
        ));
    }

    if let Some(kw) = kernel {
        description.push_str(&format!(" Kernel width: {:.4}.", kw));
    }

    vec![FailedCheck {
        check_id: "lime_explanation_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for Captum explanations.
///
/// Expects a JSON object with fields like `attribution_mean`, `attribution_threshold`,
/// `stability_gap`, `method`, `model_type`.
pub fn build_captum_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    let attribution_mean = result.get("attribution_mean").and_then(Value::as_f64)?;
    let threshold = result
        .get("attribution_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let stability_gap = result.get("stability_gap").and_then(Value::as_f64);

    let stability_exceeded = stability_gap.is_some_and(|gap| gap > threshold * 3.0);

    // Only create counterexample if below threshold or stability exceeded
    if attribution_mean >= threshold && !stability_exceeded {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "attribution_mean".to_string(),
        CounterexampleValue::Float {
            value: attribution_mean,
        },
    );

    if let Some(max_attr) = result.get("attribution_max").and_then(Value::as_f64) {
        witness.insert(
            "attribution_max".to_string(),
            CounterexampleValue::Float { value: max_attr },
        );
    }

    if let Some(gap) = stability_gap {
        witness.insert(
            "stability_gap".to_string(),
            CounterexampleValue::Float { value: gap },
        );
    }

    if let Some(top_feature) = result.get("top_feature").and_then(Value::as_u64) {
        witness.insert(
            "top_feature".to_string(),
            CounterexampleValue::Int {
                value: top_feature as i128,
                type_hint: None,
            },
        );
    }

    let failed_checks =
        build_captum_failed_checks(result, attribution_mean, threshold, stability_gap);
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

fn build_captum_failed_checks(
    result: &Value,
    attribution_mean: f64,
    threshold: f64,
    stability_gap: Option<f64>,
) -> Vec<FailedCheck> {
    let method = result
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_type = result
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let top_feature = result.get("top_feature").and_then(Value::as_u64);

    let mut description = format!(
        "Captum attribution verification failed using {} method on {} model.",
        method, model_type
    );

    description.push_str(&format!(
        " Mean attribution: {:.4}, threshold: {:.4}.",
        attribution_mean, threshold
    ));

    if attribution_mean < threshold {
        description.push_str(&format!(
            " Gap: {:.4} below threshold.",
            threshold - attribution_mean
        ));
    }

    if let Some(gap) = stability_gap {
        description.push_str(&format!(" Stability gap: {:.4}.", gap));
        let stability_threshold = threshold * 3.0;
        if gap > stability_threshold {
            description.push_str(&format!(
                " Exceeds stability threshold {:.4}. Attributions are unstable.",
                stability_threshold
            ));
        }
    }

    if let Some(feature) = top_feature {
        description.push_str(&format!(" Most important feature: index {}.", feature));
    }

    vec![FailedCheck {
        check_id: "captum_attribution_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for InterpretML explanations.
///
/// Expects a JSON object with fields like `global_importance_mean` (or `mean_importance`),
/// `local_fidelity`, `threshold` (or `importance_threshold`), `model_type`.
pub fn build_interpretml_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    // Support both global_importance_mean and mean_importance
    let global_mean = result
        .get("global_importance_mean")
        .or_else(|| result.get("mean_importance"))
        .and_then(Value::as_f64)?;
    let threshold = result
        .get("threshold")
        .or_else(|| result.get("importance_threshold"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let local_fidelity = result.get("local_fidelity").and_then(Value::as_f64);

    let local_failed = local_fidelity.is_some_and(|f| f < 0.8);

    // Only create counterexample if below threshold or local fidelity failed
    if global_mean >= threshold && !local_failed {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "global_importance_mean".to_string(),
        CounterexampleValue::Float { value: global_mean },
    );

    if let Some(fidelity) = local_fidelity {
        witness.insert(
            "local_fidelity".to_string(),
            CounterexampleValue::Float { value: fidelity },
        );
    }

    if let Some(top_feature) = result.get("top_feature").and_then(Value::as_u64) {
        witness.insert(
            "top_feature".to_string(),
            CounterexampleValue::Int {
                value: top_feature as i128,
                type_hint: None,
            },
        );
    }

    let failed_checks =
        build_interpretml_failed_checks(result, global_mean, threshold, local_fidelity);
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

fn build_interpretml_failed_checks(
    result: &Value,
    global_mean: f64,
    threshold: f64,
    local_fidelity: Option<f64>,
) -> Vec<FailedCheck> {
    let mode = result
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_type = result
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let explainer = result
        .get("explainer")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let mut description = format!(
        "InterpretML explanation verification failed in {} mode using {} for {} model.",
        mode, explainer, model_type
    );

    description.push_str(&format!(
        " Global importance mean: {:.4}, threshold: {:.4}.",
        global_mean, threshold
    ));

    if global_mean < threshold {
        description.push_str(&format!(
            " Gap: {:.4} below threshold.",
            threshold - global_mean
        ));
    }

    if let Some(fidelity) = local_fidelity {
        description.push_str(&format!(" Local fidelity: {:.4}.", fidelity));
        if fidelity < 0.8 {
            description.push_str(&format!(
                " Local fidelity gap: {:.4} below 0.80 minimum.",
                0.8 - fidelity
            ));
        }
    }

    vec![FailedCheck {
        check_id: "interpretml_explanation_failure".to_string(),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for Alibi explanations.
///
/// Expects a JSON object with fields like `explanation_coverage` (or `coverage`),
/// `fidelity` (or `precision`), `threshold` (or `coverage_threshold`/`precision_threshold`),
/// `method`, `model_type`.
pub fn build_alibi_counterexample(result: &Value) -> Option<StructuredCounterexample> {
    // Support explanation_coverage and coverage
    let coverage = result
        .get("explanation_coverage")
        .or_else(|| result.get("coverage"))
        .and_then(Value::as_f64)?;
    // Support various threshold field names
    let threshold = result
        .get("threshold")
        .or_else(|| result.get("coverage_threshold"))
        .and_then(Value::as_f64)
        .unwrap_or(0.8);
    // Support fidelity and precision
    let fidelity = result
        .get("fidelity")
        .or_else(|| result.get("precision"))
        .and_then(Value::as_f64);
    let fidelity_threshold = result
        .get("precision_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(threshold);

    let fidelity_failed = fidelity.is_some_and(|f| f < fidelity_threshold);

    // Only create counterexample if coverage below threshold or fidelity failed
    if coverage >= threshold && !fidelity_failed {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "explanation_coverage".to_string(),
        CounterexampleValue::Float { value: coverage },
    );

    if let Some(f) = fidelity {
        witness.insert(
            "fidelity".to_string(),
            CounterexampleValue::Float { value: f },
        );
    }

    if let Some(precision) = result.get("precision").and_then(Value::as_f64) {
        witness.insert(
            "precision".to_string(),
            CounterexampleValue::Float { value: precision },
        );
    }

    let failed_checks = build_alibi_failed_checks(result, coverage, threshold, fidelity);
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

fn build_alibi_failed_checks(
    result: &Value,
    coverage: f64,
    threshold: f64,
    fidelity: Option<f64>,
) -> Vec<FailedCheck> {
    let method = result
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_type = result
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let explanation_type = result
        .get("explanation_type")
        .and_then(Value::as_str)
        .unwrap_or("attribution");

    let mut description = format!(
        "Alibi {} explanation verification failed using {} method on {} model.",
        explanation_type, method, model_type
    );

    description.push_str(&format!(
        " Coverage: {:.4}, threshold: {:.4}.",
        coverage, threshold
    ));

    if coverage < threshold {
        description.push_str(&format!(
            " Coverage gap: {:.4} below threshold.",
            threshold - coverage
        ));
    }

    if let Some(f) = fidelity {
        description.push_str(&format!(" Fidelity: {:.4}.", f));
        if f < threshold {
            description.push_str(&format!(
                " Fidelity gap: {:.4} below threshold.",
                threshold - f
            ));
        }
    }

    if let Some(precision) = result.get("precision").and_then(Value::as_f64) {
        description.push_str(&format!(" Precision: {:.4}.", precision));
    }

    vec![FailedCheck {
        check_id: "alibi_explanation_failure".to_string(),
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
    fn test_shap_counterexample_pass() {
        let result = json!({
            "mean_abs_shap": 0.15,
            "importance_threshold": 0.1,
            "max_importance": 0.25
        });
        assert!(build_shap_counterexample(&result).is_none());
    }

    #[test]
    fn test_shap_counterexample_fail_threshold() {
        let result = json!({
            "mean_abs_shap": 0.05,
            "importance_threshold": 0.1,
            "max_importance": 0.08,
            "top_feature": 2,
            "explainer": "tree",
            "model_type": "classification",
            "model_score": 0.92
        });
        let cex = build_shap_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "shap_explanation_failure");
        assert!(cex.failed_checks[0].description.contains("tree"));
        assert!(cex.failed_checks[0].description.contains("0.0500"));
        assert!(cex.witness.contains_key("mean_abs_shap"));
    }

    #[test]
    fn test_shap_counterexample_stability_exceeded() {
        let result = json!({
            "mean_abs_shap": 0.15,
            "importance_threshold": 0.1,
            "stability_gap": 0.2,
            "explainer": "kernel",
            "model_type": "regression"
        });
        let cex = build_shap_counterexample(&result).unwrap();
        assert!(cex.failed_checks[0].description.contains("unstable"));
    }

    #[test]
    fn test_lime_counterexample_pass() {
        let result = json!({
            "fidelity": 0.9,
            "threshold": 0.8,
            "coverage": 0.75
        });
        assert!(build_lime_counterexample(&result).is_none());
    }

    #[test]
    fn test_lime_counterexample_fail_fidelity() {
        let result = json!({
            "fidelity": 0.6,
            "threshold": 0.8,
            "coverage": 0.7,
            "mode": "tabular",
            "model_type": "classifier",
            "kernel_width": 0.75
        });
        let cex = build_lime_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "lime_explanation_failure");
        assert!(cex.failed_checks[0].description.contains("tabular"));
        assert!(cex.failed_checks[0].description.contains("0.6000"));
    }

    #[test]
    fn test_lime_counterexample_fail_coverage() {
        let result = json!({
            "fidelity": 0.85,
            "threshold": 0.8,
            "coverage": 0.5,
            "mode": "image",
            "model_type": "cnn"
        });
        let cex = build_lime_counterexample(&result).unwrap();
        assert!(cex.failed_checks[0].description.contains("Coverage gap"));
    }

    #[test]
    fn test_captum_counterexample_pass() {
        let result = json!({
            "attribution_mean": 0.12,
            "attribution_threshold": 0.1,
            "stability_gap": 0.15
        });
        assert!(build_captum_counterexample(&result).is_none());
    }

    #[test]
    fn test_captum_counterexample_fail() {
        let result = json!({
            "attribution_mean": 0.05,
            "attribution_threshold": 0.1,
            "attribution_max": 0.08,
            "stability_gap": 0.02,
            "top_feature": 3,
            "method": "integrated_gradients",
            "model_type": "neural_network"
        });
        let cex = build_captum_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "captum_attribution_failure");
        assert!(cex.failed_checks[0]
            .description
            .contains("integrated_gradients"));
    }

    #[test]
    fn test_interpretml_counterexample_pass() {
        let result = json!({
            "global_importance_mean": 0.15,
            "threshold": 0.1,
            "local_fidelity": 0.85
        });
        assert!(build_interpretml_counterexample(&result).is_none());
    }

    #[test]
    fn test_interpretml_counterexample_fail() {
        let result = json!({
            "global_importance_mean": 0.05,
            "threshold": 0.1,
            "local_fidelity": 0.7,
            "mode": "global",
            "model_type": "ebm",
            "explainer": "EBMExplainer"
        });
        let cex = build_interpretml_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(
            cex.failed_checks[0].check_id,
            "interpretml_explanation_failure"
        );
        assert!(cex.failed_checks[0].description.contains("EBMExplainer"));
    }

    #[test]
    fn test_alibi_counterexample_pass() {
        let result = json!({
            "explanation_coverage": 0.9,
            "threshold": 0.8,
            "fidelity": 0.85
        });
        assert!(build_alibi_counterexample(&result).is_none());
    }

    #[test]
    fn test_alibi_counterexample_fail() {
        let result = json!({
            "explanation_coverage": 0.6,
            "threshold": 0.8,
            "fidelity": 0.7,
            "precision": 0.75,
            "method": "AnchorTabular",
            "model_type": "classifier",
            "explanation_type": "anchor"
        });
        let cex = build_alibi_counterexample(&result).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert_eq!(cex.failed_checks[0].check_id, "alibi_explanation_failure");
        assert!(cex.failed_checks[0].description.contains("AnchorTabular"));
        assert!(cex.failed_checks[0].description.contains("anchor"));
    }
}
