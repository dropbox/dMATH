//! Shared helpers for neural network counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by NN verifiers.

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for neural robustness backends.
///
/// Expects a JSON object with a `counterexample` field containing fields like
/// `sample_index`, `original_input`, `counterexample`, and `true_label`.
pub fn build_nn_counterexample(
    backend: &str,
    result: &Value,
    verification_rate: f64,
) -> Option<StructuredCounterexample> {
    let cex = result.get("counterexample")?;

    let mut witness = HashMap::new();

    if let Some(orig) = cex
        .get("original_input")
        .and_then(Value::as_array)
        .and_then(|values| values_to_sequence(values))
    {
        witness.insert("original_input".to_string(), orig);
    }

    if let Some(adversarial) = cex
        .get("counterexample")
        .and_then(Value::as_array)
        .and_then(|values| values_to_sequence(values))
    {
        witness.insert("adversarial_input".to_string(), adversarial);
    }

    if let Some(true_label) = cex.get("true_label").and_then(Value::as_i64) {
        witness.insert(
            "true_label".to_string(),
            CounterexampleValue::Int {
                value: true_label as i128,
                type_hint: None,
            },
        );
    }

    if let Some(pred_label) = cex.get("predicted_label").and_then(Value::as_i64) {
        witness.insert(
            "predicted_label".to_string(),
            CounterexampleValue::Int {
                value: pred_label as i128,
                type_hint: None,
            },
        );
    }

    if let Some(sample_index) = cex.get("sample_index").and_then(Value::as_i64) {
        witness.insert(
            "sample_index".to_string(),
            CounterexampleValue::Int {
                value: sample_index as i128,
                type_hint: None,
            },
        );
    }

    if let Some(epsilon) = result.get("epsilon").and_then(Value::as_f64) {
        witness.insert(
            "epsilon".to_string(),
            CounterexampleValue::Float { value: epsilon },
        );
    }

    let failed_checks = build_failed_checks(backend, result, verification_rate, cex);
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

fn values_to_sequence(values: &[Value]) -> Option<CounterexampleValue> {
    let sequence: Vec<CounterexampleValue> = values
        .iter()
        .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
        .collect();

    if sequence.is_empty() {
        None
    } else {
        Some(CounterexampleValue::Sequence(sequence))
    }
}

fn build_failed_checks(
    backend: &str,
    result: &Value,
    verification_rate: f64,
    cex: &Value,
) -> Vec<FailedCheck> {
    let total = result
        .get("total_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let verified = result
        .get("verified_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let unverified = total.saturating_sub(verified);
    let sample_index = cex.get("sample_index").and_then(Value::as_i64);
    let epsilon = result.get("epsilon").and_then(Value::as_f64);

    let mut description = if total > 0 {
        format!(
            "{} verified {}/{} samples ({:.2}%).",
            backend,
            verified,
            total,
            verification_rate * 100.0
        )
    } else {
        format!(
            "{} reported verification rate {:.2}%.",
            backend,
            verification_rate * 100.0
        )
    };

    if unverified > 0 {
        description.push_str(&format!(" {} samples were unverified.", unverified));
    }

    if let Some(idx) = sample_index {
        description.push_str(&format!(" Counterexample at sample {}.", idx));
    }

    if let Some(true_label) = cex.get("true_label").and_then(Value::as_i64) {
        description.push_str(&format!(" True label {}.", true_label));
    }

    if let Some(pred_label) = cex.get("predicted_label").and_then(Value::as_i64) {
        description.push_str(&format!(" Predicted label {}.", pred_label));
    }

    if let Some(eps) = epsilon {
        description.push_str(&format!(" Epsilon {:.4}.", eps));
    }

    vec![FailedCheck {
        check_id: format!("{}_robustness_failure", backend.to_lowercase()),
        description,
        location: None,
        function: None,
    }]
}
