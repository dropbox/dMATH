//! Shared helpers for adversarial robustness counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by robustness backends
//! (ART, Foolbox, CleverHans, TextAttack, RobustBench, AutoLiRPA, NNEnum, NNV, ERAN, Marabou).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for adversarial attack backends.
///
/// Expects a JSON object with fields like `attack_success_rate`, `robust_accuracy`,
/// `clean_accuracy`, `epsilon`, `attack_type`, and optionally `adversarial_example`.
pub fn build_adversarial_attack_counterexample(
    result: &Value,
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    let attack_success_rate = result
        .get("attack_success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    // Consider robust if attack success rate is very low
    if attack_success_rate < 0.01 {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "attack_success_rate".to_string(),
        CounterexampleValue::Float {
            value: attack_success_rate,
        },
    );

    if let Some(robust_acc) = result.get("robust_accuracy").and_then(Value::as_f64) {
        witness.insert(
            "robust_accuracy".to_string(),
            CounterexampleValue::Float { value: robust_acc },
        );
    }

    if let Some(clean_acc) = result.get("clean_accuracy").and_then(Value::as_f64) {
        witness.insert(
            "clean_accuracy".to_string(),
            CounterexampleValue::Float { value: clean_acc },
        );
    }

    if let Some(epsilon) = result.get("epsilon").and_then(Value::as_f64) {
        witness.insert(
            "epsilon".to_string(),
            CounterexampleValue::Float { value: epsilon },
        );
    }

    if let Some(num_adv) = result
        .get("num_adversarial_examples")
        .and_then(Value::as_u64)
    {
        witness.insert(
            "num_adversarial_examples".to_string(),
            CounterexampleValue::Int {
                value: num_adv as i128,
                type_hint: None,
            },
        );
    }

    if let Some(max_perturb) = result.get("max_perturbation").and_then(Value::as_f64) {
        witness.insert(
            "max_perturbation".to_string(),
            CounterexampleValue::Float { value: max_perturb },
        );
    }

    // Add adversarial example details if present
    if let Some(adv_example) = result.get("adversarial_example") {
        if let Some(orig_input) = adv_example.get("original_input").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = orig_input
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            witness.insert(
                "original_input".to_string(),
                CounterexampleValue::Sequence(values),
            );
        }
        if let Some(adv_input) = adv_example
            .get("adversarial_input")
            .and_then(Value::as_array)
        {
            let values: Vec<CounterexampleValue> = adv_input
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            witness.insert(
                "adversarial_input".to_string(),
                CounterexampleValue::Sequence(values),
            );
        }
        if let Some(orig_pred) = adv_example
            .get("original_prediction")
            .and_then(Value::as_i64)
        {
            witness.insert(
                "original_prediction".to_string(),
                CounterexampleValue::Int {
                    value: orig_pred as i128,
                    type_hint: None,
                },
            );
        }
        if let Some(adv_pred) = adv_example
            .get("adversarial_prediction")
            .and_then(Value::as_i64)
        {
            witness.insert(
                "adversarial_prediction".to_string(),
                CounterexampleValue::Int {
                    value: adv_pred as i128,
                    type_hint: None,
                },
            );
        }
        if let Some(perturb_norm) = adv_example.get("perturbation_norm").and_then(Value::as_f64) {
            witness.insert(
                "perturbation_norm".to_string(),
                CounterexampleValue::Float {
                    value: perturb_norm,
                },
            );
        }
    }

    let failed_checks = build_adversarial_failed_checks(result, backend_name);
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

fn build_adversarial_failed_checks(result: &Value, backend_name: &str) -> Vec<FailedCheck> {
    let attack_success_rate = result
        .get("attack_success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let robust_accuracy = result
        .get("robust_accuracy")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let clean_accuracy = result
        .get("clean_accuracy")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let epsilon = result.get("epsilon").and_then(Value::as_f64).unwrap_or(0.0);
    let attack_type = result
        .get("attack_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let num_adversarial = result
        .get("num_adversarial_examples")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let num_samples = result
        .get("num_samples")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let mut description = format!("{} adversarial robustness test failed.", backend_name);

    description.push_str(&format!(
        " Attack: {} with epsilon={:.4}.",
        attack_type, epsilon
    ));

    description.push_str(&format!(
        " Attack success rate: {:.1}% ({}/{} samples).",
        attack_success_rate * 100.0,
        num_adversarial,
        num_samples
    ));

    description.push_str(&format!(
        " Accuracy: clean={:.1}%, robust={:.1}%.",
        clean_accuracy * 100.0,
        robust_accuracy * 100.0
    ));

    // Calculate robustness gap
    let accuracy_drop = clean_accuracy - robust_accuracy;
    if accuracy_drop > 0.0 {
        description.push_str(&format!(" Accuracy drop: {:.1}%.", accuracy_drop * 100.0));
    }

    // Add perturbation info if available
    if let Some(max_perturb) = result.get("max_perturbation").and_then(Value::as_f64) {
        description.push_str(&format!(" Max perturbation: {:.6}.", max_perturb));
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_robustness_failure",
            backend_name.to_lowercase().replace(' ', "_")
        ),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for formal NN verification backends (ERAN, Marabou, AutoLiRPA).
///
/// Expects a JSON object with fields like `verified`, `verification_result`,
/// `property_violated`, `counterexample`, and optionally `bounds`.
pub fn build_nn_verification_counterexample(
    result: &Value,
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    // Check for verified field or verification_rate
    let verified = if let Some(v) = result.get("verified").and_then(Value::as_bool) {
        v
    } else if let Some(rate) = result.get("verification_rate").and_then(Value::as_f64) {
        rate >= 0.99
    } else {
        true
    };

    // If fully verified, no counterexample needed
    if verified {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert("verified".to_string(), CounterexampleValue::Bool(verified));

    // Add verification_rate if present
    if let Some(rate) = result.get("verification_rate").and_then(Value::as_f64) {
        witness.insert(
            "verification_rate".to_string(),
            CounterexampleValue::Float { value: rate },
        );
    }

    if let Some(property) = result.get("property_violated").and_then(Value::as_str) {
        witness.insert(
            "property_violated".to_string(),
            CounterexampleValue::String(property.to_string()),
        );
    }

    if let Some(epsilon) = result.get("epsilon").and_then(Value::as_f64) {
        witness.insert(
            "epsilon".to_string(),
            CounterexampleValue::Float { value: epsilon },
        );
    }

    // Add counterexample input/output if present
    if let Some(cex) = result.get("counterexample") {
        if let Some(input) = cex.get("input").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = input
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            if !values.is_empty() {
                witness.insert(
                    "counterexample_input".to_string(),
                    CounterexampleValue::Sequence(values),
                );
            }
        }
        if let Some(output) = cex.get("output").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = output
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            if !values.is_empty() {
                witness.insert(
                    "counterexample_output".to_string(),
                    CounterexampleValue::Sequence(values),
                );
            }
        }
        // Support for AutoLiRPA/NNEnum style counterexamples
        if let Some(orig_input) = cex.get("original_input").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = orig_input
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            if !values.is_empty() {
                witness.insert(
                    "original_input".to_string(),
                    CounterexampleValue::Sequence(values),
                );
            }
        }
        if let Some(lb) = cex.get("lower_bounds").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = lb
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            if !values.is_empty() {
                witness.insert(
                    "lower_bounds".to_string(),
                    CounterexampleValue::Sequence(values),
                );
            }
        }
        if let Some(ub) = cex.get("upper_bounds").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = ub
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            if !values.is_empty() {
                witness.insert(
                    "upper_bounds".to_string(),
                    CounterexampleValue::Sequence(values),
                );
            }
        }
        if let Some(cex_data) = cex.get("counterexample").and_then(Value::as_array) {
            let values: Vec<CounterexampleValue> = cex_data
                .iter()
                .filter_map(|v| v.as_f64().map(|f| CounterexampleValue::Float { value: f }))
                .collect();
            if !values.is_empty() {
                witness.insert(
                    "counterexample_input".to_string(),
                    CounterexampleValue::Sequence(values),
                );
            }
        }
        // Support for nnenum true_label
        if let Some(label) = cex.get("true_label").and_then(Value::as_i64) {
            witness.insert(
                "true_label".to_string(),
                CounterexampleValue::Int {
                    value: label as i128,
                    type_hint: None,
                },
            );
        }
    }

    // Add bounds info if present
    if let Some(bounds) = result.get("bounds") {
        if let Some(lower) = bounds.get("lower").and_then(Value::as_f64) {
            witness.insert(
                "output_lower_bound".to_string(),
                CounterexampleValue::Float { value: lower },
            );
        }
        if let Some(upper) = bounds.get("upper").and_then(Value::as_f64) {
            witness.insert(
                "output_upper_bound".to_string(),
                CounterexampleValue::Float { value: upper },
            );
        }
    }

    let failed_checks = build_nn_verification_failed_checks(result, backend_name);
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

fn build_nn_verification_failed_checks(result: &Value, backend_name: &str) -> Vec<FailedCheck> {
    let property = result
        .get("property_violated")
        .or_else(|| result.get("property"))
        .and_then(Value::as_str)
        .unwrap_or("robustness");
    let epsilon = result.get("epsilon").and_then(Value::as_f64);
    let verification_time = result.get("verification_time_s").and_then(Value::as_f64);
    let num_neurons = result.get("num_neurons").and_then(Value::as_u64);
    let num_layers = result.get("num_layers").and_then(Value::as_u64);

    let mut description = format!(
        "{} formal verification found property violation.",
        backend_name
    );

    description.push_str(&format!(" Property: {}.", property));

    if let Some(eps) = epsilon {
        description.push_str(&format!(" Perturbation bound (epsilon): {:.6}.", eps));
    }

    // Add network info if available
    if let (Some(neurons), Some(layers)) = (num_neurons, num_layers) {
        description.push_str(&format!(
            " Network: {} neurons, {} layers.",
            neurons, layers
        ));
    }

    // Add verification time if available
    if let Some(time) = verification_time {
        description.push_str(&format!(" Verification time: {:.2}s.", time));
    }

    // Add counterexample summary if available
    if result.get("counterexample").is_some() {
        description.push_str(" Counterexample found demonstrating property violation.");
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_verification_failure",
            backend_name.to_lowercase().replace(' ', "_")
        ),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for text adversarial attack backends (TextAttack).
///
/// Expects a JSON object with fields like `attack_success_rate`, `original_text`,
/// `perturbed_text`, `original_prediction`, `perturbed_prediction`.
pub fn build_text_attack_counterexample(
    result: &Value,
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    let attack_success_rate = result
        .get("attack_success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    if attack_success_rate < 0.01 {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "attack_success_rate".to_string(),
        CounterexampleValue::Float {
            value: attack_success_rate,
        },
    );

    if let Some(num_success) = result.get("num_successful_attacks").and_then(Value::as_u64) {
        witness.insert(
            "num_successful_attacks".to_string(),
            CounterexampleValue::Int {
                value: num_success as i128,
                type_hint: None,
            },
        );
    }

    if let Some(num_failed) = result.get("num_failed_attacks").and_then(Value::as_u64) {
        witness.insert(
            "num_failed_attacks".to_string(),
            CounterexampleValue::Int {
                value: num_failed as i128,
                type_hint: None,
            },
        );
    }

    if let Some(original_accuracy) = result.get("original_accuracy").and_then(Value::as_f64) {
        witness.insert(
            "original_accuracy".to_string(),
            CounterexampleValue::Float {
                value: original_accuracy,
            },
        );
    }

    // Add example adversarial text if available (direct or from adversarial_examples array)
    if let Some(original) = result.get("original_text").and_then(Value::as_str) {
        witness.insert(
            "original_text".to_string(),
            CounterexampleValue::String(original.to_string()),
        );
    }

    if let Some(perturbed) = result.get("perturbed_text").and_then(Value::as_str) {
        witness.insert(
            "perturbed_text".to_string(),
            CounterexampleValue::String(perturbed.to_string()),
        );
    }

    // Also check adversarial_examples array for text examples
    if let Some(examples) = result.get("adversarial_examples").and_then(Value::as_array) {
        if !examples.is_empty() {
            let first = &examples[0];
            if !witness.contains_key("original_text") {
                if let Some(orig) = first.get("original_text").and_then(Value::as_str) {
                    witness.insert(
                        "original_text".to_string(),
                        CounterexampleValue::String(orig.to_string()),
                    );
                }
            }
            if !witness.contains_key("perturbed_text") {
                if let Some(pert) = first.get("perturbed_text").and_then(Value::as_str) {
                    witness.insert(
                        "perturbed_text".to_string(),
                        CounterexampleValue::String(pert.to_string()),
                    );
                }
            }
            if let Some(orig_out) = first.get("original_output").and_then(Value::as_i64) {
                witness.insert(
                    "original_output".to_string(),
                    CounterexampleValue::Int {
                        value: orig_out as i128,
                        type_hint: None,
                    },
                );
            }
            if let Some(pert_out) = first.get("perturbed_output").and_then(Value::as_i64) {
                witness.insert(
                    "perturbed_output".to_string(),
                    CounterexampleValue::Int {
                        value: pert_out as i128,
                        type_hint: None,
                    },
                );
            }
        }
    }

    let failed_checks = build_text_attack_failed_checks(result, backend_name);
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

fn build_text_attack_failed_checks(result: &Value, backend_name: &str) -> Vec<FailedCheck> {
    let attack_success_rate = result
        .get("attack_success_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let attack_type = result
        .get("attack_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let num_successful = result
        .get("num_successful_attacks")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let num_total = result
        .get("num_examples")
        .or_else(|| result.get("num_samples"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let original_accuracy = result.get("original_accuracy").and_then(Value::as_f64);
    let perturbed_accuracy = result
        .get("perturbed_accuracy")
        .or_else(|| result.get("attack_accuracy"))
        .and_then(Value::as_f64);

    let mut description = format!("{} text adversarial robustness test failed.", backend_name);

    description.push_str(&format!(" Attack type: {}.", attack_type));

    description.push_str(&format!(
        " Attack success rate: {:.1}% ({}/{} samples).",
        attack_success_rate * 100.0,
        num_successful,
        num_total
    ));

    // Add accuracy comparison if available
    if let (Some(orig), Some(pert)) = (original_accuracy, perturbed_accuracy) {
        description.push_str(&format!(
            " Accuracy: original={:.1}%, after attack={:.1}%.",
            orig * 100.0,
            pert * 100.0
        ));
        let drop = orig - pert;
        if drop > 0.0 {
            description.push_str(&format!(" Accuracy drop: {:.1}%.", drop * 100.0));
        }
    }

    // Add example if text is short enough
    if let Some(original) = result.get("original_text").and_then(Value::as_str) {
        if original.len() <= 50 {
            description.push_str(&format!(" Example: \"{}\"", original));
            if let Some(perturbed) = result.get("perturbed_text").and_then(Value::as_str) {
                if perturbed.len() <= 50 {
                    description.push_str(&format!(" -> \"{}\".", perturbed));
                }
            }
        }
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_text_robustness_failure",
            backend_name.to_lowercase().replace(' ', "_")
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
    fn test_adversarial_attack_robust() {
        let result = json!({
            "attack_success_rate": 0.005,
            "robust_accuracy": 0.95,
            "clean_accuracy": 0.98
        });
        assert!(build_adversarial_attack_counterexample(&result, "ART").is_none());
    }

    #[test]
    fn test_adversarial_attack_failure() {
        let result = json!({
            "attack_success_rate": 0.45,
            "robust_accuracy": 0.55,
            "clean_accuracy": 0.95,
            "epsilon": 0.03,
            "attack_type": "PGD",
            "num_adversarial_examples": 45,
            "num_samples": 100,
            "max_perturbation": 0.031,
            "adversarial_example": {
                "original_prediction": 3,
                "adversarial_prediction": 7,
                "perturbation_norm": 0.029
            }
        });
        let cex = build_adversarial_attack_counterexample(&result, "Foolbox").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("foolbox"));
        assert!(cex.failed_checks[0].description.contains("PGD"));
        assert!(cex.failed_checks[0].description.contains("45.0%"));
        assert!(cex.witness.contains_key("attack_success_rate"));
        assert!(cex.witness.contains_key("original_prediction"));
    }

    #[test]
    fn test_nn_verification_verified() {
        let result = json!({
            "verified": true,
            "epsilon": 0.01
        });
        assert!(build_nn_verification_counterexample(&result, "ERAN").is_none());
    }

    #[test]
    fn test_nn_verification_failure() {
        let result = json!({
            "verified": false,
            "property_violated": "local_robustness",
            "epsilon": 0.03,
            "verification_time_s": 15.5,
            "num_neurons": 1000,
            "num_layers": 5,
            "counterexample": {
                "input": [0.1, 0.2, 0.3],
                "output": [0.8, 0.2]
            }
        });
        let cex = build_nn_verification_counterexample(&result, "Marabou").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("marabou"));
        assert!(cex.failed_checks[0]
            .description
            .contains("local_robustness"));
        assert!(cex.failed_checks[0].description.contains("1000 neurons"));
        assert!(cex.witness.contains_key("counterexample_input"));
    }

    #[test]
    fn test_text_attack_robust() {
        let result = json!({
            "attack_success_rate": 0.005
        });
        assert!(build_text_attack_counterexample(&result, "TextAttack").is_none());
    }

    #[test]
    fn test_text_attack_failure() {
        let result = json!({
            "attack_success_rate": 0.35,
            "attack_type": "TextFooler",
            "num_successful_attacks": 35,
            "num_examples": 100,
            "original_accuracy": 0.92,
            "perturbed_accuracy": 0.60,
            "original_text": "This movie was great!",
            "perturbed_text": "This film was great!"
        });
        let cex = build_text_attack_counterexample(&result, "TextAttack").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("textattack"));
        assert!(cex.failed_checks[0].description.contains("TextFooler"));
        assert!(cex.failed_checks[0].description.contains("35.0%"));
        assert!(cex.witness.contains_key("original_text"));
        assert!(cex.witness.contains_key("perturbed_text"));
    }
}
