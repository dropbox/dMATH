//! Shared helpers for model optimization counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from JSON outputs emitted by model optimization
//! backends (AIMET, Brevitas, IREE, Neural Compressor, NNCF, ONNX Runtime,
//! OpenVINO, TensorRT, Triton, TVM).

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use serde_json::Value;
use std::collections::HashMap;

/// Build a structured counterexample for quantization backends (AIMET, Brevitas, Neural Compressor, NNCF).
///
/// Expects a JSON object with fields like `output_max_diff`, `output_mse`,
/// `compression_ratio`, and optionally `latency_ms`.
pub fn build_quantization_counterexample(
    result: &Value,
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    let output_diff = result
        .get("output_max_diff")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let output_mse = result
        .get("output_mse")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    // Thresholds for "pass"
    let diff_threshold = 0.1;
    let mse_threshold = 0.01;

    if output_diff <= diff_threshold && output_mse <= mse_threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "output_max_diff".to_string(),
        CounterexampleValue::Float { value: output_diff },
    );

    witness.insert(
        "output_mse".to_string(),
        CounterexampleValue::Float { value: output_mse },
    );

    if let Some(ratio) = result.get("compression_ratio").and_then(Value::as_f64) {
        witness.insert(
            "compression_ratio".to_string(),
            CounterexampleValue::Float { value: ratio },
        );
    }

    if let Some(weight_bits) = result.get("weight_bits").and_then(Value::as_u64) {
        witness.insert(
            "weight_bits".to_string(),
            CounterexampleValue::Int {
                value: weight_bits as i128,
                type_hint: None,
            },
        );
    }

    if let Some(act_bits) = result.get("activation_bits").and_then(Value::as_u64) {
        witness.insert(
            "activation_bits".to_string(),
            CounterexampleValue::Int {
                value: act_bits as i128,
                type_hint: None,
            },
        );
    }

    // Add latency if present
    if let Some(latency) = result.get("latency_ms") {
        if let Some(mean) = latency.get("mean").and_then(Value::as_f64) {
            witness.insert(
                "mean_latency_ms".to_string(),
                CounterexampleValue::Float { value: mean },
            );
        }
    }

    let failed_checks =
        build_quantization_failed_checks(result, backend_name, diff_threshold, mse_threshold);
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

fn build_quantization_failed_checks(
    result: &Value,
    backend_name: &str,
    diff_threshold: f64,
    mse_threshold: f64,
) -> Vec<FailedCheck> {
    let output_diff = result
        .get("output_max_diff")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let output_mse = result
        .get("output_mse")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let compression_ratio = result
        .get("compression_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let weight_bits = result.get("weight_bits").and_then(Value::as_u64);
    let act_bits = result.get("activation_bits").and_then(Value::as_u64);

    let mut description = format!("{} quantization exceeds accuracy threshold.", backend_name);

    // Add bit width info if available
    if let (Some(w), Some(a)) = (weight_bits, act_bits) {
        description.push_str(&format!(" Config: W{}A{}.", w, a));
    }

    description.push_str(&format!(
        " Output max diff: {:.6} (threshold: {:.2}), MSE: {:.6} (threshold: {:.3}).",
        output_diff, diff_threshold, output_mse, mse_threshold
    ));

    // Report which threshold(s) violated
    let mut violations = Vec::new();
    if output_diff > diff_threshold {
        violations.push(format!(
            "max_diff exceeds by {:.6}",
            output_diff - diff_threshold
        ));
    }
    if output_mse > mse_threshold {
        violations.push(format!("MSE exceeds by {:.6}", output_mse - mse_threshold));
    }
    if !violations.is_empty() {
        description.push_str(&format!(" Violations: [{}].", violations.join(", ")));
    }

    description.push_str(&format!(" Compression ratio: {:.1}x.", compression_ratio));

    vec![FailedCheck {
        check_id: format!(
            "{}_quantization_accuracy_failure",
            backend_name.to_lowercase().replace(' ', "_")
        ),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for inference optimization backends (TensorRT, OpenVINO, ONNX Runtime).
///
/// Expects a JSON object with fields like `consistent_outputs`, `max_output_diff`,
/// `latency_ms`, and `throughput_ips`.
pub fn build_inference_counterexample(
    result: &Value,
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    let consistent = result
        .get("consistent_outputs")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let max_diff = result
        .get("max_output_diff")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    // Threshold for consistency
    let diff_threshold = 1e-4;

    if consistent && max_diff <= diff_threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "consistent_outputs".to_string(),
        CounterexampleValue::Bool(consistent),
    );

    witness.insert(
        "max_output_diff".to_string(),
        CounterexampleValue::Float { value: max_diff },
    );

    // Add latency metrics
    if let Some(latency) = result.get("latency_ms") {
        if let Some(mean) = latency.get("mean").and_then(Value::as_f64) {
            witness.insert(
                "mean_latency_ms".to_string(),
                CounterexampleValue::Float { value: mean },
            );
        }
        if let Some(p95) = latency.get("p95").and_then(Value::as_f64) {
            witness.insert(
                "p95_latency_ms".to_string(),
                CounterexampleValue::Float { value: p95 },
            );
        }
    }

    if let Some(throughput) = result.get("throughput_ips").and_then(Value::as_f64) {
        witness.insert(
            "throughput_ips".to_string(),
            CounterexampleValue::Float { value: throughput },
        );
    }

    let failed_checks = build_inference_failed_checks(result, backend_name, diff_threshold);
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

fn build_inference_failed_checks(
    result: &Value,
    backend_name: &str,
    diff_threshold: f64,
) -> Vec<FailedCheck> {
    let consistent = result
        .get("consistent_outputs")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let max_diff = result
        .get("max_output_diff")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let precision = result
        .get("precision")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let cuda_available = result.get("cuda_available").and_then(Value::as_bool);

    let mut description = format!("{} inference validation failed.", backend_name);

    description.push_str(&format!(" Precision mode: {}.", precision));

    if let Some(cuda) = cuda_available {
        description.push_str(&format!(
            " CUDA: {}.",
            if cuda { "enabled" } else { "simulated" }
        ));
    }

    description.push_str(&format!(
        " Output consistency: {}, max diff: {:.2e} (threshold: {:.2e}).",
        if consistent { "PASS" } else { "FAIL" },
        max_diff,
        diff_threshold
    ));

    // Report the issue
    if !consistent {
        description.push_str(" Inference outputs are non-deterministic across runs.");
    }
    if max_diff > diff_threshold {
        description.push_str(&format!(
            " Output variation exceeds threshold by {:.2e}.",
            max_diff - diff_threshold
        ));
    }

    // Add latency info
    if let Some(latency) = result.get("latency_ms") {
        if let (Some(mean), Some(p95)) = (
            latency.get("mean").and_then(Value::as_f64),
            latency.get("p95").and_then(Value::as_f64),
        ) {
            description.push_str(&format!(" Latency: mean={:.3}ms, P95={:.3}ms.", mean, p95));
        }
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_inference_consistency_failure",
            backend_name.to_lowercase().replace(' ', "_")
        ),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for compiler optimization backends (IREE, TVM, Triton).
///
/// Expects a JSON object with fields like `compilation_success`, `output_correct`,
/// `numerical_diff`, and `latency_ms`.
pub fn build_compiler_counterexample(
    result: &Value,
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    let compilation_success = result
        .get("compilation_success")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    // Support both output_correct and consistent_outputs field names
    let output_correct = result
        .get("output_correct")
        .or_else(|| result.get("consistent_outputs"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let numerical_diff = result
        .get("numerical_diff")
        .or_else(|| result.get("output_max_diff"))
        .or_else(|| result.get("max_output_diff"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    let diff_threshold = 1e-5;

    if compilation_success && output_correct && numerical_diff <= diff_threshold {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "compilation_success".to_string(),
        CounterexampleValue::Bool(compilation_success),
    );

    witness.insert(
        "output_correct".to_string(),
        CounterexampleValue::Bool(output_correct),
    );

    witness.insert(
        "numerical_diff".to_string(),
        CounterexampleValue::Float {
            value: numerical_diff,
        },
    );

    // Also add max_output_diff for compatibility
    witness.insert(
        "max_output_diff".to_string(),
        CounterexampleValue::Float {
            value: numerical_diff,
        },
    );

    // Add latency metrics
    if let Some(latency) = result.get("latency_ms") {
        if let Some(mean) = latency.get("mean").and_then(Value::as_f64) {
            witness.insert(
                "mean_latency_ms".to_string(),
                CounterexampleValue::Float { value: mean },
            );
        }
    }

    if let Some(speedup) = result.get("speedup").and_then(Value::as_f64) {
        witness.insert(
            "speedup".to_string(),
            CounterexampleValue::Float { value: speedup },
        );
    }

    let failed_checks = build_compiler_failed_checks(result, backend_name, diff_threshold);
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

fn build_compiler_failed_checks(
    result: &Value,
    backend_name: &str,
    diff_threshold: f64,
) -> Vec<FailedCheck> {
    let compilation_success = result
        .get("compilation_success")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let output_correct = result
        .get("output_correct")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let numerical_diff = result
        .get("numerical_diff")
        .or_else(|| result.get("output_max_diff"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let target = result
        .get("target")
        .or_else(|| result.get("device"))
        .and_then(Value::as_str)
        .unwrap_or("cpu");
    let optimization_level = result
        .get("optimization_level")
        .or_else(|| result.get("opt_level"))
        .and_then(Value::as_u64);

    let mut description = format!("{} compilation/optimization failed.", backend_name);

    description.push_str(&format!(" Target: {}.", target));

    if let Some(opt) = optimization_level {
        description.push_str(&format!(" Optimization level: {}.", opt));
    }

    // Report failures
    let mut issues = Vec::new();
    if !compilation_success {
        issues.push("compilation failed");
    }
    if !output_correct {
        issues.push("output mismatch");
    }
    if numerical_diff > diff_threshold {
        issues.push("numerical deviation exceeded");
    }

    if !issues.is_empty() {
        description.push_str(&format!(" Issues: [{}].", issues.join(", ")));
    }

    description.push_str(&format!(
        " Numerical diff: {:.2e} (threshold: {:.2e}).",
        numerical_diff, diff_threshold
    ));

    // Add compilation error if present
    if let Some(error) = result.get("compilation_error").and_then(Value::as_str) {
        let short_error: String = error.chars().take(100).collect();
        description.push_str(&format!(" Error: {}", short_error));
        if error.len() > 100 {
            description.push_str("...");
        }
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_compilation_failure",
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
    fn test_quantization_success() {
        let result = json!({
            "output_max_diff": 0.05,
            "output_mse": 0.005,
            "compression_ratio": 4.0
        });
        assert!(build_quantization_counterexample(&result, "AIMET").is_none());
    }

    #[test]
    fn test_quantization_failure() {
        let result = json!({
            "output_max_diff": 0.25,
            "output_mse": 0.05,
            "compression_ratio": 4.0,
            "weight_bits": 4,
            "activation_bits": 8,
            "latency_ms": {
                "mean": 1.5,
                "p95": 2.3
            }
        });
        let cex = build_quantization_counterexample(&result, "Brevitas").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("brevitas"));
        assert!(cex.failed_checks[0].description.contains("W4A8"));
        assert!(cex.failed_checks[0].description.contains("0.25"));
        assert!(cex.witness.contains_key("output_max_diff"));
        assert!(cex.witness.contains_key("weight_bits"));
    }

    #[test]
    fn test_inference_success() {
        let result = json!({
            "consistent_outputs": true,
            "max_output_diff": 1e-6,
            "latency_ms": {
                "mean": 0.5,
                "p95": 0.8
            }
        });
        assert!(build_inference_counterexample(&result, "TensorRT").is_none());
    }

    #[test]
    fn test_inference_inconsistent() {
        let result = json!({
            "consistent_outputs": false,
            "max_output_diff": 0.01,
            "precision": "FP16",
            "cuda_available": true,
            "latency_ms": {
                "mean": 0.5,
                "p95": 0.8
            },
            "throughput_ips": 2000.0
        });
        let cex = build_inference_counterexample(&result, "TensorRT").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("tensorrt"));
        assert!(cex.failed_checks[0].description.contains("FP16"));
        assert!(cex.failed_checks[0]
            .description
            .contains("non-deterministic"));
        assert!(cex.witness.contains_key("consistent_outputs"));
    }

    #[test]
    fn test_inference_high_diff() {
        let result = json!({
            "consistent_outputs": true,
            "max_output_diff": 0.001,
            "precision": "INT8"
        });
        let cex = build_inference_counterexample(&result, "OpenVINO").unwrap();
        assert!(cex.failed_checks[0]
            .description
            .contains("exceeds threshold"));
    }

    #[test]
    fn test_compiler_success() {
        let result = json!({
            "compilation_success": true,
            "output_correct": true,
            "numerical_diff": 1e-7,
            "target": "cuda"
        });
        assert!(build_compiler_counterexample(&result, "TVM").is_none());
    }

    #[test]
    fn test_compiler_failure() {
        let result = json!({
            "compilation_success": false,
            "output_correct": false,
            "numerical_diff": 0.5,
            "target": "llvm-cpu",
            "optimization_level": 3,
            "compilation_error": "Unsupported operation: CustomOp"
        });
        let cex = build_compiler_counterexample(&result, "IREE").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("iree"));
        assert!(cex.failed_checks[0].description.contains("llvm-cpu"));
        assert!(cex.failed_checks[0]
            .description
            .contains("compilation failed"));
        assert!(cex.failed_checks[0].description.contains("CustomOp"));
        assert!(cex.witness.contains_key("compilation_success"));
    }

    #[test]
    fn test_compiler_numerical_deviation() {
        let result = json!({
            "compilation_success": true,
            "output_correct": true,
            "numerical_diff": 1e-3,
            "target": "vulkan"
        });
        let cex = build_compiler_counterexample(&result, "Triton").unwrap();
        assert!(cex.failed_checks[0]
            .description
            .contains("numerical deviation"));
    }
}
