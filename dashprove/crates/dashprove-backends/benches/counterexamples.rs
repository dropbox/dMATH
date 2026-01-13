//! Benchmarks for counterexample analysis utilities
//!
//! Run with: `cargo bench -p dashprove-backends`
//! Measures the performance of:
//! - Trace diffing between large counterexamples
//! - Trace pattern compression
//! - Greedy clustering across many counterexamples
//! - Counterexample helper builder functions

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use dashprove_backends::counterexample::{
    build_adversarial_attack_counterexample, build_aequitas_counterexample,
    build_aif360_counterexample, build_alibi_counterexample, build_bmc_counterexample,
    build_captum_counterexample, build_compiler_counterexample, build_deepchecks_counterexample,
    build_evidently_counterexample, build_fairlearn_counterexample,
    build_great_expectations_counterexample, build_guardrails_ai_counterexample,
    build_guidance_counterexample, build_inference_counterexample,
    build_interpretml_counterexample, build_lime_counterexample, build_llm_eval_counterexample,
    build_model_checker_counterexample, build_nemo_guardrails_counterexample,
    build_nn_counterexample, build_nn_verification_counterexample,
    build_quantization_counterexample, build_shap_counterexample,
    build_static_analysis_counterexample, build_symbolic_execution_counterexample,
    build_text_attack_counterexample, build_whylogs_counterexample, CounterexampleClusters,
    CounterexampleValue, FailedCheck, StructuredCounterexample, TraceState,
};
use serde_json::json;

fn make_counterexample(
    states: usize,
    vars: usize,
    pattern: usize,
    offset: i128,
) -> StructuredCounterexample {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: format!("check_{}", offset),
        description: "Synthetic property violation".to_string(),
        location: None,
        function: Some("failing_function".to_string()),
    });

    ce.witness.insert(
        "threshold".to_string(),
        CounterexampleValue::Int {
            value: offset,
            type_hint: Some("i64".to_string()),
        },
    );

    for i in 0..states {
        let mut state = TraceState::new((i + 1) as u32);
        state.action = Some(format!("step_{}", i % pattern));
        for v in 0..vars {
            let value = (i as i128 * (v as i128 + 1)) + offset;
            state.variables.insert(
                format!("var{}", v),
                CounterexampleValue::Int {
                    value,
                    type_hint: Some("i64".to_string()),
                },
            );
        }

        if i % (pattern / 2 + 1) == 0 {
            state
                .variables
                .insert("flag".to_string(), CounterexampleValue::Bool(i % 2 == 0));
        }

        ce.trace.push(state);
    }

    ce
}

fn make_cluster_inputs() -> Vec<StructuredCounterexample> {
    let mut items = Vec::new();
    for i in 0..30 {
        items.push(make_counterexample(80, 4, 6, (i % 3) as i128));
    }
    for i in 0..20 {
        items.push(make_counterexample(60, 3, 4, (i % 2) as i128 + 2));
    }
    items
}

fn bench_counterexample_diff(c: &mut Criterion) {
    let ce1 = make_counterexample(128, 6, 8, 0);
    let ce2 = make_counterexample(128, 6, 8, 1);

    let mut group = c.benchmark_group("counterexample_diff");
    group.bench_function(BenchmarkId::new("large_trace", "128_states"), |b| {
        b.iter(|| ce1.diff(black_box(&ce2)))
    });
    group.finish();
}

fn bench_counterexample_compression(c: &mut Criterion) {
    let ce = make_counterexample(180, 5, 5, 0);

    let mut group = c.benchmark_group("counterexample_compression");
    group.bench_function(BenchmarkId::new("compress", "repeated_patterns"), |b| {
        b.iter(|| black_box(&ce).compress_trace())
    });
    group.finish();
}

fn bench_counterexample_clustering(c: &mut Criterion) {
    let inputs = make_cluster_inputs();

    let mut group = c.benchmark_group("counterexample_clustering");
    group.bench_function(BenchmarkId::new("greedy", "50_examples"), |b| {
        b.iter_batched(
            || inputs.clone(),
            |dataset| CounterexampleClusters::from_counterexamples(dataset, 0.6),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

// --- Helper function benchmarks ---

/// Create synthetic BMC output for benchmarking
fn make_bmc_output(states: usize) -> (String, String) {
    let mut stdout = String::from("VERIFICATION FAILED\nCounterexample trace:\n");
    for i in 0..states {
        stdout.push_str(&format!("State {}: x = {}\n", i, i * 10));
    }
    stdout.push_str("array bounds violation at test.c:42\n");
    (stdout, String::new())
}

/// Create synthetic model checker output for benchmarking
fn make_model_checker_output(states: usize, depth: usize) -> (String, String) {
    let stdout = format!(
        "pan:1: invalid end state (at depth {})\n\
         pan: wrote model.pml.trail\n\
         States explored: {}\n\
         error: assertion violated\n",
        depth, states
    );
    (stdout, String::new())
}

/// Create synthetic symbolic execution errors for benchmarking
fn make_symbolic_execution_errors(count: usize) -> (String, Vec<String>) {
    let stdout = format!("KLEE: done: total paths = {}", count * 2);
    let errors: Vec<String> = (0..count)
        .map(|i| {
            if i % 3 == 0 {
                format!(
                    "ERROR: memory error: out of bound pointer at line {}",
                    i * 10
                )
            } else if i % 3 == 1 {
                format!("ERROR: assertion failure at test_{}.c:{}", i, i * 5)
            } else {
                format!("ERROR: division by zero at compute_{}.c:{}", i, i * 7)
            }
        })
        .collect();
    (stdout, errors)
}

/// Create synthetic static analysis issues for benchmarking
fn make_static_analysis_issues(count: usize) -> Vec<serde_json::Value> {
    (0..count)
        .map(|i| {
            json!({
                "bug_type": if i % 3 == 0 { "NULL_DEREFERENCE" } else if i % 3 == 1 { "RESOURCE_LEAK" } else { "BUFFER_OVERRUN" },
                "qualifier": format!("potential issue in function_{}", i),
                "file": format!("src/module_{}.c", i % 5),
                "line": i * 10 + 1,
                "column": i % 80 + 1,
                "procedure": format!("function_{}", i)
            })
        })
        .collect()
}

/// Create synthetic NN verification result for benchmarking
fn make_nn_result(samples: usize, verified: usize) -> serde_json::Value {
    json!({
        "verified": false,
        "verification_rate": verified as f64 / samples as f64,
        "total_count": samples,
        "verified_count": verified,
        "epsilon": 0.03,
        "counterexample": {
            "sample_index": 42,
            "original_input": (0..784).map(|i| (i as f64) / 784.0).collect::<Vec<_>>(),
            "counterexample": (0..784).map(|i| ((i + 1) as f64) / 784.0).collect::<Vec<_>>(),
            "true_label": 3,
            "predicted_label": 7
        }
    })
}

/// Create synthetic data quality result for benchmarking
fn make_data_quality_result(expectations: usize, passed: usize) -> serde_json::Value {
    let validation_results: Vec<serde_json::Value> = (0..expectations)
        .map(|i| {
            json!({
                "expectation": format!("expect_column_{}_check", i),
                "success": i < passed
            })
        })
        .collect();

    json!({
        "success_rate": passed as f64 / expectations as f64,
        "expectations_passed": passed,
        "expectations_failed": expectations - passed,
        "validation_level": "standard",
        "validation_results": validation_results
    })
}

/// Create synthetic drift detection result for benchmarking
fn make_drift_result(features: usize, drifted: usize) -> serde_json::Value {
    let drifted_features: Vec<String> = (0..drifted).map(|i| format!("feature_{}", i)).collect();

    json!({
        "drift_detected": drifted > 0,
        "drift_score": drifted as f64 / features as f64,
        "drift_threshold": 0.1,
        "report_type": "data_drift",
        "total_features": features,
        "drifted_features": drifted_features
    })
}

fn bench_bmc_helper(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/bmc");

    for states in [10, 50, 100] {
        let (stdout, stderr) = make_bmc_output(states);
        group.bench_function(
            BenchmarkId::new("build", format!("{}_states", states)),
            |b| {
                b.iter(|| {
                    build_bmc_counterexample(
                        black_box(&stdout),
                        black_box(&stderr),
                        black_box("CBMC"),
                        black_box(None),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_model_checker_helper(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/model_checker");

    for (states, depth) in [(100, 10), (1000, 50), (10000, 100)] {
        let (stdout, stderr) = make_model_checker_output(states, depth);
        group.bench_function(
            BenchmarkId::new("build", format!("{}_states_{}_depth", states, depth)),
            |b| {
                b.iter(|| {
                    build_model_checker_counterexample(
                        black_box(&stdout),
                        black_box(&stderr),
                        black_box("SPIN"),
                        black_box(None),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_symbolic_execution_helper(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/symbolic_exec");

    for error_count in [5, 20, 50] {
        let (stdout, errors) = make_symbolic_execution_errors(error_count);
        group.bench_function(
            BenchmarkId::new("build", format!("{}_errors", error_count)),
            |b| {
                b.iter(|| {
                    build_symbolic_execution_counterexample(
                        black_box(&stdout),
                        black_box(""),
                        black_box("KLEE"),
                        black_box(&errors),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_static_analysis_helper(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/static_analysis");

    for issue_count in [5, 25, 100] {
        let issues = make_static_analysis_issues(issue_count);
        group.bench_function(
            BenchmarkId::new("build", format!("{}_issues", issue_count)),
            |b| {
                b.iter(|| {
                    build_static_analysis_counterexample(black_box(&issues), black_box("Infer"))
                })
            },
        );
    }

    group.finish();
}

fn bench_nn_helper(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/nn");

    for (samples, verified) in [(100, 80), (1000, 900), (10000, 9500)] {
        let result = make_nn_result(samples, verified);
        let verification_rate = verified as f64 / samples as f64;
        group.bench_function(
            BenchmarkId::new("build", format!("{}_samples", samples)),
            |b| {
                b.iter(|| {
                    build_nn_counterexample(
                        black_box("Marabou"),
                        black_box(&result),
                        black_box(verification_rate),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_data_quality_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/data_quality");

    // Great Expectations
    for (expectations, passed) in [(10, 7), (50, 35), (200, 150)] {
        let result = make_data_quality_result(expectations, passed);
        group.bench_function(
            BenchmarkId::new(
                "great_expectations",
                format!("{}_expectations", expectations),
            ),
            |b| b.iter(|| build_great_expectations_counterexample(black_box(&result))),
        );
    }

    // WhyLogs - similar structure
    for (constraints, passed) in [(10, 6), (50, 30)] {
        let result = json!({
            "success_rate": passed as f64 / constraints as f64,
            "constraints_passed": passed,
            "constraints_failed": constraints - passed,
            "profile_type": "standard",
            "constraint_type": "value",
            "constraint_results": (0..constraints).map(|i| {
                json!({
                    "feature": format!("feature_{}", i),
                    "constraint": format!("constraint_{}", i % 5),
                    "passed": i < passed
                })
            }).collect::<Vec<_>>()
        });
        group.bench_function(
            BenchmarkId::new("whylogs", format!("{}_constraints", constraints)),
            |b| b.iter(|| build_whylogs_counterexample(black_box(&result))),
        );
    }

    // Evidently drift
    for (features, drifted) in [(10, 3), (50, 15), (100, 30)] {
        let result = make_drift_result(features, drifted);
        group.bench_function(
            BenchmarkId::new(
                "evidently",
                format!("{}_features_{}_drifted", features, drifted),
            ),
            |b| b.iter(|| build_evidently_counterexample(black_box(&result))),
        );
    }

    // Deepchecks
    for (checks, passed) in [(10, 7), (50, 40)] {
        let result = json!({
            "success_rate": passed as f64 / checks as f64,
            "checks_passed": passed,
            "checks_failed": checks - passed,
            "suite_type": "data_integrity",
            "task_type": "binary",
            "check_results": (0..checks).map(|i| {
                json!({
                    "check": format!("Check_{}", i),
                    "passed": i < passed
                })
            }).collect::<Vec<_>>()
        });
        group.bench_function(
            BenchmarkId::new("deepchecks", format!("{}_checks", checks)),
            |b| b.iter(|| build_deepchecks_counterexample(black_box(&result))),
        );
    }

    group.finish();
}

// --- Fairness helper benchmarks ---

/// Create synthetic fairness audit result (Aequitas style)
fn make_aequitas_result(groups: usize, min_disparity: f64) -> serde_json::Value {
    let metrics_by_group: std::collections::HashMap<String, serde_json::Value> = (0..groups)
        .map(|i| {
            let ppr_disp = min_disparity + (i as f64) * 0.05;
            (
                format!("group_{}", i),
                json!({
                    "ppr_disparity": ppr_disp,
                    "fpr_disparity": ppr_disp * 1.1
                }),
            )
        })
        .collect();

    json!({
        "is_fair": min_disparity < 0.8,
        "min_disparity_ratio": min_disparity,
        "avg_disparity_ratio": min_disparity + 0.1,
        "max_disparity_ratio": min_disparity + 0.2,
        "disparity_tolerance": 0.8,
        "fairness_metric": "predictive_parity",
        "metrics_by_group": metrics_by_group
    })
}

/// Create synthetic AIF360 bias result
fn make_aif360_result(is_fair: bool) -> serde_json::Value {
    let primary_val = if is_fair { 0.05 } else { 0.25 };
    json!({
        "is_fair": is_fair,
        "bias_metric": "statistical_parity_difference",
        "primary_metric_value": primary_val,
        "prediction_statistical_parity_diff": primary_val,
        "prediction_disparate_impact": if is_fair { 0.95 } else { 0.7 },
        "average_odds_difference": primary_val * 0.8,
        "equal_opportunity_difference": primary_val * 0.6,
        "fairness_threshold": 0.1,
        "disparate_impact_threshold": 0.8
    })
}

/// Create synthetic Fairlearn result
fn make_fairlearn_result(metric_value: f64, threshold: f64) -> serde_json::Value {
    json!({
        "is_fair": metric_value.abs() <= threshold,
        "fairness_metric": "demographic_parity",
        "primary_metric_value": metric_value,
        "demographic_parity_diff": metric_value,
        "equalized_odds_diff": metric_value * 0.9,
        "fairness_threshold": threshold,
        "accuracy": 0.85
    })
}

fn bench_fairness_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/fairness");

    // Aequitas
    for (groups, min_disp) in [(5, 0.6), (10, 0.5), (20, 0.4)] {
        let result = make_aequitas_result(groups, min_disp);
        group.bench_function(
            BenchmarkId::new("aequitas", format!("{}_groups", groups)),
            |b| b.iter(|| build_aequitas_counterexample(black_box(&result))),
        );
    }

    // AIF360
    for is_fair in [false, true] {
        let result = make_aif360_result(is_fair);
        let label = if is_fair { "fair" } else { "unfair" };
        group.bench_function(BenchmarkId::new("aif360", label), |b| {
            b.iter(|| build_aif360_counterexample(black_box(&result)))
        });
    }

    // Fairlearn
    for (metric, threshold) in [(0.25, 0.1), (0.05, 0.1), (0.15, 0.2)] {
        let result = make_fairlearn_result(metric, threshold);
        let label = if metric.abs() <= threshold {
            "fair"
        } else {
            "unfair"
        };
        group.bench_function(
            BenchmarkId::new("fairlearn", format!("{}_{:.2}", label, metric)),
            |b| b.iter(|| build_fairlearn_counterexample(black_box(&result))),
        );
    }

    group.finish();
}

// --- Guardrails helper benchmarks ---

/// Create synthetic guardrails result
fn make_guardrails_result(passed: usize, failed: usize, errors: usize) -> serde_json::Value {
    let total = passed + failed;
    let pass_rate = passed as f64 / total as f64;
    let error_list: Vec<String> = (0..errors)
        .map(|i| format!("Case {}: validation failed", i))
        .collect();

    json!({
        "pass_rate": pass_rate,
        "pass_threshold": 0.8,
        "passed": passed,
        "failed": failed,
        "total": total,
        "guardrail_type": "schema",
        "strictness": "strict",
        "errors": error_list
    })
}

/// Create synthetic Guidance result
fn make_guidance_result(passed: usize, failed: usize) -> serde_json::Value {
    let total = passed + failed;
    let pass_rate = passed as f64 / total as f64;

    json!({
        "pass_rate": pass_rate,
        "pass_threshold": 0.85,
        "passed": passed,
        "failed": failed,
        "generation_mode": "json_schema",
        "validation_mode": "strict",
        "errors": ["Case 1: output doesn't match pattern", "Case 2: invalid JSON"]
    })
}

/// Create synthetic NeMo Guardrails result
fn make_nemo_result(passed: usize, failed: usize) -> serde_json::Value {
    let total = passed + failed;
    let pass_rate = passed as f64 / total as f64;

    json!({
        "pass_rate": pass_rate,
        "pass_threshold": 0.85,
        "passed": passed,
        "failed": failed,
        "rail_type": "input",
        "jailbreak_detection": true,
        "topical_rail": true,
        "fact_checking": true,
        "errors": ["Input 1: potential jailbreak", "Input 2: off-topic"]
    })
}

fn bench_guardrails_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/guardrails");

    // GuardrailsAI
    for (passed, failed, errors) in [(5, 5, 3), (7, 3, 2), (3, 7, 5)] {
        let result = make_guardrails_result(passed, failed, errors);
        group.bench_function(
            BenchmarkId::new(
                "guardrails_ai",
                format!("{}_passed_{}_errors", passed, errors),
            ),
            |b| b.iter(|| build_guardrails_ai_counterexample(black_box(&result))),
        );
    }

    // Guidance
    for (passed, failed) in [(6, 4), (8, 2), (4, 6)] {
        let result = make_guidance_result(passed, failed);
        group.bench_function(
            BenchmarkId::new("guidance", format!("{}_passed", passed)),
            |b| b.iter(|| build_guidance_counterexample(black_box(&result))),
        );
    }

    // NeMo Guardrails
    for (passed, failed) in [(5, 5), (7, 3), (3, 7)] {
        let result = make_nemo_result(passed, failed);
        group.bench_function(
            BenchmarkId::new("nemo", format!("{}_passed", passed)),
            |b| b.iter(|| build_nemo_guardrails_counterexample(black_box(&result))),
        );
    }

    group.finish();
}

// --- Interpretability helper benchmarks ---

/// Create synthetic SHAP result
fn make_shap_result(mean_abs: f64, threshold: f64, with_stability: bool) -> serde_json::Value {
    let mut result = json!({
        "mean_abs_shap": mean_abs,
        "importance_threshold": threshold,
        "max_importance": mean_abs * 1.5,
        "top_feature": 2,
        "explainer": "tree",
        "model_type": "classification",
        "model_score": 0.92
    });
    if with_stability {
        result["stability_gap"] = json!(threshold * 1.5);
    }
    result
}

/// Create synthetic LIME result
fn make_lime_result(fidelity: f64, coverage: f64) -> serde_json::Value {
    json!({
        "fidelity": fidelity,
        "threshold": 0.8,
        "coverage": coverage,
        "mode": "tabular",
        "model_type": "classifier",
        "kernel_width": 0.75,
        "num_features": 10
    })
}

/// Create synthetic Captum result
fn make_captum_result(attribution_mean: f64, threshold: f64) -> serde_json::Value {
    json!({
        "attribution_mean": attribution_mean,
        "attribution_threshold": threshold,
        "attribution_max": attribution_mean * 2.0,
        "stability_gap": 0.02,
        "top_feature": 3,
        "method": "integrated_gradients",
        "model_type": "neural_network"
    })
}

/// Create synthetic InterpretML result
fn make_interpretml_result(global_mean: f64, local_fidelity: f64) -> serde_json::Value {
    json!({
        "global_importance_mean": global_mean,
        "threshold": 0.1,
        "local_fidelity": local_fidelity,
        "mode": "global",
        "model_type": "ebm",
        "explainer": "EBMExplainer",
        "top_feature": 1
    })
}

/// Create synthetic Alibi result
fn make_alibi_result(coverage: f64, fidelity: f64) -> serde_json::Value {
    json!({
        "explanation_coverage": coverage,
        "threshold": 0.8,
        "fidelity": fidelity,
        "precision": fidelity * 0.9,
        "method": "AnchorTabular",
        "model_type": "classifier",
        "explanation_type": "anchor"
    })
}

fn bench_interpretability_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/interpretability");

    // SHAP
    for (mean_abs, threshold, stability) in
        [(0.05, 0.1, false), (0.15, 0.1, true), (0.08, 0.1, false)]
    {
        let result = make_shap_result(mean_abs, threshold, stability);
        let label = if stability { "with_stability" } else { "basic" };
        group.bench_function(
            BenchmarkId::new("shap", format!("{:.2}_{}", mean_abs, label)),
            |b| b.iter(|| build_shap_counterexample(black_box(&result))),
        );
    }

    // LIME
    for (fidelity, coverage) in [(0.6, 0.7), (0.85, 0.5), (0.9, 0.8)] {
        let result = make_lime_result(fidelity, coverage);
        group.bench_function(
            BenchmarkId::new("lime", format!("f{:.2}_c{:.2}", fidelity, coverage)),
            |b| b.iter(|| build_lime_counterexample(black_box(&result))),
        );
    }

    // Captum
    for (mean, threshold) in [(0.05, 0.1), (0.12, 0.1), (0.08, 0.15)] {
        let result = make_captum_result(mean, threshold);
        group.bench_function(
            BenchmarkId::new("captum", format!("{:.2}_{:.2}", mean, threshold)),
            |b| b.iter(|| build_captum_counterexample(black_box(&result))),
        );
    }

    // InterpretML
    for (global, local) in [(0.05, 0.7), (0.15, 0.85), (0.08, 0.75)] {
        let result = make_interpretml_result(global, local);
        group.bench_function(
            BenchmarkId::new("interpretml", format!("g{:.2}_l{:.2}", global, local)),
            |b| b.iter(|| build_interpretml_counterexample(black_box(&result))),
        );
    }

    // Alibi
    for (coverage, fidelity) in [(0.6, 0.7), (0.9, 0.85), (0.75, 0.6)] {
        let result = make_alibi_result(coverage, fidelity);
        group.bench_function(
            BenchmarkId::new("alibi", format!("c{:.2}_f{:.2}", coverage, fidelity)),
            |b| b.iter(|| build_alibi_counterexample(black_box(&result))),
        );
    }

    group.finish();
}

// --- Robustness helper benchmarks ---

/// Create synthetic adversarial attack result
fn make_adversarial_result(success_rate: f64, num_samples: usize) -> serde_json::Value {
    let num_adversarial = (success_rate * num_samples as f64) as u64;
    json!({
        "attack_success_rate": success_rate,
        "robust_accuracy": 1.0 - success_rate,
        "clean_accuracy": 0.95,
        "epsilon": 0.03,
        "attack_type": "PGD",
        "num_adversarial_examples": num_adversarial,
        "num_samples": num_samples,
        "max_perturbation": 0.031,
        "adversarial_example": {
            "original_prediction": 3,
            "adversarial_prediction": 7,
            "perturbation_norm": 0.029
        }
    })
}

/// Create synthetic NN verification result
fn make_nn_verification_result(verified: bool, with_counterexample: bool) -> serde_json::Value {
    let mut result = json!({
        "verified": verified,
        "property_violated": "local_robustness",
        "epsilon": 0.03,
        "verification_time_s": 15.5,
        "num_neurons": 1000,
        "num_layers": 5
    });

    if with_counterexample && !verified {
        result["counterexample"] = json!({
            "input": (0..50).map(|i| i as f64 * 0.02).collect::<Vec<_>>(),
            "output": [0.8, 0.2]
        });
    }

    result
}

/// Create synthetic text attack result
fn make_text_attack_result(success_rate: f64, num_samples: usize) -> serde_json::Value {
    let num_successful = (success_rate * num_samples as f64) as u64;
    json!({
        "attack_success_rate": success_rate,
        "attack_type": "TextFooler",
        "num_successful_attacks": num_successful,
        "num_examples": num_samples,
        "original_accuracy": 0.92,
        "perturbed_accuracy": 0.92 * (1.0 - success_rate),
        "original_text": "This movie was great!",
        "perturbed_text": "This film was great!"
    })
}

fn bench_robustness_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/robustness");

    // Adversarial attack
    for (success_rate, samples) in [(0.45, 100), (0.2, 500), (0.6, 200)] {
        let result = make_adversarial_result(success_rate, samples);
        group.bench_function(
            BenchmarkId::new(
                "adversarial",
                format!("{:.0}pct_{}_samples", success_rate * 100.0, samples),
            ),
            |b| {
                b.iter(|| {
                    build_adversarial_attack_counterexample(black_box(&result), black_box("ART"))
                })
            },
        );
    }

    // NN verification
    for (verified, has_cex) in [(false, true), (false, false), (true, false)] {
        let result = make_nn_verification_result(verified, has_cex);
        let label = if verified {
            "verified"
        } else if has_cex {
            "failed_with_cex"
        } else {
            "failed_no_cex"
        };
        group.bench_function(BenchmarkId::new("nn_verification", label), |b| {
            b.iter(|| build_nn_verification_counterexample(black_box(&result), black_box("ERAN")))
        });
    }

    // Text attack
    for (success_rate, samples) in [(0.35, 100), (0.5, 200), (0.15, 500)] {
        let result = make_text_attack_result(success_rate, samples);
        group.bench_function(
            BenchmarkId::new(
                "text_attack",
                format!("{:.0}pct_{}_samples", success_rate * 100.0, samples),
            ),
            |b| {
                b.iter(|| {
                    build_text_attack_counterexample(black_box(&result), black_box("TextAttack"))
                })
            },
        );
    }

    group.finish();
}

// --- LLM eval helper benchmarks ---

/// Create synthetic LLM eval result
fn make_llm_eval_result(pass_rate: f64, total: usize, with_errors: bool) -> serde_json::Value {
    let passed = (pass_rate * total as f64) as u64;
    let failed = total as u64 - passed;
    let errors: Vec<String> = if with_errors {
        (0..failed.min(5))
            .map(|i| format!("Case {}: score {:.2} < threshold 0.8", i, pass_rate * 0.9))
            .collect()
    } else {
        vec![]
    };

    json!({
        "pass_rate": pass_rate,
        "pass_threshold": 0.8,
        "passed": passed,
        "failed": failed,
        "total": total,
        "metric": "answer_relevancy",
        "avg_score": pass_rate * 0.95,
        "errors": errors
    })
}

fn bench_llm_eval_helper(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/llm_eval");

    for (pass_rate, total, with_errors) in [(0.6, 10, true), (0.5, 50, true), (0.7, 100, false)] {
        let result = make_llm_eval_result(pass_rate, total, with_errors);
        let label = if with_errors {
            "with_errors"
        } else {
            "no_errors"
        };
        group.bench_function(
            BenchmarkId::new(
                "llm_eval",
                format!("{:.0}pct_{}_{}", pass_rate * 100.0, total, label),
            ),
            |b| b.iter(|| build_llm_eval_counterexample(black_box("DeepEval"), black_box(&result))),
        );
    }

    group.finish();
}

// --- Model optimization helper benchmarks ---

/// Create synthetic quantization result
fn make_quantization_result(max_diff: f64, mse: f64, weight_bits: u64) -> serde_json::Value {
    json!({
        "output_max_diff": max_diff,
        "output_mse": mse,
        "compression_ratio": 4.0,
        "weight_bits": weight_bits,
        "activation_bits": 8,
        "latency_ms": {
            "mean": 1.5,
            "p95": 2.3
        }
    })
}

/// Create synthetic inference result
fn make_inference_result(consistent: bool, max_diff: f64) -> serde_json::Value {
    json!({
        "consistent_outputs": consistent,
        "max_output_diff": max_diff,
        "precision": "FP16",
        "cuda_available": true,
        "latency_ms": {
            "mean": 0.5,
            "p95": 0.8
        },
        "throughput_ips": 2000.0
    })
}

/// Create synthetic compiler result
fn make_compiler_result(
    compilation_success: bool,
    output_correct: bool,
    diff: f64,
) -> serde_json::Value {
    let mut result = json!({
        "compilation_success": compilation_success,
        "output_correct": output_correct,
        "numerical_diff": diff,
        "target": "llvm-cpu",
        "optimization_level": 3,
        "latency_ms": {
            "mean": 2.0
        },
        "speedup": 1.5
    });
    if !compilation_success {
        result["compilation_error"] = json!("Unsupported operation: CustomOp");
    }
    result
}

fn bench_model_optimization_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_helpers/model_opt");

    // Quantization
    for (max_diff, mse, bits) in [(0.25, 0.05, 4), (0.15, 0.02, 8), (0.3, 0.08, 2)] {
        let result = make_quantization_result(max_diff, mse, bits);
        group.bench_function(
            BenchmarkId::new("quantization", format!("W{}_diff{:.2}", bits, max_diff)),
            |b| {
                b.iter(|| build_quantization_counterexample(black_box(&result), black_box("AIMET")))
            },
        );
    }

    // Inference
    for (consistent, diff) in [(false, 0.01), (true, 0.001), (true, 1e-6)] {
        let result = make_inference_result(consistent, diff);
        let label = if consistent && diff <= 1e-4 {
            "pass"
        } else if !consistent {
            "inconsistent"
        } else {
            "high_diff"
        };
        group.bench_function(BenchmarkId::new("inference", label), |b| {
            b.iter(|| build_inference_counterexample(black_box(&result), black_box("TensorRT")))
        });
    }

    // Compiler
    for (comp_ok, out_ok, diff) in [(false, false, 0.5), (true, false, 0.1), (true, true, 1e-3)] {
        let result = make_compiler_result(comp_ok, out_ok, diff);
        let label = if !comp_ok {
            "comp_fail"
        } else if !out_ok {
            "out_mismatch"
        } else {
            "num_deviation"
        };
        group.bench_function(BenchmarkId::new("compiler", label), |b| {
            b.iter(|| build_compiler_counterexample(black_box(&result), black_box("TVM")))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_counterexample_diff,
    bench_counterexample_compression,
    bench_counterexample_clustering,
    bench_bmc_helper,
    bench_model_checker_helper,
    bench_symbolic_execution_helper,
    bench_static_analysis_helper,
    bench_nn_helper,
    bench_data_quality_helpers,
    bench_fairness_helpers,
    bench_guardrails_helpers,
    bench_interpretability_helpers,
    bench_robustness_helpers,
    bench_llm_eval_helper,
    bench_model_optimization_helpers,
);
criterion_main!(benches);
