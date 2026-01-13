//! Verify execution traces against TLA+ specifications

use anyhow::{Context, Result};
use dashprove_async::{ExecutionTrace, TlaTraceConfig, TlaTraceVerifier};
use dashprove_monitor::{
    invariant::Invariant,
    liveness::{check_liveness, LivenessProperty},
};
use std::time::Duration;

/// Configuration for verify-trace command
pub struct VerifyTraceConfig<'a> {
    /// Path to the trace JSON file
    pub trace_path: &'a str,
    /// Path to TLA+ specification (optional)
    pub spec_path: Option<&'a str>,
    /// List of invariants to check (by name)
    pub invariants: Vec<String>,
    /// List of liveness properties to check
    pub liveness: Vec<String>,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Whether to show verbose output
    pub verbose: bool,
    /// Output format (text, json)
    pub format: &'a str,
}

/// Run the verify-trace command
pub fn run_verify_trace(config: VerifyTraceConfig<'_>) -> Result<()> {
    // Load the trace file
    if config.verbose {
        println!("Loading trace from: {}", config.trace_path);
    }

    let trace_content = std::fs::read_to_string(config.trace_path)
        .context(format!("Failed to read trace file: {}", config.trace_path))?;

    let trace: ExecutionTrace =
        serde_json::from_str(&trace_content).context("Failed to parse trace as JSON")?;

    if config.verbose {
        println!(
            "Trace loaded: {} transitions, initial state: {}",
            trace.len(),
            serde_json::to_string_pretty(&trace.initial_state)?
        );
    }

    let mut total_checks = 0;
    let mut passed_checks = 0;
    let mut results = Vec::new();

    // Run TLA+ verification if spec provided
    if let Some(spec_path) = config.spec_path {
        if config.verbose {
            println!("\nVerifying against TLA+ spec: {}", spec_path);
        }

        let tla_config = TlaTraceConfig::new(spec_path)
            .with_timeout(Duration::from_secs(config.timeout_secs))
            .with_invariants(config.invariants.clone());

        let verifier = TlaTraceVerifier::new(tla_config);

        match tokio::runtime::Runtime::new()?.block_on(verifier.verify_trace(&trace)) {
            Ok(result) => {
                total_checks += 1;
                if result.valid {
                    passed_checks += 1;
                    results.push(CheckResult {
                        check_type: "tla+".to_string(),
                        name: "spec_conformance".to_string(),
                        passed: true,
                        message: None,
                        details: None,
                    });
                    if config.verbose {
                        println!("  ✓ Trace conforms to specification");
                    }
                } else {
                    let msg = if let Some(idx) = result.divergence_state {
                        format!("Diverged at state {}", idx)
                    } else {
                        result
                            .violations
                            .first()
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| "Unknown violation".to_string())
                    };
                    results.push(CheckResult {
                        check_type: "tla+".to_string(),
                        name: "spec_conformance".to_string(),
                        passed: false,
                        message: Some(msg.clone()),
                        details: Some(serde_json::json!({
                            "divergence_state": result.divergence_state,
                            "expected_actions": result.expected_actions,
                            "violations": result.violations.len(),
                        })),
                    });
                    if config.verbose {
                        println!("  ✗ {}", msg);
                    }
                }
            }
            Err(e) => {
                results.push(CheckResult {
                    check_type: "tla+".to_string(),
                    name: "spec_conformance".to_string(),
                    passed: false,
                    message: Some(format!("Verification failed: {}", e)),
                    details: None,
                });
                if config.verbose {
                    println!("  ✗ Verification error: {}", e);
                }
            }
        }
    }

    // Run built-in invariant checks
    if !config.invariants.is_empty() && config.spec_path.is_none() {
        if config.verbose {
            println!("\nChecking invariants on trace:");
        }

        for inv_name in &config.invariants {
            if let Some(inv) = get_builtin_invariant(inv_name) {
                total_checks += 1;
                let result = dashprove_monitor::invariant::check_invariant(&trace, &inv);

                if result.satisfied {
                    passed_checks += 1;
                    results.push(CheckResult {
                        check_type: "invariant".to_string(),
                        name: inv_name.clone(),
                        passed: true,
                        message: None,
                        details: None,
                    });
                    if config.verbose {
                        println!("  ✓ {}", inv_name);
                    }
                } else {
                    results.push(CheckResult {
                        check_type: "invariant".to_string(),
                        name: inv_name.clone(),
                        passed: false,
                        message: Some(format!(
                            "First violation at state {}",
                            result.first_violation_index.unwrap_or(0)
                        )),
                        details: Some(serde_json::json!({
                            "violation_count": result.violation_count,
                            "first_violation_index": result.first_violation_index,
                        })),
                    });
                    if config.verbose {
                        println!(
                            "  ✗ {} (violated at state {})",
                            inv_name,
                            result.first_violation_index.unwrap_or(0)
                        );
                    }
                }
            } else {
                println!("  Warning: Unknown invariant '{}'", inv_name);
            }
        }
    }

    // Run liveness checks
    if !config.liveness.is_empty() {
        if config.verbose {
            println!("\nChecking liveness properties:");
        }

        for prop_name in &config.liveness {
            if let Some(prop) = get_builtin_liveness(prop_name) {
                total_checks += 1;
                let result = check_liveness(&trace, &prop);

                if result.satisfied {
                    passed_checks += 1;
                    results.push(CheckResult {
                        check_type: "liveness".to_string(),
                        name: prop_name.clone(),
                        passed: true,
                        message: None,
                        details: result.satisfaction_index.map(|idx| {
                            serde_json::json!({
                                "satisfied_at": idx,
                                "steps": result.steps_to_satisfaction,
                            })
                        }),
                    });
                    if config.verbose {
                        if let Some(idx) = result.satisfaction_index {
                            println!("  ✓ {} (satisfied at state {})", prop_name, idx);
                        } else {
                            println!("  ✓ {} (vacuously satisfied)", prop_name);
                        }
                    }
                } else {
                    results.push(CheckResult {
                        check_type: "liveness".to_string(),
                        name: prop_name.clone(),
                        passed: false,
                        message: result.error_message.clone(),
                        details: Some(serde_json::json!({
                            "trigger_index": result.trigger_index,
                            "final_state": result.final_state,
                        })),
                    });
                    if config.verbose {
                        let msg = result
                            .error_message
                            .as_deref()
                            .unwrap_or("not satisfied by end of trace");
                        println!("  ✗ {} ({})", prop_name, msg);
                    }
                }
            } else {
                println!("  Warning: Unknown liveness property '{}'", prop_name);
            }
        }
    }

    // Output results
    if config.format == "json" {
        let output = VerifyTraceOutput {
            trace_path: config.trace_path.to_string(),
            spec_path: config.spec_path.map(String::from),
            total_checks,
            passed_checks,
            passed: passed_checks == total_checks,
            results,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!();
        if passed_checks == total_checks {
            println!("✓ All {} checks passed", total_checks);
        } else {
            println!(
                "✗ {} of {} checks failed",
                total_checks - passed_checks,
                total_checks
            );
        }
    }

    if passed_checks == total_checks {
        Ok(())
    } else {
        anyhow::bail!("Trace verification failed")
    }
}

/// Result of a single check
#[derive(serde::Serialize)]
struct CheckResult {
    check_type: String,
    name: String,
    passed: bool,
    message: Option<String>,
    details: Option<serde_json::Value>,
}

/// Overall output structure
#[derive(serde::Serialize)]
struct VerifyTraceOutput {
    trace_path: String,
    spec_path: Option<String>,
    total_checks: usize,
    passed_checks: usize,
    passed: bool,
    results: Vec<CheckResult>,
}

/// Get a built-in invariant by name
fn get_builtin_invariant(name: &str) -> Option<Invariant> {
    match name {
        "positive" | "non_negative" => Some(
            dashprove_monitor::invariant::patterns::field_positive("value"),
        ),
        "bounded" => Some(dashprove_monitor::invariant::patterns::field_in_range(
            "value", 0, 1000,
        )),
        "monotonic" => Some(dashprove_monitor::invariant::patterns::monotonic_increasing("value")),
        _ => {
            // Try to interpret as a field name for positive check
            if name.ends_with("_positive") {
                let field = name.trim_end_matches("_positive");
                Some(dashprove_monitor::invariant::patterns::field_positive(
                    field,
                ))
            } else if name.ends_with("_monotonic") {
                let field = name.trim_end_matches("_monotonic");
                Some(dashprove_monitor::invariant::patterns::monotonic_increasing(field))
            } else {
                None
            }
        }
    }
}

/// Get a built-in liveness property by name
fn get_builtin_liveness(name: &str) -> Option<LivenessProperty> {
    match name {
        "terminates" | "done" => Some(LivenessProperty::eventually("terminates", |s| {
            s.get("done")
                .or_else(|| s.get("terminated"))
                .or_else(|| s.get("complete"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })),
        "progress" => Some(LivenessProperty::eventually("progress", |s| {
            // Check if any state value changed (progress made)
            s.as_object()
                .is_some_and(|o| o.values().any(|v| !v.is_null()))
        })),
        _ => {
            // Try to interpret as field_eventually pattern
            if name.ends_with("_eventually") {
                let field = name.trim_end_matches("_eventually").to_string();
                Some(LivenessProperty::eventually(name, move |s| {
                    s.get(&field).and_then(|v| v.as_bool()).unwrap_or(false)
                }))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_invariant() {
        let inv = get_builtin_invariant("positive").unwrap();
        assert_eq!(inv.name, "value_positive");

        let inv2 = get_builtin_invariant("counter_positive").unwrap();
        assert_eq!(inv2.name, "counter_positive");
    }

    #[test]
    fn test_builtin_liveness() {
        let prop = get_builtin_liveness("terminates").unwrap();
        assert_eq!(prop.name, "terminates");

        let prop2 = get_builtin_liveness("ready_eventually").unwrap();
        assert_eq!(prop2.name, "ready_eventually");
    }
}
