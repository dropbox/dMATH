//! Output parsing for ERAN
//!
//! Parses ERAN verification output and extracts counterexamples.

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use crate::traits::VerificationStatus;
use std::collections::HashMap;

/// Parse ERAN output to determine verification status
///
/// Returns (status, optional certification percentage)
pub fn parse_output(stdout: &str, stderr: &str) -> (VerificationStatus, Option<f64>) {
    let combined = format!("{}\n{}", stdout, stderr);

    // ERAN outputs certification percentages
    // Look for patterns like "certified: 95.3%"
    let certified_pct = combined
        .lines()
        .find(|l| l.contains("certified") || l.contains("verified"))
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|w| w.trim_end_matches('%').parse::<f64>().ok())
        });

    if let Some(pct) = certified_pct {
        if pct >= 99.99 {
            (VerificationStatus::Proven, Some(pct))
        } else if pct < 0.01 {
            (VerificationStatus::Disproven, Some(pct))
        } else {
            (
                VerificationStatus::Partial {
                    verified_percentage: pct,
                },
                Some(pct),
            )
        }
    } else if combined.contains("timeout") {
        (
            VerificationStatus::Unknown {
                reason: "Verification timed out".to_string(),
            },
            None,
        )
    } else {
        (
            VerificationStatus::Unknown {
                reason: "Could not parse ERAN output".to_string(),
            },
            None,
        )
    }
}

/// Parse counterexample from ERAN output
///
/// ERAN outputs adversarial examples in several formats:
/// 1. When certification fails: per-input bounds or samples
/// 2. Labeled format: `x0 = 0.5` or `input[0] = 0.5`
/// 3. Array format: `[0.1, 0.2, 0.3]`
pub fn parse_counterexample(stdout: &str) -> Option<StructuredCounterexample> {
    let lower = stdout.to_lowercase();

    // Only parse if we have a failure/unsafe result (not verified/certified)
    // ERAN outputs "certified" or "verified" for safe results
    if lower.contains("certified") && !lower.contains("not certified") {
        return None;
    }
    if lower.contains("verified") && !lower.contains("not verified") {
        return None;
    }
    if lower.contains("safe") && !lower.contains("unsafe") {
        return None;
    }

    // Check for failure indicators
    if !lower.contains("unsafe")
        && !lower.contains("not certified")
        && !lower.contains("not verified")
        && !lower.contains("fail")
        && !lower.contains("counterexample")
        && !lower.contains("adversarial")
    {
        return None;
    }

    let mut witness: HashMap<String, CounterexampleValue> = HashMap::new();

    // Pattern 1: Array format like `[0.1, 0.2, 0.3]` or `adversarial: [...]`
    for line in stdout.lines() {
        let trimmed = line.trim();

        // Look for labeled arrays
        if let Some(pos) = trimmed.find(':') {
            let label = trimmed[..pos].trim().to_lowercase();
            let rest = trimmed[pos + 1..].trim();

            if label.contains("adv")
                || label.contains("input")
                || label.contains("example")
                || label.contains("counter")
                || label.contains("perturbation")
            {
                if let Some(values) = parse_array_values(rest) {
                    for (i, val) in values.into_iter().enumerate() {
                        witness
                            .insert(format!("x{}", i), CounterexampleValue::Float { value: val });
                    }
                }
            }
        }

        // Pattern 2: Variable assignments like "x0 = 0.5" or "input[0] = 0.5"
        if let Some(ce_value) = parse_assignment_line(trimmed) {
            witness.insert(ce_value.0, ce_value.1);
        }
    }

    // Pattern 3: Standalone array on a line
    if witness.is_empty() {
        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                if let Some(values) = parse_array_values(trimmed) {
                    for (i, val) in values.into_iter().enumerate() {
                        witness
                            .insert(format!("x{}", i), CounterexampleValue::Float { value: val });
                    }
                    break;
                }
            }
        }
    }

    // Build failed_checks with ERAN-specific details
    let failed_checks = vec![FailedCheck {
        check_id: "eran_robustness_verification_failure".to_string(),
        description: build_eran_failure_description(stdout),
        location: None,
        function: None,
    }];

    if witness.is_empty() {
        // Return raw output as fallback if we detected a failure
        let mut cex = StructuredCounterexample::from_raw(stdout.to_string());
        cex.failed_checks = failed_checks;
        Some(cex)
    } else {
        Some(StructuredCounterexample {
            witness,
            failed_checks,
            playback_test: None,
            trace: vec![],
            raw: Some(stdout.to_string()),
            minimized: false,
        })
    }
}

/// Build a description of the ERAN verification failure from output
fn build_eran_failure_description(stdout: &str) -> String {
    let lower = stdout.to_lowercase();
    let mut description =
        "ERAN neural network verification found robustness violation.".to_string();

    // Extract failure type
    if lower.contains("unsafe") {
        description
            .push_str(" Property: unsafe - adversarial example found within perturbation bounds.");
    } else if lower.contains("not certified") {
        description.push_str(" Property: not certified - could not verify robustness.");
    } else if lower.contains("not verified") {
        description.push_str(" Property: not verified - verification incomplete.");
    }

    // Try to extract epsilon/perturbation info
    for line in stdout.lines() {
        let line_lower = line.to_lowercase();
        if line_lower.contains("epsilon") || line_lower.contains("perturbation") {
            let short_line: String = line.chars().take(100).collect();
            description.push_str(&format!(" {}", short_line.trim()));
            if line.len() > 100 {
                description.push_str("...");
            }
            break;
        }
    }

    description
}

/// Parse array values from a string like "[0.1, 0.2, 0.3]" or "[[0.1, 0.2]]"
pub fn parse_array_values(s: &str) -> Option<Vec<f64>> {
    // Remove outer brackets and whitespace
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }

    // Handle nested arrays [[...]]
    let inner = if s.starts_with("[[") {
        let inner = s.trim_start_matches('[').trim_end_matches(']');
        inner.trim_start_matches('[').trim_end_matches(']')
    } else {
        &s[1..s.len() - 1]
    };

    let values: Vec<f64> = inner
        .split(',')
        .filter_map(|v| v.trim().parse::<f64>().ok())
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Parse an assignment line like "x0 = 0.5" or "input[0] = 0.5"
fn parse_assignment_line(line: &str) -> Option<(String, CounterexampleValue)> {
    // Look for = separator
    let parts: Vec<&str> = line.splitn(2, '=').collect();
    if parts.len() != 2 {
        return None;
    }

    let var_raw = parts[0].trim();
    let value_str = parts[1].trim();

    // Skip status lines
    if var_raw.is_empty() || var_raw.contains("certified") || var_raw.contains("verified") {
        return None;
    }

    // Normalize variable name
    let var_name = normalize_var_name(var_raw);

    // Parse value
    if let Ok(value) = value_str.parse::<f64>() {
        Some((var_name, CounterexampleValue::Float { value }))
    } else {
        None
    }
}

/// Normalize variable names like "input[0]", "Input_0", "x0" to "x0" format
pub fn normalize_var_name(name: &str) -> String {
    let lower = name.to_lowercase();

    // Handle input[0] format
    if lower.contains("input")
        || lower.contains("output")
        || lower.starts_with("x")
        || lower.starts_with("y")
    {
        if let Some(idx) = extract_index_from_name(name) {
            if lower.contains("output") || lower.starts_with("y") {
                return format!("y{}", idx);
            } else {
                return format!("x{}", idx);
            }
        }
    }

    // Return as-is if no pattern matches
    name.to_lowercase().replace(['[', ']', '_'], "")
}

/// Extract numeric index from a variable name
fn extract_index_from_name(name: &str) -> Option<usize> {
    let digit_start = name.find(|c: char| c.is_ascii_digit())?;
    let num_str: String = name[digit_start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}
