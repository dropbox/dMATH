//! Output parsing for alpha-beta-CROWN

use crate::counterexample::{CounterexampleValue, StructuredCounterexample};
use crate::traits::VerificationStatus;
use std::collections::HashMap;

/// Parse alpha-beta-CROWN output
pub fn parse_output(stdout: &str, stderr: &str) -> VerificationStatus {
    let combined = format!("{}\n{}", stdout, stderr);
    let lower = combined.to_lowercase();

    // Check UNSAT first so it is not shadowed by "sat" substring
    if lower.contains("unsat") || lower.contains("verified") || lower.contains("safe") {
        VerificationStatus::Proven
    } else if lower.contains("sat")
        || lower.contains("falsified")
        || lower.contains("unsafe")
        || lower.contains("counterexample")
    {
        VerificationStatus::Disproven
    } else if lower.contains("timeout") {
        VerificationStatus::Unknown {
            reason: "Verification timed out".to_string(),
        }
    } else if lower.contains("unknown") {
        VerificationStatus::Unknown {
            reason: "Verification result is unknown".to_string(),
        }
    } else {
        VerificationStatus::Unknown {
            reason: "Could not determine verification result".to_string(),
        }
    }
}

/// Parse counterexample from alpha-beta-CROWN output
///
/// alpha-beta-CROWN outputs adversarial examples in several formats:
/// 1. Array format: `input: [0.1, 0.2, 0.3]` or `adv_example: [[...]]`
/// 2. Variable assignments: `X_0 = 0.5` or `x0 = 0.5`
/// 3. Raw numeric arrays on separate lines
pub fn parse_counterexample(stdout: &str) -> Option<StructuredCounterexample> {
    let lower = stdout.to_lowercase();

    // Only parse if we have a SAT/falsified result (not UNSAT which means proven)
    // Check for UNSAT first since it contains "sat" as a substring
    if lower.contains("unsat") || lower.contains("verified") || lower.contains("safe") {
        return None;
    }

    // Check for SAT/disproven indicators
    if !lower.contains("sat")
        && !lower.contains("falsified")
        && !lower.contains("unsafe")
        && !lower.contains("counterexample")
    {
        return None;
    }

    let mut witness: HashMap<String, CounterexampleValue> = HashMap::new();

    // Pattern 1: Array format like `adv_example: [[0.1, 0.2, 0.3]]` or `input: [...]`
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
            {
                if let Some(values) = parse_array_values(rest) {
                    for (i, val) in values.into_iter().enumerate() {
                        witness
                            .insert(format!("x{}", i), CounterexampleValue::Float { value: val });
                    }
                }
            }
        }

        // Pattern 2: Variable assignments like `X_0 = 0.5` or `x0 = 0.5`
        if let Some(pos) = trimmed.find('=') {
            let var_part = trimmed[..pos].trim();
            let val_part = trimmed[pos + 1..].trim();

            // Parse variable name
            let var_lower = var_part.to_lowercase();
            if var_lower.starts_with('x')
                || var_lower.starts_with("input")
                || var_lower.starts_with('y')
                || var_lower.starts_with("output")
            {
                if let Ok(val) = val_part.parse::<f64>() {
                    // Normalize variable name
                    let normalized = normalize_var_name(var_part);
                    witness.insert(normalized, CounterexampleValue::Float { value: val });
                }
            }
        }
    }

    // Pattern 3: Standalone arrays (just numbers in brackets)
    if witness.is_empty() {
        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                if let Some(values) = parse_array_values(trimmed) {
                    for (i, val) in values.into_iter().enumerate() {
                        witness
                            .insert(format!("x{}", i), CounterexampleValue::Float { value: val });
                    }
                    break; // Only use first array found
                }
            }
        }
    }

    if witness.is_empty() {
        // Return raw output as fallback
        Some(StructuredCounterexample {
            raw: Some(stdout.to_string()),
            ..Default::default()
        })
    } else {
        Some(StructuredCounterexample {
            witness,
            raw: Some(stdout.to_string()),
            ..Default::default()
        })
    }
}

/// Parse array values from a string like `[0.1, 0.2, 0.3]` or `[[0.1, 0.2]]`
pub fn parse_array_values(s: &str) -> Option<Vec<f64>> {
    let trimmed = s.trim();

    // Handle nested arrays by stripping outer brackets
    let mut content = trimmed;
    while content.starts_with('[') && content.ends_with(']') && content.len() > 2 {
        let inner = &content[1..content.len() - 1].trim();
        if inner.starts_with('[') {
            content = inner;
        } else {
            content = inner;
            break;
        }
    }

    // Now parse comma-separated values
    let mut values = Vec::new();
    for part in content.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // Strip any remaining brackets
        let cleaned = part.trim_matches(|c| c == '[' || c == ']');
        if let Ok(val) = cleaned.parse::<f64>() {
            values.push(val);
        }
    }

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Normalize a variable name to standard form (x0, x1, y0, y1, etc.)
pub fn normalize_var_name(name: &str) -> String {
    let lower = name.to_lowercase();

    // Handle X_0, x_0, X0, x0 formats
    if lower.starts_with('x') || lower.starts_with("input") || lower.starts_with("in") {
        if let Some(idx) = extract_index_static(&lower) {
            return format!("x{}", idx);
        }
    }

    if lower.starts_with('y') || lower.starts_with("output") || lower.starts_with("out") {
        if let Some(idx) = extract_index_static(&lower) {
            return format!("y{}", idx);
        }
    }

    // Fallback to original
    name.to_string()
}

/// Static version of extract_index for use in static methods
pub fn extract_index_static(name: &str) -> Option<usize> {
    let digit_start = name.find(|c: char| c.is_ascii_digit())?;
    let num_str: String = name[digit_start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}
