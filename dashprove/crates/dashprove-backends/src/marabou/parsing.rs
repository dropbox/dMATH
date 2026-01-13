//! Output parsing and counterexample extraction for Marabou backend

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use crate::traits::VerificationStatus;
use std::collections::HashMap;

/// Parse Marabou output and return status
pub fn parse_output(stdout: &str, stderr: &str) -> VerificationStatus {
    let combined = format!("{}\n{}", stdout, stderr);

    // Check UNSAT first since "unsat" contains "sat"
    if combined.contains("unsat") || combined.contains("UNSAT") {
        // UNSAT means property holds
        VerificationStatus::Proven
    } else if combined.contains("sat") || combined.contains("SAT") {
        // SAT means counterexample found (property violated)
        VerificationStatus::Disproven
    } else if combined.contains("TIMEOUT") || combined.contains("timeout") {
        VerificationStatus::Unknown {
            reason: "Verification timed out".to_string(),
        }
    } else if combined.contains("ERROR") || combined.contains("error") {
        VerificationStatus::Unknown {
            reason: format!(
                "Marabou error: {}",
                stderr.lines().next().unwrap_or("unknown")
            ),
        }
    } else {
        VerificationStatus::Unknown {
            reason: "Could not determine verification result".to_string(),
        }
    }
}

/// Parse counterexample from Marabou SAT output
///
/// Marabou outputs counterexamples in several formats:
/// 1. Variable assignments: `x0 = 0.5` or `Input 0 = 0.5`
/// 2. Input/output sections with values
/// 3. Raw numeric lines for each variable
pub fn parse_counterexample(stdout: &str) -> Option<StructuredCounterexample> {
    let mut witness: HashMap<String, CounterexampleValue> = HashMap::new();

    // Track whether we found any SAT-related content
    let combined_lower = stdout.to_lowercase();
    if !combined_lower.contains("sat") || combined_lower.contains("unsat") {
        return None;
    }

    // Parse variable assignments in various formats
    for line in stdout.lines() {
        let line = line.trim();

        // Skip empty lines and result status lines
        if line.is_empty() || line.eq_ignore_ascii_case("sat") || line.eq_ignore_ascii_case("unsat")
        {
            continue;
        }

        // Check for labeled format first: "Input 0 = 0.5" or "Output 0 = 0.5"
        // Must come before generic assignment parsing to avoid treating "Input 0" as var name
        if let Some(ce_value) = parse_labeled_variable(line) {
            witness.insert(ce_value.0, ce_value.1);
            continue;
        }

        // Format: "x0 = 0.5" or "x0 = 0.500000"
        if let Some(ce_value) = parse_assignment_line(line, "=") {
            witness.insert(ce_value.0, ce_value.1);
            continue;
        }

        // Format: "x0 : 0.5" (alternative separator)
        if let Some(ce_value) = parse_assignment_line(line, ":") {
            witness.insert(ce_value.0, ce_value.1);
        }
    }

    if witness.is_empty() {
        // Try to parse a more structured format (one value per line after SAT)
        let mut in_values = false;
        let mut var_index = 0;
        for line in stdout.lines() {
            let line = line.trim();
            if line.eq_ignore_ascii_case("sat") {
                in_values = true;
                continue;
            }
            if in_values {
                if let Ok(value) = line.parse::<f64>() {
                    let var_name = format!("x{var_index}");
                    witness.insert(var_name, CounterexampleValue::Float { value });
                    var_index += 1;
                }
            }
        }
    }

    // Build failed_checks with Marabou-specific details
    let failed_checks = vec![FailedCheck {
        check_id: "marabou_verification_failure".to_string(),
        description: build_marabou_failure_description(stdout, &witness),
        location: None,
        function: None,
    }];

    if witness.is_empty() {
        // Return raw output as fallback
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

/// Build a description of the Marabou verification failure from output
fn build_marabou_failure_description(
    stdout: &str,
    witness: &HashMap<String, CounterexampleValue>,
) -> String {
    let mut description =
        "Marabou SMT-based neural network verification found property violation.".to_string();

    // Add SAT indicator
    if stdout.contains("SAT") {
        description.push_str(
            " Result: SAT - counterexample found demonstrating property can be violated.",
        );
    }

    // Add witness size info
    if !witness.is_empty() {
        description.push_str(&format!(
            " Counterexample has {} variable assignments.",
            witness.len()
        ));
    }

    // Try to extract any property/bound information
    for line in stdout.lines() {
        let line_lower = line.to_lowercase();
        if line_lower.contains("property")
            || line_lower.contains("query")
            || line_lower.contains("bound")
        {
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

/// Parse an assignment line with a given separator (= or :)
pub fn parse_assignment_line(line: &str, separator: &str) -> Option<(String, CounterexampleValue)> {
    let parts: Vec<&str> = line.splitn(2, separator).collect();
    if parts.len() != 2 {
        return None;
    }

    let var_name = parts[0].trim().to_string();
    let value_str = parts[1].trim();

    // Skip if var_name is empty or doesn't look like a variable
    if var_name.is_empty() {
        return None;
    }

    // Parse the value
    let value = parse_value(value_str)?;
    Some((var_name, value))
}

/// Parse labeled variable format: "Input 0 = 0.5" or "Output 0 = 0.5"
pub fn parse_labeled_variable(line: &str) -> Option<(String, CounterexampleValue)> {
    let line_trimmed = line.trim();
    let lower = line_trimmed.to_lowercase();

    // Check for "Input" or "Output" prefix and extract the rest
    let (prefix, rest) = if lower.starts_with("input") {
        // Handle both "Input0" and "Input 0" and "Input[0]"
        ("x", &line_trimmed[5..])
    } else if lower.starts_with("output") {
        ("y", &line_trimmed[6..])
    } else {
        return None;
    };

    // Parse remaining: " 0 = 0.5" or "[0] = 0.5" or "0 = 0.5"
    let rest = rest.trim();

    // Try to find the index and value using '=' separator
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        return None;
    }

    // Index part might be " 0", "[0]", "0"
    let index_str = parts[0]
        .trim()
        .trim_matches(|c: char| c == '[' || c == ']' || c.is_whitespace());
    let index: usize = index_str.parse().ok()?;
    let value_str = parts[1].trim();

    let var_name = format!("{prefix}{index}");
    let value = parse_value(value_str)?;

    Some((var_name, value))
}

/// Parse a numeric value string into a CounterexampleValue
pub fn parse_value(value_str: &str) -> Option<CounterexampleValue> {
    let value_str = value_str.trim();

    // Try parsing as float first (most common for neural networks)
    if let Ok(value) = value_str.parse::<f64>() {
        return Some(CounterexampleValue::Float { value });
    }

    // Try parsing as integer
    if let Ok(value) = value_str.parse::<i128>() {
        return Some(CounterexampleValue::Int {
            value,
            type_hint: None,
        });
    }

    // Handle special values
    let lower = value_str.to_lowercase();
    if lower == "inf" || lower == "infinity" || lower == "+inf" {
        return Some(CounterexampleValue::Float {
            value: f64::INFINITY,
        });
    }
    if lower == "-inf" || lower == "-infinity" {
        return Some(CounterexampleValue::Float {
            value: f64::NEG_INFINITY,
        });
    }
    if lower == "nan" {
        return Some(CounterexampleValue::Float { value: f64::NAN });
    }

    // If nothing else works, store as unknown
    if !value_str.is_empty() {
        return Some(CounterexampleValue::Unknown(value_str.to_string()));
    }

    None
}
