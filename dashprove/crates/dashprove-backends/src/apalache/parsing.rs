//! Apalache output parsing

use crate::traits::{
    BackendId, BackendResult, CounterexampleValue, FailedCheck, StructuredCounterexample,
    TraceState, VerificationStatus,
};
use std::collections::HashMap;

use super::execution::ApalacheOutput;

/// Parse Apalache output into verification result
pub fn parse_output(output: &ApalacheOutput) -> BackendResult {
    let combined = format!("{}\n{}", output.stdout, output.stderr);

    // Apalache exit codes:
    // 0: success (property holds)
    // 12: counterexample found
    // Other: various errors

    // Check for successful verification (no counterexample)
    if output.exit_code == Some(0) {
        // Check if it actually verified something
        if combined.contains("The outcome is: NoError")
            || combined.contains("PASS")
            || combined.contains("Checker reports no error")
        {
            return BackendResult {
                backend: BackendId::Apalache,
                status: VerificationStatus::Proven,
                proof: Some("Symbolic model checking completed successfully".to_string()),
                counterexample: None,
                diagnostics: extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }
    }

    // Check for counterexample found
    if output.exit_code == Some(12)
        || combined.contains("The outcome is: Error")
        || combined.contains("FAIL")
        || combined.contains("Found an error")
        || combined.contains("counterexample")
    {
        let counterexample = parse_counterexample(&combined);
        return BackendResult {
            backend: BackendId::Apalache,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(counterexample),
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Check for type errors
    if combined.contains("Type error") || combined.contains("TYPEERROR") {
        let error_msg = extract_type_error(&combined);
        return BackendResult {
            backend: BackendId::Apalache,
            status: VerificationStatus::Unknown {
                reason: format!("Type error: {}", error_msg),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![error_msg],
            time_taken: output.duration,
        };
    }

    // Check for parse errors
    if combined.contains("Parse error") || combined.contains("PARSEERROR") {
        let error_msg = extract_parse_error(&combined);
        return BackendResult {
            backend: BackendId::Apalache,
            status: VerificationStatus::Unknown {
                reason: format!("Parse error: {}", error_msg),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![error_msg],
            time_taken: output.duration,
        };
    }

    // Check for timeout
    if combined.contains("Timeout") || combined.contains("timeout") {
        return BackendResult {
            backend: BackendId::Apalache,
            status: VerificationStatus::Unknown {
                reason: "Verification timed out".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Check for out of memory
    if combined.contains("OutOfMemoryError") || combined.contains("out of memory") {
        return BackendResult {
            backend: BackendId::Apalache,
            status: VerificationStatus::Unknown {
                reason: "Out of memory".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Unknown result
    BackendResult {
        backend: BackendId::Apalache,
        status: VerificationStatus::Unknown {
            reason: "Could not determine verification result".to_string(),
        },
        proof: None,
        counterexample: None,
        diagnostics: vec![combined],
        time_taken: output.duration,
    }
}

/// Parse counterexample from Apalache output
fn parse_counterexample(output: &str) -> StructuredCounterexample {
    let mut trace = Vec::new();
    let mut failed_checks = Vec::new();
    let mut witness = HashMap::new();

    // Extract violated invariant name
    for line in output.lines() {
        if line.contains("violated") {
            if let Some(inv_name) = extract_violated_invariant(line) {
                failed_checks.push(FailedCheck {
                    check_id: inv_name.clone(),
                    description: format!("Invariant {} violated", inv_name),
                    location: None,
                    function: None,
                });
            }
        }
    }

    // Parse state trace from Apalache's ITF JSON output or text output
    // Apalache outputs counterexamples in ITF (Informal Trace Format) JSON
    // or in a text format showing state assignments
    let mut current_state: Option<TraceState> = None;
    let mut state_num = 0u32;

    for line in output.lines() {
        let trimmed = line.trim();

        // Look for state markers
        if trimmed.starts_with("State") || trimmed.contains("state") && trimmed.contains(':') {
            // Save previous state if any
            if let Some(state) = current_state.take() {
                trace.push(state);
            }

            // Start new state
            let action = extract_action_name(trimmed);
            current_state = Some(TraceState {
                state_num,
                action,
                variables: HashMap::new(),
            });
            state_num += 1;
        }

        // Look for variable assignments (var = value or var := value)
        if let Some((var, value)) = parse_variable_assignment(trimmed) {
            if let Some(ref mut state) = current_state {
                state.variables.insert(var.clone(), value.clone());
            }
            // Also add to witness
            witness.insert(var, value);
        }
    }

    // Push final state if any
    if let Some(state) = current_state {
        trace.push(state);
    }

    // If no failed checks were found, add a generic one
    if failed_checks.is_empty() {
        failed_checks.push(FailedCheck {
            check_id: "unknown".to_string(),
            description: "Property violated".to_string(),
            location: None,
            function: None,
        });
    }

    StructuredCounterexample {
        witness,
        trace,
        failed_checks,
        raw: Some(output.to_string()),
        minimized: false,
        playback_test: None,
    }
}

/// Extract violated invariant name from a line
fn extract_violated_invariant(line: &str) -> Option<String> {
    // Patterns like "Invariant Inv1 violated" or "inv=Inv1 violated"
    let patterns = ["Invariant ", "inv=", "INV="];

    for pattern in patterns {
        if let Some(idx) = line.find(pattern) {
            let start = idx + pattern.len();
            let rest = &line[start..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

/// Extract action name from state line
fn extract_action_name(line: &str) -> Option<String> {
    // Patterns like "State 1: Action" or "state: ActionName"
    if let Some(idx) = line.find(':') {
        let after = line[idx + 1..].trim();
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

/// Parse a variable assignment line
fn parse_variable_assignment(line: &str) -> Option<(String, CounterexampleValue)> {
    // Patterns: "var = value" or "var := value" or "/\\ var = value"
    let line = line.trim_start_matches("/\\").trim();

    let eq_patterns = [" := ", " = "];

    for pattern in eq_patterns {
        if let Some(idx) = line.find(pattern) {
            let var = line[..idx].trim().to_string();
            let value_str = line[idx + pattern.len()..].trim();

            // Skip empty or system variables
            if var.is_empty() || var.starts_with('_') {
                continue;
            }

            let value = parse_value(value_str);
            return Some((var, value));
        }
    }
    None
}

/// Parse a TLA+ value string into CounterexampleValue
fn parse_value(s: &str) -> CounterexampleValue {
    let s = s.trim();

    // Boolean
    if s == "TRUE" || s == "true" {
        return CounterexampleValue::Bool(true);
    }
    if s == "FALSE" || s == "false" {
        return CounterexampleValue::Bool(false);
    }

    // Integer
    if let Ok(n) = s.parse::<i128>() {
        return CounterexampleValue::Int {
            value: n,
            type_hint: None,
        };
    }

    // String (quoted)
    if s.starts_with('"') && s.ends_with('"') {
        return CounterexampleValue::String(s[1..s.len() - 1].to_string());
    }

    // Set
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len() - 1];
        if inner.is_empty() {
            return CounterexampleValue::Set(Vec::new());
        }
        let elements: Vec<CounterexampleValue> =
            inner.split(',').map(|e| parse_value(e.trim())).collect();
        return CounterexampleValue::Set(elements);
    }

    // Sequence
    if s.starts_with("<<") && s.ends_with(">>") {
        let inner = &s[2..s.len() - 2];
        if inner.is_empty() {
            return CounterexampleValue::Sequence(Vec::new());
        }
        let elements: Vec<CounterexampleValue> =
            inner.split(',').map(|e| parse_value(e.trim())).collect();
        return CounterexampleValue::Sequence(elements);
    }

    // Record
    if s.starts_with('[') && s.ends_with(']') && s.contains("|->") {
        let inner = &s[1..s.len() - 1];
        let mut fields = HashMap::new();
        for field in inner.split(',') {
            if let Some(idx) = field.find("|->") {
                let key = field[..idx].trim().to_string();
                let val = parse_value(field[idx + 3..].trim());
                fields.insert(key, val);
            }
        }
        return CounterexampleValue::Record(fields);
    }

    // Unknown - store as string
    CounterexampleValue::Unknown(s.to_string())
}

/// Extract diagnostics from output
fn extract_diagnostics(output: &str) -> Vec<String> {
    let mut diagnostics = Vec::new();

    for line in output.lines() {
        let trimmed = line.trim();

        // Collect warning and info messages
        if trimmed.starts_with("WARN")
            || trimmed.starts_with("INFO")
            || trimmed.starts_with("WARNING")
        {
            diagnostics.push(trimmed.to_string());
        }
    }

    // Limit to reasonable number
    diagnostics.truncate(50);
    diagnostics
}

/// Extract type error message
fn extract_type_error(output: &str) -> String {
    for line in output.lines() {
        if line.contains("Type error") || line.contains("TYPEERROR") {
            return line.trim().to_string();
        }
    }
    "Type checking failed".to_string()
}

/// Extract parse error message
fn extract_parse_error(output: &str) -> String {
    for line in output.lines() {
        if line.contains("Parse error") || line.contains("PARSEERROR") {
            return line.trim().to_string();
        }
    }
    "Parsing failed".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_parse_success() {
        let output = ApalacheOutput {
            stdout: "The outcome is: NoError\nChecker reports no error".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(1),
        };

        let result = parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Proven));
        assert_eq!(result.backend, BackendId::Apalache);
    }

    #[test]
    fn test_parse_counterexample() {
        let output = ApalacheOutput {
            stdout: "Found an error\nInvariant TypeOK violated\nState 0: Init\nx = 1\ny = 2"
                .to_string(),
            stderr: String::new(),
            exit_code: Some(12),
            duration: Duration::from_secs(1),
        };

        let result = parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
        assert!(result.counterexample.is_some());

        let ce = result.counterexample.unwrap();
        assert!(!ce.failed_checks.is_empty());
        assert_eq!(ce.failed_checks[0].check_id, "TypeOK");
    }

    #[test]
    fn test_parse_variable_assignment() {
        let result = parse_variable_assignment("x = 42");
        assert!(result.is_some());
        let (var, val) = result.unwrap();
        assert_eq!(var, "x");
        assert!(matches!(val, CounterexampleValue::Int { value: 42, .. }));
    }

    #[test]
    fn test_parse_value_set() {
        let val = parse_value("{1, 2, 3}");
        if let CounterexampleValue::Set(elements) = val {
            assert_eq!(elements.len(), 3);
        } else {
            panic!("Expected Set");
        }
    }

    #[test]
    fn test_parse_value_sequence() {
        let val = parse_value("<<a, b, c>>");
        if let CounterexampleValue::Sequence(elements) = val {
            assert_eq!(elements.len(), 3);
        } else {
            panic!("Expected Sequence");
        }
    }
}
