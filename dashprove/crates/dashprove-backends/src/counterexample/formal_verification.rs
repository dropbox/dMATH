//! Shared helpers for traditional formal verification counterexamples
//!
//! These helpers construct structured counterexamples with populated
//! `failed_checks` entries from outputs emitted by traditional formal
//! verification backends (CBMC, SPIN, KLEE, Infer, DIVINE, SMACK, etc.).

use crate::counterexample::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample,
};
use std::collections::HashMap;

/// Build a structured counterexample for bounded model checker backends (CBMC, SMACK, ESBMC).
///
/// Parses trace output to extract variable assignments and property violations.
pub fn build_bmc_counterexample(
    stdout: &str,
    stderr: &str,
    backend_name: &str,
    property_type: Option<&str>,
) -> Option<StructuredCounterexample> {
    let combined = format!("{}\n{}", stdout, stderr);

    // Look for counterexample indicators
    let has_counterexample = combined.contains("VERIFICATION FAILED")
        || combined.contains("FAILURE")
        || combined.contains("Counterexample")
        || combined.contains("counterexample")
        || combined.contains("State")
        || combined.contains("Trace");

    if !has_counterexample {
        return None;
    }

    let mut witness = HashMap::new();

    // Extract property violation type
    let property = property_type
        .map(String::from)
        .or_else(|| extract_property_type(&combined))
        .unwrap_or_else(|| "assertion".to_string());

    witness.insert(
        "property_violated".to_string(),
        CounterexampleValue::String(property.clone()),
    );

    // Extract trace states if present
    let trace_lines: Vec<String> = combined
        .lines()
        .filter(|l| {
            l.contains("State")
                || l.contains("Trace")
                || l.contains("Assignment")
                || l.contains("=")
        })
        .take(20) // Limit trace size
        .map(String::from)
        .collect();

    // Try to extract variable assignments from trace
    for line in &trace_lines {
        if let Some((var, val)) = parse_assignment(line) {
            witness.insert(var, val);
        }
    }

    // Extract loop bound if unwinding-related
    if let Some(bound) = extract_unwind_bound(&combined) {
        witness.insert(
            "unwind_bound".to_string(),
            CounterexampleValue::Int {
                value: bound as i128,
                type_hint: None,
            },
        );
    }

    // Extract source location if present
    let location = extract_source_location(&combined);

    let failed_checks =
        build_bmc_failed_checks(&combined, backend_name, &property, location.as_deref());
    let raw = if trace_lines.is_empty() {
        Some(combined.lines().take(50).collect::<Vec<_>>().join("\n"))
    } else {
        Some(trace_lines.join("\n"))
    };

    Some(StructuredCounterexample {
        witness,
        failed_checks,
        playback_test: None,
        trace: vec![],
        raw,
        minimized: false,
    })
}

fn extract_property_type(output: &str) -> Option<String> {
    let patterns = [
        ("array bounds", "array_bounds_violation"),
        ("buffer overflow", "buffer_overflow"),
        ("null pointer", "null_pointer_dereference"),
        ("division by zero", "division_by_zero"),
        ("assertion", "assertion_violation"),
        ("memory leak", "memory_leak"),
        ("use after free", "use_after_free"),
        ("double free", "double_free"),
        ("uninitialized", "uninitialized_variable"),
        ("overflow", "arithmetic_overflow"),
        ("deadlock", "deadlock"),
        ("data race", "data_race"),
    ];

    let lower = output.to_lowercase();
    for (pattern, name) in patterns {
        if lower.contains(pattern) {
            return Some(name.to_string());
        }
    }
    None
}

fn parse_assignment(line: &str) -> Option<(String, CounterexampleValue)> {
    // Try patterns like "x = 42" or "var: 3.14"
    let patterns = ["=", ":"];
    for sep in patterns {
        if let Some(idx) = line.find(sep) {
            let var_part = line[..idx].trim();
            let val_part = line[idx + 1..].trim();

            // Extract variable name (last word before separator)
            let var_name = var_part.split_whitespace().last()?;
            if var_name.is_empty() || var_name.starts_with("//") {
                continue;
            }

            // Try to parse value
            if let Ok(i) = val_part
                .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '-')
                .parse::<i64>()
            {
                return Some((
                    var_name.to_string(),
                    CounterexampleValue::Int {
                        value: i as i128,
                        type_hint: None,
                    },
                ));
            }
            if let Ok(f) = val_part
                .trim_end_matches(|c: char| {
                    !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E'
                })
                .parse::<f64>()
            {
                return Some((
                    var_name.to_string(),
                    CounterexampleValue::Float { value: f },
                ));
            }
            // String value
            if val_part.starts_with('"') || val_part.starts_with('\'') {
                let cleaned = val_part.trim_matches(|c| c == '"' || c == '\'');
                return Some((
                    var_name.to_string(),
                    CounterexampleValue::String(cleaned.to_string()),
                ));
            }
        }
    }
    None
}

fn extract_unwind_bound(output: &str) -> Option<usize> {
    // Look for patterns like "unwinding iteration 10" or "loop bound: 5"
    for line in output.lines() {
        let lower = line.to_lowercase();
        if lower.contains("unwind") || lower.contains("loop bound") {
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<usize>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn extract_source_location(output: &str) -> Option<String> {
    // Look for patterns like "file.c:42" or "at line 42"
    for line in output.lines() {
        // Pattern: filename:line
        if let Some(caps) = line
            .split_whitespace()
            .find(|w| w.contains(':') && w.chars().filter(|c| *c == ':').count() == 1)
        {
            let parts: Vec<&str> = caps.split(':').collect();
            if parts.len() == 2 && parts[1].chars().all(|c| c.is_ascii_digit()) {
                return Some(caps.to_string());
            }
        }
    }
    None
}

fn build_bmc_failed_checks(
    output: &str,
    backend_name: &str,
    property: &str,
    location: Option<&str>,
) -> Vec<FailedCheck> {
    let mut description = format!(
        "{} bounded model checking found property violation.",
        backend_name
    );

    description.push_str(&format!(" Property: {}.", property));

    // Add location if available
    if let Some(loc) = location {
        description.push_str(&format!(" Location: {}.", loc));
    }

    // Add details based on property type
    match property {
        "array_bounds_violation" | "buffer_overflow" => {
            description.push_str(" Array index out of bounds or buffer overflow detected.");
        }
        "null_pointer_dereference" => {
            description.push_str(" Null pointer dereference detected.");
        }
        "division_by_zero" => {
            description.push_str(" Division by zero detected.");
        }
        "memory_leak" => {
            description.push_str(" Memory leak detected - allocated memory not freed.");
        }
        "arithmetic_overflow" => {
            description.push_str(" Arithmetic overflow detected.");
        }
        _ => {}
    }

    // Check for unwinding issues
    if output.to_lowercase().contains("unwinding assertion") {
        description
            .push_str(" Note: Unwinding assertion violated - loop bound may be insufficient.");
    }

    // Parse location string into SourceLocation if present
    let source_location = location.and_then(|loc| {
        let parts: Vec<&str> = loc.split(':').collect();
        if parts.len() >= 2 {
            if let Ok(line) = parts[1].parse::<u32>() {
                return Some(SourceLocation {
                    file: parts[0].to_string(),
                    line,
                    column: parts.get(2).and_then(|c| c.parse().ok()),
                });
            }
        }
        None
    });

    vec![FailedCheck {
        check_id: format!(
            "{}_{}_failure",
            backend_name.to_lowercase().replace(' ', "_"),
            property.replace(' ', "_")
        ),
        description,
        location: source_location,
        function: None,
    }]
}

/// Build a structured counterexample for model checker backends with LTL support (SPIN, DIVINE, mCRL2).
///
/// Parses trail/trace output to extract state sequences and property violations.
pub fn build_model_checker_counterexample(
    stdout: &str,
    stderr: &str,
    backend_name: &str,
    trail_content: Option<&str>,
) -> Option<StructuredCounterexample> {
    let combined = format!("{}\n{}", stdout, stderr);
    let lower = combined.to_lowercase();

    // Check for success indicators first - if we see these, no counterexample
    let is_success = (lower.contains("verified") || lower.contains("properties"))
        && !lower.contains("violated")
        && !lower.contains("error:")
        && !lower.contains("invalid");

    if is_success {
        return None;
    }

    // Look for counterexample indicators
    // Be more specific to avoid false positives
    let has_counterexample = lower.contains("error:")
        || lower.contains("violated")
        || lower.contains("trail")
        || lower.contains("counterexample")
        || lower.contains("deadlock")
        || lower.contains("invalid end state")
        || (lower.contains("assertion") && lower.contains("fail"));

    if !has_counterexample {
        return None;
    }

    let mut witness = HashMap::new();

    // Determine violation type
    let violation_type = if combined.to_lowercase().contains("deadlock") {
        "deadlock"
    } else if combined.to_lowercase().contains("liveness") || combined.contains("acceptance cycle")
    {
        "liveness_violation"
    } else if combined.to_lowercase().contains("assertion") {
        "assertion_violation"
    } else {
        "property_violation"
    };

    witness.insert(
        "violation_type".to_string(),
        CounterexampleValue::String(violation_type.to_string()),
    );

    // Extract state count if available
    if let Some(states) = extract_state_count(&combined) {
        witness.insert(
            "states_explored".to_string(),
            CounterexampleValue::Int {
                value: states as i128,
                type_hint: None,
            },
        );
    }

    // Extract depth if available
    if let Some(depth) = extract_search_depth(&combined) {
        witness.insert(
            "search_depth".to_string(),
            CounterexampleValue::Int {
                value: depth as i128,
                type_hint: None,
            },
        );
    }

    let errors: Vec<String> = combined
        .lines()
        .filter(|l| {
            l.to_lowercase().contains("error")
                || l.to_lowercase().contains("violated")
                || l.to_lowercase().contains("assertion")
        })
        .take(10)
        .map(String::from)
        .collect();

    let failed_checks = build_model_checker_failed_checks(&combined, backend_name, violation_type);
    let raw = if let Some(trail) = trail_content {
        Some(trail.to_string())
    } else if !errors.is_empty() {
        Some(errors.join("\n"))
    } else {
        Some(combined.lines().take(30).collect::<Vec<_>>().join("\n"))
    };

    Some(StructuredCounterexample {
        witness,
        failed_checks,
        playback_test: None,
        trace: vec![],
        raw,
        minimized: false,
    })
}

fn extract_state_count(output: &str) -> Option<u64> {
    for line in output.lines() {
        let lower = line.to_lowercase();
        if lower.contains("state")
            && (lower.contains("explored") || lower.contains("stored") || lower.contains("visited"))
        {
            for word in line.split_whitespace() {
                if let Ok(n) = word.replace(',', "").parse::<u64>() {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

fn extract_search_depth(output: &str) -> Option<u64> {
    for line in output.lines() {
        let lower = line.to_lowercase();
        if lower.contains("depth") {
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<u64>() {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

fn build_model_checker_failed_checks(
    output: &str,
    backend_name: &str,
    violation_type: &str,
) -> Vec<FailedCheck> {
    let mut description = format!(
        "{} model checking found {}.",
        backend_name,
        violation_type.replace('_', " ")
    );

    // Add details based on violation type
    match violation_type {
        "deadlock" => {
            description.push_str(" System reached a state with no enabled transitions.");
        }
        "liveness_violation" => {
            description.push_str(" An acceptance cycle was found - liveness property violated.");
        }
        "assertion_violation" => {
            description.push_str(" An assertion in the model was violated.");
        }
        _ => {}
    }

    // Add state space info if available
    if let Some(states) = extract_state_count(output) {
        description.push_str(&format!(" States explored: {}.", states));
    }
    if let Some(depth) = extract_search_depth(output) {
        description.push_str(&format!(" Search depth: {}.", depth));
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_{}_failure",
            backend_name.to_lowercase().replace(' ', "_"),
            violation_type
        ),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for symbolic execution backends (KLEE, Manticore, Angr).
///
/// Parses error output to extract test case inputs that trigger bugs.
pub fn build_symbolic_execution_counterexample(
    stdout: &str,
    stderr: &str,
    backend_name: &str,
    errors: &[String],
) -> Option<StructuredCounterexample> {
    if errors.is_empty() {
        return None;
    }

    let combined = format!("{}\n{}", stdout, stderr);
    let mut witness = HashMap::new();

    // Categorize error types
    let mut error_types: Vec<String> = Vec::new();
    for error in errors {
        let lower = error.to_lowercase();
        if lower.contains("memory") || lower.contains("heap") || lower.contains("buffer") {
            if !error_types.contains(&"memory_error".to_string()) {
                error_types.push("memory_error".to_string());
            }
        } else if lower.contains("assert") {
            if !error_types.contains(&"assertion_failure".to_string()) {
                error_types.push("assertion_failure".to_string());
            }
        } else if lower.contains("division") || lower.contains("div") {
            if !error_types.contains(&"division_error".to_string()) {
                error_types.push("division_error".to_string());
            }
        } else if !error_types.contains(&"execution_error".to_string()) {
            error_types.push("execution_error".to_string());
        }
    }

    witness.insert(
        "error_types".to_string(),
        CounterexampleValue::Sequence(
            error_types
                .iter()
                .map(|s| CounterexampleValue::String(s.clone()))
                .collect(),
        ),
    );

    witness.insert(
        "num_errors".to_string(),
        CounterexampleValue::Int {
            value: errors.len() as i128,
            type_hint: None,
        },
    );

    // Extract execution stats if available
    if let Some(paths) = extract_paths_explored(&combined) {
        witness.insert(
            "paths_explored".to_string(),
            CounterexampleValue::Int {
                value: paths as i128,
                type_hint: None,
            },
        );
    }

    if let Some(tests) = extract_tests_generated(&combined) {
        witness.insert(
            "tests_generated".to_string(),
            CounterexampleValue::Int {
                value: tests as i128,
                type_hint: None,
            },
        );
    }

    let failed_checks = build_symbolic_execution_failed_checks(backend_name, errors, &error_types);
    let raw = Some(errors.join("\n"));

    Some(StructuredCounterexample {
        witness,
        failed_checks,
        playback_test: None,
        trace: vec![],
        raw,
        minimized: false,
    })
}

fn extract_paths_explored(output: &str) -> Option<u64> {
    for line in output.lines() {
        let lower = line.to_lowercase();
        if lower.contains("path") && (lower.contains("completed") || lower.contains("explored")) {
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<u64>() {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

fn extract_tests_generated(output: &str) -> Option<u64> {
    for line in output.lines() {
        let lower = line.to_lowercase();
        if lower.contains("test") && lower.contains("generated") {
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<u64>() {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

fn build_symbolic_execution_failed_checks(
    backend_name: &str,
    errors: &[String],
    error_types: &[String],
) -> Vec<FailedCheck> {
    let mut description = format!(
        "{} symbolic execution found {} error(s).",
        backend_name,
        errors.len()
    );

    if !error_types.is_empty() {
        description.push_str(&format!(" Error types: [{}].", error_types.join(", ")));
    }

    // Add first few error messages
    let sample_errors: Vec<&str> = errors.iter().take(3).map(|s| s.as_str()).collect();
    if !sample_errors.is_empty() {
        for (i, err) in sample_errors.iter().enumerate() {
            let short_err: String = err.chars().take(80).collect();
            description.push_str(&format!(" [{}] {}", i + 1, short_err));
            if err.len() > 80 {
                description.push_str("...");
            }
        }
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_symbolic_execution_failure",
            backend_name.to_lowercase().replace(' ', "_")
        ),
        description,
        location: None,
        function: None,
    }]
}

/// Build a structured counterexample for static analyzer backends (Infer, Astree, Polyspace).
///
/// Parses issue reports to extract bug descriptions and locations.
pub fn build_static_analysis_counterexample(
    issues: &[serde_json::Value],
    backend_name: &str,
) -> Option<StructuredCounterexample> {
    if issues.is_empty() {
        return None;
    }

    let mut witness = HashMap::new();

    witness.insert(
        "num_issues".to_string(),
        CounterexampleValue::Int {
            value: issues.len() as i128,
            type_hint: None,
        },
    );

    // Categorize issues by type
    let mut issue_types: HashMap<String, u64> = HashMap::new();
    for issue in issues {
        let bug_type = issue
            .get("bug_type")
            .or_else(|| issue.get("type"))
            .or_else(|| issue.get("category"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        *issue_types.entry(bug_type.to_string()).or_insert(0) += 1;
    }

    witness.insert(
        "issue_types".to_string(),
        CounterexampleValue::Sequence(
            issue_types
                .keys()
                .map(|s| CounterexampleValue::String(s.clone()))
                .collect(),
        ),
    );

    // Extract first issue details
    if let Some(first) = issues.first() {
        if let Some(file) = first.get("file").and_then(|v| v.as_str()) {
            witness.insert(
                "file".to_string(),
                CounterexampleValue::String(file.to_string()),
            );
        }
        if let Some(line) = first.get("line").and_then(|v| v.as_u64()) {
            witness.insert(
                "line".to_string(),
                CounterexampleValue::Int {
                    value: line as i128,
                    type_hint: None,
                },
            );
        }
    }

    let failed_checks = build_static_analysis_failed_checks(issues, backend_name, &issue_types);
    let raw = Some(
        issues
            .iter()
            .take(5)
            .filter_map(|i| serde_json::to_string(i).ok())
            .collect::<Vec<_>>()
            .join("\n"),
    );

    Some(StructuredCounterexample {
        witness,
        failed_checks,
        playback_test: None,
        trace: vec![],
        raw,
        minimized: false,
    })
}

fn build_static_analysis_failed_checks(
    issues: &[serde_json::Value],
    backend_name: &str,
    issue_types: &HashMap<String, u64>,
) -> Vec<FailedCheck> {
    let mut description = format!(
        "{} static analysis found {} issue(s).",
        backend_name,
        issues.len()
    );

    // Add issue type breakdown
    if !issue_types.is_empty() {
        let type_summary: Vec<String> = issue_types
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect();
        description.push_str(&format!(" Issue breakdown: [{}].", type_summary.join(", ")));
    }

    // Add first issue details
    if let Some(first) = issues.first() {
        let bug_type = first
            .get("bug_type")
            .or_else(|| first.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let qualifier = first
            .get("qualifier")
            .or_else(|| first.get("message"))
            .or_else(|| first.get("description"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let file = first.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let line = first.get("line").and_then(|v| v.as_u64()).unwrap_or(0);

        if !qualifier.is_empty() {
            let short_qualifier: String = qualifier.chars().take(100).collect();
            description.push_str(&format!(" First issue: {} - {}", bug_type, short_qualifier));
            if qualifier.len() > 100 {
                description.push_str("...");
            }
        }

        if !file.is_empty() {
            description.push_str(&format!(" at {}:{}", file, line));
        }
    }

    vec![FailedCheck {
        check_id: format!(
            "{}_static_analysis_failure",
            backend_name.to_lowercase().replace(' ', "_")
        ),
        description,
        location: issues.first().and_then(|i| {
            let file = i.get("file").and_then(|v| v.as_str())?;
            let line = i.get("line").and_then(|v| v.as_u64())?;
            Some(SourceLocation {
                file: file.to_string(),
                line: line as u32,
                column: i.get("column").and_then(|v| v.as_u64()).map(|c| c as u32),
            })
        }),
        function: issues.first().and_then(|i| {
            i.get("procedure")
                .or_else(|| i.get("function"))
                .and_then(|v| v.as_str())
                .map(String::from)
        }),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_bmc_no_counterexample() {
        let stdout = "VERIFICATION SUCCESSFUL\nAll properties verified";
        let stderr = "";
        assert!(build_bmc_counterexample(stdout, stderr, "CBMC", None).is_none());
    }

    #[test]
    fn test_bmc_counterexample() {
        let stdout = r#"
VERIFICATION FAILED
Counterexample trace:
State 0: x = 0
State 1: x = 10
array bounds violation at test.c:42
        "#;
        let stderr = "";
        let cex = build_bmc_counterexample(stdout, stderr, "CBMC", None).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("cbmc"));
        assert!(cex.failed_checks[0].description.contains("array"));
        assert!(cex.witness.contains_key("property_violated"));
    }

    #[test]
    fn test_bmc_null_pointer() {
        let stdout = "FAILURE: null pointer dereference\nptr = NULL";
        let stderr = "";
        let cex = build_bmc_counterexample(stdout, stderr, "ESBMC", None).unwrap();
        // Description uses "Null pointer" (capitalized)
        assert!(cex.failed_checks[0]
            .description
            .to_lowercase()
            .contains("null pointer"));
    }

    #[test]
    fn test_model_checker_no_error() {
        // "verified" without error indicators should not produce a counterexample
        let stdout = "All properties verified successfully\n0 errors found";
        let stderr = "";
        assert!(build_model_checker_counterexample(stdout, stderr, "SPIN", None).is_none());
    }

    #[test]
    fn test_model_checker_deadlock() {
        let stdout = r#"
pan:1: invalid end state (at depth 45)
pan: wrote model.pml.trail
States explored: 1234
        "#;
        let stderr = "";
        let cex = build_model_checker_counterexample(stdout, stderr, "SPIN", None).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("spin"));
    }

    #[test]
    fn test_model_checker_with_trail() {
        let stdout = "error: assertion violated";
        let stderr = "";
        let trail = "proc 0 (init) line 10\nproc 0 (init) line 15";
        let cex = build_model_checker_counterexample(stdout, stderr, "SPIN", Some(trail)).unwrap();
        assert!(cex.raw.unwrap().contains("line 10"));
    }

    #[test]
    fn test_symbolic_execution_no_errors() {
        let errors: Vec<String> = vec![];
        assert!(build_symbolic_execution_counterexample("", "", "KLEE", &errors).is_none());
    }

    #[test]
    fn test_symbolic_execution_with_errors() {
        let errors = vec![
            "ERROR: memory error: out of bound pointer".to_string(),
            "ERROR: assertion failure".to_string(),
        ];
        let stdout = "KLEE: done: total paths = 42";
        let cex = build_symbolic_execution_counterexample(stdout, "", "KLEE", &errors).unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("klee"));
        assert_eq!(
            cex.witness.get("num_errors"),
            Some(&CounterexampleValue::Int {
                value: 2,
                type_hint: None
            })
        );
    }

    #[test]
    fn test_static_analysis_no_issues() {
        let issues: Vec<serde_json::Value> = vec![];
        assert!(build_static_analysis_counterexample(&issues, "Infer").is_none());
    }

    #[test]
    fn test_static_analysis_with_issues() {
        let issues = vec![
            json!({
                "bug_type": "NULL_DEREFERENCE",
                "qualifier": "pointer `p` may be null",
                "file": "test.c",
                "line": 42,
                "procedure": "foo"
            }),
            json!({
                "bug_type": "RESOURCE_LEAK",
                "qualifier": "memory allocated here may be leaked",
                "file": "test.c",
                "line": 50,
                "procedure": "bar"
            }),
        ];
        let cex = build_static_analysis_counterexample(&issues, "Infer").unwrap();
        assert!(!cex.failed_checks.is_empty());
        assert!(cex.failed_checks[0].check_id.contains("infer"));
        assert!(cex.failed_checks[0].description.contains("2 issue"));
        assert!(cex.failed_checks[0]
            .description
            .contains("NULL_DEREFERENCE"));
        assert_eq!(
            cex.failed_checks[0].location,
            Some(SourceLocation {
                file: "test.c".to_string(),
                line: 42,
                column: None,
            })
        );
        assert_eq!(cex.failed_checks[0].function, Some("foo".to_string()));
    }

    #[test]
    fn test_parse_assignment() {
        assert!(parse_assignment("x = 42").is_some());
        assert!(parse_assignment("value: 3.14").is_some());
        assert!(parse_assignment("no assignment here").is_none());
    }

    #[test]
    fn test_extract_property_type() {
        assert_eq!(
            extract_property_type("array bounds violation"),
            Some("array_bounds_violation".to_string())
        );
        assert_eq!(
            extract_property_type("null pointer dereference detected"),
            Some("null_pointer_dereference".to_string())
        );
        assert_eq!(
            extract_property_type("deadlock found"),
            Some("deadlock".to_string())
        );
    }
}
