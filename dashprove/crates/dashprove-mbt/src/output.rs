//! Test output formatters
//!
//! This module provides formatters to convert generated tests into various
//! output formats like JSON, Rust test code, and human-readable reports.

use serde::{Deserialize, Serialize};

use crate::generator::{CoverageReport, GenerationResult, TestCase};
use crate::model::{ModelAction, ModelState, ModelValue};

/// Output format for generated tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON format
    Json,
    /// Rust test code
    Rust,
    /// Python test code
    Python,
    /// Plain text report
    Text,
    /// Markdown report
    Markdown,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Rust => write!(f, "rust"),
            OutputFormat::Python => write!(f, "python"),
            OutputFormat::Text => write!(f, "text"),
            OutputFormat::Markdown => write!(f, "markdown"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "rust" | "rs" => Ok(OutputFormat::Rust),
            "python" | "py" => Ok(OutputFormat::Python),
            "text" | "txt" => Ok(OutputFormat::Text),
            "markdown" | "md" => Ok(OutputFormat::Markdown),
            _ => Err(format!("Unknown output format: {s}")),
        }
    }
}

/// Format generation results
pub fn format_results(result: &GenerationResult, format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => format_json(result),
        OutputFormat::Rust => format_rust(result),
        OutputFormat::Python => format_python(result),
        OutputFormat::Text => format_text(result),
        OutputFormat::Markdown => format_markdown(result),
    }
}

/// Format as JSON
fn format_json(result: &GenerationResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("Error: {e}"))
}

/// Format as Rust test code
fn format_rust(result: &GenerationResult) -> String {
    let mut output = String::new();

    output.push_str("//! Auto-generated model-based tests\n");
    output.push_str("//!\n");
    output.push_str(&format!(
        "//! Generated {} tests with {:.1}% state coverage\n",
        result.stats.tests_generated,
        result.coverage.state_coverage_pct()
    ));
    output.push('\n');
    output.push_str("#[cfg(test)]\n");
    output.push_str("mod mbt_tests {\n");
    output.push_str("    use super::*;\n\n");

    for test in &result.tests {
        output.push_str(&format_rust_test(test));
        output.push('\n');
    }

    output.push_str("}\n");
    output
}

/// Format a single test as Rust code
fn format_rust_test(test: &TestCase) -> String {
    let mut output = String::new();

    output.push_str(&format!("    /// {}\n", test.description));
    output.push_str("    #[test]\n");
    output.push_str(&format!("    fn {}() {{\n", sanitize_ident(&test.id)));

    // Setup initial state
    output.push_str("        // Initial state\n");
    output.push_str(&format_rust_state(&test.initial_state, "        "));

    // Execute actions and verify states
    for (i, (action, expected)) in test
        .actions
        .iter()
        .zip(test.expected_states.iter())
        .enumerate()
    {
        output.push_str(&format!("\n        // Step {}: {}\n", i + 1, action.name));
        output.push_str(&format!(
            "        // Execute: {}\n",
            format_rust_action(action)
        ));
        output.push_str("        // Expected state:\n");
        output.push_str(&format_rust_state_assertions(expected, "        "));
    }

    output.push_str("    }\n");
    output
}

/// Format model state as Rust code
fn format_rust_state(state: &ModelState, indent: &str) -> String {
    let mut output = String::new();
    for (name, value) in &state.variables {
        output.push_str(&format!(
            "{}let {} = {};\n",
            indent,
            sanitize_ident(name),
            format_rust_value(value)
        ));
    }
    output
}

/// Format state assertions as Rust code
fn format_rust_state_assertions(state: &ModelState, indent: &str) -> String {
    let mut output = String::new();
    for (name, value) in &state.variables {
        output.push_str(&format!(
            "{}assert_eq!({}, {});\n",
            indent,
            sanitize_ident(name),
            format_rust_value(value)
        ));
    }
    output
}

/// Format a model value as Rust expression
fn format_rust_value(value: &ModelValue) -> String {
    match value {
        ModelValue::Bool(b) => b.to_string(),
        ModelValue::Int(i) => i.to_string(),
        ModelValue::String(s) => format!("\"{s}\".to_string()"),
        ModelValue::Set(elems) => {
            let items: Vec<_> = elems.iter().map(format_rust_value).collect();
            format!(
                "vec![{}].into_iter().collect::<HashSet<_>>()",
                items.join(", ")
            )
        }
        ModelValue::Sequence(elems) => {
            let items: Vec<_> = elems.iter().map(format_rust_value).collect();
            format!("vec![{}]", items.join(", "))
        }
        ModelValue::Record(fields) => {
            let items: Vec<_> = fields
                .iter()
                .map(|(k, v)| format!("{}: {}", sanitize_ident(k), format_rust_value(v)))
                .collect();
            format!("{{ {} }}", items.join(", "))
        }
        ModelValue::Function(mappings) => {
            let items: Vec<_> = mappings
                .iter()
                .map(|(k, v)| format!("({}, {})", format_rust_value(k), format_rust_value(v)))
                .collect();
            format!(
                "vec![{}].into_iter().collect::<HashMap<_, _>>()",
                items.join(", ")
            )
        }
        ModelValue::Null => "None".to_string(),
    }
}

/// Format an action as Rust method call
fn format_rust_action(action: &ModelAction) -> String {
    if action.parameters.is_empty() {
        format!("{}()", sanitize_ident(&action.name))
    } else {
        let params: Vec<_> = action.parameters.iter().map(format_rust_value).collect();
        format!("{}({})", sanitize_ident(&action.name), params.join(", "))
    }
}

/// Sanitize a string for use as Rust identifier
fn sanitize_ident(s: &str) -> String {
    s.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
        .to_lowercase()
}

/// Format as Python test code
fn format_python(result: &GenerationResult) -> String {
    let mut output = String::new();

    output.push_str("\"\"\"Auto-generated model-based tests\n");
    output.push_str(&format!(
        "Generated {} tests with {:.1}% state coverage\n",
        result.stats.tests_generated,
        result.coverage.state_coverage_pct()
    ));
    output.push_str("\"\"\"\n\n");
    output.push_str("import pytest\n\n");

    for test in &result.tests {
        output.push_str(&format_python_test(test));
        output.push('\n');
    }

    output
}

/// Format a single test as Python code
fn format_python_test(test: &TestCase) -> String {
    let mut output = String::new();

    output.push_str(&format!("def test_{}():\n", sanitize_ident(&test.id)));
    output.push_str(&format!("    \"\"\"{}.\"\"\"\n", test.description));

    // Setup initial state
    output.push_str("    # Initial state\n");
    output.push_str(&format_python_state(&test.initial_state, "    "));

    // Execute actions and verify states
    for (i, (action, expected)) in test
        .actions
        .iter()
        .zip(test.expected_states.iter())
        .enumerate()
    {
        output.push_str(&format!("\n    # Step {}: {}\n", i + 1, action.name));
        output.push_str(&format!("    # {}\n", format_python_action(action)));
        for (name, value) in &expected.variables {
            output.push_str(&format!(
                "    assert {} == {}\n",
                sanitize_ident(name),
                format_python_value(value)
            ));
        }
    }

    output
}

/// Format model state as Python code
fn format_python_state(state: &ModelState, indent: &str) -> String {
    let mut output = String::new();
    for (name, value) in &state.variables {
        output.push_str(&format!(
            "{}{} = {}\n",
            indent,
            sanitize_ident(name),
            format_python_value(value)
        ));
    }
    output
}

/// Format a model value as Python expression
fn format_python_value(value: &ModelValue) -> String {
    match value {
        ModelValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        ModelValue::Int(i) => i.to_string(),
        ModelValue::String(s) => format!("\"{s}\""),
        ModelValue::Set(elems) => {
            let items: Vec<_> = elems.iter().map(format_python_value).collect();
            format!("{{{}}}", items.join(", "))
        }
        ModelValue::Sequence(elems) => {
            let items: Vec<_> = elems.iter().map(format_python_value).collect();
            format!("[{}]", items.join(", "))
        }
        ModelValue::Record(fields) => {
            let items: Vec<_> = fields
                .iter()
                .map(|(k, v)| format!("\"{}\": {}", k, format_python_value(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        ModelValue::Function(mappings) => {
            let items: Vec<_> = mappings
                .iter()
                .map(|(k, v)| format!("{}: {}", format_python_value(k), format_python_value(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        ModelValue::Null => "None".to_string(),
    }
}

/// Format an action as Python method call
fn format_python_action(action: &ModelAction) -> String {
    if action.parameters.is_empty() {
        format!("{}()", sanitize_ident(&action.name))
    } else {
        let params: Vec<_> = action.parameters.iter().map(format_python_value).collect();
        format!("{}({})", sanitize_ident(&action.name), params.join(", "))
    }
}

/// Format as plain text report
fn format_text(result: &GenerationResult) -> String {
    let mut output = String::new();

    output.push_str("Model-Based Test Generation Report\n");
    output.push_str("===================================\n\n");

    // Summary
    output.push_str("Summary\n");
    output.push_str("-------\n");
    output.push_str(&format!(
        "Tests generated: {}\n",
        result.stats.tests_generated
    ));
    output.push_str(&format!("Total steps: {}\n", result.stats.total_steps));
    output.push_str(&format!(
        "Average test length: {:.1}\n",
        result.stats.avg_test_length
    ));
    output.push_str(&format!(
        "Max test length: {}\n",
        result.stats.max_test_length
    ));
    output.push_str(&format!(
        "Generation time: {}ms\n\n",
        result.stats.duration_ms
    ));

    // Coverage
    output.push_str(&format_text_coverage(&result.coverage));

    // Tests
    output.push_str("\nTest Cases\n");
    output.push_str("----------\n");
    for test in &result.tests {
        output.push_str(&format_text_test(test));
    }

    output
}

/// Format coverage as text
fn format_text_coverage(coverage: &CoverageReport) -> String {
    let mut output = String::new();

    output.push_str("Coverage\n");
    output.push_str("--------\n");
    output.push_str(&format!(
        "States: {}/{} ({:.1}%)\n",
        coverage.states_covered,
        coverage.states_total,
        coverage.state_coverage_pct()
    ));
    output.push_str(&format!(
        "Transitions: {}/{} ({:.1}%)\n",
        coverage.transitions_covered,
        coverage.transitions_total,
        coverage.transition_coverage_pct()
    ));
    output.push_str(&format!(
        "Actions: {}/{}\n",
        coverage.actions_covered, coverage.actions_total
    ));

    if !coverage.uncovered.is_empty() {
        output.push_str(&format!(
            "\nUncovered goals: {}\n",
            coverage.uncovered.len()
        ));
    }

    output
}

/// Format a single test as text
fn format_text_test(test: &TestCase) -> String {
    let mut output = String::new();

    output.push_str(&format!("\n{}: {}\n", test.id, test.description));
    output.push_str(&format!(
        "  Initial: {}\n",
        test.initial_state.canonical_string()
    ));

    for (i, (action, state)) in test
        .actions
        .iter()
        .zip(test.expected_states.iter())
        .enumerate()
    {
        output.push_str(&format!(
            "  Step {}: {} -> {}\n",
            i + 1,
            action.signature(),
            state.canonical_string()
        ));
    }

    output
}

/// Format as Markdown report
fn format_markdown(result: &GenerationResult) -> String {
    let mut output = String::new();

    output.push_str("# Model-Based Test Generation Report\n\n");

    // Summary
    output.push_str("## Summary\n\n");
    output.push_str("| Metric | Value |\n");
    output.push_str("|--------|-------|\n");
    output.push_str(&format!(
        "| Tests generated | {} |\n",
        result.stats.tests_generated
    ));
    output.push_str(&format!("| Total steps | {} |\n", result.stats.total_steps));
    output.push_str(&format!(
        "| Average test length | {:.1} |\n",
        result.stats.avg_test_length
    ));
    output.push_str(&format!(
        "| Max test length | {} |\n",
        result.stats.max_test_length
    ));
    output.push_str(&format!(
        "| Generation time | {}ms |\n",
        result.stats.duration_ms
    ));
    output.push('\n');

    // Coverage
    output.push_str("## Coverage\n\n");
    output.push_str("| Metric | Covered | Total | Percentage |\n");
    output.push_str("|--------|---------|-------|------------|\n");
    output.push_str(&format!(
        "| States | {} | {} | {:.1}% |\n",
        result.coverage.states_covered,
        result.coverage.states_total,
        result.coverage.state_coverage_pct()
    ));
    output.push_str(&format!(
        "| Transitions | {} | {} | {:.1}% |\n",
        result.coverage.transitions_covered,
        result.coverage.transitions_total,
        result.coverage.transition_coverage_pct()
    ));
    output.push_str(&format!(
        "| Actions | {} | {} | - |\n",
        result.coverage.actions_covered, result.coverage.actions_total
    ));
    output.push('\n');

    // Tests
    output.push_str("## Test Cases\n\n");
    for test in &result.tests {
        output.push_str(&format!("### {}\n\n", test.id));
        output.push_str(&format!("**Description:** {}\n\n", test.description));
        output.push_str(&format!(
            "**Initial State:** `{}`\n\n",
            test.initial_state.canonical_string()
        ));

        if !test.actions.is_empty() {
            output.push_str("**Steps:**\n\n");
            output.push_str("| Step | Action | Expected State |\n");
            output.push_str("|------|--------|----------------|\n");
            for (i, (action, state)) in test
                .actions
                .iter()
                .zip(test.expected_states.iter())
                .enumerate()
            {
                output.push_str(&format!(
                    "| {} | `{}` | `{}` |\n",
                    i + 1,
                    action.signature(),
                    state.canonical_string()
                ));
            }
        }
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{GenerationStats, TestCase};

    fn create_sample_result() -> GenerationResult {
        let mut test = TestCase::new("test_1", "Sample test");
        let mut initial = ModelState::new();
        initial.set("x", ModelValue::Int(0));
        test = test.with_initial_state(initial);

        let mut next = ModelState::new();
        next.set("x", ModelValue::Int(1));
        test.add_step(ModelAction::new("increment"), next);

        GenerationResult {
            tests: vec![test],
            coverage: CoverageReport {
                states_covered: 2,
                states_total: 3,
                transitions_covered: 1,
                transitions_total: 2,
                actions_covered: 1,
                actions_total: 2,
                boundaries_covered: 0,
                boundaries_total: 0,
                uncovered: vec![],
            },
            stats: GenerationStats {
                tests_generated: 1,
                total_steps: 1,
                avg_test_length: 1.0,
                max_test_length: 1,
                duration_ms: 10,
            },
        }
    }

    #[test]
    fn test_json_output() {
        let result = create_sample_result();
        let output = format_results(&result, OutputFormat::Json);
        assert!(output.contains("\"tests\""));
        assert!(output.contains("test_1"));
    }

    #[test]
    fn test_rust_output() {
        let result = create_sample_result();
        let output = format_results(&result, OutputFormat::Rust);
        assert!(output.contains("#[test]"));
        assert!(output.contains("fn test_1"));
    }

    #[test]
    fn test_python_output() {
        let result = create_sample_result();
        let output = format_results(&result, OutputFormat::Python);
        assert!(output.contains("def test_"));
        assert!(output.contains("pytest"));
    }

    #[test]
    fn test_text_output() {
        let result = create_sample_result();
        let output = format_results(&result, OutputFormat::Text);
        assert!(output.contains("Model-Based Test Generation Report"));
        assert!(output.contains("Coverage"));
    }

    #[test]
    fn test_markdown_output() {
        let result = create_sample_result();
        let output = format_results(&result, OutputFormat::Markdown);
        assert!(output.contains("# Model-Based Test Generation Report"));
        assert!(output.contains("| Metric |"));
    }
}
