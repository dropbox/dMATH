//! Analyze command implementations for counterexample trace analysis

use dashprove::backends::{
    AbstractedTrace, CompressedTrace, StructuredCounterexample, TraceAbstractionSegment, TraceDiff,
    TraceInterleaving, TraceSegment, TraceState,
};
use dashprove_cli::cli::AnalyzeAction;
use std::path::Path;

/// Run analyze command - analyze counterexample traces
pub fn run_analyze(path: &str, action: AnalyzeAction) -> Result<(), Box<dyn std::error::Error>> {
    let ce_path = Path::new(path);
    if !ce_path.exists() {
        return Err(format!("Counterexample file not found: {}", path).into());
    }

    // Read and parse the counterexample JSON
    let content = std::fs::read_to_string(ce_path)?;
    let counterexample: StructuredCounterexample = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse counterexample JSON: {}", e))?;

    match action {
        AnalyzeAction::Suggest { format } => run_analyze_suggest(&counterexample, &format),
        AnalyzeAction::Compress { format, output } => {
            run_analyze_compress(&counterexample, &format, output.as_deref())
        }
        AnalyzeAction::Interleavings { format, output } => {
            run_analyze_interleavings(&counterexample, &format, output.as_deref())
        }
        AnalyzeAction::Minimize { max_states, output } => {
            run_analyze_minimize(&counterexample, max_states, output.as_deref())
        }
        AnalyzeAction::Abstract {
            min_group_size,
            format,
            output,
        } => run_analyze_abstract(&counterexample, min_group_size, &format, output.as_deref()),
        AnalyzeAction::Diff {
            other,
            format,
            output,
        } => run_analyze_diff(&counterexample, &other, &format, output.as_deref()),
    }
}

/// Run suggest analysis - find patterns and suggest filters
fn run_analyze_suggest(
    counterexample: &StructuredCounterexample,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let suggestions = counterexample.suggest_patterns();

    if suggestions.is_empty() {
        println!("No patterns or suggestions found for this counterexample.");
        return Ok(());
    }

    match format.to_lowercase().as_str() {
        "json" => {
            // TraceSuggestion now implements Serialize
            println!("{}", serde_json::to_string_pretty(&suggestions)?);
        }
        _ => {
            println!("=== Pattern Suggestions ===\n");
            println!("Found {} suggestions:\n", suggestions.len());

            for (i, suggestion) in suggestions.iter().enumerate() {
                let severity_icon = match suggestion.severity {
                    dashprove::backends::SuggestionSeverity::Low => "[LOW]",
                    dashprove::backends::SuggestionSeverity::Medium => "[MEDIUM]",
                    dashprove::backends::SuggestionSeverity::High => "[HIGH]",
                    dashprove::backends::SuggestionSeverity::Critical => "[CRITICAL]",
                };
                println!(
                    "{}. {} {} (confidence: {:.0}%)",
                    i + 1,
                    severity_icon,
                    suggestion.description,
                    suggestion.confidence * 100.0
                );
                println!("   Kind: {:?}", suggestion.kind);
                println!("   Suggestion: {}\n", suggestion.suggested_action);
            }
        }
    }

    Ok(())
}

/// Run compress analysis - detect repeating patterns
fn run_analyze_compress(
    counterexample: &StructuredCounterexample,
    format: &str,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let compressed = counterexample.compress_trace();

    let result = match format.to_lowercase().as_str() {
        "json" => serde_json::to_string_pretty(&compressed)?,
        "mermaid" | "mmd" => compressed.to_mermaid(),
        "dot" | "graphviz" => compressed.to_dot(),
        "html" | "htm" => compressed.to_html(Some("Compressed Trace")),
        _ => format_compressed_text(&compressed),
    };

    output_result(&result, output, format)?;
    Ok(())
}

/// Format compressed trace as text
fn format_compressed_text(compressed: &CompressedTrace) -> String {
    let mut output = String::new();
    output.push_str("=== Compressed Trace ===\n\n");
    output.push_str(&format!(
        "Original states: {}\n",
        compressed.original_length
    ));
    output.push_str(&format!(
        "Compressed to: {} segments ({:.1}% compression)\n\n",
        compressed.segments.len(),
        compressed.compression_ratio() * 100.0
    ));

    for (i, segment) in compressed.segments.iter().enumerate() {
        match segment {
            TraceSegment::Single(state) => {
                output.push_str(&format!(
                    "Segment {}: State {} (single)\n",
                    i + 1,
                    state.state_num
                ));
                if let Some(ref action) = state.action {
                    output.push_str(&format!("  Action: {}\n", action));
                }
            }
            TraceSegment::Repeated { pattern, count } => {
                output.push_str(&format!(
                    "Segment {}: {} states repeated {} times\n",
                    i + 1,
                    pattern.len(),
                    count
                ));
                output.push_str(&format!(
                    "  Pattern covers {} total states\n",
                    pattern.len() * count
                ));
            }
        }
    }

    output
}

/// Run interleavings analysis - detect actor interleavings
fn run_analyze_interleavings(
    counterexample: &StructuredCounterexample,
    format: &str,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let interleavings = counterexample.detect_interleavings();

    let result = match format.to_lowercase().as_str() {
        "json" => serde_json::to_string_pretty(&interleavings)?,
        "mermaid" | "mmd" => interleavings.to_mermaid(),
        "html" | "htm" => interleavings.to_html(Some("Trace Interleavings")),
        _ => format_interleavings_text(&interleavings),
    };

    output_result(&result, output, format)?;
    Ok(())
}

/// Format interleavings as text
fn format_interleavings_text(interleavings: &TraceInterleaving) -> String {
    let mut output = String::new();
    output.push_str("=== Trace Interleavings ===\n\n");
    output.push_str(&format!(
        "Total states: {}\n",
        interleavings.original_trace.len()
    ));
    output.push_str(&format!("Actors detected: {}\n", interleavings.lanes.len()));
    output.push_str(&format!(
        "Coverage: {:.1}%\n\n",
        interleavings.coverage() * 100.0
    ));

    for lane in &interleavings.lanes {
        output.push_str(&format!(
            "Actor '{}': {} transitions\n",
            lane.actor,
            lane.states.len()
        ));
        for state in &lane.states {
            output.push_str(&format!("  State {}", state.state_num));
            if let Some(ref action) = state.action {
                output.push_str(&format!(": {}", action));
            }
            output.push('\n');
        }
        output.push('\n');
    }

    if !interleavings.unassigned_states.is_empty() {
        output.push_str(&format!(
            "Unassigned states: {}\n",
            interleavings.unassigned_states.len()
        ));
    }

    output
}

/// Run minimize analysis - remove irrelevant variables and states
fn run_analyze_minimize(
    counterexample: &StructuredCounterexample,
    max_states: usize,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // First minimize by removing constant variables
    let mut minimized = counterexample.minimize();

    // Then optionally limit trace length
    if max_states > 0 {
        minimized = minimized.minimize_trace_length(max_states);
    }

    let result = serde_json::to_string_pretty(&minimized)?;

    if let Some(output_path) = output {
        std::fs::write(output_path, &result)?;
        println!("Minimized counterexample written to {}", output_path);
        println!(
            "Original: {} states, {} variables",
            counterexample.trace.len(),
            counterexample
                .trace
                .first()
                .map(|s| s.variables.len())
                .unwrap_or(0)
        );
        println!(
            "Minimized: {} states, {} variables",
            minimized.trace.len(),
            minimized
                .trace
                .first()
                .map(|s| s.variables.len())
                .unwrap_or(0)
        );
    } else {
        println!("{}", result);
    }

    Ok(())
}

/// Run abstract analysis - group consecutive similar states
fn run_analyze_abstract(
    counterexample: &StructuredCounterexample,
    min_group_size: usize,
    format: &str,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let abstracted = counterexample.abstract_trace(min_group_size);

    let result = match format.to_lowercase().as_str() {
        "json" => serde_json::to_string_pretty(&abstracted)?,
        "mermaid" | "mmd" => abstracted.to_mermaid(),
        "dot" | "graphviz" => abstracted.to_dot(),
        "html" | "htm" => abstracted.to_html(Some("Abstracted Trace")),
        _ => format_abstracted_text(&abstracted),
    };

    output_result(&result, output, format)?;
    Ok(())
}

/// Format abstracted trace as text
fn format_abstracted_text(abstracted: &AbstractedTrace) -> String {
    let mut output = String::new();
    output.push_str("=== Abstracted Trace ===\n\n");
    output.push_str(&format!(
        "Original states: {}\n",
        abstracted.original_length
    ));
    output.push_str(&format!(
        "Abstracted to: {} segments ({:.1}% compression)\n\n",
        abstracted.segments.len(),
        abstracted.compression_ratio * 100.0
    ));

    for (i, segment) in abstracted.segments.iter().enumerate() {
        match segment {
            TraceAbstractionSegment::Concrete(state) => {
                output.push_str(&format!(
                    "Segment {}: State {} (concrete)\n",
                    i + 1,
                    state.state_num
                ));
                if let Some(ref action) = state.action {
                    output.push_str(&format!("  Action: {}\n", action));
                }
            }
            TraceAbstractionSegment::Abstracted(abs_state) => {
                let (first_idx, last_idx) = if abs_state.original_indices.is_empty() {
                    (0, 0)
                } else {
                    (
                        *abs_state.original_indices.first().unwrap(),
                        *abs_state.original_indices.last().unwrap(),
                    )
                };
                output.push_str(&format!(
                    "Segment {}: {} states (indices {}-{}) [abstracted]\n",
                    i + 1,
                    abs_state.count,
                    first_idx,
                    last_idx
                ));
                output.push_str(&format!("  Description: {}\n", abs_state.description));
                if let Some(ref action) = abs_state.common_action {
                    output.push_str(&format!("  Common action: {}\n", action));
                }
                if !abs_state.variables.is_empty() {
                    output.push_str("  Variables:\n");
                    for (var, val) in &abs_state.variables {
                        output.push_str(&format!("    {}: {:?}\n", var, val));
                    }
                }
            }
        }
        output.push('\n');
    }

    output
}

/// Run diff analysis - compare two counterexamples
fn run_analyze_diff(
    counterexample: &StructuredCounterexample,
    other_path: &str,
    format: &str,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let other_ce_path = Path::new(other_path);
    if !other_ce_path.exists() {
        return Err(format!("Second counterexample file not found: {}", other_path).into());
    }

    let other_content = std::fs::read_to_string(other_ce_path)?;
    let other: StructuredCounterexample = serde_json::from_str(&other_content)
        .map_err(|e| format!("Failed to parse second counterexample JSON: {}", e))?;

    let diff = counterexample.diff_traces(&other);

    let result = match format.to_lowercase().as_str() {
        "mermaid" | "mmd" => {
            diff.to_mermaid(&counterexample.trace, &other.trace, "Trace 1", "Trace 2")
        }
        "dot" | "graphviz" => {
            diff.to_dot(&counterexample.trace, &other.trace, "Trace 1", "Trace 2")
        }
        "html" | "htm" => diff.to_html(
            &counterexample.trace,
            &other.trace,
            "Trace 1",
            "Trace 2",
            Some("Trace Diff"),
        ),
        _ => format_diff_text(&diff, &counterexample.trace, &other.trace),
    };

    output_result(&result, output, format)?;
    Ok(())
}

/// Format trace diff as text
fn format_diff_text(diff: &TraceDiff, trace1: &[TraceState], trace2: &[TraceState]) -> String {
    let mut output = String::new();
    output.push_str("=== Trace Diff ===\n\n");
    output.push_str(&format!("Trace 1 length: {}\n", trace1.len()));
    output.push_str(&format!("Trace 2 length: {}\n", trace2.len()));
    output.push_str(&format!(
        "Identical states: {}\n",
        diff.identical_states.len()
    ));
    output.push_str(&format!(
        "States only in trace 1: {}\n",
        diff.states_only_in_first.len()
    ));
    output.push_str(&format!(
        "States only in trace 2: {}\n",
        diff.states_only_in_second.len()
    ));
    output.push_str(&format!(
        "States with differences: {}\n\n",
        diff.state_diffs.len()
    ));

    output.push_str(&diff.summary());
    output.push('\n');

    if !diff.states_only_in_first.is_empty() {
        output.push_str("\nStates only in trace 1: ");
        let nums: Vec<String> = diff
            .states_only_in_first
            .iter()
            .map(|n| n.to_string())
            .collect();
        output.push_str(&nums.join(", "));
        output.push('\n');
    }

    if !diff.states_only_in_second.is_empty() {
        output.push_str("\nStates only in trace 2: ");
        let nums: Vec<String> = diff
            .states_only_in_second
            .iter()
            .map(|n| n.to_string())
            .collect();
        output.push_str(&nums.join(", "));
        output.push('\n');
    }

    if !diff.state_diffs.is_empty() {
        output.push_str("\nStates with differences:\n");
        for (state_num, state_diff) in &diff.state_diffs {
            if !state_diff.is_empty() {
                output.push_str(&format!("  State {}:\n", state_num));

                if !state_diff.vars_only_in_first.is_empty() {
                    output.push_str("    Variables only in trace 1:\n");
                    for (var, val) in &state_diff.vars_only_in_first {
                        output.push_str(&format!("      {} = {}\n", var, val));
                    }
                }

                if !state_diff.vars_only_in_second.is_empty() {
                    output.push_str("    Variables only in trace 2:\n");
                    for (var, val) in &state_diff.vars_only_in_second {
                        output.push_str(&format!("      {} = {}\n", var, val));
                    }
                }

                if !state_diff.value_diffs.is_empty() {
                    output.push_str("    Value differences:\n");
                    for (var, (val1, val2)) in &state_diff.value_diffs {
                        output.push_str(&format!("      {}: {} -> {}\n", var, val1, val2));
                    }
                }

                if let Some((act1, act2)) = &state_diff.action_diff {
                    output.push_str(&format!(
                        "    Action: {} -> {}\n",
                        act1.as_deref().unwrap_or("(none)"),
                        act2.as_deref().unwrap_or("(none)")
                    ));
                }
            }
        }
    }

    output
}

/// Helper to output result to file or stdout
pub fn output_result(
    result: &str,
    output: Option<&str>,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(output_path) = output {
        std::fs::write(output_path, result)?;
        let format_upper = format.to_uppercase();
        println!("{} output written to {}", format_upper, output_path);
    } else {
        println!("{}", result);
    }
    Ok(())
}
