//! Bisimulation verification command
//!
//! Runs bisimulation checks between an oracle and subject implementation.

use anyhow::{anyhow, Result};
use dashprove_bisim::{
    BisimulationChecker, BisimulationConfig, DefaultBisimulationChecker, Difference,
    EquivalenceCriteria, NondeterminismStrategy, OracleConfig, TestInput, TestSubjectConfig,
};
use std::time::{Duration, Instant};

/// Configuration for the bisim command
pub struct BisimConfig<'a> {
    /// Path to oracle binary or trace directory
    pub oracle: &'a str,
    /// Path to subject binary
    pub subject: &'a str,
    /// Path to test inputs file (JSON)
    pub inputs: Option<&'a str>,
    /// Whether oracle is a recorded trace directory
    pub recorded_traces: bool,
    /// Timeout for each test in seconds
    pub timeout_secs: u64,
    /// Similarity threshold (0.0 to 1.0)
    pub threshold: f64,
    /// Allow timing differences
    pub ignore_timing: bool,
    /// Allow nondeterministic differences
    pub nondeterminism: &'a str,
    /// Show verbose output
    pub verbose: bool,
}

/// Aggregated result from multiple bisimulation tests
pub struct AggregatedResult {
    /// Whether all tests passed
    pub equivalent: bool,
    /// Overall similarity score (average)
    pub similarity_score: f64,
    /// Number of tests run
    pub test_count: usize,
    /// Total duration
    pub total_duration: Duration,
    /// All differences found
    pub differences: Vec<DifferenceInfo>,
}

/// Information about a single difference
pub struct DifferenceInfo {
    /// Test name where difference occurred
    pub test_name: String,
    /// Description of the difference
    pub description: String,
    /// Oracle value (if applicable)
    pub oracle_value: Option<String>,
    /// Subject value (if applicable)
    pub subject_value: Option<String>,
}

impl AggregatedResult {
    fn new() -> Self {
        Self {
            equivalent: true,
            similarity_score: 1.0,
            test_count: 0,
            total_duration: Duration::ZERO,
            differences: vec![],
        }
    }
}

fn format_difference(test_name: &str, diff: &Difference) -> DifferenceInfo {
    match diff {
        Difference::OutputMismatch {
            oracle,
            subject,
            similarity,
        } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!("Output mismatch (similarity: {:.1}%)", similarity * 100.0),
            oracle_value: Some(truncate(oracle, 100)),
            subject_value: Some(truncate(subject, 100)),
        },
        Difference::ApiRequestMismatch { index, oracle, .. } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!(
                "API request mismatch at index {}: {} {}",
                index, oracle.method, oracle.url
            ),
            oracle_value: None,
            subject_value: None,
        },
        Difference::ToolCallMismatch { index, oracle, .. } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!("Tool call mismatch at index {}: {}", index, oracle.name),
            oracle_value: None,
            subject_value: None,
        },
        Difference::SequenceLengthMismatch {
            sequence_type,
            oracle_count,
            subject_count,
        } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!(
                "{} count mismatch: oracle={}, subject={}",
                sequence_type, oracle_count, subject_count
            ),
            oracle_value: None,
            subject_value: None,
        },
        Difference::MissingEvent {
            index,
            event_type,
            description,
        } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!("Missing {} at index {}: {}", event_type, index, description),
            oracle_value: None,
            subject_value: None,
        },
        Difference::ExtraEvent {
            index,
            event_type,
            description,
        } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!("Extra {} at index {}: {}", event_type, index, description),
            oracle_value: None,
            subject_value: None,
        },
        Difference::TimingViolation {
            oracle_ms,
            subject_ms,
            tolerance,
        } => DifferenceInfo {
            test_name: test_name.to_string(),
            description: format!(
                "Timing violation: oracle={}ms, subject={}ms, tolerance={}%",
                oracle_ms,
                subject_ms,
                tolerance * 100.0
            ),
            oracle_value: None,
            subject_value: None,
        },
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

/// Run bisimulation verification
pub async fn run_bisim(config: BisimConfig<'_>) -> Result<()> {
    println!("DashProve Bisimulation Checker");
    println!("==============================\n");

    // Build oracle config
    let oracle = if config.recorded_traces {
        OracleConfig::traces(config.oracle)
    } else {
        OracleConfig::binary(config.oracle)
    };

    // Build subject config
    let subject = TestSubjectConfig::binary(config.subject);

    // Build equivalence criteria
    let mut equivalence_criteria = EquivalenceCriteria::default();
    if config.ignore_timing {
        equivalence_criteria = equivalence_criteria.with_timing_tolerance(1.0);
    }

    // Parse nondeterminism strategy
    let nondeterminism_strategy = match config.nondeterminism {
        "strict" => NondeterminismStrategy::ExactMatch,
        "semantic" => NondeterminismStrategy::semantic(config.threshold),
        "distribution" => NondeterminismStrategy::distribution(10),
        _ => {
            return Err(anyhow!(
                "Invalid nondeterminism strategy: {}. Use: strict, semantic, or distribution",
                config.nondeterminism
            ))
        }
    };

    // Build full config
    let bisim_config = BisimulationConfig {
        oracle,
        subject,
        equivalence_criteria,
        nondeterminism_strategy,
    };

    if config.verbose {
        println!("Configuration:");
        println!("  Oracle: {}", config.oracle);
        println!("  Subject: {}", config.subject);
        println!("  Threshold: {:.2}", config.threshold);
        println!("  Ignore timing: {}", config.ignore_timing);
        println!("  Nondeterminism: {}", config.nondeterminism);
        println!("  Timeout: {}s", config.timeout_secs);
        println!();
    }

    // Create checker
    let checker = DefaultBisimulationChecker::new(bisim_config);

    // Load or generate test inputs
    let inputs: Vec<TestInput> = if let Some(inputs_path) = config.inputs {
        let contents = std::fs::read_to_string(inputs_path)?;
        // Parse as JSON array of {name, input} objects
        let raw: Vec<serde_json::Value> = serde_json::from_str(&contents)?;
        raw.iter()
            .map(|v| {
                let name = v
                    .get("id")
                    .or_else(|| v.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("unnamed");
                let input = v
                    .get("input")
                    .or_else(|| v.get("data"))
                    .map(|d| d.to_string())
                    .unwrap_or_default();
                TestInput::new(name, input).with_timeout(Duration::from_secs(config.timeout_secs))
            })
            .collect()
    } else {
        // Default to empty test inputs for now
        vec![]
    };

    if inputs.is_empty() {
        println!("Warning: No test inputs provided. Use --inputs to specify a JSON file with test cases.");
        println!(r#"Expected format: [{{"id": "test1", "input": "..."}} , ...]"#);
        return Ok(());
    }

    println!("Running {} test cases...\n", inputs.len());

    // Run bisimulation checks
    let start = Instant::now();
    let results = checker.check_batch(&inputs).await?;
    let total_duration = start.elapsed();

    // Aggregate results
    let mut aggregated = AggregatedResult::new();
    aggregated.test_count = results.len();
    aggregated.total_duration = total_duration;

    let mut total_confidence = 0.0;
    for (result, input) in results.iter().zip(inputs.iter()) {
        total_confidence += result.confidence;
        if !result.equivalent {
            aggregated.equivalent = false;
            for diff in &result.differences {
                aggregated
                    .differences
                    .push(format_difference(&input.name, diff));
            }
        }
    }
    aggregated.similarity_score = total_confidence / results.len() as f64;

    // Print results
    println!("Results:");
    println!("--------");
    println!(
        "  Equivalent: {}",
        if aggregated.equivalent { "YES" } else { "NO" }
    );
    println!("  Similarity: {:.1}%", aggregated.similarity_score * 100.0);
    println!("  Tests run: {}", aggregated.test_count);
    println!(
        "  Duration: {:.2}s",
        aggregated.total_duration.as_secs_f64()
    );

    if !aggregated.differences.is_empty() {
        println!("\nDifferences found:");
        for diff in &aggregated.differences {
            println!("  - [{}] {}", diff.test_name, diff.description);
            if config.verbose {
                if let Some(ref oracle_val) = diff.oracle_value {
                    println!("    Oracle: {}", oracle_val);
                }
                if let Some(ref subject_val) = diff.subject_value {
                    println!("    Subject: {}", subject_val);
                }
            }
        }
    }

    if !aggregated.equivalent {
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisim_config_defaults() {
        let config = BisimConfig {
            oracle: "/bin/echo",
            subject: "/bin/echo",
            inputs: None,
            recorded_traces: false,
            timeout_secs: 60,
            threshold: 0.95,
            ignore_timing: true,
            nondeterminism: "strict",
            verbose: false,
        };
        assert_eq!(config.threshold, 0.95);
        assert!(config.ignore_timing);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world this is long", 10), "hello worl...");
    }
}
