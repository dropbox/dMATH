//! MIRI undefined behavior detection CLI command

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use dashprove_miri::{
    detect_miri, parse_miri_output, run_miri, setup_miri, HarnessConfig, HarnessGenerator,
    MiriConfig, MiriFlags, ParsedMiriOutput,
};

/// Configuration for MIRI command
pub struct MiriCmdConfig<'a> {
    /// Path to project or file to verify
    pub path: &'a str,
    /// Test filter pattern
    pub test_filter: Option<&'a str>,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Disable MIRI isolation
    pub disable_isolation: bool,
    /// Skip stacked borrows checking
    pub skip_stacked_borrows: bool,
    /// Skip data race detection
    pub skip_data_races: bool,
    /// Track raw pointers
    pub track_raw_pointers: bool,
    /// Seed for deterministic execution
    pub seed: Option<u64>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Output file
    pub output: Option<&'a str>,
    /// Show verbose output
    pub verbose: bool,
    /// Generate harness for a function
    pub generate_harness: Option<&'a str>,
    /// Just setup MIRI (don't run tests)
    pub setup_only: bool,
}

/// Run the MIRI command
pub async fn run_miri_cmd(config: MiriCmdConfig<'_>) -> Result<()> {
    let path = Path::new(config.path);

    // Build MIRI configuration
    let miri_config = MiriConfig {
        timeout: std::time::Duration::from_secs(config.timeout_secs),
        flags: MiriFlags {
            disable_isolation: config.disable_isolation,
            check_stacked_borrows: !config.skip_stacked_borrows,
            check_data_races: !config.skip_data_races,
            track_raw_pointers: config.track_raw_pointers,
            seed: config.seed,
            ..Default::default()
        },
        ..Default::default()
    };

    if config.verbose {
        println!("Detecting MIRI installation...");
    }

    // Detect MIRI
    let detection = detect_miri(&miri_config).await;

    if !detection.is_available() {
        return Err(anyhow::anyhow!(
            "MIRI is not available. Install with: rustup +nightly component add miri"
        ));
    }

    if config.verbose {
        if let Some(version) = detection.version() {
            println!("MIRI detected: {}", version.version_string);
            if let Some(ref rust_ver) = version.rust_version {
                println!("Rust version: {}", rust_ver);
            }
        }
    }

    // Handle setup-only mode
    if config.setup_only {
        if config.verbose {
            println!("Running MIRI setup...");
        }
        let setup_output = setup_miri(&miri_config, &detection).await?;
        if setup_output.exit_code == Some(0) {
            println!("MIRI setup complete!");
        } else {
            eprintln!("MIRI setup output:\n{}", setup_output.stderr);
            return Err(anyhow::anyhow!("MIRI setup failed"));
        }
        return Ok(());
    }

    // Handle harness generation mode
    if let Some(function_name) = config.generate_harness {
        return generate_harness_for_function(path, function_name, config.output, config.verbose);
    }

    // Validate project path
    if !path.exists() {
        return Err(anyhow::anyhow!("Path does not exist: {}", path.display()));
    }

    if config.verbose {
        println!("Running MIRI on: {}", path.display());
        if let Some(filter) = config.test_filter {
            println!("Test filter: {}", filter);
        }
        println!("MIRIFLAGS: {}", miri_config.flags.to_miriflags());
    }

    // Run MIRI
    let output = run_miri(&miri_config, &detection, path, config.test_filter).await?;

    if config.verbose {
        println!("MIRI completed in {:?}", output.duration);
        println!("Exit code: {:?}", output.exit_code);
    }

    // Parse output
    let parsed = parse_miri_output(&output)?;

    // Format and output results
    let formatted = format_results(&parsed, config.format)?;
    output_result(&formatted, config.output)?;

    // Return error if UB was found
    if parsed.has_undefined_behavior() {
        return Err(anyhow::anyhow!(
            "MIRI detected {} undefined behavior instance(s)",
            parsed.undefined_behaviors.len()
        ));
    }

    if !parsed.all_tests_passed() {
        return Err(anyhow::anyhow!("{} test(s) failed", parsed.summary.failed));
    }

    Ok(())
}

/// Generate a harness for a specific function
fn generate_harness_for_function(
    source_path: &Path,
    function_name: &str,
    output_path: Option<&str>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Generating MIRI harness for function: {}", function_name);
    }

    // Read source file to find function signature
    let source_content = fs::read_to_string(source_path)
        .with_context(|| format!("Failed to read source file: {}", source_path.display()))?;

    // Find function signature (simple approach)
    let signature = find_function_signature(&source_content, function_name).with_context(|| {
        format!(
            "Could not find function '{}' in {}",
            function_name,
            source_path.display()
        )
    })?;

    if verbose {
        println!("Found signature: {}", signature);
    }

    // Generate harness
    let generator = HarnessGenerator::new(HarnessConfig::default());
    let harness = generator
        .generate_function_harness(function_name, &signature, None)
        .map_err(|e| anyhow::anyhow!("Failed to generate harness: {}", e))?;

    if verbose {
        println!("Generated harness: {}", harness.name);
        println!("Test inputs: {} parameters", harness.inputs.len());
    }

    // Output
    output_result(&harness.code, output_path)?;

    Ok(())
}

/// Find function signature in source code
fn find_function_signature(source: &str, function_name: &str) -> Option<String> {
    // Simple pattern matching for function definitions
    // This is a simplified implementation - a real one would use syn or tree-sitter
    let patterns = [
        format!("fn {}(", function_name),
        format!("pub fn {}(", function_name),
        format!("async fn {}(", function_name),
        format!("pub async fn {}(", function_name),
        format!("unsafe fn {}(", function_name),
        format!("pub unsafe fn {}(", function_name),
    ];

    for line in source.lines() {
        let trimmed = line.trim();
        for pattern in &patterns {
            if trimmed.contains(pattern.as_str()) {
                // Extract until closing paren and return type
                let mut result = String::new();
                let mut found = false;
                for c in source
                    .chars()
                    .skip(source.find(pattern.as_str()).unwrap_or(0))
                {
                    result.push(c);
                    if c == '{' || (c == ';' && found) {
                        break;
                    }
                    if c == ')' {
                        found = true;
                    }
                }
                return Some(result.trim().trim_end_matches('{').trim().to_string());
            }
        }
    }

    None
}

/// Format results based on output format
fn format_results(parsed: &ParsedMiriOutput, format: &str) -> Result<String> {
    match format.to_lowercase().as_str() {
        "json" => {
            serde_json::to_string_pretty(parsed).context("Failed to serialize results to JSON")
        }
        _ => Ok(format_text_output(parsed)),
    }
}

/// Format results as text
fn format_text_output(parsed: &ParsedMiriOutput) -> String {
    let mut output = String::new();

    // Summary header
    output.push_str("=== MIRI Verification Results ===\n\n");

    // Test summary
    output.push_str(&format!(
        "Tests: {} total, {} passed, {} failed, {} ignored\n",
        parsed.summary.total_tests,
        parsed.summary.passed,
        parsed.summary.failed,
        parsed.summary.ignored
    ));

    // UB summary
    if parsed.has_undefined_behavior() {
        output.push_str(&format!(
            "\n⚠️  {} UNDEFINED BEHAVIOR DETECTED:\n",
            parsed.undefined_behaviors.len()
        ));

        for (i, ub) in parsed.undefined_behaviors.iter().enumerate() {
            output.push_str(&format!("\n[UB #{}] {:?}\n", i + 1, ub.kind));
            output.push_str(&format!("  Message: {}\n", ub.message));
            if let Some(ref loc) = ub.location {
                output.push_str(&format!("  Location: {}\n", loc));
            }
            if !ub.notes.is_empty() {
                output.push_str("  Notes:\n");
                for note in &ub.notes {
                    output.push_str(&format!("    - {}\n", note));
                }
            }
            if !ub.backtrace.is_empty() {
                output.push_str("  Backtrace:\n");
                for frame in ub.backtrace.iter().take(5) {
                    output.push_str(&format!("    {}\n", frame));
                }
                if ub.backtrace.len() > 5 {
                    output.push_str(&format!(
                        "    ... ({} more frames)\n",
                        ub.backtrace.len() - 5
                    ));
                }
            }
        }

        // UB by kind summary
        if !parsed.summary.ub_by_kind.is_empty() {
            output.push_str("\nUB by kind:\n");
            for (kind, count) in &parsed.summary.ub_by_kind {
                output.push_str(&format!("  {}: {}\n", kind, count));
            }
        }
    } else {
        output.push_str("\n✓ No undefined behavior detected\n");
    }

    // Test results
    if !parsed.test_results.is_empty() {
        output.push_str("\nTest Results:\n");
        for result in &parsed.test_results {
            let status_icon = match result.status {
                dashprove_miri::MiriTestStatus::Passed => "✓",
                dashprove_miri::MiriTestStatus::Failed => "✗",
                dashprove_miri::MiriTestStatus::Ignored => "○",
                dashprove_miri::MiriTestStatus::UndefinedBehavior => "⚠",
                dashprove_miri::MiriTestStatus::TimedOut => "⏱",
            };
            let duration = result
                .duration_ms
                .map(|d| format!(" ({}ms)", d))
                .unwrap_or_default();
            output.push_str(&format!("  {} {}{}\n", status_icon, result.name, duration));
        }
    }

    output
}

/// Output the result to file or stdout
fn output_result(content: &str, output_path: Option<&str>) -> Result<()> {
    if let Some(path) = output_path {
        fs::write(path, content).context("Failed to write output file")?;
        println!("Output written to: {}", path);
    } else {
        println!("{}", content);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_function_signature() {
        let source = r#"
fn simple_add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn public_func(x: usize) {
    println!("{}", x);
}

pub async fn async_func() -> Result<()> {
    Ok(())
}
"#;

        let sig = find_function_signature(source, "simple_add");
        assert!(sig.is_some());
        let sig = sig.unwrap();
        assert!(sig.contains("fn simple_add"));
        assert!(sig.contains("i32"));

        let sig = find_function_signature(source, "public_func");
        assert!(sig.is_some());

        let sig = find_function_signature(source, "async_func");
        assert!(sig.is_some());

        let sig = find_function_signature(source, "nonexistent");
        assert!(sig.is_none());
    }
}
