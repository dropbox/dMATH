//! Model-based testing CLI command

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use dashprove_mbt::{
    explorer::{ExplorationConfig, PrecomputedExecutor, StateExplorer},
    generator::{generate_boundary_tests, GenerationStrategy, GeneratorConfig, TestGenerator},
    output::{format_results, OutputFormat},
    tlaplus::parse_tlaplus_spec,
};

/// Configuration for MBT command
pub struct MbtConfig<'a> {
    pub spec_path: &'a str,
    pub coverage: &'a str,
    pub format: &'a str,
    pub output: Option<&'a str>,
    pub max_states: usize,
    pub max_depth: usize,
    pub max_test_length: usize,
    pub max_tests: usize,
    pub seed: Option<u64>,
    pub timeout_secs: u64,
    pub verbose: bool,
}

/// Run the MBT command
pub fn run_mbt(config: MbtConfig<'_>) -> Result<()> {
    let spec_path = Path::new(config.spec_path);

    if config.verbose {
        println!("Reading specification from: {}", spec_path.display());
    }

    // Read and parse the specification
    let spec_content =
        fs::read_to_string(spec_path).context("Failed to read specification file")?;

    let model = parse_tlaplus_spec(&spec_content).context("Failed to parse TLA+ specification")?;

    if config.verbose {
        println!("Parsed model: {}", model.name);
        println!("  Variables: {}", model.variables.len());
        println!("  Actions: {}", model.actions.len());
        println!("  Invariants: {}", model.invariants.len());
        println!("  Initial states: {}", model.initial_states.len());
    }

    // Parse coverage strategy
    let strategy = match config.coverage.to_lowercase().as_str() {
        "state" => GenerationStrategy::StateCoverage,
        "transition" => GenerationStrategy::TransitionCoverage,
        "boundary" => GenerationStrategy::BoundaryValue,
        "random" => GenerationStrategy::RandomWalk,
        _ => GenerationStrategy::Combined,
    };

    // Parse output format
    let output_format: OutputFormat = config
        .format
        .parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    // Handle boundary tests specially (they don't need exploration)
    if strategy == GenerationStrategy::BoundaryValue {
        if config.verbose {
            println!("Generating boundary value tests...");
        }

        let tests = generate_boundary_tests(&model).context("Failed to generate boundary tests")?;

        let result = dashprove_mbt::generator::GenerationResult {
            tests,
            coverage: dashprove_mbt::generator::CoverageReport {
                states_covered: 0,
                states_total: 0,
                transitions_covered: 0,
                transitions_total: 0,
                actions_covered: 0,
                actions_total: model.actions.len(),
                boundaries_covered: model.variables.len(),
                boundaries_total: model.variables.len(),
                uncovered: vec![],
            },
            stats: dashprove_mbt::generator::GenerationStats {
                tests_generated: model.variables.len(),
                total_steps: 0,
                avg_test_length: 0.0,
                max_test_length: 0,
                duration_ms: 0,
            },
        };

        let output = format_results(&result, output_format);
        output_result(&output, config.output)?;
        return Ok(());
    }

    // Set up exploration
    let exploration_config = ExplorationConfig {
        max_states: config.max_states,
        max_depth: config.max_depth,
        timeout_ms: config.timeout_secs * 1000,
        compute_transitions: true,
        verbose: config.verbose,
    };

    if config.verbose {
        println!("Exploring state space...");
        println!("  Max states: {}", config.max_states);
        println!("  Max depth: {}", config.max_depth);
        println!("  Timeout: {}s", config.timeout_secs);
    }

    // Build executor from model
    // For now, we use a simple precomputed executor
    // In practice, this would be connected to a TLC backend or symbolic executor
    let executor = PrecomputedExecutor::new();
    let explorer = StateExplorer::with_config(executor, exploration_config);

    // Explore from initial states
    let exploration = explorer
        .explore(&model.initial_states)
        .context("State space exploration failed")?;

    if config.verbose {
        println!("Exploration complete:");
        println!("  States discovered: {}", exploration.states.len());
        println!(
            "  Transitions discovered: {}",
            exploration.transitions.len()
        );
        println!("  Max depth reached: {}", exploration.max_depth_reached);
        println!("  Duration: {}ms", exploration.duration_ms);
        if let Some(reason) = &exploration.incomplete_reason {
            println!("  Note: Exploration incomplete - {}", reason);
        }
    }

    // Generate tests
    let mut gen_config = GeneratorConfig::new()
        .with_strategy(strategy)
        .with_max_length(config.max_test_length)
        .with_max_tests(config.max_tests);

    if let Some(seed) = config.seed {
        gen_config = gen_config.with_seed(seed);
    }

    if config.verbose {
        println!("Generating tests...");
        println!("  Strategy: {:?}", strategy);
        println!("  Max test length: {}", config.max_test_length);
        println!("  Max tests: {}", config.max_tests);
    }

    let mut generator = TestGenerator::with_config(gen_config);
    let result = generator
        .generate(&exploration)
        .context("Test generation failed")?;

    if config.verbose {
        println!("Generation complete:");
        println!("  Tests generated: {}", result.stats.tests_generated);
        println!("  Total steps: {}", result.stats.total_steps);
        println!(
            "  State coverage: {:.1}%",
            result.coverage.state_coverage_pct()
        );
        println!(
            "  Transition coverage: {:.1}%",
            result.coverage.transition_coverage_pct()
        );
    }

    // Format and output
    let output = format_results(&result, output_format);
    output_result(&output, config.output)?;

    Ok(())
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
