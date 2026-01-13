//! Monitor generation command implementation

use dashprove::{
    monitor::{MonitorConfig, MonitorTarget, RuntimeMonitor},
    usl::{parse, typecheck},
};
use std::path::Path;
use tracing::info;

/// Configuration for the monitor command
#[derive(Debug)]
pub struct MonitorCmdConfig<'a> {
    pub path: &'a str,
    pub target: &'a str,
    pub output: Option<&'a str>,
    pub assertions: bool,
    pub logging: bool,
    pub metrics: bool,
}

/// Run monitor generation command
pub fn run_monitor(config: MonitorCmdConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let MonitorCmdConfig {
        path,
        target,
        output,
        assertions,
        logging,
        metrics,
    } = config;

    // Read specification file
    let spec_path = Path::new(path);
    if !spec_path.exists() {
        return Err(format!("Specification file not found: {}", path).into());
    }

    let spec_content = std::fs::read_to_string(spec_path)?;
    info!("Read specification from {}", path);

    // Parse and type-check
    let spec = parse(&spec_content).map_err(|e| format!("Parse error: {:?}", e))?;
    info!("Parsed {} properties", spec.properties.len());

    let typed_spec = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;
    info!("Type checking passed");

    // Parse target language
    let monitor_target = match target.to_lowercase().as_str() {
        "rust" | "rs" => MonitorTarget::Rust,
        "typescript" | "ts" => MonitorTarget::TypeScript,
        "python" | "py" => MonitorTarget::Python,
        _ => {
            return Err(format!("Unknown target: {}. Use: rust, typescript, python", target).into())
        }
    };

    // Create monitor configuration
    let monitor_config = MonitorConfig {
        generate_assertions: assertions,
        generate_logging: logging,
        generate_metrics: metrics,
        target: monitor_target,
    };

    // Generate runtime monitor
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &monitor_config);

    // Print info
    println!(
        "// Generated runtime monitor for {} properties",
        monitor.property_count()
    );
    println!("// Target: {:?}", monitor.target);
    if assertions {
        println!("// Assertions: enabled");
    }
    if logging {
        println!("// Logging: enabled");
    }
    if metrics {
        println!("// Metrics: enabled");
    }
    println!();

    // Output
    if let Some(output_path) = output {
        std::fs::write(output_path, &monitor.code)?;
        println!("Monitor written to {}", output_path);
    } else {
        println!("{}", monitor.code);
    }

    Ok(())
}
