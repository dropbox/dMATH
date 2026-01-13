//! Verify-code command implementation
//!
//! Verifies Rust code against USL contracts using Kani.

use dashprove::{DashProve, DashProveConfig};
use std::io::{self, Read};
use std::path::Path;

/// Configuration for the verify-code command
#[derive(Debug)]
pub struct VerifyCodeConfig<'a> {
    /// Path to Rust code file (if None, read from stdin)
    pub code_path: Option<&'a str>,
    /// Path to USL specification file containing contracts
    pub spec_path: &'a str,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Show verbose output
    pub verbose: bool,
}

/// Helper macro for verbose output
macro_rules! verbose_println {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!("[verbose] {}", format!($($arg)*));
        }
    };
}

/// Run verify-code command
pub async fn run_verify_code(
    config: VerifyCodeConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let VerifyCodeConfig {
        code_path,
        spec_path,
        timeout_secs,
        verbose,
    } = config;

    // Read the Rust code (from file or stdin)
    let code = match code_path {
        Some(path) => {
            let code_path = Path::new(path);
            if !code_path.exists() {
                return Err(format!("Code file not found: {}", path).into());
            }
            verbose_println!(verbose, "Reading code from {}", path);
            std::fs::read_to_string(code_path)?
        }
        None => {
            verbose_println!(verbose, "Reading code from stdin...");
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            buffer
        }
    };

    if code.trim().is_empty() {
        return Err("No code provided".into());
    }
    verbose_println!(verbose, "Code size: {} bytes", code.len());

    // Read the specification file
    let spec_file = Path::new(spec_path);
    if !spec_file.exists() {
        return Err(format!("Specification file not found: {}", spec_path).into());
    }

    verbose_println!(verbose, "Reading specification from {}", spec_path);
    let spec_content = std::fs::read_to_string(spec_file)?;
    verbose_println!(verbose, "Specification size: {} bytes", spec_content.len());

    // Create DashProve client with Kani backend
    verbose_println!(verbose, "Initializing DashProve client with Kani backend");
    let mut dashprove_config = DashProveConfig::with_backend(dashprove::backends::BackendId::Kani);
    dashprove_config.dispatcher.task_timeout = std::time::Duration::from_secs(timeout_secs);
    let mut client = DashProve::new(dashprove_config);

    // Run verification
    println!("Verifying code against specification...");
    verbose_println!(verbose, "Timeout: {}s", timeout_secs);

    let result = client.verify_code(&code, &spec_content).await?;

    // Print results
    println!("\n=== Verification Results ===");
    let status_str = match &result.status {
        dashprove::backends::VerificationStatus::Proven => "PROVEN",
        dashprove::backends::VerificationStatus::Disproven => "DISPROVEN",
        dashprove::backends::VerificationStatus::Unknown { reason } => {
            println!("Status: UNKNOWN");
            println!("Reason: {}", reason);
            return Ok(());
        }
        dashprove::backends::VerificationStatus::Partial {
            verified_percentage,
        } => {
            println!(
                "Status: PARTIAL ({:.0}% verified)",
                verified_percentage * 100.0
            );
            return Ok(());
        }
    };

    println!("Status: {}", status_str);
    println!("Confidence: {:.1}%", result.confidence * 100.0);

    // Print property results
    for prop in &result.properties {
        let prop_status = match &prop.status {
            dashprove::backends::VerificationStatus::Proven => "PROVEN".to_string(),
            dashprove::backends::VerificationStatus::Disproven => "DISPROVEN".to_string(),
            dashprove::backends::VerificationStatus::Unknown { reason } => {
                format!("UNKNOWN ({})", reason)
            }
            dashprove::backends::VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("PARTIAL ({:.0}%)", verified_percentage * 100.0)
            }
        };
        println!("\nContract: {}", prop.name);
        println!("  Status: {}", prop_status);
        println!("  Backends: {:?}", prop.backends_used);

        if let Some(ref proof) = prop.proof {
            println!("  Proof: {}", proof);
        }

        if let Some(ref ce) = prop.counterexample {
            println!("  Counterexample: {}", ce);
        }
    }

    // Show counterexample if disproven
    if let Some(ref ce) = result.counterexample {
        println!("\n=== Counterexample ===");
        println!("{}", ce);
    }

    // Show suggestions if any
    if !result.suggestions.is_empty() {
        println!("\n=== Suggestions ===");
        for suggestion in &result.suggestions {
            println!("  - {}", suggestion);
        }
    }

    // Exit with non-zero if not proven
    if !result.is_proven() {
        std::process::exit(1);
    }

    Ok(())
}
