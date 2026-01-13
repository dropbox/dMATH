//! Kani Fast detection logic

use super::config::{KaniFastConfig, KaniFastDetection};
use tokio::process::Command;
use tracing::debug;

/// Detect if kani-fast CLI is available
pub async fn detect_kani_fast(config: &KaniFastConfig) -> KaniFastDetection {
    // First check if a custom path is provided
    if let Some(ref cli_path) = config.cli_path {
        if cli_path.exists() {
            return KaniFastDetection::Available {
                cli_path: cli_path.clone(),
            };
        }
        return KaniFastDetection::NotFound(format!(
            "kani-fast not found at specified path: {}",
            cli_path.display()
        ));
    }

    // Try to find kani-fast in PATH
    match which::which("kani-fast") {
        Ok(path) => {
            debug!("Found kani-fast at {:?}", path);

            // Verify it actually works
            match Command::new(&path).arg("--version").output().await {
                Ok(output) if output.status.success() => {
                    let version = String::from_utf8_lossy(&output.stdout);
                    debug!("kani-fast version: {}", version.trim());
                    KaniFastDetection::Available { cli_path: path }
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    KaniFastDetection::NotFound(format!(
                        "kani-fast found but failed to run: {}",
                        stderr.trim()
                    ))
                }
                Err(e) => KaniFastDetection::NotFound(format!(
                    "kani-fast found but failed to execute: {}",
                    e
                )),
            }
        }
        Err(_) => {
            // Check if cargo is available (for installation instructions)
            if which::which("cargo").is_ok() {
                debug!("kani-fast not in PATH, cargo is available");
                KaniFastDetection::NotFound(
                    "kani-fast not found in PATH. Install with: cargo install --path ~/kani_fast/crates/kani-fast-cli"
                        .to_string(),
                )
            } else {
                KaniFastDetection::NotFound(
                    "kani-fast not found: neither kani-fast nor cargo are available in PATH"
                        .to_string(),
                )
            }
        }
    }
}
