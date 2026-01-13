//! Tamarin installation detection

use super::config::TamarinConfig;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

/// Detect Tamarin installation
pub async fn detect_tamarin(config: &TamarinConfig) -> Result<PathBuf, String> {
    let tamarin_path = config
        .tamarin_path
        .clone()
        .or_else(|| which::which("tamarin-prover").ok())
        .ok_or("Tamarin not found. Install from https://tamarin-prover.com/")?;

    let output = Command::new(&tamarin_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute tamarin-prover: {}", e))?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        debug!("Detected Tamarin version: {}", version.trim());
        Ok(tamarin_path)
    } else {
        Err("tamarin-prover --version failed".to_string())
    }
}
