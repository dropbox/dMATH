//! Storm installation detection

use super::config::StormConfig;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

/// Detect Storm installation
pub async fn detect_storm(config: &StormConfig) -> Result<PathBuf, String> {
    let storm_path = config
        .storm_path
        .clone()
        .or_else(|| which::which("storm").ok())
        .or_else(|| which::which("storm-dft").ok())
        .ok_or("Storm not found. Install from https://www.stormchecker.org/")?;

    let output = Command::new(&storm_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute storm: {}", e))?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        debug!("Detected Storm version: {}", version.trim());
        Ok(storm_path)
    } else {
        Err("storm --version failed".to_string())
    }
}
