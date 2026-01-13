//! Verifpal installation detection

use super::config::VerifpalConfig;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

/// Detect Verifpal installation
pub async fn detect_verifpal(config: &VerifpalConfig) -> Result<PathBuf, String> {
    let verifpal_path = config
        .verifpal_path
        .clone()
        .or_else(|| which::which("verifpal").ok())
        .ok_or("Verifpal not found. Install from https://verifpal.com/")?;

    let output = Command::new(&verifpal_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute verifpal: {}", e))?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        debug!("Detected Verifpal version: {}", version.trim());
        Ok(verifpal_path)
    } else {
        Err("verifpal --version failed".to_string())
    }
}
