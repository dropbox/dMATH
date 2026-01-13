//! PRISM installation detection

use super::config::PrismConfig;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

/// Detect PRISM installation
pub async fn detect_prism(config: &PrismConfig) -> Result<PathBuf, String> {
    let prism_path = config
        .prism_path
        .clone()
        .or_else(|| which::which("prism").ok())
        .or_else(|| {
            // Check common installation locations
            let candidates = vec![
                PathBuf::from("/opt/prism/bin/prism"),
                PathBuf::from("/usr/local/prism/bin/prism"),
                dirs::home_dir()
                    .map(|h| h.join("prism/bin/prism"))
                    .unwrap_or_default(),
            ];
            candidates.into_iter().find(|p| p.exists())
        })
        .ok_or("PRISM not found. Install from https://www.prismmodelchecker.org/")?;

    let output = Command::new(&prism_path)
        .arg("-version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute prism: {}", e))?;

    if output.status.success() || !output.stdout.is_empty() {
        let version = String::from_utf8_lossy(&output.stdout);
        debug!("Detected PRISM version: {}", version.trim());
        Ok(prism_path)
    } else {
        Err("prism -version failed".to_string())
    }
}
