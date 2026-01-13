//! ERAN availability detection
//!
//! Detects whether ERAN is installed and available.

use super::EranConfig;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;

/// Detect whether ERAN is available
///
/// Returns (python_path, eran_path) if available, error message otherwise.
pub async fn detect_eran(config: &EranConfig) -> Result<(PathBuf, PathBuf), String> {
    let python_path = config
        .python_path
        .clone()
        .or_else(|| which::which("python3").ok())
        .or_else(|| which::which("python").ok())
        .ok_or("Python not found")?;

    let eran_path = if let Some(ref path) = config.eran_path {
        if path.exists() {
            path.clone()
        } else {
            return Err(format!("ERAN path does not exist: {:?}", path));
        }
    } else {
        // Check for ERAN in common locations
        let candidates = vec![
            PathBuf::from("./eran"),
            PathBuf::from("~/eran"),
            PathBuf::from("/opt/eran"),
        ];

        candidates
            .into_iter()
            .find(|p| p.exists())
            .ok_or("ERAN not found. Clone from https://github.com/eth-sri/eran".to_string())?
    };

    // Verify ELINA is available (required dependency)
    let check = Command::new(&python_path)
        .arg("-c")
        .arg("import numpy; print('ERAN deps OK')")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to check ERAN dependencies: {}", e))?;

    if !check.status.success() {
        return Err("ERAN dependencies not satisfied (numpy required)".to_string());
    }

    Ok((python_path, eran_path))
}
