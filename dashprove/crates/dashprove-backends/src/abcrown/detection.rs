//! alpha-beta-CROWN installation detection

use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

use super::AbCrownConfig;

/// Detect whether alpha-beta-CROWN is available
pub async fn detect_abcrown(config: &AbCrownConfig) -> Result<(PathBuf, PathBuf), String> {
    // Find Python
    let python_path = if let Some(ref path) = config.python_path {
        if path.exists() {
            path.clone()
        } else {
            return Err(format!("Configured Python path does not exist: {:?}", path));
        }
    } else {
        which::which("python3")
            .or_else(|_| which::which("python"))
            .map_err(|_| "Python not found. Install Python 3.8+ for alpha-beta-CROWN.")?
    };

    // Find abcrown.py
    let abcrown_path = if let Some(ref path) = config.abcrown_path {
        if path.exists() {
            path.clone()
        } else {
            return Err(format!(
                "Configured abcrown.py path does not exist: {:?}",
                path
            ));
        }
    } else {
        // Try common locations
        let candidates = vec![
            PathBuf::from("./alpha-beta-CROWN/abcrown.py"),
            PathBuf::from("~/alpha-beta-CROWN/abcrown.py"),
            PathBuf::from("/opt/alpha-beta-CROWN/abcrown.py"),
        ];

        candidates
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| {
                "alpha-beta-CROWN not found. Clone from https://github.com/Verified-Intelligence/alpha-beta-CROWN"
                    .to_string()
            })?
    };

    // Verify Python can import required packages
    let output = Command::new(&python_path)
        .arg("-c")
        .arg("import torch; print(torch.__version__)")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to check PyTorch: {}", e))?;

    if !output.status.success() {
        return Err("PyTorch not available. Install with: pip install torch".to_string());
    }

    let torch_version = String::from_utf8_lossy(&output.stdout);
    debug!("Detected PyTorch version: {}", torch_version.trim());

    Ok((python_path, abcrown_path))
}
