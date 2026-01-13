//! ART detection and availability checking

use super::config::ArtConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect ART installation
///
/// Returns the path to the Python interpreter with ART installed.
pub async fn detect_art(config: &ArtConfig) -> Result<PathBuf, String> {
    // Try configured Python path first
    let python_paths = if let Some(ref p) = config.python_path {
        vec![p.clone()]
    } else {
        vec![
            PathBuf::from("python3"),
            PathBuf::from("python"),
            PathBuf::from("/usr/bin/python3"),
            PathBuf::from("/usr/local/bin/python3"),
        ]
    };

    for python_path in python_paths {
        if check_art_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("ART not found. Install with: pip install adversarial-robustness-toolbox".to_string())
}

/// Check if ART is available via the given Python interpreter
async fn check_art_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import art; print(art.__version__)"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Get ART version
#[allow(dead_code)]
pub async fn get_art_version(python_path: &PathBuf) -> Option<String> {
    let result = Command::new(python_path)
        .args(["-c", "import art; print(art.__version__)"])
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        _ => None,
    }
}
