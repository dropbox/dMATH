//! Foolbox detection and availability checking

use super::config::FoolboxConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect Foolbox installation
///
/// Returns the path to the Python interpreter with Foolbox installed.
pub async fn detect_foolbox(config: &FoolboxConfig) -> Result<PathBuf, String> {
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
        if check_foolbox_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("Foolbox not found. Install with: pip install foolbox".to_string())
}

/// Check if Foolbox is available via the given Python interpreter
async fn check_foolbox_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import foolbox; print(foolbox.__version__)"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
