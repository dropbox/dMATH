//! AutoLiRPA detection and availability checking

use super::config::AutoLirpaConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect AutoLiRPA installation
pub async fn detect_autolirpa(config: &AutoLirpaConfig) -> Result<PathBuf, String> {
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
        if check_autolirpa_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("Auto-LiRPA not found. Install with: pip install auto_lirpa".to_string())
}

async fn check_autolirpa_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import auto_LiRPA; print('Auto-LiRPA available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
