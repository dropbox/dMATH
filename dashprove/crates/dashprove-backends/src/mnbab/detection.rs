//! MNBaB detection and availability checking

use super::config::MNBaBConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect MNBaB installation
pub async fn detect_mnbab(config: &MNBaBConfig) -> Result<PathBuf, String> {
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
        if check_mnbab_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("MNBaB not found. Install from: https://github.com/eth-sri/mn-bab".to_string())
}

async fn check_mnbab_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import mnbab; print('MNBaB available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
