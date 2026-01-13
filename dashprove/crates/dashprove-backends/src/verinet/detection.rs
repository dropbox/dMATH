//! VeriNet detection and availability checking

use super::config::VeriNetConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect VeriNet installation
pub async fn detect_verinet(config: &VeriNetConfig) -> Result<PathBuf, String> {
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
        if check_verinet_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err(
        "VeriNet not found. Install from: https://github.com/vas-group-imperial/VeriNet"
            .to_string(),
    )
}

async fn check_verinet_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import verinet; print('VeriNet available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
