//! Venus detection and availability checking

use super::config::VenusConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect Venus installation
pub async fn detect_venus(config: &VenusConfig) -> Result<PathBuf, String> {
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
        if check_venus_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("Venus not found. Install with: pip install venus-verification".to_string())
}

async fn check_venus_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import venus; print('Venus available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
