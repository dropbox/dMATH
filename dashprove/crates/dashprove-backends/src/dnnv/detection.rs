//! DNNV detection and availability checking

use super::config::DnnvConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect DNNV installation
pub async fn detect_dnnv(config: &DnnvConfig) -> Result<PathBuf, String> {
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
        if check_dnnv_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("DNNV not found. Install with: pip install dnnv".to_string())
}

async fn check_dnnv_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import dnnv; print('DNNV available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
