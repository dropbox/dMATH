//! ReluVal detection and availability checking

use super::config::ReluValConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect ReluVal installation
pub async fn detect_reluval(config: &ReluValConfig) -> Result<PathBuf, String> {
    // Check custom path first
    if let Some(ref path) = config.reluval_path {
        if check_reluval_binary(path).await {
            return Ok(path.clone());
        }
    }

    // Check common installation paths
    let reluval_paths = vec![
        PathBuf::from("reluval"),
        PathBuf::from("/usr/local/bin/reluval"),
        PathBuf::from("/opt/reluval/reluval"),
    ];

    for path in reluval_paths {
        if check_reluval_binary(&path).await {
            return Ok(path);
        }
    }

    // Check Python wrapper
    let python_paths = vec![PathBuf::from("python3"), PathBuf::from("python")];

    for python_path in python_paths {
        if check_reluval_python(&python_path).await {
            return Ok(python_path);
        }
    }

    Err(
        "ReluVal not found. Install from: https://github.com/tcwangshiqi-columbia/ReluVal"
            .to_string(),
    )
}

async fn check_reluval_binary(path: &PathBuf) -> bool {
    let result = Command::new(path).args(["--version"]).output().await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

async fn check_reluval_python(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import reluval; print('ReluVal available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
