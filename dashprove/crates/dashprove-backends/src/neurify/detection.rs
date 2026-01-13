//! Neurify detection and availability checking

use super::config::NeurifyConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect Neurify installation
pub async fn detect_neurify(config: &NeurifyConfig) -> Result<PathBuf, String> {
    // Check custom path first
    if let Some(ref path) = config.neurify_path {
        if check_neurify_binary(path).await {
            return Ok(path.clone());
        }
    }

    // Check common installation paths
    let neurify_paths = vec![
        PathBuf::from("neurify"),
        PathBuf::from("/usr/local/bin/neurify"),
        PathBuf::from("/opt/neurify/neurify"),
    ];

    for path in neurify_paths {
        if check_neurify_binary(&path).await {
            return Ok(path);
        }
    }

    // Check Python wrapper
    let python_paths = vec![PathBuf::from("python3"), PathBuf::from("python")];

    for python_path in python_paths {
        if check_neurify_python(&python_path).await {
            return Ok(python_path);
        }
    }

    Err(
        "Neurify not found. Install from: https://github.com/tcwangshiqi-columbia/Neurify"
            .to_string(),
    )
}

async fn check_neurify_binary(path: &PathBuf) -> bool {
    let result = Command::new(path).args(["--version"]).output().await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

async fn check_neurify_python(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import neurify; print('Neurify available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
