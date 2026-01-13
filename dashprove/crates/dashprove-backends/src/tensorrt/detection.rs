//! TensorRT detection and availability checking

use super::config::TensorRTConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect TensorRT installation
pub async fn detect_tensorrt(config: &TensorRTConfig) -> Result<PathBuf, String> {
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
        if check_tensorrt_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("TensorRT not found. Install from NVIDIA SDK or pip install tensorrt".to_string())
}

async fn check_tensorrt_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args([
            "-c",
            "import tensorrt; print(f'TensorRT {tensorrt.__version__}')",
        ])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
