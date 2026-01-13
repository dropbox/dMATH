//! ONNX Runtime detection and availability checking

use super::config::OnnxRuntimeConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect ONNX Runtime installation
pub async fn detect_onnxruntime(config: &OnnxRuntimeConfig) -> Result<PathBuf, String> {
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
        if check_onnxruntime_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("ONNX Runtime not found. Install with: pip install onnxruntime".to_string())
}

async fn check_onnxruntime_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args([
            "-c",
            "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__}')",
        ])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
