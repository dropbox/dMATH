//! NNV detection and availability checking

use super::config::NnvConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect NNV installation
///
/// Returns the path to the Python interpreter with NNV installed.
pub async fn detect_nnv(config: &NnvConfig) -> Result<PathBuf, String> {
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
        if check_nnv_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("NNV not found. Install with: pip install nnv".to_string())
}

/// Check if NNV is available via the given Python interpreter
async fn check_nnv_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import nnv; print('NNV available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Get NNV version
#[allow(dead_code)]
pub async fn get_nnv_version(python_path: &PathBuf) -> Option<String> {
    let result = Command::new(python_path)
        .args([
            "-c",
            "import nnv; print(nnv.__version__ if hasattr(nnv, '__version__') else '0.0.0')",
        ])
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        _ => None,
    }
}
