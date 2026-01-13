//! Nnenum detection and availability checking

use super::config::NnenumConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect nnenum installation
///
/// Returns the path to the Python interpreter with nnenum installed.
pub async fn detect_nnenum(config: &NnenumConfig) -> Result<PathBuf, String> {
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
        if check_nnenum_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("nnenum not found. Install with: pip install nnenum".to_string())
}

/// Check if nnenum is available via the given Python interpreter
async fn check_nnenum_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import nnenum; print('nnenum available')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Get nnenum version
#[allow(dead_code)]
pub async fn get_nnenum_version(python_path: &PathBuf) -> Option<String> {
    let result = Command::new(python_path)
        .args(["-c", "import nnenum; print(nnenum.__version__ if hasattr(nnenum, '__version__') else '0.0.0')"])
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        _ => None,
    }
}
