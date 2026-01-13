//! Apache TVM detection and availability checking

use super::config::TVMConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect TVM installation
pub async fn detect_tvm(config: &TVMConfig) -> Result<PathBuf, String> {
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
        if check_tvm_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("Apache TVM not found. Install with: pip install apache-tvm".to_string())
}

async fn check_tvm_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import tvm; print(f'TVM {tvm.__version__}')"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
