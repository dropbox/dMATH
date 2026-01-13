//! TextAttack detection and availability checking

use super::config::TextAttackConfig;
use std::path::PathBuf;
use tokio::process::Command;

/// Detect TextAttack installation
///
/// Returns the path to the Python interpreter with TextAttack installed.
pub async fn detect_textattack(config: &TextAttackConfig) -> Result<PathBuf, String> {
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
        if check_textattack_available(&python_path).await {
            return Ok(python_path);
        }
    }

    Err("TextAttack not found. Install with: pip install textattack".to_string())
}

/// Check if TextAttack is available via the given Python interpreter
async fn check_textattack_available(python_path: &PathBuf) -> bool {
    let result = Command::new(python_path)
        .args(["-c", "import textattack; print(textattack.__version__)"])
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Get TextAttack version
#[allow(dead_code)]
pub async fn get_textattack_version(python_path: &PathBuf) -> Option<String> {
    let result = Command::new(python_path)
        .args(["-c", "import textattack; print(textattack.__version__)"])
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        _ => None,
    }
}
