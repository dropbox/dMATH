//! Fairlearn installation detection

use super::config::FairlearnConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Fairlearn installation and return Python path
pub async fn detect_fairlearn(config: &FairlearnConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Fairlearn is available
    let check_script = r#"
import sys
try:
    import fairlearn
    print(f"FAIRLEARN_VERSION:{fairlearn.__version__}")
except ImportError as e:
    print(f"FAIRLEARN_ERROR:{e}")
    sys.exit(1)
"#;

    let output = Command::new(&python)
        .arg("-c")
        .arg(check_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to run Python: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() && stdout.contains("FAIRLEARN_VERSION:") {
        Ok(python)
    } else if stdout.contains("FAIRLEARN_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("Fairlearn not installed. Install with: pip install fairlearn".to_string())
    } else {
        Err(format!(
            "Fairlearn detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
