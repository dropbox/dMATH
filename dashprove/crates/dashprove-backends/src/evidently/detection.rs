//! Evidently installation detection

use super::config::EvidentlyConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Evidently installation and return Python path
pub async fn detect_evidently(config: &EvidentlyConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Evidently is available
    let check_script = r#"
import sys
try:
    import evidently
    print(f"EVIDENTLY_VERSION:{evidently.__version__}")
except ImportError as e:
    print(f"EVIDENTLY_ERROR:{e}")
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

    if output.status.success() && stdout.contains("EVIDENTLY_VERSION:") {
        Ok(python)
    } else if stdout.contains("EVIDENTLY_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("Evidently not installed. Install with: pip install evidently".to_string())
    } else {
        Err(format!(
            "Evidently detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
