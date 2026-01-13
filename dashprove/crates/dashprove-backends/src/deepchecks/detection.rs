//! Deepchecks installation detection

use super::config::DeepchecksConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Deepchecks installation and return Python path
pub async fn detect_deepchecks(config: &DeepchecksConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Deepchecks is available
    let check_script = r#"
import sys
try:
    import deepchecks
    print(f"DEEPCHECKS_VERSION:{deepchecks.__version__}")
except ImportError as e:
    print(f"DEEPCHECKS_ERROR:{e}")
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

    if output.status.success() && stdout.contains("DEEPCHECKS_VERSION:") {
        Ok(python)
    } else if stdout.contains("DEEPCHECKS_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("Deepchecks not installed. Install with: pip install deepchecks".to_string())
    } else {
        Err(format!(
            "Deepchecks detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
