//! NNCF installation detection

use super::config::NNCFConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect NNCF installation and return Python path
pub async fn detect_nncf(config: &NNCFConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if NNCF is available
    let check_script = r#"
import sys
try:
    import nncf
    print(f"NNCF_VERSION:{nncf.__version__}")
except ImportError as e:
    print(f"NNCF_ERROR:{e}")
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

    if output.status.success() && stdout.contains("NNCF_VERSION:") {
        Ok(python)
    } else if stdout.contains("NNCF_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("NNCF not installed. Install with: pip install nncf".to_string())
    } else {
        Err(format!(
            "NNCF detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
