//! AIMET installation detection

use super::config::AimetConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect AIMET installation and return Python path
pub async fn detect_aimet(config: &AimetConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if AIMET is available
    let check_script = r#"
import sys
try:
    import aimet_common
    import aimet_torch
    print(f"AIMET_VERSION:{aimet_common.__version__}")
except ImportError as e:
    print(f"AIMET_ERROR:{e}")
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

    if output.status.success() && stdout.contains("AIMET_VERSION:") {
        Ok(python)
    } else if stdout.contains("AIMET_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("AIMET not installed. Install from: https://github.com/quic/aimet".to_string())
    } else {
        Err(format!(
            "AIMET detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
