//! Aequitas installation detection

use super::config::AequitasConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Aequitas installation and return Python path
pub async fn detect_aequitas(config: &AequitasConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Aequitas is available
    let check_script = r#"
import sys
try:
    import aequitas
    print(f"AEQUITAS_VERSION:{aequitas.__version__}")
except ImportError as e:
    print(f"AEQUITAS_ERROR:{e}")
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

    if output.status.success() && stdout.contains("AEQUITAS_VERSION:") {
        Ok(python)
    } else if stdout.contains("AEQUITAS_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("Aequitas not installed. Install with: pip install aequitas".to_string())
    } else {
        Err(format!(
            "Aequitas detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
