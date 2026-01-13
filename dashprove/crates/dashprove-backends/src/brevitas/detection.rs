//! Brevitas installation detection

use super::config::BrevitasConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Brevitas installation and return Python path
pub async fn detect_brevitas(config: &BrevitasConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Brevitas is available
    let check_script = r#"
import sys
try:
    import brevitas
    print(f"BREVITAS_VERSION:{brevitas.__version__}")
except ImportError as e:
    print(f"BREVITAS_ERROR:{e}")
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

    if output.status.success() && stdout.contains("BREVITAS_VERSION:") {
        Ok(python)
    } else if stdout.contains("BREVITAS_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("Brevitas not installed. Install with: pip install brevitas".to_string())
    } else {
        Err(format!(
            "Brevitas detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
