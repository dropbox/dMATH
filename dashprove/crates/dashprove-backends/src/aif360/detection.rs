//! AIF360 installation detection

use super::config::AIF360Config;
use std::process::Stdio;
use tokio::process::Command;

/// Detect AIF360 installation and return Python path
pub async fn detect_aif360(config: &AIF360Config) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if AIF360 is available
    let check_script = r#"
import sys
try:
    import aif360
    print(f"AIF360_VERSION:{aif360.__version__}")
except ImportError as e:
    print(f"AIF360_ERROR:{e}")
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

    if output.status.success() && stdout.contains("AIF360_VERSION:") {
        Ok(python)
    } else if stdout.contains("AIF360_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("AIF360 not installed. Install with: pip install aif360".to_string())
    } else {
        Err(format!(
            "AIF360 detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
