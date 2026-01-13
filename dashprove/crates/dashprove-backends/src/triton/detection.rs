//! Triton installation detection

use super::config::TritonConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Triton installation and return Python path
pub async fn detect_triton(config: &TritonConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Triton is available
    let check_script = r#"
import sys
try:
    import triton
    print(f"TRITON_VERSION:{triton.__version__}")
except ImportError as e:
    print(f"TRITON_ERROR:{e}")
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

    if output.status.success() && stdout.contains("TRITON_VERSION:") {
        Ok(python)
    } else if stdout.contains("TRITON_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("Triton not installed. Install with: pip install triton".to_string())
    } else {
        Err(format!(
            "Triton detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
