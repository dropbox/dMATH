//! IREE installation detection

use super::config::IREEConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect IREE installation and return Python path
pub async fn detect_iree(config: &IREEConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if IREE runtime is available
    let check_script = r#"
import sys
try:
    import iree.runtime as rt
    import iree.compiler as compiler
    print(f"IREE_VERSION:{rt.version}")
except ImportError as e:
    print(f"IREE_ERROR:{e}")
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

    if output.status.success() && stdout.contains("IREE_VERSION:") {
        Ok(python)
    } else if stdout.contains("IREE_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("IREE not installed. Install with: pip install iree-compiler iree-runtime".to_string())
    } else {
        Err(format!(
            "IREE detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
