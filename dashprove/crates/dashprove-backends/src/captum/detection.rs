//! Captum installation detection

use super::config::CaptumConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Captum installation and return Python path
pub async fn detect_captum(config: &CaptumConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import torch
    import captum
    print(f"CAPTUM_VERSION:{captum.__version__}")
except ImportError as e:
    print(f"CAPTUM_ERROR:{e}")
    print("CAPTUM_ERROR: Install with: pip install torch captum")
    sys.exit(1)
except Exception as e:
    print(f"CAPTUM_ERROR:{e}")
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

    if output.status.success() && stdout.contains("CAPTUM_VERSION:") {
        Ok(python)
    } else if stdout.contains("CAPTUM_ERROR:") || stderr.contains("No module named") {
        Err("Captum not installed. Install with: pip install torch captum".to_string())
    } else {
        Err(format!(
            "Captum detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
