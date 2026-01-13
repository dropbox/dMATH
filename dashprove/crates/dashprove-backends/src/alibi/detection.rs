//! Alibi installation detection

use super::config::AlibiConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Alibi installation and return Python path
pub async fn detect_alibi(config: &AlibiConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import alibi
    import numpy
    print(f"ALIBI_VERSION:{alibi.__version__}")
except ImportError as e:
    print(f"ALIBI_ERROR:{e}")
    print("ALIBI_ERROR: Install with: pip install alibi scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"ALIBI_ERROR:{e}")
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

    if output.status.success() && stdout.contains("ALIBI_VERSION:") {
        Ok(python)
    } else if stdout.contains("ALIBI_ERROR:") || stderr.contains("No module named") {
        Err("Alibi not installed. Install with: pip install alibi scikit-learn".to_string())
    } else {
        Err(format!(
            "Alibi detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
