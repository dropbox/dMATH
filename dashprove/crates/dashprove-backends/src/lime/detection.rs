//! LIME installation detection

use super::config::LimeConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect LIME installation and return Python path
pub async fn detect_lime(config: &LimeConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import lime
    import sklearn
    print(f"LIME_VERSION:{lime.__version__}")
except ImportError as e:
    print(f"LIME_ERROR:{e}")
    print("LIME_ERROR: Install with: pip install lime scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"LIME_ERROR:{e}")
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

    if output.status.success() && stdout.contains("LIME_VERSION:") {
        Ok(python)
    } else if stdout.contains("LIME_ERROR:") || stderr.contains("No module named") {
        Err("LIME not installed. Install with: pip install lime scikit-learn".to_string())
    } else {
        Err(format!(
            "LIME detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
