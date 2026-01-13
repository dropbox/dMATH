//! SHAP installation detection

use super::config::ShapConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect SHAP installation and return Python path
pub async fn detect_shap(config: &ShapConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import shap
    import numpy
    print(f"SHAP_VERSION:{shap.__version__}")
except ImportError as e:
    print(f"SHAP_ERROR:{e}")
    print("SHAP_ERROR: Install with: pip install shap scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"SHAP_ERROR:{e}")
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

    if output.status.success() && stdout.contains("SHAP_VERSION:") {
        Ok(python)
    } else if stdout.contains("SHAP_ERROR:") || stderr.contains("No module named") {
        Err("SHAP not installed. Install with: pip install shap scikit-learn".to_string())
    } else {
        Err(format!(
            "SHAP detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
