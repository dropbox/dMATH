//! Great Expectations installation detection

use super::config::GreatExpectationsConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Great Expectations installation and return Python path
pub async fn detect_great_expectations(config: &GreatExpectationsConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Great Expectations is available
    let check_script = r#"
import sys
try:
    import great_expectations as gx
    print(f"GX_VERSION:{gx.__version__}")
except ImportError as e:
    print(f"GX_ERROR:{e}")
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

    if output.status.success() && stdout.contains("GX_VERSION:") {
        Ok(python)
    } else if stdout.contains("GX_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err(
            "Great Expectations not installed. Install with: pip install great_expectations"
                .to_string(),
        )
    } else {
        Err(format!(
            "Great Expectations detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
