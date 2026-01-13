//! WhyLogs installation detection

use super::config::WhyLogsConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect WhyLogs installation and return Python path
pub async fn detect_whylogs(config: &WhyLogsConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if WhyLogs is available
    let check_script = r#"
import sys
try:
    import whylogs
    print(f"WHYLOGS_VERSION:{whylogs.__version__}")
except ImportError as e:
    print(f"WHYLOGS_ERROR:{e}")
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

    if output.status.success() && stdout.contains("WHYLOGS_VERSION:") {
        Ok(python)
    } else if stdout.contains("WHYLOGS_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err("WhyLogs not installed. Install with: pip install whylogs".to_string())
    } else {
        Err(format!(
            "WhyLogs detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
