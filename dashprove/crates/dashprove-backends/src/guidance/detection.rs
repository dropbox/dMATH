//! Guidance installation detection

use super::config::GuidanceConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Guidance installation and return Python path
pub async fn detect_guidance(config: &GuidanceConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import guidance
    print(f"GUIDANCE_VERSION:{guidance.__version__}")
except ImportError as e:
    print(f"GUIDANCE_ERROR:{e}")
    print("GUIDANCE_ERROR: Install with: pip install guidance")
    sys.exit(1)
except Exception as e:
    print(f"GUIDANCE_ERROR:{e}")
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

    if output.status.success() && stdout.contains("GUIDANCE_VERSION:") {
        Ok(python)
    } else if stdout.contains("GUIDANCE_ERROR:") || stderr.contains("No module named") {
        Err("Guidance not installed. Install with: pip install guidance".to_string())
    } else {
        Err(format!(
            "Guidance detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
