//! InterpretML installation detection

use super::config::InterpretMlConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect InterpretML installation and return Python path
pub async fn detect_interpretml(config: &InterpretMlConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import interpret
    from interpret import show
    print(f"INTERPRET_VERSION:{interpret.__version__}")
except ImportError as e:
    print(f"INTERPRET_ERROR:{e}")
    print("INTERPRET_ERROR: Install with: pip install interpret scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"INTERPRET_ERROR:{e}")
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

    if output.status.success() && stdout.contains("INTERPRET_VERSION:") {
        Ok(python)
    } else if stdout.contains("INTERPRET_ERROR:") || stderr.contains("No module named") {
        Err(
            "InterpretML not installed. Install with: pip install interpret scikit-learn"
                .to_string(),
        )
    } else {
        Err(format!(
            "InterpretML detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
