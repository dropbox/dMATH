//! TruLens installation detection

use super::config::TruLensConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect TruLens installation and return Python path
pub async fn detect_trulens(config: &TruLensConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import trulens_eval
    print(f"TRULENS_VERSION:{trulens_eval.__version__}")
except ImportError:
    try:
        import trulens
        print(f"TRULENS_VERSION:{trulens.__version__}")
    except ImportError as e:
        print(f"TRULENS_ERROR:{e}")
        print("TRULENS_ERROR: Install with: pip install trulens-eval")
        sys.exit(1)
except Exception as e:
    print(f"TRULENS_ERROR:{e}")
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

    if output.status.success() && stdout.contains("TRULENS_VERSION:") {
        Ok(python)
    } else if stdout.contains("TRULENS_ERROR:") || stderr.contains("No module named") {
        Err("TruLens not installed. Install with: pip install trulens-eval".to_string())
    } else {
        Err(format!(
            "TruLens detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
