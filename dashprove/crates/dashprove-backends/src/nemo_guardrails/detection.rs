//! NeMo Guardrails installation detection

use super::config::NeMoGuardrailsConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect NeMo Guardrails installation and return Python path
pub async fn detect_nemo_guardrails(config: &NeMoGuardrailsConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import nemoguardrails
    print(f"NEMO_VERSION:{nemoguardrails.__version__}")
except ImportError as e:
    print(f"NEMO_ERROR:{e}")
    print("NEMO_ERROR: Install with: pip install nemoguardrails")
    sys.exit(1)
except Exception as e:
    print(f"NEMO_ERROR:{e}")
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

    if output.status.success() && stdout.contains("NEMO_VERSION:") {
        Ok(python)
    } else if stdout.contains("NEMO_ERROR:") || stderr.contains("No module named") {
        Err("NeMo Guardrails not installed. Install with: pip install nemoguardrails".to_string())
    } else {
        Err(format!(
            "NeMo Guardrails detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
