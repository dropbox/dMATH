//! GuardrailsAI installation detection

use super::config::GuardrailsAIConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect GuardrailsAI installation and return Python path
pub async fn detect_guardrails_ai(config: &GuardrailsAIConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    let check_script = r#"
import sys
try:
    import guardrails
    print(f"GUARDRAILS_VERSION:{guardrails.__version__}")
except ImportError as e:
    print(f"GUARDRAILS_ERROR:{e}")
    print("GUARDRAILS_ERROR: Install with: pip install guardrails-ai")
    sys.exit(1)
except Exception as e:
    print(f"GUARDRAILS_ERROR:{e}")
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

    if output.status.success() && stdout.contains("GUARDRAILS_VERSION:") {
        Ok(python)
    } else if stdout.contains("GUARDRAILS_ERROR:") || stderr.contains("No module named") {
        Err("GuardrailsAI not installed. Install with: pip install guardrails-ai".to_string())
    } else {
        Err(format!(
            "GuardrailsAI detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
