//! Intel Neural Compressor installation detection

use super::config::NeuralCompressorConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Neural Compressor installation and return Python path
pub async fn detect_neural_compressor(config: &NeuralCompressorConfig) -> Result<String, String> {
    let python = config
        .python_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "python3".to_string());

    // Check if Neural Compressor is available
    let check_script = r#"
import sys
try:
    import neural_compressor
    print(f"NC_VERSION:{neural_compressor.__version__}")
except ImportError as e:
    print(f"NC_ERROR:{e}")
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

    if output.status.success() && stdout.contains("NC_VERSION:") {
        Ok(python)
    } else if stdout.contains("NC_ERROR:") || stderr.contains("ModuleNotFoundError") {
        Err(
            "Neural Compressor not installed. Install with: pip install neural-compressor"
                .to_string(),
        )
    } else {
        Err(format!(
            "Neural Compressor detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
