//! Promptfoo installation detection

use super::config::PromptfooConfig;
use std::process::Stdio;
use tokio::process::Command;

/// Detect Promptfoo installation and return node path
pub async fn detect_promptfoo(config: &PromptfooConfig) -> Result<String, String> {
    let node = config
        .node_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "node".to_string());

    // Check if node is available
    let node_check = Command::new(&node)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to run Node.js: {}", e))?;

    if !node_check.status.success() {
        return Err("Node.js not available. Install with: brew install node".to_string());
    }

    // Check if promptfoo is installed
    let check_script = r#"
try {
    const promptfoo = require('promptfoo');
    console.log('PROMPTFOO_VERSION:' + (promptfoo.version || 'installed'));
} catch (e) {
    console.log('PROMPTFOO_ERROR:' + e.message);
    process.exit(1);
}
"#;

    let output = Command::new(&node)
        .arg("-e")
        .arg(check_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to check promptfoo: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() && stdout.contains("PROMPTFOO_VERSION:") {
        Ok(node)
    } else if stdout.contains("PROMPTFOO_ERROR:") || stderr.contains("Cannot find module") {
        Err("Promptfoo not installed. Install with: npm install -g promptfoo".to_string())
    } else {
        Err(format!(
            "Promptfoo detection failed: {}{}",
            stdout.trim(),
            stderr.trim()
        ))
    }
}
