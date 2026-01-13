//! Kani installation detection

use super::config::{KaniConfig, KaniDetection};
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

/// Detect whether cargo-kani is available
pub async fn detect_kani(config: &KaniConfig) -> KaniDetection {
    // Resolve cargo path
    let cargo_path = if let Some(ref path) = config.cargo_path {
        if path.exists() {
            path.clone()
        } else {
            return KaniDetection::NotFound(format!(
                "Configured cargo path does not exist: {:?}",
                path
            ));
        }
    } else {
        match which::which("cargo") {
            Ok(path) => path,
            Err(_) => {
                return KaniDetection::NotFound(
                    "cargo not found. Install Rust and cargo to use Kani.".to_string(),
                )
            }
        }
    };

    // Verify that cargo supports the kani subcommand
    let mut cmd = Command::new(&cargo_path);
    cmd.arg("kani")
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let result = tokio::time::timeout(config.timeout, cmd.output()).await;
    match result {
        Ok(Ok(output)) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected cargo-kani version: {}", version.trim());
            KaniDetection::Available { cargo_path }
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            KaniDetection::NotFound(format!("cargo kani --version failed: {}", stderr.trim()))
        }
        Ok(Err(e)) => KaniDetection::NotFound(format!("Failed to execute cargo: {}", e)),
        Err(_) => KaniDetection::NotFound("cargo kani --version timed out".to_string()),
    }
}
