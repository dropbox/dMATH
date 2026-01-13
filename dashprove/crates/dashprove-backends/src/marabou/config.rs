//! Configuration and detection for Marabou backend

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

/// Configuration for Marabou backend
#[derive(Debug, Clone)]
pub struct MarabouConfig {
    /// Path to Marabou binary (if not in PATH)
    pub marabou_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Number of threads to use
    pub num_threads: Option<usize>,
    /// Enable split-and-conquer mode
    pub split_and_conquer: bool,
}

impl Default for MarabouConfig {
    fn default() -> Self {
        Self {
            marabou_path: None,
            timeout: Duration::from_secs(300),
            num_threads: None,
            split_and_conquer: false,
        }
    }
}

/// Detect whether Marabou is available
pub async fn detect_marabou(config: &MarabouConfig) -> Result<PathBuf, String> {
    let marabou_path = if let Some(ref path) = config.marabou_path {
        if path.exists() {
            path.clone()
        } else {
            return Err(format!(
                "Configured Marabou path does not exist: {:?}",
                path
            ));
        }
    } else {
        // Try common installation paths
        let candidates = vec![
            which::which("Marabou").ok(),
            which::which("marabou").ok(),
            Some(PathBuf::from("/usr/local/bin/Marabou")),
            Some(PathBuf::from("/opt/marabou/Marabou")),
        ];

        candidates
            .into_iter()
            .flatten()
            .find(|p| p.exists())
            .ok_or_else(|| {
                "Marabou not found. Install from https://github.com/NeuralNetworkVerification/Marabou"
                    .to_string()
            })?
    };

    // Verify it works
    let output = Command::new(&marabou_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute Marabou: {}", e))?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        debug!("Detected Marabou version: {}", version.trim());
        Ok(marabou_path)
    } else {
        Err("Marabou --version failed".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = MarabouConfig::default();
        assert!(config.marabou_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(!config.split_and_conquer);
    }
}
