//! MIRI installation detection

use crate::config::MiriConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

/// MIRI version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriVersion {
    /// Full version string
    pub version_string: String,
    /// MIRI version (if parseable)
    pub miri_version: Option<String>,
    /// Rust version MIRI is built against
    pub rust_version: Option<String>,
}

/// Result of MIRI detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MiriDetection {
    /// MIRI is available
    Available {
        /// Path to cargo executable
        cargo_path: PathBuf,
        /// MIRI version information
        version: MiriVersion,
    },
    /// MIRI is not available
    NotFound(String),
}

impl MiriDetection {
    /// Check if MIRI is available
    pub fn is_available(&self) -> bool {
        matches!(self, MiriDetection::Available { .. })
    }

    /// Get the cargo path if available
    pub fn cargo_path(&self) -> Option<&PathBuf> {
        match self {
            MiriDetection::Available { cargo_path, .. } => Some(cargo_path),
            MiriDetection::NotFound(_) => None,
        }
    }

    /// Get the version info if available
    pub fn version(&self) -> Option<&MiriVersion> {
        match self {
            MiriDetection::Available { version, .. } => Some(version),
            MiriDetection::NotFound(_) => None,
        }
    }
}

/// Detect whether MIRI is available
pub async fn detect_miri(config: &MiriConfig) -> MiriDetection {
    // Resolve cargo path
    let cargo_path = if let Some(ref path) = config.cargo_path {
        if path.exists() {
            path.clone()
        } else {
            return MiriDetection::NotFound(format!(
                "Configured cargo path does not exist: {:?}",
                path
            ));
        }
    } else {
        match which::which("cargo") {
            Ok(path) => path,
            Err(_) => {
                return MiriDetection::NotFound(
                    "cargo not found. Install Rust and cargo to use MIRI.".to_string(),
                )
            }
        }
    };

    // Check if miri component is installed via rustup
    let rustup_check = check_miri_component().await;
    if let Err(reason) = rustup_check {
        return MiriDetection::NotFound(reason);
    }

    // Verify that cargo miri works
    let mut cmd = Command::new(&cargo_path);
    cmd.arg("miri")
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let result = tokio::time::timeout(config.timeout, cmd.output()).await;
    match result {
        Ok(Ok(output)) if output.status.success() => {
            let version_string = String::from_utf8_lossy(&output.stdout).trim().to_string();
            debug!("Detected cargo miri version: {}", version_string);

            let version = parse_miri_version(&version_string);
            MiriDetection::Available {
                cargo_path,
                version,
            }
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            MiriDetection::NotFound(format!(
                "cargo miri --version failed: {} {}",
                stderr.trim(),
                stdout.trim()
            ))
        }
        Ok(Err(e)) => MiriDetection::NotFound(format!("Failed to execute cargo: {}", e)),
        Err(_) => MiriDetection::NotFound("cargo miri --version timed out".to_string()),
    }
}

/// Check if miri component is installed via rustup
async fn check_miri_component() -> Result<(), String> {
    // Try to find rustup
    let rustup_path = match which::which("rustup") {
        Ok(path) => path,
        Err(_) => {
            // No rustup means we can't check components, but miri might still work
            // if installed some other way
            return Ok(());
        }
    };

    let mut cmd = Command::new(&rustup_path);
    cmd.arg("component")
        .arg("list")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run rustup: {}", e))?;

    if !output.status.success() {
        // Can't check components, but continue anyway
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let has_miri = stdout
        .lines()
        .any(|line| line.starts_with("miri") && line.contains("installed"));

    if has_miri {
        Ok(())
    } else {
        Err(
            "MIRI component not installed. Install with: rustup +nightly component add miri"
                .to_string(),
        )
    }
}

/// Parse MIRI version string into structured format
fn parse_miri_version(version_string: &str) -> MiriVersion {
    // Example: "miri 0.1.0 (abc1234 2024-01-01)"
    let miri_version = version_string
        .split_whitespace()
        .nth(1)
        .map(|s| s.to_string());

    // Try to get Rust version from rustc
    let rust_version = std::process::Command::new("rustc")
        .arg("+nightly")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });

    MiriVersion {
        version_string: version_string.to_string(),
        miri_version,
        rust_version,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_miri_version() {
        let version = parse_miri_version("miri 0.1.0 (abc1234 2024-01-01)");
        assert_eq!(version.miri_version, Some("0.1.0".to_string()));
        assert!(version.version_string.contains("miri"));
    }

    #[test]
    fn test_detection_is_available() {
        let available = MiriDetection::Available {
            cargo_path: PathBuf::from("/usr/bin/cargo"),
            version: MiriVersion {
                version_string: "miri 0.1.0".to_string(),
                miri_version: Some("0.1.0".to_string()),
                rust_version: None,
            },
        };
        assert!(available.is_available());

        let not_found = MiriDetection::NotFound("not installed".to_string());
        assert!(!not_found.is_available());
    }

    #[test]
    fn test_detection_cargo_path() {
        let available = MiriDetection::Available {
            cargo_path: PathBuf::from("/usr/bin/cargo"),
            version: MiriVersion {
                version_string: "miri 0.1.0".to_string(),
                miri_version: Some("0.1.0".to_string()),
                rust_version: None,
            },
        };
        assert_eq!(
            available.cargo_path(),
            Some(&PathBuf::from("/usr/bin/cargo"))
        );

        let not_found = MiriDetection::NotFound("not installed".to_string());
        assert!(not_found.cargo_path().is_none());
    }

    #[tokio::test]
    async fn test_detect_miri_with_missing_cargo() {
        let config = MiriConfig {
            cargo_path: Some(PathBuf::from("/nonexistent/cargo")),
            ..Default::default()
        };
        let detection = detect_miri(&config).await;
        assert!(!detection.is_available());
    }
}
