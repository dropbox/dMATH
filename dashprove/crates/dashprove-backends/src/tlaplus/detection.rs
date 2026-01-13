//! TLC installation detection

use crate::util::expand_home_dir;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;

use super::TlaPlusConfig;

/// Detection result for TLC model checker
#[derive(Debug, Clone)]
pub enum TlcDetection {
    /// TLC available as a standalone command
    Standalone(PathBuf),
    /// TLC available via tla2tools.jar
    Jar {
        java_path: PathBuf,
        jar_path: PathBuf,
    },
    /// TLC not found
    NotFound(String),
}

/// Detect TLC installation based on configuration
pub async fn detect_tlc(config: &TlaPlusConfig) -> TlcDetection {
    // 1. Check if tlc_path is explicitly configured
    if let Some(ref path) = config.tlc_path {
        if path.exists() {
            if path.extension().map(|e| e == "jar").unwrap_or(false) {
                // It's a JAR file
                return TlcDetection::Jar {
                    java_path: config.java_path.clone(),
                    jar_path: path.clone(),
                };
            } else {
                return TlcDetection::Standalone(path.clone());
            }
        }
    }

    // 2. Check for tlc in PATH
    if let Ok(path) = which::which("tlc") {
        return TlcDetection::Standalone(path);
    }

    // 3. Check for tla2tools.jar in common locations
    let jar_locations = [
        // User's home directory
        expand_home_dir("~/.tla/tla2tools.jar"),
        expand_home_dir("~/tla2tools.jar"),
        // System locations
        Some(PathBuf::from("/usr/local/lib/tla2tools.jar")),
        Some(PathBuf::from("/opt/tla/tla2tools.jar")),
        Some(PathBuf::from("/opt/homebrew/lib/tla2tools.jar")),
    ];

    for jar_path in jar_locations.into_iter().flatten() {
        if jar_path.exists() {
            // Verify Java is available
            if Command::new(&config.java_path)
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
                .is_ok()
            {
                return TlcDetection::Jar {
                    java_path: config.java_path.clone(),
                    jar_path,
                };
            }
        }
    }

    TlcDetection::NotFound(
        "TLC not found. Install TLA+ Toolbox or place tla2tools.jar in ~/.tla/".to_string(),
    )
}
