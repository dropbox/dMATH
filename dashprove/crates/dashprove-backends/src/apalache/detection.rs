//! Apalache installation detection

use crate::util::expand_home_dir;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;

use super::ApalacheConfig;

/// Detection result for Apalache
#[derive(Debug, Clone)]
pub enum ApalacheDetection {
    /// Apalache available as a standalone command
    Standalone(PathBuf),
    /// Apalache available via JAR
    Jar {
        java_path: PathBuf,
        jar_path: PathBuf,
    },
    /// Apalache not found
    NotFound(String),
}

/// Detect Apalache installation based on configuration
pub async fn detect_apalache(config: &ApalacheConfig) -> ApalacheDetection {
    // 1. Check if apalache_path is explicitly configured
    if let Some(ref path) = config.apalache_path {
        if path.exists() {
            if path.extension().map(|e| e == "jar").unwrap_or(false) {
                // It's a JAR file
                return ApalacheDetection::Jar {
                    java_path: config.java_path.clone(),
                    jar_path: path.clone(),
                };
            } else {
                return ApalacheDetection::Standalone(path.clone());
            }
        }
    }

    // 2. Check for apalache-mc in PATH
    if let Ok(path) = which::which("apalache-mc") {
        return ApalacheDetection::Standalone(path);
    }

    // 3. Check for apalache in PATH (alternative name)
    if let Ok(path) = which::which("apalache") {
        return ApalacheDetection::Standalone(path);
    }

    // 4. Check for Apalache JAR in common locations
    let jar_locations = [
        // User's home directory
        expand_home_dir("~/.apalache/lib/apalache.jar"),
        expand_home_dir("~/.apalache/apalache.jar"),
        expand_home_dir("~/apalache/lib/apalache.jar"),
        // System locations
        Some(PathBuf::from("/usr/local/lib/apalache/lib/apalache.jar")),
        Some(PathBuf::from("/opt/apalache/lib/apalache.jar")),
        Some(PathBuf::from("/opt/homebrew/lib/apalache/lib/apalache.jar")),
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
                return ApalacheDetection::Jar {
                    java_path: config.java_path.clone(),
                    jar_path,
                };
            }
        }
    }

    ApalacheDetection::NotFound(
        "Apalache not found. Install from https://github.com/informalsystems/apalache or place apalache.jar in ~/.apalache/".to_string(),
    )
}
