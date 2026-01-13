//! MIRI execution logic

use crate::config::MiriConfig;
use crate::detection::MiriDetection;
use crate::error::MiriError;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::debug;

/// Output from a MIRI execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriOutput {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code (None if terminated by signal)
    pub exit_code: Option<i32>,
    /// Execution duration
    pub duration: Duration,
    /// Whether MIRI found any issues
    pub has_errors: bool,
}

impl MiriOutput {
    /// Check if the execution succeeded without UB
    pub fn success(&self) -> bool {
        self.exit_code == Some(0) && !self.has_errors
    }
}

/// Run cargo miri test on a project
pub async fn run_miri(
    config: &MiriConfig,
    detection: &MiriDetection,
    manifest_path: &Path,
    test_filter: Option<&str>,
) -> Result<MiriOutput, MiriError> {
    let cargo_path = match detection {
        MiriDetection::Available { cargo_path, .. } => cargo_path,
        MiriDetection::NotFound(reason) => return Err(MiriError::NotAvailable(reason.clone())),
    };

    // Validate project path
    if !manifest_path.exists() {
        return Err(MiriError::ProjectNotFound(manifest_path.to_path_buf()));
    }

    // Ensure it's a Cargo project
    let cargo_toml = if manifest_path.is_file() && manifest_path.ends_with("Cargo.toml") {
        manifest_path.to_path_buf()
    } else if manifest_path.is_dir() {
        let toml = manifest_path.join("Cargo.toml");
        if !toml.exists() {
            return Err(MiriError::NotACargoProject(manifest_path.to_path_buf()));
        }
        toml
    } else {
        return Err(MiriError::NotACargoProject(manifest_path.to_path_buf()));
    };

    let mut cmd = Command::new(cargo_path);
    cmd.arg("+nightly")
        .arg("miri")
        .arg("test")
        .arg("--manifest-path")
        .arg(&cargo_toml);

    // Add test filter if provided
    if let Some(filter) = test_filter {
        cmd.arg("--").arg(filter);
    }

    // Set MIRIFLAGS environment variable
    let miriflags = config.flags.to_miriflags();
    if !miriflags.is_empty() {
        cmd.env("MIRIFLAGS", &miriflags);
        debug!("MIRIFLAGS: {}", miriflags);
    }

    // Add additional environment variables
    for (key, value) in &config.env_vars {
        cmd.env(key, value);
    }

    // Set job count if specified
    if let Some(jobs) = config.jobs {
        cmd.arg("-j").arg(jobs.to_string());
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // Set working directory to project root
    if let Some(parent) = cargo_toml.parent() {
        cmd.current_dir(parent);
    }

    let start = Instant::now();
    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| MiriError::Timeout(config.timeout))?
        .map_err(|e| MiriError::ExecutionFailed(format!("Failed to execute cargo miri: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    debug!("cargo miri test stdout:\n{}", stdout);
    if !stderr.is_empty() {
        debug!("cargo miri test stderr:\n{}", stderr);
    }

    // Check for MIRI errors in output
    let has_errors = stderr.contains("Undefined Behavior")
        || stderr.contains("error[")
        || stderr.contains("Miri evaluation error")
        || stderr.contains("memory access")
        || output.status.code() != Some(0);

    Ok(MiriOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
        has_errors,
    })
}

/// Run cargo miri on a single file (creates a temporary project)
pub async fn run_miri_on_file(
    config: &MiriConfig,
    detection: &MiriDetection,
    file_path: &Path,
) -> Result<MiriOutput, MiriError> {
    use std::fs;

    let cargo_path = match detection {
        MiriDetection::Available { cargo_path, .. } => cargo_path,
        MiriDetection::NotFound(reason) => return Err(MiriError::NotAvailable(reason.clone())),
    };

    if !file_path.exists() {
        return Err(MiriError::IoError(format!(
            "Source file not found: {}",
            file_path.display()
        )));
    }

    // Create temporary directory
    let temp_dir = tempfile::tempdir()
        .map_err(|e| MiriError::IoError(format!("Failed to create temp directory: {}", e)))?;

    // Create minimal Cargo.toml
    let cargo_toml_content = r#"
[package]
name = "miri_test"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[[test]]
name = "miri_test"
path = "tests/test.rs"
"#;

    fs::create_dir_all(temp_dir.path().join("src"))
        .map_err(|e| MiriError::IoError(format!("Failed to create src directory: {}", e)))?;
    fs::create_dir_all(temp_dir.path().join("tests"))
        .map_err(|e| MiriError::IoError(format!("Failed to create tests directory: {}", e)))?;

    fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml_content)
        .map_err(|e| MiriError::IoError(format!("Failed to write Cargo.toml: {}", e)))?;

    // Copy source file to lib.rs
    fs::copy(file_path, temp_dir.path().join("src/lib.rs"))
        .map_err(|e| MiriError::IoError(format!("Failed to copy source file: {}", e)))?;

    // Create a minimal test file
    let test_content = r#"
use miri_test::*;

#[test]
fn miri_main_test() {
    // This test exists to make MIRI run on the library
}
"#;
    fs::write(temp_dir.path().join("tests/test.rs"), test_content)
        .map_err(|e| MiriError::IoError(format!("Failed to write test file: {}", e)))?;

    // Run miri on the temporary project
    let mut cmd = Command::new(cargo_path);
    cmd.arg("+nightly")
        .arg("miri")
        .arg("test")
        .arg("--manifest-path")
        .arg(temp_dir.path().join("Cargo.toml"));

    let miriflags = config.flags.to_miriflags();
    if !miriflags.is_empty() {
        cmd.env("MIRIFLAGS", &miriflags);
    }

    for (key, value) in &config.env_vars {
        cmd.env(key, value);
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(temp_dir.path());

    let start = Instant::now();
    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| MiriError::Timeout(config.timeout))?
        .map_err(|e| MiriError::ExecutionFailed(format!("Failed to execute cargo miri: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    let has_errors = stderr.contains("Undefined Behavior")
        || stderr.contains("error[")
        || stderr.contains("Miri evaluation error")
        || output.status.code() != Some(0);

    Ok(MiriOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
        has_errors,
    })
}

/// Run cargo miri setup to prepare the sysroot
pub async fn setup_miri(
    config: &MiriConfig,
    detection: &MiriDetection,
) -> Result<MiriOutput, MiriError> {
    let cargo_path = match detection {
        MiriDetection::Available { cargo_path, .. } => cargo_path,
        MiriDetection::NotFound(reason) => return Err(MiriError::NotAvailable(reason.clone())),
    };

    let mut cmd = Command::new(cargo_path);
    cmd.arg("+nightly").arg("miri").arg("setup");

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let start = Instant::now();
    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| MiriError::Timeout(config.timeout))?
        .map_err(|e| {
            MiriError::ExecutionFailed(format!("Failed to execute cargo miri setup: {}", e))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    Ok(MiriOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
        has_errors: output.status.code() != Some(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miri_output_success() {
        let output = MiriOutput {
            stdout: "test result: ok".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(1),
            has_errors: false,
        };
        assert!(output.success());
    }

    #[test]
    fn test_miri_output_failure() {
        let output = MiriOutput {
            stdout: String::new(),
            stderr: "Undefined Behavior: memory access error".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
            has_errors: true,
        };
        assert!(!output.success());
    }
}
