//! Apalache execution

use crate::traits::BackendError;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::debug;

use super::{ApalacheConfig, ApalacheDetection};

/// Apalache execution output
pub struct ApalacheOutput {
    pub stdout: String,
    pub stderr: String,
    /// Exit code captured for determining success/failure
    pub exit_code: Option<i32>,
    pub duration: Duration,
}

/// Execute Apalache and capture output
pub async fn run_apalache(
    config: &ApalacheConfig,
    detection: &ApalacheDetection,
    tla_path: &Path,
    cfg_path: Option<&Path>,
) -> Result<ApalacheOutput, BackendError> {
    let start = Instant::now();

    let mut cmd = match detection {
        ApalacheDetection::Standalone(path) => {
            let mut cmd = Command::new(path);
            cmd.arg(config.mode.command());
            cmd
        }
        ApalacheDetection::Jar {
            java_path,
            jar_path,
        } => {
            let mut cmd = Command::new(java_path);
            // Add JVM memory settings
            if let Some(ref mem) = config.jvm_memory {
                cmd.arg(mem);
            }
            cmd.arg("-jar").arg(jar_path);
            cmd.arg(config.mode.command());
            cmd
        }
        ApalacheDetection::NotFound(reason) => {
            return Err(BackendError::Unavailable(reason.clone()));
        }
    };

    // Add config file if provided
    if let Some(cfg) = cfg_path {
        cmd.arg("--config").arg(cfg);
    }

    // Add length bound
    if let Some(length) = config.length {
        cmd.arg("--length").arg(length.to_string());
    }

    // Add init predicate
    if let Some(ref init) = config.init {
        cmd.arg("--init").arg(init);
    }

    // Add next relation
    if let Some(ref next) = config.next {
        cmd.arg("--next").arg(next);
    }

    // Add invariants
    for inv in &config.inv {
        cmd.arg("--inv").arg(inv);
    }

    // Add temporal properties
    for temporal in &config.temporal {
        cmd.arg("--temporal").arg(temporal);
    }

    // Set SMT solver
    cmd.arg("--smt-encoding").arg("arrays");

    // Add debug flag
    if config.debug {
        cmd.arg("--debug");
    }

    // Add output directory
    if let Some(ref out_dir) = config.out_dir {
        cmd.arg("--out-dir").arg(out_dir);
    }

    // Add the TLA+ spec file (must be last)
    cmd.arg(tla_path);

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(tla_path.parent().unwrap_or(Path::new(".")));

    debug!("Running Apalache: {:?}", cmd);

    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| BackendError::Timeout(config.timeout))?
        .map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to execute Apalache: {}", e))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    debug!("Apalache stdout:\n{}", stdout);
    if !stderr.is_empty() {
        debug!("Apalache stderr:\n{}", stderr);
    }

    Ok(ApalacheOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
    })
}
