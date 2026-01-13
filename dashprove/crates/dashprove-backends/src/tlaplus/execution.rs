//! TLC execution

use crate::traits::BackendError;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::debug;

use super::{TlaPlusConfig, TlcDetection};

/// TLC execution output
pub struct TlcOutput {
    pub stdout: String,
    pub stderr: String,
    /// Exit code captured for potential future use (TLC output parsing doesn't require it)
    #[allow(dead_code)]
    pub exit_code: Option<i32>,
    pub duration: Duration,
}

/// Execute TLC and capture output
pub async fn run_tlc(
    config: &TlaPlusConfig,
    detection: &TlcDetection,
    tla_path: &Path,
    cfg_path: &Path,
) -> Result<TlcOutput, BackendError> {
    let start = Instant::now();

    let mut cmd = match detection {
        TlcDetection::Standalone(path) => {
            let mut cmd = Command::new(path);
            cmd.arg("-config").arg(cfg_path);
            cmd.arg("-workers").arg(config.workers.to_string());
            if let Some(depth) = config.depth_limit {
                cmd.arg("-depth").arg(depth.to_string());
            }
            cmd.arg(tla_path);
            cmd
        }
        TlcDetection::Jar {
            java_path,
            jar_path,
        } => {
            let mut cmd = Command::new(java_path);
            cmd.arg("-jar").arg(jar_path);
            cmd.arg("-config").arg(cfg_path);
            cmd.arg("-workers").arg(config.workers.to_string());
            if let Some(depth) = config.depth_limit {
                cmd.arg("-depth").arg(depth.to_string());
            }
            cmd.arg(tla_path);
            cmd
        }
        TlcDetection::NotFound(reason) => {
            return Err(BackendError::Unavailable(reason.clone()));
        }
    };

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(tla_path.parent().unwrap_or(Path::new(".")));

    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| BackendError::Timeout(config.timeout))?
        .map_err(|e| BackendError::VerificationFailed(format!("Failed to execute TLC: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    debug!("TLC stdout:\n{}", stdout);
    if !stderr.is_empty() {
        debug!("TLC stderr:\n{}", stderr);
    }

    Ok(TlcOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
    })
}
