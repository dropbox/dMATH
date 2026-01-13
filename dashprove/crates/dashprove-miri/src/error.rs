//! Error types for MIRI operations

use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Result type for MIRI operations
pub type MiriResult<T> = Result<T, MiriError>;

/// Errors that can occur during MIRI operations
#[derive(Error, Debug, Clone)]
pub enum MiriError {
    /// MIRI is not available
    #[error("MIRI not available: {0}")]
    NotAvailable(String),

    /// MIRI execution failed
    #[error("MIRI execution failed: {0}")]
    ExecutionFailed(String),

    /// MIRI execution timed out
    #[error("MIRI execution timed out after {0:?}")]
    Timeout(Duration),

    /// Failed to parse MIRI output
    #[error("Failed to parse MIRI output: {0}")]
    ParseError(String),

    /// Project path does not exist
    #[error("Project path does not exist: {}", .0.display())]
    ProjectNotFound(PathBuf),

    /// Cargo.toml not found in project
    #[error("Cargo.toml not found in project: {}", .0.display())]
    NotACargoProject(PathBuf),

    /// Failed to generate harness
    #[error("Failed to generate harness: {0}")]
    HarnessGenerationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

impl From<std::io::Error> for MiriError {
    fn from(err: std::io::Error) -> Self {
        MiriError::IoError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MiriError::NotAvailable("rustup not found".to_string());
        assert!(err.to_string().contains("MIRI not available"));
        assert!(err.to_string().contains("rustup not found"));
    }

    #[test]
    fn test_timeout_error() {
        let err = MiriError::Timeout(Duration::from_secs(60));
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_project_not_found() {
        let err = MiriError::ProjectNotFound(PathBuf::from("/nonexistent/path"));
        assert!(err.to_string().contains("Project path does not exist"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let miri_err: MiriError = io_err.into();
        assert!(matches!(miri_err, MiriError::IoError(_)));
    }
}
