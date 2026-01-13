//! Error types for bisimulation checking

use thiserror::Error;

/// Errors that can occur during bisimulation checking
#[derive(Error, Debug)]
pub enum BisimError {
    /// Error executing the oracle
    #[error("Oracle execution error: {0}")]
    OracleExecution(String),

    /// Error executing the subject
    #[error("Subject execution error: {0}")]
    SubjectExecution(String),

    /// Timeout during execution
    #[error("Execution timeout after {0}ms")]
    Timeout(u64),

    /// Error parsing execution trace
    #[error("Trace parsing error: {0}")]
    TraceParsing(String),

    /// Error comparing traces
    #[error("Trace comparison error: {0}")]
    TraceComparison(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Process error (binary execution)
    #[error("Process error: {code:?} - {stderr}")]
    Process {
        /// Exit code if available
        code: Option<i32>,
        /// Standard error output
        stderr: String,
    },
}

impl BisimError {
    /// Create an oracle execution error
    pub fn oracle(msg: impl Into<String>) -> Self {
        Self::OracleExecution(msg.into())
    }

    /// Create a subject execution error
    pub fn subject(msg: impl Into<String>) -> Self {
        Self::SubjectExecution(msg.into())
    }

    /// Create a timeout error
    pub fn timeout(millis: u64) -> Self {
        Self::Timeout(millis)
    }

    /// Create a trace parsing error
    pub fn parsing(msg: impl Into<String>) -> Self {
        Self::TraceParsing(msg.into())
    }

    /// Create a trace comparison error
    pub fn comparison(msg: impl Into<String>) -> Self {
        Self::TraceComparison(msg.into())
    }

    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a process error
    pub fn process(code: Option<i32>, stderr: impl Into<String>) -> Self {
        Self::Process {
            code,
            stderr: stderr.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BisimError::oracle("failed to start");
        assert!(err.to_string().contains("Oracle execution error"));

        let err = BisimError::timeout(5000);
        assert!(err.to_string().contains("5000ms"));

        let err = BisimError::process(Some(1), "command not found");
        assert!(err.to_string().contains("command not found"));
    }
}
