//! Error types for platform API verification

use thiserror::Error;

/// Result type for platform API operations
pub type Result<T> = std::result::Result<T, PlatformApiError>;

/// Errors that can occur during platform API verification
#[derive(Debug, Error)]
pub enum PlatformApiError {
    /// Unknown state referenced
    #[error("Unknown state '{state}' in API '{api}'")]
    UnknownState { api: String, state: String },

    /// Unknown transition referenced
    #[error("Unknown transition '{transition}' in API '{api}'")]
    UnknownTransition { api: String, transition: String },

    /// Invalid state transition attempted
    #[error("Invalid transition '{transition}' from state '{from}' (allowed from: {allowed:?})")]
    InvalidTransition {
        transition: String,
        from: String,
        allowed: Vec<String>,
    },

    /// Constraint violation detected
    #[error("Constraint violation: {message}")]
    ConstraintViolation { message: String },

    /// Parse error in source code analysis
    #[error("Parse error: {message}")]
    ParseError { message: String },

    /// State machine is invalid (e.g., unreachable states)
    #[error("Invalid state machine: {message}")]
    InvalidStateMachine { message: String },
}
