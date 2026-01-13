//! Error types for model-based testing

use thiserror::Error;

/// Errors that can occur during model-based testing
#[derive(Debug, Error)]
pub enum MbtError {
    /// Error parsing a specification
    #[error("Failed to parse specification: {0}")]
    ParseError(String),

    /// Error in model exploration
    #[error("Model exploration failed: {0}")]
    ExplorationError(String),

    /// Invalid model structure
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// No initial state defined
    #[error("Model has no initial state")]
    NoInitialState,

    /// State space exhausted without finding target coverage
    #[error("State space exhausted: explored {states} states, {transitions} transitions")]
    StateSpaceExhausted { states: usize, transitions: usize },

    /// Timeout during exploration
    #[error("Exploration timeout after {0}ms")]
    Timeout(u64),

    /// Maximum depth reached during exploration
    #[error("Maximum depth {0} reached during exploration")]
    MaxDepthReached(usize),

    /// Variable not found in state
    #[error("Variable '{0}' not found in state")]
    VariableNotFound(String),

    /// Invalid variable type
    #[error("Invalid type for variable '{name}': expected {expected}, got {actual}")]
    InvalidVariableType {
        name: String,
        expected: String,
        actual: String,
    },

    /// Test generation failed
    #[error("Test generation failed: {0}")]
    TestGenerationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result type for MBT operations
pub type MbtResult<T> = Result<T, MbtError>;
