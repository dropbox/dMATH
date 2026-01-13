//! Error types for SMT operations

use thiserror::Error;

/// Result type alias for SMT operations
pub type SmtResult<T> = std::result::Result<T, SmtError>;

/// Errors that can occur during SMT operations
#[derive(Debug, Error)]
pub enum SmtError {
    /// Expression cannot be translated to SMT
    #[error("cannot translate expression to SMT: {0}")]
    UntranslatableExpr(String),

    /// Variable type mismatch
    #[error("type mismatch for variable '{name}': expected {expected}, got {actual}")]
    TypeMismatch {
        name: String,
        expected: String,
        actual: String,
    },

    /// Unknown variable reference
    #[error("unknown variable: {0}")]
    UnknownVariable(String),

    /// Unsupported operation
    #[error("unsupported SMT operation: {0}")]
    UnsupportedOp(String),

    /// Z3 solver error
    #[error("Z3 error: {0}")]
    Z3Error(String),

    /// Timeout during solving
    #[error("SMT solver timed out after {0}ms")]
    Timeout(u64),

    /// Model extraction error
    #[error("failed to extract model: {0}")]
    ModelError(String),
}
