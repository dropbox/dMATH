//! Error types for CHC solving

use thiserror::Error;

/// CHC solver errors
#[derive(Debug, Error)]
pub enum ChcError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("undefined predicate: {0}")]
    UndefinedPredicate(String),

    #[error("arity mismatch for predicate {name}: expected {expected}, got {actual}")]
    ArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("sort mismatch: expected {expected:?}, got {actual:?}")]
    SortMismatch { expected: String, actual: String },

    #[error("no query clause found")]
    NoQuery,

    #[error("timeout after {0} iterations")]
    Timeout(usize),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("verification error: {0}")]
    Verification(String),
}

/// Result type for CHC operations
pub type ChcResult<T> = Result<T, ChcError>;
