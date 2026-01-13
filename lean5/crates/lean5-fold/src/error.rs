//! Error types for folding operations

use thiserror::Error;

/// Errors that can occur during folding operations
#[derive(Debug, Error)]
pub enum FoldError {
    /// Shape mismatch between instances being folded
    #[error("shape mismatch: expected {expected} constraints, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// Invalid witness for R1CS instance
    #[error("invalid witness: {0}")]
    InvalidWitness(String),

    /// Dimension mismatch in linear combination
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Commitment error
    #[error("commitment error: {0}")]
    CommitmentError(String),

    /// Unsupported certificate type
    #[error("unsupported certificate: {0}")]
    UnsupportedCert(String),
}

/// Errors that can occur during IVC operations
#[derive(Debug, Error)]
pub enum IvcError {
    /// Folding operation failed
    #[error("folding error: {0}")]
    FoldingError(#[from] FoldError),

    /// Verification failed
    #[error("verification failed: {0}")]
    VerificationFailed(String),

    /// Invalid step counter
    #[error("invalid step: expected {expected}, got {got}")]
    InvalidStep { expected: u64, got: u64 },

    /// Serialization error
    #[error("serialization error: {0}")]
    SerializationError(String),
}
