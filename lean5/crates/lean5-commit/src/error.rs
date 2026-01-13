//! Error types for polynomial commitments

use thiserror::Error;

/// Error during commitment operations
#[derive(Error, Debug)]
pub enum CommitError {
    /// Certificate is too large for the configured scheme
    #[error("Certificate degree {0} exceeds maximum {1}")]
    DegreeTooLarge(usize, usize),

    /// Encoding failed
    #[error("Failed to encode certificate: {0}")]
    EncodingFailed(String),

    /// Arkworks error during commitment
    #[error("Commitment computation failed: {0}")]
    CommitmentFailed(String),

    /// Invalid polynomial degree
    #[error("Invalid polynomial degree: {0}")]
    InvalidDegree(String),

    /// Encoding error for unsupported constructs
    #[error("Encoding error: {0}")]
    EncodingError(String),
}

/// Error during verification operations
#[derive(Error, Debug)]
pub enum VerifyError {
    /// Commitment format is invalid
    #[error("Invalid commitment format: {0}")]
    InvalidCommitment(String),

    /// Proof format is invalid
    #[error("Invalid proof format: {0}")]
    InvalidProof(String),

    /// Verification computation failed
    #[error("Verification computation failed: {0}")]
    VerificationFailed(String),

    /// Batch size mismatch
    #[error("Batch size mismatch: expected {0}, got {1}")]
    BatchSizeMismatch(usize, usize),
}
