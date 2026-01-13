//! Error types for the self-improvement infrastructure

use thiserror::Error;

/// Result type for self-improvement operations
pub type SelfImpResult<T> = Result<T, SelfImpError>;

/// Errors that can occur during self-improvement operations
#[derive(Debug, Error)]
pub enum SelfImpError {
    /// Verification gate rejected the improvement
    #[error("Verification gate rejected improvement: {reason}")]
    GateRejection {
        /// Why the improvement was rejected
        reason: String,
        /// Which check failed
        failed_check: Option<String>,
    },

    /// Soundness violation detected
    #[error("Soundness violation: {0}")]
    SoundnessViolation(String),

    /// Capability regression detected
    #[error("Capability regression: {capability} decreased from {old_value} to {new_value}")]
    CapabilityRegression {
        /// The capability that regressed
        capability: String,
        /// Previous value
        old_value: String,
        /// New (lower) value
        new_value: String,
    },

    /// Rollback failed
    #[error("Rollback failed: {0}")]
    RollbackFailed(String),

    /// No previous version to rollback to
    #[error("No previous version available for rollback")]
    NoPreviousVersion,

    /// Version not found in history
    #[error("Version not found: {0}")]
    VersionNotFound(String),

    /// Invalid improvement proposal
    #[error("Invalid improvement: {0}")]
    InvalidImprovement(String),

    /// Proof certificate invalid
    #[error("Invalid proof certificate: {0}")]
    InvalidCertificate(String),

    /// Certificate chain broken
    #[error("Certificate chain broken at version {version}: {reason}")]
    BrokenCertificateChain {
        /// Version where chain broke
        version: String,
        /// Why the chain is broken
        reason: String,
    },

    /// Verification timeout
    #[error("Verification timed out after {0} seconds")]
    VerificationTimeout(u64),

    /// Backend verification failed
    #[error("Backend verification failed: {backend}: {reason}")]
    BackendVerificationFailed {
        /// Which backend failed
        backend: String,
        /// Why it failed
        reason: String,
    },

    /// History corruption detected
    #[error("History corruption detected: {0}")]
    HistoryCorruption(String),

    /// Concurrent modification detected
    #[error("Concurrent modification detected: another improvement is in progress")]
    ConcurrentModification,

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Internal error (should not happen in normal operation)
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Dispatcher error during verification
    #[error("Dispatcher error: {0}")]
    DispatcherError(String),

    /// USL parsing error
    #[error("USL parsing error: {0}")]
    UslParseError(String),

    /// Type checking error
    #[error("Type checking error: {0}")]
    TypeCheckError(String),
}

impl SelfImpError {
    /// Create a gate rejection error
    pub fn gate_rejection(reason: impl Into<String>) -> Self {
        Self::GateRejection {
            reason: reason.into(),
            failed_check: None,
        }
    }

    /// Create a gate rejection error with a specific failed check
    pub fn gate_rejection_with_check(reason: impl Into<String>, check: impl Into<String>) -> Self {
        Self::GateRejection {
            reason: reason.into(),
            failed_check: Some(check.into()),
        }
    }

    /// Create a capability regression error
    pub fn capability_regression(
        capability: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) -> Self {
        Self::CapabilityRegression {
            capability: capability.into(),
            old_value: old_value.into(),
            new_value: new_value.into(),
        }
    }

    /// Create a backend verification failed error
    pub fn backend_failed(backend: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::BackendVerificationFailed {
            backend: backend.into(),
            reason: reason.into(),
        }
    }

    /// Returns true if this error indicates a verification failure (not a system error)
    pub fn is_verification_failure(&self) -> bool {
        matches!(
            self,
            Self::GateRejection { .. }
                | Self::SoundnessViolation(_)
                | Self::CapabilityRegression { .. }
                | Self::BackendVerificationFailed { .. }
        )
    }

    /// Returns true if this error is recoverable (rollback should work)
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            Self::HistoryCorruption(_)
                | Self::BrokenCertificateChain { .. }
                | Self::InternalError(_)
        )
    }
}

impl From<std::io::Error> for SelfImpError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for SelfImpError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}
