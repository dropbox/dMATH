//! Error types for the proof manager

use thiserror::Error;
use tla_core::span::Span;

/// Result type for proof operations
pub type ProofResult<T> = Result<T, ProofError>;

/// Errors that can occur during proof checking
#[derive(Debug, Error)]
pub enum ProofError {
    /// Failed to extract proof obligation
    #[error("failed to extract obligation: {message}")]
    ObligationError { message: String, span: Option<Span> },

    /// Backend failed to prove obligation
    #[error("proof failed: {message}")]
    ProofFailed { message: String, span: Option<Span> },

    /// Backend returned unknown result
    #[error("proof inconclusive: {reason}")]
    Inconclusive { reason: String, span: Option<Span> },

    /// SMT backend error
    #[error("SMT error: {0}")]
    SmtError(#[from] tla_smt::SmtError),

    /// Missing definition
    #[error("undefined: {name}")]
    Undefined { name: String, span: Option<Span> },

    /// Invalid proof structure
    #[error("invalid proof: {message}")]
    InvalidProof { message: String, span: Option<Span> },

    /// Unsupported construct
    #[error("unsupported: {feature}")]
    Unsupported { feature: String, span: Option<Span> },
}

impl ProofError {
    pub fn obligation(message: impl Into<String>, span: Option<Span>) -> Self {
        ProofError::ObligationError {
            message: message.into(),
            span,
        }
    }

    pub fn failed(message: impl Into<String>, span: Option<Span>) -> Self {
        ProofError::ProofFailed {
            message: message.into(),
            span,
        }
    }

    pub fn inconclusive(reason: impl Into<String>, span: Option<Span>) -> Self {
        ProofError::Inconclusive {
            reason: reason.into(),
            span,
        }
    }

    pub fn undefined(name: impl Into<String>, span: Option<Span>) -> Self {
        ProofError::Undefined {
            name: name.into(),
            span,
        }
    }

    pub fn invalid(message: impl Into<String>, span: Option<Span>) -> Self {
        ProofError::InvalidProof {
            message: message.into(),
            span,
        }
    }

    pub fn unsupported(feature: impl Into<String>, span: Option<Span>) -> Self {
        ProofError::Unsupported {
            feature: feature.into(),
            span,
        }
    }

    /// Get the span associated with this error, if any
    pub fn span(&self) -> Option<Span> {
        match self {
            ProofError::ObligationError { span, .. } => *span,
            ProofError::ProofFailed { span, .. } => *span,
            ProofError::Inconclusive { span, .. } => *span,
            ProofError::SmtError(_) => None,
            ProofError::Undefined { span, .. } => *span,
            ProofError::InvalidProof { span, .. } => *span,
            ProofError::Unsupported { span, .. } => *span,
        }
    }
}
