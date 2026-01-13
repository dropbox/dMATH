//! Error types for tla-core

use crate::span::Span;
use thiserror::Error;

/// Result type for tla-core operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during parsing or analysis
#[derive(Debug, Error)]
pub enum Error {
    #[error("Syntax error: {message}")]
    Syntax { message: String, span: Span },

    #[error("Undefined name: {name}")]
    UndefinedName { name: String, span: Span },

    #[error("Duplicate definition: {name}")]
    DuplicateDefinition {
        name: String,
        original: Span,
        duplicate: Span,
    },

    #[error("Type error: {message}")]
    Type { message: String, span: Span },

    #[error("Arity mismatch: expected {expected} arguments, got {got}")]
    ArityMismatch {
        expected: usize,
        got: usize,
        span: Span,
    },

    #[error("Module not found: {name}")]
    ModuleNotFound { name: String, span: Span },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Get the primary span for this error
    pub fn span(&self) -> Option<Span> {
        match self {
            Error::Syntax { span, .. } => Some(*span),
            Error::UndefinedName { span, .. } => Some(*span),
            Error::DuplicateDefinition { duplicate, .. } => Some(*duplicate),
            Error::Type { span, .. } => Some(*span),
            Error::ArityMismatch { span, .. } => Some(*span),
            Error::ModuleNotFound { span, .. } => Some(*span),
            Error::Io(_) => None,
        }
    }
}
