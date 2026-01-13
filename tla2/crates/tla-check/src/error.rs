//! Evaluation errors for the TLA+ model checker

use crate::value::Value;
use thiserror::Error;
use tla_core::Span;

/// Evaluation error
#[derive(Debug, Clone, Error)]
pub enum EvalError {
    /// Type mismatch in operation
    #[error("Type error: expected {expected}, got {got}")]
    TypeError {
        expected: &'static str,
        got: &'static str,
        span: Option<Span>,
    },

    /// Division by zero
    #[error("Division by zero")]
    DivisionByZero { span: Option<Span> },

    /// Undefined variable reference
    #[error("Undefined variable: {name}")]
    UndefinedVar { name: String, span: Option<Span> },

    /// Undefined operator reference
    #[error("Undefined operator: {name}")]
    UndefinedOp { name: String, span: Option<Span> },

    /// Function applied to value not in domain
    #[error("Function application error: {arg} not in domain")]
    NotInDomain { arg: String, span: Option<Span> },

    /// Record field not found
    #[error("Record has no field: {field}")]
    NoSuchField { field: String, span: Option<Span> },

    /// Sequence index out of bounds
    #[error("Sequence index out of bounds: {index} not in 1..{len}")]
    IndexOutOfBounds {
        index: i64,
        len: usize,
        span: Option<Span>,
    },

    /// CHOOSE found no witness
    #[error("CHOOSE failed: no value satisfies predicate")]
    ChooseFailed { span: Option<Span> },

    /// Arity mismatch in operator application
    #[error("Arity mismatch: {op} expects {expected} arguments, got {got}")]
    ArityMismatch {
        op: String,
        expected: usize,
        got: usize,
        span: Option<Span>,
    },

    /// Set too large to enumerate
    #[error("Set too large to enumerate (infinite or > limit)")]
    SetTooLarge { span: Option<Span> },

    /// Internal evaluation error (bug in evaluator)
    #[error("Internal error: {message}")]
    Internal { message: String, span: Option<Span> },
}

impl EvalError {
    pub fn type_error(expected: &'static str, got: &Value, span: Option<Span>) -> Self {
        EvalError::TypeError {
            expected,
            got: got.type_name(),
            span,
        }
    }

    pub fn span(&self) -> Option<Span> {
        match self {
            EvalError::TypeError { span, .. } => *span,
            EvalError::DivisionByZero { span } => *span,
            EvalError::UndefinedVar { span, .. } => *span,
            EvalError::UndefinedOp { span, .. } => *span,
            EvalError::NotInDomain { span, .. } => *span,
            EvalError::NoSuchField { span, .. } => *span,
            EvalError::IndexOutOfBounds { span, .. } => *span,
            EvalError::ChooseFailed { span } => *span,
            EvalError::ArityMismatch { span, .. } => *span,
            EvalError::SetTooLarge { span } => *span,
            EvalError::Internal { span, .. } => *span,
        }
    }
}

pub type EvalResult<T> = Result<T, EvalError>;

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use tla_core::{FileId, Span};

    // ============================================================================
    // SNAPSHOT TESTS - Error message format stability
    // These tests ensure error messages don't change unexpectedly.
    // ============================================================================

    #[test]
    fn snapshot_type_error() {
        let err = EvalError::TypeError {
            expected: "integer",
            got: "set",
            span: Some(Span::new(FileId(0), 10, 20)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_division_by_zero() {
        let err = EvalError::DivisionByZero {
            span: Some(Span::new(FileId(0), 15, 20)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_undefined_var() {
        let err = EvalError::UndefinedVar {
            name: "myVar".to_string(),
            span: Some(Span::new(FileId(0), 5, 10)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_undefined_op() {
        let err = EvalError::UndefinedOp {
            name: "MyOperator".to_string(),
            span: Some(Span::new(FileId(0), 25, 35)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_not_in_domain() {
        let err = EvalError::NotInDomain {
            arg: "42".to_string(),
            span: Some(Span::new(FileId(0), 30, 35)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_no_such_field() {
        let err = EvalError::NoSuchField {
            field: "missingField".to_string(),
            span: Some(Span::new(FileId(0), 40, 52)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_index_out_of_bounds() {
        let err = EvalError::IndexOutOfBounds {
            index: 10,
            len: 5,
            span: Some(Span::new(FileId(0), 50, 55)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_choose_failed() {
        let err = EvalError::ChooseFailed {
            span: Some(Span::new(FileId(0), 60, 75)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_arity_mismatch() {
        let err = EvalError::ArityMismatch {
            op: "Add".to_string(),
            expected: 2,
            got: 1,
            span: Some(Span::new(FileId(0), 80, 85)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_set_too_large() {
        let err = EvalError::SetTooLarge {
            span: Some(Span::new(FileId(0), 90, 100)),
        };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_internal_error() {
        let err = EvalError::Internal {
            message: "unexpected state in evaluator".to_string(),
            span: Some(Span::new(FileId(0), 100, 110)),
        };
        assert_snapshot!(err.to_string());
    }
}
