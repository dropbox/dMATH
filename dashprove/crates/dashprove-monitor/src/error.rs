//! Error types for the dashprove-monitor crate

use thiserror::Error;

/// Errors that can occur during monitoring operations
#[derive(Debug, Error)]
pub enum MonitorError {
    /// Monitor was not properly initialized
    #[error("Monitor not initialized: {0}")]
    NotInitialized(String),

    /// Recording error
    #[error("Recording error: {0}")]
    RecordingError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invariant compilation error
    #[error("Invariant compilation error: {0}")]
    InvariantCompilationError(String),

    /// Invariant evaluation error
    #[error("Invariant evaluation error: {0}")]
    InvariantEvaluationError(String),

    /// Liveness property error
    #[error("Liveness property error: {0}")]
    LivenessError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl MonitorError {
    /// Create a not initialized error
    pub fn not_initialized(msg: impl Into<String>) -> Self {
        Self::NotInitialized(msg.into())
    }

    /// Create a recording error
    pub fn recording(msg: impl Into<String>) -> Self {
        Self::RecordingError(msg.into())
    }

    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::SerializationError(msg.into())
    }

    /// Create an invariant compilation error
    pub fn invariant_compilation(msg: impl Into<String>) -> Self {
        Self::InvariantCompilationError(msg.into())
    }

    /// Create an invariant evaluation error
    pub fn invariant_evaluation(msg: impl Into<String>) -> Self {
        Self::InvariantEvaluationError(msg.into())
    }

    /// Create a liveness error
    pub fn liveness(msg: impl Into<String>) -> Self {
        Self::LivenessError(msg.into())
    }
}

/// Result type for monitor operations
pub type MonitorResult<T> = Result<T, MonitorError>;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn not_initialized_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
            let err = MonitorError::not_initialized(msg.clone());
            prop_assert!(err.to_string().contains(&msg));
        }

        #[test]
        fn recording_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
            let err = MonitorError::recording(msg.clone());
            prop_assert!(err.to_string().contains(&msg));
        }

        #[test]
        fn serialization_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
            let err = MonitorError::serialization(msg.clone());
            prop_assert!(err.to_string().contains(&msg));
        }

        #[test]
        fn invariant_compilation_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
            let err = MonitorError::invariant_compilation(msg.clone());
            prop_assert!(err.to_string().contains(&msg));
        }

        #[test]
        fn invariant_evaluation_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
            let err = MonitorError::invariant_evaluation(msg.clone());
            prop_assert!(err.to_string().contains(&msg));
        }

        #[test]
        fn liveness_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
            let err = MonitorError::liveness(msg.clone());
            prop_assert!(err.to_string().contains(&msg));
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify MonitorError::not_initialized produces NotInitialized variant
    #[kani::proof]
    fn kani_not_initialized_variant() {
        let err = MonitorError::not_initialized("test");
        assert!(matches!(err, MonitorError::NotInitialized(_)));
    }

    /// Verify MonitorError::recording produces RecordingError variant
    #[kani::proof]
    fn kani_recording_variant() {
        let err = MonitorError::recording("test");
        assert!(matches!(err, MonitorError::RecordingError(_)));
    }

    /// Verify MonitorError::serialization produces SerializationError variant
    #[kani::proof]
    fn kani_serialization_variant() {
        let err = MonitorError::serialization("test");
        assert!(matches!(err, MonitorError::SerializationError(_)));
    }

    /// Verify MonitorError::invariant_compilation produces InvariantCompilationError variant
    #[kani::proof]
    fn kani_invariant_compilation_variant() {
        let err = MonitorError::invariant_compilation("test");
        assert!(matches!(err, MonitorError::InvariantCompilationError(_)));
    }

    /// Verify MonitorError::invariant_evaluation produces InvariantEvaluationError variant
    #[kani::proof]
    fn kani_invariant_evaluation_variant() {
        let err = MonitorError::invariant_evaluation("test");
        assert!(matches!(err, MonitorError::InvariantEvaluationError(_)));
    }

    /// Verify MonitorError::liveness produces LivenessError variant
    #[kani::proof]
    fn kani_liveness_variant() {
        let err = MonitorError::liveness("test");
        assert!(matches!(err, MonitorError::LivenessError(_)));
    }

    /// Verify all error factory methods produce non-panic errors
    #[kani::proof]
    fn kani_all_error_factories_non_panic() {
        let _e1 = MonitorError::not_initialized("");
        let _e2 = MonitorError::recording("");
        let _e3 = MonitorError::serialization("");
        let _e4 = MonitorError::invariant_compilation("");
        let _e5 = MonitorError::invariant_evaluation("");
        let _e6 = MonitorError::liveness("");
    }

    /// Verify error display formatting doesn't panic
    #[kani::proof]
    fn kani_error_display_non_panic() {
        let err = MonitorError::not_initialized("x");
        let _ = format!("{}", err);
    }
}
