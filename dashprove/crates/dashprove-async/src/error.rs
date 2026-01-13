//! Error types for async verification

use thiserror::Error;

/// Errors that can occur during async verification
#[derive(Error, Debug)]
pub enum AsyncVerifyError {
    /// Timeout during verification
    #[error("Verification timeout after {0}ms")]
    Timeout(u64),

    /// State machine error
    #[error("State machine error: {0}")]
    StateMachine(String),

    /// Invalid state transition
    #[error("Invalid state transition: {from} -> {to} via {event}")]
    InvalidTransition {
        from: String,
        to: String,
        event: String,
    },

    /// Invariant check error
    #[error("Invariant check error: {0}")]
    InvariantCheck(String),

    /// Loom model error
    #[error("Loom model error: {0}")]
    LoomModel(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// TLA+ trace verification error
    #[error("TLA+ verification error: {0}")]
    TlaVerification(String),
}

impl AsyncVerifyError {
    /// Create a state machine error
    pub fn state_machine(msg: impl Into<String>) -> Self {
        Self::StateMachine(msg.into())
    }

    /// Create an invalid transition error
    pub fn invalid_transition(
        from: impl Into<String>,
        to: impl Into<String>,
        event: impl Into<String>,
    ) -> Self {
        Self::InvalidTransition {
            from: from.into(),
            to: to.into(),
            event: event.into(),
        }
    }

    /// Create an invariant check error
    pub fn invariant(msg: impl Into<String>) -> Self {
        Self::InvariantCheck(msg.into())
    }

    /// Create a timeout error
    pub fn timeout(millis: u64) -> Self {
        Self::Timeout(millis)
    }

    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a loom model error
    pub fn loom(msg: impl Into<String>) -> Self {
        Self::LoomModel(msg.into())
    }

    /// Create a TLA+ verification error
    pub fn tla_verification(msg: impl Into<String>) -> Self {
        Self::TlaVerification(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AsyncVerifyError::timeout(5000);
        assert!(err.to_string().contains("5000ms"));

        let err = AsyncVerifyError::invalid_transition("idle", "running", "start");
        assert!(err.to_string().contains("idle"));
        assert!(err.to_string().contains("running"));
        assert!(err.to_string().contains("start"));

        let err = AsyncVerifyError::state_machine("panic in state handler");
        assert!(err.to_string().contains("panic in state handler"));
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn timeout_error_contains_millis(ms in 0u64..1_000_000u64) {
                let err = AsyncVerifyError::timeout(ms);
                let expected = format!("{}ms", ms);
                prop_assert!(err.to_string().contains(&expected));
            }

            #[test]
            fn state_machine_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
                let err = AsyncVerifyError::state_machine(msg.clone());
                prop_assert!(err.to_string().contains(&msg));
            }

            #[test]
            fn invalid_transition_contains_all_fields(
                from in "[a-z]{1,10}",
                to in "[a-z]{1,10}",
                event in "[a-z]{1,10}"
            ) {
                let err = AsyncVerifyError::invalid_transition(from.clone(), to.clone(), event.clone());
                let display = err.to_string();
                prop_assert!(display.contains(&from));
                prop_assert!(display.contains(&to));
                prop_assert!(display.contains(&event));
            }

            #[test]
            fn invariant_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
                let err = AsyncVerifyError::invariant(msg.clone());
                prop_assert!(err.to_string().contains(&msg));
            }

            #[test]
            fn config_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
                let err = AsyncVerifyError::config(msg.clone());
                prop_assert!(err.to_string().contains(&msg));
            }

            #[test]
            fn loom_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
                let err = AsyncVerifyError::loom(msg.clone());
                prop_assert!(err.to_string().contains(&msg));
            }

            #[test]
            fn tla_verification_error_contains_message(msg in "[a-zA-Z0-9 ]{1,50}") {
                let err = AsyncVerifyError::tla_verification(msg.clone());
                prop_assert!(err.to_string().contains(&msg));
            }
        }
    }
}
