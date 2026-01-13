//! # dashprove-async
//!
//! Async/concurrent state machine verification with Loom for DashProve.
//!
//! This crate provides infrastructure for verifying async state machines,
//! detecting race conditions, and exploring all possible interleavings
//! of concurrent operations.
//!
//! ## Key Features
//!
//! - **AsyncStateMachine trait**: Define async state machines for verification
//! - **Loom integration**: Explore all thread interleavings via the `loom` feature
//! - **Interleaving exploration**: Find race conditions and verify invariants
//! - **TLA+ trace verification**: Verify async traces against TLA+ specs

mod error;
mod interleaving;
mod state_machine;
pub mod tlaplus;
mod verifier;

pub use error::*;
pub use interleaving::*;
pub use state_machine::*;
pub use tlaplus::*;
pub use verifier::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_passed() {
        let result = VerificationResult::passed();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_verification_result_failed() {
        let violations = vec![Violation::InvariantViolation {
            invariant: "no_deadlock".to_string(),
            state: serde_json::json!({"locked": true}),
            message: "Deadlock detected".to_string(),
        }];
        let result = VerificationResult::failed(violations.clone());

        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_state_transition() {
        let trans = StateTransition::new(
            serde_json::json!({"state": "idle"}),
            "start".to_string(),
            serde_json::json!({"state": "running"}),
        );

        assert_eq!(trans.event, "start");
    }
}
