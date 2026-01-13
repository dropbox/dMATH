//! Platform API Constraint Verification
//!
//! This crate provides verification of external platform API contracts that formal
//! verification tools (TLA+, CBMC, Coq) cannot verify because they assume the
//! environment behaves correctly.
//!
//! # Motivation
//!
//! During the MPS verification project, a critical SIGABRT crash was caused by
//! violating Metal's `addCompletedHandler:` constraint - this must be called BEFORE
//! `commit`. No formal tool caught this because it's an external API contract.
//!
//! This crate provides:
//! - State machine models for platform APIs (Metal, CUDA, Vulkan, POSIX)
//! - Static analysis to verify API usage patterns
//! - Integration with the DashProve dispatcher
//!
//! # Example
//!
//! ```
//! use dashprove_platform_api::{PlatformApi, ApiState, StateTransition, ApiConstraint};
//!
//! // Define Metal's MTLCommandBuffer state machine
//! let mut metal_cmd_buffer = PlatformApi::new("Metal", "MTLCommandBuffer");
//!
//! metal_cmd_buffer.add_state(ApiState::new("Created"));
//! metal_cmd_buffer.add_state(ApiState::new("Encoding"));
//! metal_cmd_buffer.add_state(ApiState::new("Committed"));
//! metal_cmd_buffer.add_state(ApiState::new("Completed"));
//!
//! // commit() can only be called from Created or Encoding
//! metal_cmd_buffer.add_transition(StateTransition::new(
//!     "commit",
//!     vec!["Created", "Encoding"],
//!     "Committed",
//! ));
//!
//! // addCompletedHandler() must be called before commit (from Created or Encoding)
//! metal_cmd_buffer.add_transition(StateTransition::new(
//!     "addCompletedHandler",
//!     vec!["Created", "Encoding"],
//!     "Created",  // Handler registration doesn't change state
//! ));
//! ```

mod api;
mod catalog;
mod checker;
mod constraint;
mod error;
mod state_machine;

pub use api::{ApiState, PlatformApi, StateTransition};
pub use catalog::{BuiltinCatalog, CudaCatalog, MetalCatalog, PosixCatalog};
pub use checker::{ApiChecker, CheckResult, Violation, ViolationSeverity};
pub use constraint::{ApiConstraint, ConstraintKind, TemporalRelation};
pub use error::{PlatformApiError, Result};
pub use state_machine::{StateId, StateMachine, TransitionId};

/// Check if a sequence of API calls satisfies platform constraints
pub fn check_api_sequence(api: &PlatformApi, calls: &[&str]) -> CheckResult {
    let checker = ApiChecker::new(api);
    checker.check_sequence(calls)
}

/// Check source code for platform API constraint violations
pub fn check_source_code(api: &PlatformApi, source: &str) -> CheckResult {
    let checker = ApiChecker::new(api);
    checker.check_source(source)
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::constraint::Severity;

    // Severity invariants
    #[kani::proof]
    fn verify_severity_as_str_non_empty() {
        let severity: Severity = kani::any();
        let s = severity.as_str();
        kani::assert(!s.is_empty(), "Severity as_str must not be empty");
    }

    #[kani::proof]
    fn verify_severity_equality_reflexive() {
        let severity: Severity = kani::any();
        kani::assert(severity == severity, "Severity equality must be reflexive");
    }

    // ViolationSeverity invariants
    #[kani::proof]
    fn verify_violation_severity_equality_reflexive() {
        let severity: ViolationSeverity = kani::any();
        kani::assert(
            severity == severity,
            "ViolationSeverity equality must be reflexive",
        );
    }

    #[kani::proof]
    fn verify_severity_to_violation_severity_mapping() {
        let severity: Severity = kani::any();
        let violation_severity: ViolationSeverity = severity.into();
        // Mapping is one-to-one
        let matches = match (severity, violation_severity) {
            (Severity::Critical, ViolationSeverity::Critical) => true,
            (Severity::Error, ViolationSeverity::Error) => true,
            (Severity::Warning, ViolationSeverity::Warning) => true,
            (Severity::Info, ViolationSeverity::Info) => true,
            _ => false,
        };
        kani::assert(
            matches,
            "Severity must map to corresponding ViolationSeverity",
        );
    }

    // TemporalRelation invariants
    #[kani::proof]
    fn verify_temporal_relation_equality_reflexive() {
        let relation: TemporalRelation = kani::any();
        kani::assert(
            relation == relation,
            "TemporalRelation equality must be reflexive",
        );
    }

    // ApiState invariants
    #[kani::proof]
    fn verify_api_state_new_defaults() {
        let state = ApiState::new("test");
        kani::assert(state.name == "test", "ApiState name must be set");
        kani::assert(
            state.description.is_none(),
            "ApiState default description is None",
        );
        kani::assert(!state.is_error, "ApiState default is_error is false");
        kani::assert(!state.is_terminal, "ApiState default is_terminal is false");
    }

    #[kani::proof]
    fn verify_api_state_as_error() {
        let state = ApiState::new("test").as_error();
        kani::assert(state.is_error, "as_error must set is_error to true");
    }

    #[kani::proof]
    fn verify_api_state_as_terminal() {
        let state = ApiState::new("test").as_terminal();
        kani::assert(
            state.is_terminal,
            "as_terminal must set is_terminal to true",
        );
    }

    // CheckResult invariants
    #[kani::proof]
    fn verify_check_result_pass_has_no_violations() {
        let result = CheckResult::pass("final", 10);
        kani::assert(result.passed, "pass() must set passed=true");
        kani::assert(
            result.violations.is_empty(),
            "pass() must have empty violations",
        );
        kani::assert(result.final_state.is_some(), "pass() must have final_state");
        kani::assert(
            result.calls_checked == 10,
            "pass() must preserve calls_checked",
        );
    }

    #[kani::proof]
    fn verify_check_result_fail_not_passed() {
        let result = CheckResult::fail(Vec::new(), 5);
        kani::assert(!result.passed, "fail() must set passed=false");
        kani::assert(
            result.final_state.is_none(),
            "fail() must have no final_state",
        );
        kani::assert(
            result.calls_checked == 5,
            "fail() must preserve calls_checked",
        );
    }

    // Violation invariants
    #[kani::proof]
    fn verify_violation_is_critical() {
        let v1 = Violation::new("test", "method", 0, "msg", ViolationSeverity::Critical);
        kani::assert(v1.is_critical(), "Critical severity must be critical");

        let v2 = Violation::new("test", "method", 0, "msg", ViolationSeverity::Error);
        kani::assert(!v2.is_critical(), "Error severity must not be critical");

        let v3 = Violation::new("test", "method", 0, "msg", ViolationSeverity::Warning);
        kani::assert(!v3.is_critical(), "Warning severity must not be critical");

        let v4 = Violation::new("test", "method", 0, "msg", ViolationSeverity::Info);
        kani::assert(!v4.is_critical(), "Info severity must not be critical");
    }

    // ApiConstraint invariants
    #[kani::proof]
    fn verify_api_constraint_must_call_before_is_temporal() {
        let constraint = ApiConstraint::must_call_before("a", "b", "msg");
        kani::assert(
            constraint.is_temporal(),
            "must_call_before creates temporal constraint",
        );
        kani::assert(
            constraint.method_b.is_some(),
            "must_call_before has method_b",
        );
        kani::assert(
            constraint.severity == Severity::Critical,
            "must_call_before is Critical",
        );
    }

    #[kani::proof]
    fn verify_api_constraint_at_most_once_not_temporal() {
        let constraint = ApiConstraint::at_most_once("method", "msg");
        kani::assert(!constraint.is_temporal(), "at_most_once is not temporal");
        kani::assert(
            constraint.method_b.is_none(),
            "at_most_once has no method_b",
        );
    }

    #[kani::proof]
    fn verify_api_constraint_exactly_once_not_temporal() {
        let constraint = ApiConstraint::exactly_once("method", "msg");
        kani::assert(!constraint.is_temporal(), "exactly_once is not temporal");
        kani::assert(
            constraint.method_b.is_none(),
            "exactly_once has no method_b",
        );
        kani::assert(
            constraint.severity == Severity::Critical,
            "exactly_once is Critical",
        );
    }

    #[kani::proof]
    fn verify_api_constraint_paired_not_temporal() {
        let constraint = ApiConstraint::paired("a", "b", "msg");
        kani::assert(!constraint.is_temporal(), "paired is not temporal");
        kani::assert(constraint.method_b.is_some(), "paired has method_b");
        kani::assert(
            constraint.severity == Severity::Error,
            "paired is Error severity",
        );
    }

    #[kani::proof]
    fn verify_api_constraint_with_severity() {
        let severity: Severity = kani::any();
        let constraint = ApiConstraint::at_most_once("method", "msg").with_severity(severity);
        kani::assert(
            constraint.severity == severity,
            "with_severity must preserve value",
        );
    }

    // PlatformApi invariants
    #[kani::proof]
    fn verify_platform_api_new_empty() {
        let api = PlatformApi::new("Metal", "MTLCommandBuffer");
        kani::assert(api.platform == "Metal", "Platform must be preserved");
        kani::assert(
            api.api_object == "MTLCommandBuffer",
            "API object must be preserved",
        );
        kani::assert(api.states.is_empty(), "New API has no states");
        kani::assert(api.transitions.is_empty(), "New API has no transitions");
        kani::assert(api.constraints.is_empty(), "New API has no constraints");
        kani::assert(api.initial_state.is_none(), "New API has no initial state");
    }
}
