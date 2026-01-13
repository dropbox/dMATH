//! # dashprove-monitor
//!
//! Runtime monitoring and trace recording for DashProve.
//!
//! This crate provides infrastructure for instrumenting Rust code to record
//! execution traces and verify them against specifications at runtime.
//!
//! ## Features
//!
//! - **Trace Recording**: Record execution traces with states and transitions
//! - **Traceable Trait**: Instrument Rust types for automatic state capture
//! - **Runtime Monitor**: Check invariants at runtime during execution
//! - **Invariant Checking**: Verify traces against state and transition invariants
//! - **Liveness Checking**: Verify progress and liveness properties
//! - **Macros**: Convenient macros for trace recording
//! - **Derive Macro**: `#[derive(Monitored)]` for automatic `Traceable` impl (with `macros` feature)
//!
//! ## Quick Start
//!
//! ```rust
//! use dashprove_monitor::{
//!     RuntimeMonitor, Traceable, TraceRecorder, trace_state,
//!     invariant::patterns,
//! };
//! use serde::{Serialize, Deserialize};
//!
//! // Define a traceable type
//! #[derive(Debug, Clone, Serialize, Deserialize)]
//! struct Counter {
//!     value: i32,
//! }
//!
//! impl Traceable for Counter {
//!     fn trace_name(&self) -> &str {
//!         "Counter"
//!     }
//!
//!     fn capture_state(&self) -> serde_json::Value {
//!         serde_json::json!({ "value": self.value })
//!     }
//! }
//!
//! // Create a monitor with invariants
//! let mut monitor = RuntimeMonitor::new();
//! monitor.add_simple_invariant("positive", |state| {
//!     state.get("value").and_then(|v| v.as_i64()).is_some_and(|n| n >= 0)
//! });
//!
//! // Check invariants
//! let counter = Counter { value: 10 };
//! let violations = monitor.check_traceable(&counter).unwrap();
//! assert!(violations.is_empty());
//! ```
//!
//! ## Trace Recording
//!
//! ```rust
//! use dashprove_monitor::{TraceRecorder, trace_state};
//! use serde_json::json;
//!
//! // Record a trace
//! let mut recorder = TraceRecorder::start("counter_trace", json!({"value": 0}));
//!
//! // Record state changes
//! recorder.record("increment", json!({"value": 1}));
//! recorder.record("increment", json!({"value": 2}));
//!
//! // Get the completed trace
//! let trace = recorder.finish();
//! assert_eq!(trace.len(), 2);
//! ```
//!
//! ## Invariant Checking
//!
//! ```rust
//! use dashprove_monitor::invariant::{Invariant, check_invariant, patterns};
//! use dashprove_async::{ExecutionTrace, StateTransition};
//! use serde_json::json;
//!
//! // Create a trace
//! let mut trace = ExecutionTrace::new(json!({"balance": 100}));
//! trace.add_transition(StateTransition::new(
//!     json!({"balance": 100}),
//!     "withdraw".to_string(),
//!     json!({"balance": 50}),
//! ));
//!
//! // Check invariant
//! let inv = patterns::field_positive("balance");
//! let result = check_invariant(&trace, &inv);
//! assert!(result.satisfied);
//! ```
//!
//! ## Liveness Properties
//!
//! ```rust
//! use dashprove_monitor::liveness::{LivenessProperty, check_liveness};
//! use dashprove_async::{ExecutionTrace, StateTransition};
//! use serde_json::json;
//!
//! // Create a trace
//! let mut trace = ExecutionTrace::new(json!({"done": false}));
//! trace.add_transition(StateTransition::new(
//!     json!({"done": false}),
//!     "complete".to_string(),
//!     json!({"done": true}),
//! ));
//!
//! // Check liveness property
//! let prop = LivenessProperty::eventually("completion", |s| {
//!     s.get("done").and_then(|v| v.as_bool()).unwrap_or(false)
//! });
//! let result = check_liveness(&trace, &prop);
//! assert!(result.satisfied);
//! ```

mod error;
pub mod invariant;
pub mod liveness;
mod macros;
mod monitor;
mod trace;
mod traceable;

// Re-export main types
pub use error::{MonitorError, MonitorResult};
pub use monitor::{
    CompiledInvariant, MonitorConfig, MonitorStatistics, RuntimeMonitor, RuntimeViolation,
    ScopedMonitor,
};
pub use trace::{
    ExecutionTrace, MonitoredTrace, RecordedAction, SourceLocation, StateSnapshot, StateTransition,
    TraceRecorder,
};
pub use traceable::{MonitoredType, TraceContext, TraceGuard, Traceable, TraceableActions, Traced};

// Re-export from dashprove-async for convenience
pub use dashprove_async::Violation;

// Re-export derive macros when the `macros` feature is enabled
#[cfg(feature = "macros")]
pub use dashprove_monitor_macros::{monitor_action, Monitored};

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestState {
        counter: i32,
        message: String,
    }

    impl Traceable for TestState {
        fn trace_name(&self) -> &str {
            "TestState"
        }

        fn capture_state(&self) -> serde_json::Value {
            serde_json::json!({
                "counter": self.counter,
                "message": self.message
            })
        }

        fn tracked_variables(&self) -> Vec<&str> {
            vec!["counter", "message"]
        }
    }

    #[test]
    fn test_integration_traceable_with_monitor() {
        let mut state = TestState {
            counter: 0,
            message: "initial".to_string(),
        };

        let mut monitor = RuntimeMonitor::new();
        monitor.add_simple_invariant("counter_positive", |s| {
            s.get("counter")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n >= 0)
        });

        // Check initial state
        let violations = monitor.check_traceable(&state).unwrap();
        assert!(violations.is_empty());

        // Update state and check again
        state.counter = 5;
        let violations = monitor.check_traceable(&state).unwrap();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_integration_trace_with_invariants() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"balance": 100}));

        trace.add_transition(StateTransition::new(
            serde_json::json!({"balance": 100}),
            "deposit".to_string(),
            serde_json::json!({"balance": 150}),
        ));

        trace.add_transition(StateTransition::new(
            serde_json::json!({"balance": 150}),
            "withdraw".to_string(),
            serde_json::json!({"balance": 50}),
        ));

        // Check with invariant checker
        let mut checker = invariant::TraceInvariantChecker::new();
        checker.add(invariant::patterns::field_positive("balance"));

        let result = checker.check(&trace);
        assert!(result.passed);
    }

    #[test]
    fn test_integration_monitored_trace() {
        let mut trace = MonitoredTrace::new("test_trace", serde_json::json!({"x": 0}))
            .with_tag("integration")
            .with_metadata("version", serde_json::json!("1.0"));

        trace.record_transition(
            "set",
            serde_json::json!({"x": 0}),
            serde_json::json!({"x": 10}),
        );

        trace.complete();

        assert!(trace.is_complete());
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.tags, vec!["integration"]);
    }

    #[test]
    fn test_integration_recorder_with_liveness() {
        let mut recorder = TraceRecorder::start("progress", serde_json::json!({"done": false}));

        recorder.record("working", serde_json::json!({"done": false}));
        recorder.record("working", serde_json::json!({"done": false}));
        recorder.record("complete", serde_json::json!({"done": true}));

        let trace = recorder.finish();

        // Check liveness
        let prop = liveness::LivenessProperty::eventually("completion", |s| {
            s.get("done").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        let result = liveness::check_liveness(trace.as_execution_trace(), &prop);
        assert!(result.satisfied);
        assert_eq!(result.satisfaction_index, Some(3));
    }

    #[test]
    fn test_macros() {
        let x = 10;
        let y = 20;

        // Test trace_state macro
        let state = trace_state!(x, y);
        assert_eq!(state, serde_json::json!({"x": 10, "y": 20}));

        // Test source_location macro
        let loc = source_location!();
        assert!(loc.file.contains("lib.rs"));
    }
}
