//! Traceable trait for instrumenting Rust code
//!
//! This module defines the `Traceable` trait which allows Rust types
//! to be instrumented for runtime monitoring. Types implementing this
//! trait can have their state changes automatically recorded to traces.

use serde::Serialize;
use std::fmt::Debug;

use crate::trace::{MonitoredTrace, RecordedAction, StateSnapshot};

/// Trait for types that can be traced/monitored at runtime
///
/// Implementing this trait allows a type to participate in runtime
/// verification by exposing its state for recording and validation.
///
/// # Example
///
/// ```rust
/// use dashprove_monitor::{Traceable, StateSnapshot};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// struct Counter {
///     value: i32,
/// }
///
/// impl Traceable for Counter {
///     fn trace_name(&self) -> &str {
///         "Counter"
///     }
///
///     fn capture_state(&self) -> serde_json::Value {
///         serde_json::json!({
///             "value": self.value
///         })
///     }
/// }
/// ```
pub trait Traceable: Send + Sync {
    /// Get the name of this traceable type (used in trace labels)
    fn trace_name(&self) -> &str;

    /// Capture the current state as a JSON value
    fn capture_state(&self) -> serde_json::Value;

    /// Get a list of variable names that are tracked
    fn tracked_variables(&self) -> Vec<&str> {
        vec![]
    }

    /// Check if a specific variable is being tracked
    fn is_tracking(&self, variable: &str) -> bool {
        self.tracked_variables().contains(&variable)
    }

    /// Create a state snapshot of the current state
    fn snapshot(&self) -> StateSnapshot {
        StateSnapshot::new(self.capture_state())
    }

    /// Create a labeled state snapshot
    fn snapshot_with_label(&self, label: impl Into<String>) -> StateSnapshot {
        StateSnapshot::new(self.capture_state()).with_label(label)
    }
}

/// Extension trait for Traceable types that can also capture actions
pub trait TraceableActions: Traceable {
    /// Record that an action was performed
    fn record_action(&self, action: &str) -> RecordedAction {
        RecordedAction::success(action)
    }

    /// Record a failed action
    fn record_failed_action(&self, action: &str, error: &str) -> RecordedAction {
        RecordedAction::failure(action, error)
    }
}

/// Blanket implementation for all Traceable types
impl<T: Traceable> TraceableActions for T {}

/// A wrapper that adds tracing to any Serialize + Debug type
#[derive(Debug, Clone)]
pub struct Traced<T> {
    inner: T,
    trace_name: String,
}

impl<T> Traced<T> {
    /// Create a new traced wrapper
    pub fn new(inner: T, name: impl Into<String>) -> Self {
        Self {
            inner,
            trace_name: name.into(),
        }
    }

    /// Get a reference to the inner value
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner value
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Unwrap and return the inner value
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: Serialize + Send + Sync> Traceable for Traced<T> {
    fn trace_name(&self) -> &str {
        &self.trace_name
    }

    fn capture_state(&self) -> serde_json::Value {
        serde_json::to_value(&self.inner).unwrap_or(serde_json::Value::Null)
    }
}

/// Context for recording traces from traceable objects
pub struct TraceContext {
    trace: MonitoredTrace,
    last_state: serde_json::Value,
}

impl TraceContext {
    /// Create a new trace context for a traceable object
    pub fn new<T: Traceable>(traceable: &T) -> Self {
        let state = traceable.capture_state();
        Self {
            trace: MonitoredTrace::new(traceable.trace_name(), state.clone()),
            last_state: state,
        }
    }

    /// Create with a custom trace name
    pub fn with_name<T: Traceable>(traceable: &T, name: impl Into<String>) -> Self {
        let state = traceable.capture_state();
        Self {
            trace: MonitoredTrace::new(name, state.clone()),
            last_state: state,
        }
    }

    /// Record a state transition
    pub fn record<T: Traceable>(&mut self, traceable: &T, action: impl Into<String>) {
        let new_state = traceable.capture_state();
        self.trace
            .record_transition(action, self.last_state.clone(), new_state.clone());
        self.last_state = new_state;
    }

    /// Record a state transition with the action result
    pub fn record_with_result<T: Traceable, R: Serialize>(
        &mut self,
        traceable: &T,
        action: impl Into<String>,
        result: &R,
    ) {
        let new_state = traceable.capture_state();
        let action_name = action.into();

        // Add result to metadata
        if let Ok(result_json) = serde_json::to_value(result) {
            self.trace
                .metadata
                .insert(format!("result_{}", self.trace.len()), result_json);
        }

        self.trace
            .record_transition(action_name, self.last_state.clone(), new_state.clone());
        self.last_state = new_state;
    }

    /// Complete the trace and return it
    pub fn complete(mut self) -> MonitoredTrace {
        self.trace.complete();
        self.trace
    }

    /// Get the current trace (without completing it)
    pub fn current_trace(&self) -> &MonitoredTrace {
        &self.trace
    }

    /// Get the last recorded state
    pub fn last_state(&self) -> &serde_json::Value {
        &self.last_state
    }
}

/// Guard that automatically records state transitions
pub struct TraceGuard<'a, T: Traceable> {
    traceable: &'a T,
    context: &'a mut TraceContext,
    action: String,
    recorded: bool,
}

impl<'a, T: Traceable> TraceGuard<'a, T> {
    /// Create a new trace guard
    pub fn new(traceable: &'a T, context: &'a mut TraceContext, action: impl Into<String>) -> Self {
        Self {
            traceable,
            context,
            action: action.into(),
            recorded: false,
        }
    }

    /// Mark the action as complete (records the transition)
    pub fn complete(mut self) {
        if !self.recorded {
            self.context.record(self.traceable, &self.action);
            self.recorded = true;
        }
    }
}

impl<'a, T: Traceable> Drop for TraceGuard<'a, T> {
    fn drop(&mut self) {
        // Auto-record on drop if not already recorded
        if !self.recorded {
            self.context.record(self.traceable, &self.action);
        }
    }
}

/// Trait for types that can be monitored with invariant checking
pub trait MonitoredType: Traceable {
    /// List of invariants that should hold for this type
    #[allow(clippy::type_complexity)]
    fn invariants(&self) -> Vec<Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>> {
        vec![]
    }

    /// Names of the invariants
    fn invariant_names(&self) -> Vec<&str> {
        vec![]
    }

    /// Check all invariants on the current state
    fn check_invariants(&self) -> Vec<String> {
        let state = self.capture_state();
        let mut violations = vec![];

        for (i, invariant) in self.invariants().iter().enumerate() {
            if !invariant(&state) {
                let name = self.invariant_names().get(i).copied().unwrap_or("unknown");
                violations.push(name.to_string());
            }
        }

        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestCounter {
        value: i32,
    }

    impl Traceable for TestCounter {
        fn trace_name(&self) -> &str {
            "TestCounter"
        }

        fn capture_state(&self) -> serde_json::Value {
            serde_json::json!({
                "value": self.value
            })
        }

        fn tracked_variables(&self) -> Vec<&str> {
            vec!["value"]
        }
    }

    #[test]
    fn test_traceable_basic() {
        let counter = TestCounter { value: 42 };

        assert_eq!(counter.trace_name(), "TestCounter");
        assert_eq!(counter.capture_state(), serde_json::json!({"value": 42}));
        assert!(counter.is_tracking("value"));
        assert!(!counter.is_tracking("other"));
    }

    #[test]
    fn test_traceable_snapshot() {
        let counter = TestCounter { value: 10 };

        let snapshot = counter.snapshot();
        assert_eq!(snapshot.state, serde_json::json!({"value": 10}));

        let labeled = counter.snapshot_with_label("initial");
        assert_eq!(labeled.label, Some("initial".to_string()));
    }

    #[test]
    fn test_traced_wrapper() {
        let traced = Traced::new(vec![1, 2, 3], "numbers");

        assert_eq!(traced.trace_name(), "numbers");
        assert_eq!(traced.capture_state(), serde_json::json!([1, 2, 3]));
        assert_eq!(traced.inner(), &vec![1, 2, 3]);
    }

    #[test]
    fn test_trace_context() {
        let mut counter = TestCounter { value: 0 };
        let mut ctx = TraceContext::new(&counter);

        counter.value = 1;
        ctx.record(&counter, "increment");

        counter.value = 2;
        ctx.record(&counter, "increment");

        let trace = ctx.complete();

        assert!(trace.is_complete());
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.events(), vec!["increment", "increment"]);
    }

    #[test]
    fn test_trace_context_custom_name() {
        let counter = TestCounter { value: 0 };
        let ctx = TraceContext::with_name(&counter, "my_counter_trace");

        assert_eq!(ctx.current_trace().name, "my_counter_trace");
    }

    #[test]
    fn test_traceable_actions() {
        let counter = TestCounter { value: 0 };

        let action = counter.record_action("increment");
        assert!(action.success);
        assert_eq!(action.name, "increment");

        let failed = counter.record_failed_action("divide", "division by zero");
        assert!(!failed.success);
        assert_eq!(failed.error, Some("division by zero".to_string()));
    }

    #[test]
    fn test_traced_into_inner() {
        let traced = Traced::new(42i32, "number");
        assert_eq!(traced.into_inner(), 42);
    }

    #[test]
    fn test_traced_inner_mut() {
        let mut traced = Traced::new(vec![1, 2], "numbers");
        traced.inner_mut().push(3);
        assert_eq!(traced.inner(), &vec![1, 2, 3]);
    }

    // Mutation-killing tests for Traceable::tracked_variables returning vec![""] or vec!["xyzzy"]
    #[test]
    fn test_traceable_tracked_variables_content() {
        let counter = TestCounter { value: 42 };
        let tracked = counter.tracked_variables();

        // Verify the actual content of tracked variables
        assert_eq!(tracked, vec!["value"]);
        assert_ne!(tracked, vec![""]);
        assert_ne!(tracked, vec!["xyzzy"]);
    }

    #[test]
    fn test_traceable_default_tracked_variables() {
        // Test default implementation returns empty vec
        struct NoTracking;
        impl Traceable for NoTracking {
            fn trace_name(&self) -> &str {
                "NoTracking"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({})
            }
        }

        let no_tracking = NoTracking;
        assert!(no_tracking.tracked_variables().is_empty());
    }

    // Mutation-killing tests for TraceContext::record_with_result (line 166)
    #[test]
    fn test_trace_context_record_with_result() {
        let mut counter = TestCounter { value: 0 };
        let mut ctx = TraceContext::new(&counter);

        counter.value = 10;
        ctx.record_with_result(&counter, "set_value", &"success");

        let trace = ctx.complete();
        assert_eq!(trace.len(), 1);

        // Verify result was stored in metadata
        assert!(trace.metadata.contains_key("result_0"));
    }

    // Mutation-killing tests for TraceContext::last_state (lines 193-195)
    #[test]
    fn test_trace_context_last_state() {
        let counter = TestCounter { value: 42 };
        let ctx = TraceContext::new(&counter);

        // last_state should match the initial state
        assert_eq!(ctx.last_state(), &serde_json::json!({"value": 42}));
    }

    #[test]
    fn test_trace_context_last_state_updates() {
        let mut counter = TestCounter { value: 0 };
        let mut ctx = TraceContext::new(&counter);

        counter.value = 100;
        ctx.record(&counter, "update");

        // last_state should now reflect the new state
        assert_eq!(ctx.last_state(), &serde_json::json!({"value": 100}));
    }

    // Mutation-killing tests for TraceGuard::complete and Drop (lines 219-232)
    #[test]
    fn test_trace_guard_complete() {
        use std::sync::atomic::{AtomicI32, Ordering};

        #[derive(Debug)]
        struct AtomicCounter {
            value: AtomicI32,
        }

        impl Traceable for AtomicCounter {
            fn trace_name(&self) -> &str {
                "AtomicCounter"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({"value": self.value.load(Ordering::SeqCst)})
            }
        }

        let counter = AtomicCounter {
            value: AtomicI32::new(0),
        };
        let mut ctx = TraceContext::new(&counter);

        {
            let guard = TraceGuard::new(&counter, &mut ctx, "action");
            counter.value.store(5, Ordering::SeqCst);
            guard.complete(); // Explicit complete
        }

        // Should have recorded exactly once
        let trace = ctx.complete();
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_trace_guard_drop_records() {
        use std::sync::atomic::{AtomicI32, Ordering};

        #[derive(Debug)]
        struct AtomicCounter {
            value: AtomicI32,
        }

        impl Traceable for AtomicCounter {
            fn trace_name(&self) -> &str {
                "AtomicCounter"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({"value": self.value.load(Ordering::SeqCst)})
            }
        }

        let counter = AtomicCounter {
            value: AtomicI32::new(0),
        };
        let mut ctx = TraceContext::new(&counter);

        {
            let _guard = TraceGuard::new(&counter, &mut ctx, "action");
            counter.value.store(5, Ordering::SeqCst);
            // Don't call complete - let drop handle it
        }

        // Should still have recorded
        let trace = ctx.complete();
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_trace_guard_no_double_record() {
        use std::sync::atomic::{AtomicI32, Ordering};

        #[derive(Debug)]
        struct AtomicCounter {
            value: AtomicI32,
        }

        impl Traceable for AtomicCounter {
            fn trace_name(&self) -> &str {
                "AtomicCounter"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({"value": self.value.load(Ordering::SeqCst)})
            }
        }

        let counter = AtomicCounter {
            value: AtomicI32::new(0),
        };
        let mut ctx = TraceContext::new(&counter);

        {
            let guard = TraceGuard::new(&counter, &mut ctx, "action");
            counter.value.store(5, Ordering::SeqCst);
            guard.complete(); // Explicit complete should prevent double recording on drop
        }

        // Should have recorded exactly once (not twice)
        let trace = ctx.complete();
        assert_eq!(trace.len(), 1);
    }

    // Mutation-killing tests for MonitoredType::invariant_names (lines 244-246)
    #[test]
    fn test_monitored_type_invariant_names() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct Monitored {
            value: i32,
        }

        impl Traceable for Monitored {
            fn trace_name(&self) -> &str {
                "Monitored"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({"value": self.value})
            }
        }

        impl MonitoredType for Monitored {
            fn invariants(&self) -> Vec<Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>> {
                vec![Box::new(|s| {
                    s.get("value")
                        .and_then(|v| v.as_i64())
                        .is_some_and(|n| n >= 0)
                })]
            }

            fn invariant_names(&self) -> Vec<&str> {
                vec!["non_negative"]
            }
        }

        let m = Monitored { value: 10 };
        let names = m.invariant_names();

        // Verify actual content
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "non_negative");
        assert_ne!(names, vec![""]);
        assert_ne!(names, vec!["xyzzy"]);
    }

    // Mutation-killing tests for MonitoredType::check_invariants (lines 249-261)
    #[test]
    fn test_monitored_type_check_invariants_violations() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct Monitored {
            value: i32,
        }

        impl Traceable for Monitored {
            fn trace_name(&self) -> &str {
                "Monitored"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({"value": self.value})
            }
        }

        impl MonitoredType for Monitored {
            fn invariants(&self) -> Vec<Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>> {
                vec![Box::new(|s| {
                    s.get("value")
                        .and_then(|v| v.as_i64())
                        .is_some_and(|n| n >= 0)
                })]
            }

            fn invariant_names(&self) -> Vec<&str> {
                vec!["non_negative"]
            }
        }

        // Should pass when value >= 0
        let m_pass = Monitored { value: 10 };
        let violations = m_pass.check_invariants();
        assert!(violations.is_empty());

        // Should fail when value < 0
        let m_fail = Monitored { value: -5 };
        let violations = m_fail.check_invariants();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0], "non_negative");
    }

    #[test]
    fn test_monitored_type_default_invariant_names() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct DefaultMonitored;

        impl Traceable for DefaultMonitored {
            fn trace_name(&self) -> &str {
                "DefaultMonitored"
            }
            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({})
            }
        }

        impl MonitoredType for DefaultMonitored {}

        let m = DefaultMonitored;
        // Default implementation should return empty vec
        assert!(m.invariant_names().is_empty());
        assert!(m.invariants().is_empty());
        assert!(m.check_invariants().is_empty());
    }
}
