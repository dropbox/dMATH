//! Trace recording macros
//!
//! This module provides macros for easily recording trace events
//! in instrumented code. These macros provide a convenient way to
//! capture state snapshots and record actions.

/// Capture the current state as a JSON value
///
/// # Examples
///
/// ```rust
/// use dashprove_monitor::trace_state;
///
/// let x = 10;
/// let y = "hello";
/// let state = trace_state!(x, y);
/// // state == json!({"x": 10, "y": "hello"})
/// ```
#[macro_export]
macro_rules! trace_state {
    // Single variable
    ($var:ident) => {
        serde_json::json!({
            stringify!($var): $var
        })
    };

    // Multiple variables
    ($($var:ident),+ $(,)?) => {
        serde_json::json!({
            $(stringify!($var): $var),+
        })
    };

    // Named fields
    ($($name:literal : $value:expr),+ $(,)?) => {
        serde_json::json!({
            $($name: $value),+
        })
    };

    // Mixed: named and auto
    ($($var:ident),+ ; $($name:literal : $value:expr),+ $(,)?) => {
        serde_json::json!({
            $(stringify!($var): $var,)+
            $($name: $value),+
        })
    };
}

/// Record an action with optional before/after state capture
///
/// # Examples
///
/// ```rust,ignore
/// use dashprove_monitor::{trace_action, TraceRecorder};
///
/// let mut recorder = TraceRecorder::start("example", json!({"x": 0}));
///
/// // Simple action recording
/// let x = 10;
/// trace_action!(recorder, "set_x", x);
///
/// // With explicit state
/// trace_action!(recorder, "update", json!({"x": 20}));
/// ```
#[macro_export]
macro_rules! trace_action {
    // Action with auto-captured variables
    ($recorder:expr, $action:expr, $($var:ident),+ $(,)?) => {
        $recorder.record($action, $crate::trace_state!($($var),+))
    };

    // Action with explicit state
    ($recorder:expr, $action:expr, $state:expr) => {
        $recorder.record($action, $state)
    };
}

/// Create a source location from the current position
///
/// # Examples
///
/// ```rust
/// use dashprove_monitor::source_location;
///
/// let loc = source_location!();
/// // loc.file == current file
/// // loc.line == current line
/// ```
#[macro_export]
macro_rules! source_location {
    () => {
        $crate::SourceLocation::new(file!(), line!(), column!())
    };

    ($function:expr) => {
        $crate::SourceLocation::new(file!(), line!(), column!()).with_function($function)
    };
}

/// Monitor a block of code and record any state changes
///
/// # Examples
///
/// ```rust,ignore
/// use dashprove_monitor::{monitor_block, RuntimeMonitor, Traceable};
///
/// let monitor = RuntimeMonitor::new();
/// let mut counter = Counter::new();
///
/// monitor_block!(monitor, counter, "increment", {
///     counter.value += 1;
/// });
/// ```
#[macro_export]
macro_rules! monitor_block {
    ($monitor:expr, $traceable:expr, $action:expr, $block:block) => {{
        let _guard = $crate::ScopedMonitor::new(&$monitor, &$traceable, $action);
        $block
    }};
}

/// Check invariants and panic if any are violated
///
/// # Examples
///
/// ```rust,ignore
/// use dashprove_monitor::{assert_invariants, RuntimeMonitor};
///
/// let monitor = RuntimeMonitor::new();
/// // ... add invariants ...
///
/// assert_invariants!(monitor, json!({"value": 10}));
/// ```
#[macro_export]
macro_rules! assert_invariants {
    ($monitor:expr, $state:expr) => {
        match $monitor.check(&$state) {
            Ok(violations) if !violations.is_empty() => {
                let msgs: Vec<_> = violations.iter().map(|v| &v.message).collect();
                panic!("Invariant violations: {:?}", msgs);
            }
            Err(e) => panic!("Monitor error: {}", e),
            _ => {}
        }
    };

    ($monitor:expr, $traceable:expr) => {
        match $monitor.check_traceable(&$traceable) {
            Ok(violations) if !violations.is_empty() => {
                let msgs: Vec<_> = violations.iter().map(|v| &v.message).collect();
                panic!("Invariant violations: {:?}", msgs);
            }
            Err(e) => panic!("Monitor error: {}", e),
            _ => {}
        }
    };
}

/// Debug-only invariant check (no-op in release builds)
///
/// # Examples
///
/// ```rust,ignore
/// use dashprove_monitor::{debug_check_invariants, RuntimeMonitor};
///
/// let monitor = RuntimeMonitor::new();
/// debug_check_invariants!(monitor, json!({"value": 10}));
/// ```
#[macro_export]
macro_rules! debug_check_invariants {
    ($monitor:expr, $state:expr) => {
        #[cfg(debug_assertions)]
        {
            let _ = $monitor.check(&$state);
        }
    };

    ($monitor:expr, $traceable:expr) => {
        #[cfg(debug_assertions)]
        {
            let _ = $monitor.check_traceable(&$traceable);
        }
    };
}

/// Create a traced wrapper around a value
///
/// # Examples
///
/// ```rust
/// use dashprove_monitor::traced;
///
/// let counter = traced!(vec![1, 2, 3], "my_vec");
/// ```
#[macro_export]
macro_rules! traced {
    ($value:expr, $name:expr) => {
        $crate::Traced::new($value, $name)
    };
}

/// Log a trace event (similar to log crate macros)
///
/// # Examples
///
/// ```rust,ignore
/// use dashprove_monitor::{trace_event, TraceRecorder};
///
/// let mut recorder = TraceRecorder::start("example", json!({}));
/// trace_event!(recorder, "user_login", user_id = 123, success = true);
/// ```
#[macro_export]
macro_rules! trace_event {
    ($recorder:expr, $event:expr, $($key:ident = $value:expr),* $(,)?) => {{
        let state = serde_json::json!({
            $(stringify!($key): $value),*
        });
        $recorder.record($event, state);
    }};
}

#[cfg(test)]
mod tests {
    use crate::{RuntimeMonitor, TraceRecorder};
    use serde_json::json;

    #[test]
    fn test_trace_state_single() {
        let x = 42;
        let state = trace_state!(x);
        assert_eq!(state, json!({"x": 42}));
    }

    #[test]
    fn test_trace_state_multiple() {
        let x = 1;
        let y = 2;
        let z = 3;
        let state = trace_state!(x, y, z);
        assert_eq!(state, json!({"x": 1, "y": 2, "z": 3}));
    }

    #[test]
    fn test_trace_state_named() {
        let state = trace_state!("count": 10, "name": "test");
        assert_eq!(state, json!({"count": 10, "name": "test"}));
    }

    #[test]
    fn test_source_location() {
        let loc = source_location!();
        assert!(loc.file.contains("macros.rs"));
        assert!(loc.line > 0);
        assert!(loc.column > 0);
    }

    #[test]
    fn test_source_location_with_function() {
        let loc = source_location!("test_function");
        assert_eq!(loc.function, Some("test_function".to_string()));
    }

    #[test]
    fn test_traced_macro() {
        let v = traced!(vec![1, 2, 3], "numbers");
        assert_eq!(v.inner(), &vec![1, 2, 3]);
    }

    #[test]
    fn test_trace_action() {
        let mut recorder = TraceRecorder::start("test", json!({"value": 0}));

        let value = 10;
        trace_action!(recorder, "set", value);

        let trace = recorder.finish();
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_trace_event() {
        let mut recorder = TraceRecorder::start("test", json!({}));

        trace_event!(recorder, "login", user_id = 123, success = true);

        let trace = recorder.finish();
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_debug_check_invariants() {
        let mut monitor = RuntimeMonitor::new();
        monitor.add_simple_invariant("always_pass", |_| true);

        // This should not panic
        debug_check_invariants!(monitor, json!({}));
    }
}
