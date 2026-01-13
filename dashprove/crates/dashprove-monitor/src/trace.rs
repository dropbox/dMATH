//! Extended trace types for runtime monitoring
//!
//! This module provides enhanced trace recording capabilities that extend
//! the basic `ExecutionTrace` from `dashprove-async` with additional
//! metadata useful for runtime verification.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Re-export base types from dashprove-async
pub use dashprove_async::{ExecutionTrace, StateTransition};

/// Global trace ID counter
static TRACE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique trace ID
fn next_trace_id() -> u64 {
    TRACE_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Extended execution trace with additional runtime metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoredTrace {
    /// Unique trace identifier
    pub trace_id: u64,

    /// Name/description of this trace
    pub name: String,

    /// The underlying execution trace
    pub execution_trace: ExecutionTrace,

    /// When the trace recording started
    pub started_at: DateTime<Utc>,

    /// When the trace recording ended (if complete)
    #[serde(default)]
    pub ended_at: Option<DateTime<Utc>>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Thread/task ID that recorded this trace
    #[serde(default)]
    pub thread_id: Option<String>,

    /// Source location where trace was started
    #[serde(default)]
    pub source_location: Option<SourceLocation>,

    /// Tags for categorizing traces
    #[serde(default)]
    pub tags: Vec<String>,
}

impl MonitoredTrace {
    /// Create a new monitored trace
    pub fn new(name: impl Into<String>, initial_state: serde_json::Value) -> Self {
        Self {
            trace_id: next_trace_id(),
            name: name.into(),
            execution_trace: ExecutionTrace::new(initial_state),
            started_at: Utc::now(),
            ended_at: None,
            metadata: HashMap::new(),
            thread_id: None,
            source_location: None,
            tags: vec![],
        }
    }

    /// Create with a specific trace ID
    pub fn with_id(id: u64, name: impl Into<String>, initial_state: serde_json::Value) -> Self {
        Self {
            trace_id: id,
            name: name.into(),
            execution_trace: ExecutionTrace::new(initial_state),
            started_at: Utc::now(),
            ended_at: None,
            metadata: HashMap::new(),
            thread_id: None,
            source_location: None,
            tags: vec![],
        }
    }

    /// Set thread ID
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Set source location
    pub fn with_source_location(mut self, location: SourceLocation) -> Self {
        self.source_location = Some(location);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add a transition to the trace
    pub fn add_transition(&mut self, transition: StateTransition) {
        self.execution_trace.add_transition(transition);
    }

    /// Record a state transition with the current timestamp
    pub fn record_transition(
        &mut self,
        event: impl Into<String>,
        from_state: serde_json::Value,
        to_state: serde_json::Value,
    ) {
        let now = Utc::now();
        let timestamp_ms = now.timestamp_millis() as u64;

        let transition =
            StateTransition::new(from_state, event.into(), to_state).with_timestamp(timestamp_ms);

        self.add_transition(transition);
    }

    /// Mark the trace as complete
    pub fn complete(&mut self) {
        self.ended_at = Some(Utc::now());

        // Calculate total duration
        if let Some(ended) = self.ended_at {
            let duration_ms = (ended - self.started_at).num_milliseconds() as u64;
            self.execution_trace.total_duration_ms = Some(duration_ms);
        }
    }

    /// Check if the trace is complete
    pub fn is_complete(&self) -> bool {
        self.ended_at.is_some()
    }

    /// Get the number of transitions
    pub fn len(&self) -> usize {
        self.execution_trace.len()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.execution_trace.is_empty()
    }

    /// Get the initial state
    pub fn initial_state(&self) -> &serde_json::Value {
        &self.execution_trace.initial_state
    }

    /// Get the final state
    pub fn final_state(&self) -> &serde_json::Value {
        &self.execution_trace.final_state
    }

    /// Get all events in order
    pub fn events(&self) -> Vec<&str> {
        self.execution_trace.events()
    }

    /// Get duration in milliseconds (if complete)
    pub fn duration_ms(&self) -> Option<u64> {
        self.execution_trace.total_duration_ms
    }

    /// Convert to the underlying ExecutionTrace
    pub fn into_execution_trace(self) -> ExecutionTrace {
        self.execution_trace
    }

    /// Get reference to underlying ExecutionTrace
    pub fn as_execution_trace(&self) -> &ExecutionTrace {
        &self.execution_trace
    }
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u32,
    /// Function/method name
    #[serde(default)]
    pub function: Option<String>,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(file: impl Into<String>, line: u32, column: u32) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            function: None,
        }
    }

    /// Set function name
    pub fn with_function(mut self, function: impl Into<String>) -> Self {
        self.function = Some(function.into());
        self
    }
}

/// A recorded action within a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedAction {
    /// Action name/type
    pub name: String,

    /// When the action occurred
    pub timestamp: DateTime<Utc>,

    /// Duration of the action (if measured)
    #[serde(default)]
    pub duration: Option<Duration>,

    /// Action arguments/parameters
    #[serde(default)]
    pub arguments: HashMap<String, serde_json::Value>,

    /// Action result (if any)
    #[serde(default)]
    pub result: Option<serde_json::Value>,

    /// Whether the action succeeded
    pub success: bool,

    /// Error message if action failed
    #[serde(default)]
    pub error: Option<String>,

    /// Source location
    #[serde(default)]
    pub source_location: Option<SourceLocation>,
}

impl RecordedAction {
    /// Create a successful action
    pub fn success(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: Utc::now(),
            duration: None,
            arguments: HashMap::new(),
            result: None,
            success: true,
            error: None,
            source_location: None,
        }
    }

    /// Create a failed action
    pub fn failure(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: Utc::now(),
            duration: None,
            arguments: HashMap::new(),
            result: None,
            success: false,
            error: Some(error.into()),
            source_location: None,
        }
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Add an argument
    pub fn with_argument(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.arguments.insert(key.into(), value);
        self
    }

    /// Set result
    pub fn with_result(mut self, result: serde_json::Value) -> Self {
        self.result = Some(result);
        self
    }

    /// Set source location
    pub fn with_source_location(mut self, location: SourceLocation) -> Self {
        self.source_location = Some(location);
        self
    }
}

/// A recorded state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,

    /// The state values
    pub state: serde_json::Value,

    /// Optional label for this snapshot
    #[serde(default)]
    pub label: Option<String>,

    /// Source location where snapshot was taken
    #[serde(default)]
    pub source_location: Option<SourceLocation>,
}

impl StateSnapshot {
    /// Create a new state snapshot
    pub fn new(state: serde_json::Value) -> Self {
        Self {
            timestamp: Utc::now(),
            state,
            label: None,
            source_location: None,
        }
    }

    /// Set label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set source location
    pub fn with_source_location(mut self, location: SourceLocation) -> Self {
        self.source_location = Some(location);
        self
    }
}

/// Builder for recording traces incrementally
pub struct TraceRecorder {
    trace: MonitoredTrace,
    start_time: Instant,
    current_state: serde_json::Value,
}

impl TraceRecorder {
    /// Start recording a new trace
    pub fn start(name: impl Into<String>, initial_state: serde_json::Value) -> Self {
        Self {
            trace: MonitoredTrace::new(name, initial_state.clone()),
            start_time: Instant::now(),
            current_state: initial_state,
        }
    }

    /// Record a state change with an action
    pub fn record(&mut self, action: impl Into<String>, new_state: serde_json::Value) {
        let duration = Duration::from_millis(self.start_time.elapsed().as_millis() as u64);
        let transition =
            StateTransition::new(self.current_state.clone(), action.into(), new_state.clone())
                .with_duration(duration);

        self.trace.add_transition(transition);
        self.current_state = new_state;
    }

    /// Add metadata to the trace
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.trace.metadata.insert(key.into(), value);
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.trace.tags.push(tag.into());
    }

    /// Finish recording and return the trace
    pub fn finish(mut self) -> MonitoredTrace {
        self.trace.complete();
        self.trace
    }

    /// Get the current state
    pub fn current_state(&self) -> &serde_json::Value {
        &self.current_state
    }

    /// Get elapsed time since recording started
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitored_trace_creation() {
        let trace = MonitoredTrace::new("test_trace", serde_json::json!({"x": 0}));

        assert_eq!(trace.name, "test_trace");
        assert!(trace.is_empty());
        assert!(!trace.is_complete());
        assert_eq!(trace.initial_state(), &serde_json::json!({"x": 0}));
    }

    #[test]
    fn test_monitored_trace_transitions() {
        let mut trace = MonitoredTrace::new("counter", serde_json::json!({"count": 0}));

        trace.record_transition(
            "increment",
            serde_json::json!({"count": 0}),
            serde_json::json!({"count": 1}),
        );

        trace.record_transition(
            "increment",
            serde_json::json!({"count": 1}),
            serde_json::json!({"count": 2}),
        );

        assert_eq!(trace.len(), 2);
        assert_eq!(trace.events(), vec!["increment", "increment"]);
        assert_eq!(trace.final_state(), &serde_json::json!({"count": 2}));
    }

    #[test]
    fn test_monitored_trace_completion() {
        let mut trace = MonitoredTrace::new("test", serde_json::json!({}));
        assert!(!trace.is_complete());

        trace.complete();
        assert!(trace.is_complete());
        assert!(trace.ended_at.is_some());
    }

    #[test]
    fn test_monitored_trace_metadata() {
        let trace = MonitoredTrace::new("test", serde_json::json!({}))
            .with_thread_id("main")
            .with_tag("integration")
            .with_tag("slow")
            .with_metadata("version", serde_json::json!("1.0.0"));

        assert_eq!(trace.thread_id, Some("main".to_string()));
        assert_eq!(trace.tags, vec!["integration", "slow"]);
        assert_eq!(
            trace.metadata.get("version"),
            Some(&serde_json::json!("1.0.0"))
        );
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("src/main.rs", 42, 5).with_function("process");

        assert_eq!(loc.file, "src/main.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.column, 5);
        assert_eq!(loc.function, Some("process".to_string()));
    }

    #[test]
    fn test_recorded_action() {
        let action = RecordedAction::success("api_call")
            .with_argument("endpoint", serde_json::json!("/users"))
            .with_result(serde_json::json!({"status": 200}))
            .with_duration(Duration::from_millis(150));

        assert!(action.success);
        assert_eq!(action.name, "api_call");
        assert_eq!(
            action.arguments.get("endpoint"),
            Some(&serde_json::json!("/users"))
        );
        assert_eq!(action.duration, Some(Duration::from_millis(150)));
    }

    #[test]
    fn test_recorded_action_failure() {
        let action = RecordedAction::failure("db_query", "Connection timeout");

        assert!(!action.success);
        assert_eq!(action.error, Some("Connection timeout".to_string()));
    }

    #[test]
    fn test_state_snapshot() {
        let snapshot =
            StateSnapshot::new(serde_json::json!({"balance": 100})).with_label("after_deposit");

        assert_eq!(snapshot.state, serde_json::json!({"balance": 100}));
        assert_eq!(snapshot.label, Some("after_deposit".to_string()));
    }

    #[test]
    fn test_trace_recorder() {
        let mut recorder = TraceRecorder::start("counter", serde_json::json!({"value": 0}));

        recorder.record("increment", serde_json::json!({"value": 1}));
        recorder.record("double", serde_json::json!({"value": 2}));
        recorder.add_tag("test");
        recorder.add_metadata("env", serde_json::json!("test"));

        let trace = recorder.finish();

        assert!(trace.is_complete());
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.tags, vec!["test"]);
        assert_eq!(trace.metadata.get("env"), Some(&serde_json::json!("test")));
    }

    #[test]
    fn test_unique_trace_ids() {
        let trace1 = MonitoredTrace::new("trace1", serde_json::json!({}));
        let trace2 = MonitoredTrace::new("trace2", serde_json::json!({}));

        assert_ne!(trace1.trace_id, trace2.trace_id);
    }

    #[test]
    fn test_into_execution_trace() {
        let mut trace = MonitoredTrace::new("test", serde_json::json!({"x": 0}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"x": 0}),
            "inc".to_string(),
            serde_json::json!({"x": 1}),
        ));

        let exec_trace = trace.into_execution_trace();
        assert_eq!(exec_trace.len(), 1);
        assert_eq!(exec_trace.final_state, serde_json::json!({"x": 1}));
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // SourceLocation property tests

            #[test]
            fn source_location_preserves_fields(
                file in "[a-z/]{1,30}\\.rs",
                line in 1u32..10000,
                column in 1u32..200
            ) {
                let loc = SourceLocation::new(file.clone(), line, column);
                prop_assert_eq!(loc.file, file);
                prop_assert_eq!(loc.line, line);
                prop_assert_eq!(loc.column, column);
            }

            #[test]
            fn source_location_function_preserved(func in "[a-z_]{1,30}") {
                let loc = SourceLocation::new("test.rs", 1, 1).with_function(func.clone());
                prop_assert_eq!(loc.function, Some(func));
            }

            // MonitoredTrace property tests

            #[test]
            fn monitored_trace_name_preserved(name in "[a-z_]{1,30}") {
                let trace = MonitoredTrace::new(name.clone(), serde_json::json!({}));
                prop_assert_eq!(trace.name, name);
            }

            #[test]
            fn monitored_trace_with_id_preserved(id in 0u64..1000000) {
                let trace = MonitoredTrace::with_id(id, "test", serde_json::json!({}));
                prop_assert_eq!(trace.trace_id, id);
            }

            #[test]
            fn monitored_trace_thread_id_preserved(thread_id in "[a-z0-9_]{1,20}") {
                let trace = MonitoredTrace::new("test", serde_json::json!({}))
                    .with_thread_id(thread_id.clone());
                prop_assert_eq!(trace.thread_id, Some(thread_id));
            }

            #[test]
            fn monitored_trace_tags_accumulated(tags in prop::collection::vec("[a-z]{1,10}", 0..5)) {
                let mut trace = MonitoredTrace::new("test", serde_json::json!({}));
                for tag in &tags {
                    trace = trace.with_tag(tag.clone());
                }
                prop_assert_eq!(trace.tags, tags);
            }

            #[test]
            fn monitored_trace_len_equals_transitions(n in 0usize..20) {
                let mut trace = MonitoredTrace::new("test", serde_json::json!({}));
                for i in 0..n {
                    trace.add_transition(StateTransition::new(
                        serde_json::json!({"i": i}),
                        "step".to_string(),
                        serde_json::json!({"i": i + 1}),
                    ));
                }
                prop_assert_eq!(trace.len(), n);
                prop_assert_eq!(trace.is_empty(), n == 0);
            }

            #[test]
            fn monitored_trace_complete_sets_ended_at(_dummy in 0..1i32) {
                let mut trace = MonitoredTrace::new("test", serde_json::json!({}));
                prop_assert!(!trace.is_complete());
                trace.complete();
                prop_assert!(trace.is_complete());
                prop_assert!(trace.ended_at.is_some());
            }

            // RecordedAction property tests

            #[test]
            fn recorded_action_success_name_preserved(name in "[a-z_]{1,30}") {
                let action = RecordedAction::success(name.clone());
                prop_assert_eq!(action.name, name);
                prop_assert!(action.success);
                prop_assert!(action.error.is_none());
            }

            #[test]
            fn recorded_action_failure_preserves_error(name in "[a-z_]{1,30}", error in "[a-zA-Z0-9 ]{1,50}") {
                let action = RecordedAction::failure(name.clone(), error.clone());
                prop_assert_eq!(action.name, name);
                prop_assert!(!action.success);
                prop_assert_eq!(action.error, Some(error));
            }

            #[test]
            fn recorded_action_duration_preserved(ms in 0u64..10000) {
                let action = RecordedAction::success("test")
                    .with_duration(Duration::from_millis(ms));
                prop_assert_eq!(action.duration, Some(Duration::from_millis(ms)));
            }

            #[test]
            fn recorded_action_argument_preserved(key in "[a-z]{1,10}", val in 0i64..1000) {
                let json_val = serde_json::json!(val);
                let action = RecordedAction::success("test")
                    .with_argument(key.clone(), json_val.clone());
                prop_assert_eq!(action.arguments.get(&key), Some(&json_val));
            }

            // StateSnapshot property tests

            #[test]
            fn state_snapshot_label_preserved(label in "[a-z_]{1,30}") {
                let snapshot = StateSnapshot::new(serde_json::json!({}))
                    .with_label(label.clone());
                prop_assert_eq!(snapshot.label, Some(label));
            }

            #[test]
            fn state_snapshot_state_preserved(val in 0i64..1000) {
                let state = serde_json::json!({"value": val});
                let snapshot = StateSnapshot::new(state.clone());
                prop_assert_eq!(snapshot.state, state);
            }

            // TraceRecorder property tests

            #[test]
            fn trace_recorder_tracks_transitions(n in 0usize..10) {
                let mut recorder = TraceRecorder::start("test", serde_json::json!({"x": 0}));
                for i in 0..n {
                    recorder.record("step", serde_json::json!({"x": i + 1}));
                }
                let trace = recorder.finish();
                prop_assert_eq!(trace.len(), n);
                prop_assert!(trace.is_complete());
            }

            #[test]
            fn trace_recorder_current_state_updated(val in 0i64..1000) {
                let mut recorder = TraceRecorder::start("test", serde_json::json!({"x": 0}));
                recorder.record("set", serde_json::json!({"x": val}));
                let expected = serde_json::json!({"x": val});
                prop_assert_eq!(recorder.current_state(), &expected);
            }

            #[test]
            fn trace_recorder_metadata_preserved(key in "[a-z]{1,10}", val in 0i64..1000) {
                let mut recorder = TraceRecorder::start("test", serde_json::json!({}));
                recorder.add_metadata(key.clone(), serde_json::json!(val));
                let trace = recorder.finish();
                prop_assert_eq!(trace.metadata.get(&key), Some(&serde_json::json!(val)));
            }

            #[test]
            fn trace_recorder_tags_preserved(tags in prop::collection::vec("[a-z]{1,10}", 0..5)) {
                let mut recorder = TraceRecorder::start("test", serde_json::json!({}));
                for tag in &tags {
                    recorder.add_tag(tag.clone());
                }
                let trace = recorder.finish();
                prop_assert_eq!(trace.tags, tags);
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // Note: Many types in this module use chrono::Utc::now() which calls clock_gettime,
    // a foreign function not supported by Kani. We only verify types that don't use time.

    /// Proves that SourceLocation::new preserves all fields.
    #[kani::proof]
    fn verify_source_location_new_fields() {
        let line: u32 = kani::any();
        let column: u32 = kani::any();
        kani::assume(line < 100000);
        kani::assume(column < 1000);
        let loc = SourceLocation::new("test.rs", line, column);
        kani::assert(loc.file == "test.rs", "file should be preserved");
        kani::assert(loc.line == line, "line should be preserved");
        kani::assert(loc.column == column, "column should be preserved");
        kani::assert(loc.function.is_none(), "function should be None by default");
    }

    /// Proves that SourceLocation::with_function preserves the function name.
    #[kani::proof]
    fn verify_source_location_with_function() {
        let loc = SourceLocation::new("test.rs", 1, 1).with_function("my_func");
        kani::assert(
            loc.function == Some("my_func".to_string()),
            "function should be preserved",
        );
    }

    /// Proves that SourceLocation line and column are preserved exactly.
    #[kani::proof]
    fn verify_source_location_line_column_exact() {
        let line: u32 = kani::any();
        let column: u32 = kani::any();
        let loc = SourceLocation::new("file.rs", line, column);
        kani::assert(loc.line == line, "line should be exact");
        kani::assert(loc.column == column, "column should be exact");
    }

    /// Proves that SourceLocation file is preserved.
    #[kani::proof]
    fn verify_source_location_file_preserved() {
        let loc = SourceLocation::new("path/to/file.rs", 42, 10);
        kani::assert(
            loc.file == "path/to/file.rs",
            "file path should be preserved",
        );
    }

    /// Proves that SourceLocation with_function can be chained.
    #[kani::proof]
    fn verify_source_location_chaining() {
        let loc = SourceLocation::new("test.rs", 1, 2).with_function("test_fn");
        kani::assert(loc.line == 1, "line should be preserved after chaining");
        kani::assert(loc.column == 2, "column should be preserved after chaining");
        kani::assert(
            loc.function == Some("test_fn".to_string()),
            "function should be set",
        );
    }

    /// Proves that SourceLocation with_source_location sets location on StateSnapshot.
    #[kani::proof]
    fn verify_source_location_default_function_none() {
        let loc = SourceLocation::new("a.rs", 0, 0);
        kani::assert(loc.function.is_none(), "function should default to None");
    }
}
