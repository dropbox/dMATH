//! Async state machine trait and types

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::Duration;

use crate::AsyncVerifyError;

/// Trait for async state machines that can be verified
#[async_trait]
pub trait AsyncStateMachine: Send + Sync {
    /// The state type
    type State: Clone + Send + Debug + Serialize;

    /// The event type
    type Event: Clone + Send + Debug + Serialize;

    /// Get the current state
    fn current_state(&self) -> Self::State;

    /// Get all possible events from current state
    fn possible_events(&self) -> Vec<Self::Event>;

    /// Process an event and transition to new state
    async fn process_event(&mut self, event: Self::Event) -> Result<(), AsyncVerifyError>;

    /// Reset to initial state
    async fn reset(&mut self) -> Result<(), AsyncVerifyError>;

    /// Check if the state machine is in a terminal state
    fn is_terminal(&self) -> bool {
        self.possible_events().is_empty()
    }
}

/// A recorded state transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// State before the transition
    pub from_state: serde_json::Value,
    /// Event that triggered the transition
    pub event: String,
    /// State after the transition
    pub to_state: serde_json::Value,
    /// Duration of the transition
    #[serde(default)]
    pub duration_ms: Option<u64>,
    /// Timestamp when transition occurred
    #[serde(default)]
    pub timestamp_ms: Option<u64>,
}

impl StateTransition {
    /// Create a new state transition
    pub fn new(from_state: serde_json::Value, event: String, to_state: serde_json::Value) -> Self {
        Self {
            from_state,
            event,
            to_state,
            duration_ms: None,
            timestamp_ms: None,
        }
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = Some(duration.as_millis() as u64);
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = Some(timestamp_ms);
        self
    }
}

/// An execution trace of state transitions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Initial state
    pub initial_state: serde_json::Value,
    /// Sequence of transitions
    pub transitions: Vec<StateTransition>,
    /// Final state
    pub final_state: serde_json::Value,
    /// Total duration
    #[serde(default)]
    pub total_duration_ms: Option<u64>,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new(initial_state: serde_json::Value) -> Self {
        Self {
            initial_state: initial_state.clone(),
            transitions: vec![],
            final_state: initial_state,
            total_duration_ms: None,
        }
    }

    /// Add a transition
    pub fn add_transition(&mut self, transition: StateTransition) {
        self.final_state = transition.to_state.clone();
        self.transitions.push(transition);
    }

    /// Get the number of transitions
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Get all events in order
    pub fn events(&self) -> Vec<&str> {
        self.transitions.iter().map(|t| t.event.as_str()).collect()
    }
}

/// A concurrent operation that can be interleaved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentOperation {
    /// Operation name
    pub name: String,
    /// Thread/task ID
    pub thread_id: usize,
    /// Operation type
    pub operation_type: OperationType,
    /// Associated data
    pub data: serde_json::Value,
}

/// Types of concurrent operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Lock acquire
    LockAcquire,
    /// Lock release
    LockRelease,
    /// Send on channel
    Send,
    /// Receive on channel
    Receive,
    /// Spawn task/thread
    Spawn,
    /// Join task/thread
    Join,
    /// Atomic operation
    Atomic,
    /// Custom operation
    Custom,
}

impl ConcurrentOperation {
    /// Create a new concurrent operation
    pub fn new(name: impl Into<String>, thread_id: usize, operation_type: OperationType) -> Self {
        Self {
            name: name.into(),
            thread_id,
            operation_type,
            data: serde_json::Value::Null,
        }
    }

    /// Set operation data
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = data;
        self
    }

    /// Check if this operation can conflict with another
    pub fn can_conflict_with(&self, other: &ConcurrentOperation) -> bool {
        // Different threads accessing same resource
        if self.thread_id == other.thread_id {
            return false;
        }

        // Write-write or read-write conflicts
        matches!(
            (self.operation_type, other.operation_type),
            (OperationType::Write, OperationType::Write)
                | (OperationType::Read, OperationType::Write)
                | (OperationType::Write, OperationType::Read)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_transition_builder() {
        let trans = StateTransition::new(
            serde_json::json!({"counter": 0}),
            "increment".to_string(),
            serde_json::json!({"counter": 1}),
        )
        .with_duration(Duration::from_millis(50))
        .with_timestamp(1000);

        assert_eq!(trans.event, "increment");
        assert_eq!(trans.duration_ms, Some(50));
        assert_eq!(trans.timestamp_ms, Some(1000));
    }

    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"value": 0}));

        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 0}),
            "set".to_string(),
            serde_json::json!({"value": 1}),
        ));

        trace.add_transition(StateTransition::new(
            serde_json::json!({"value": 1}),
            "increment".to_string(),
            serde_json::json!({"value": 2}),
        ));

        assert_eq!(trace.len(), 2);
        assert_eq!(trace.events(), vec!["set", "increment"]);
        assert_eq!(trace.final_state, serde_json::json!({"value": 2}));
    }

    #[test]
    fn test_concurrent_operation_conflict() {
        let read1 = ConcurrentOperation::new("read_x", 1, OperationType::Read);
        let write1 = ConcurrentOperation::new("write_x", 2, OperationType::Write);
        let read2 = ConcurrentOperation::new("read_x", 1, OperationType::Read);

        // Read-write conflict (different threads)
        assert!(read1.can_conflict_with(&write1));

        // Same thread - no conflict
        assert!(!read1.can_conflict_with(&read2));
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // StateTransition property tests

            #[test]
            fn state_transition_preserves_event(event in "[a-z_]{1,20}") {
                let trans = StateTransition::new(
                    serde_json::json!({}),
                    event.clone(),
                    serde_json::json!({}),
                );
                prop_assert_eq!(trans.event, event);
            }

            #[test]
            fn state_transition_duration_preserved(ms in 0u64..1_000_000u64) {
                let trans = StateTransition::new(
                    serde_json::json!({}),
                    "event".to_string(),
                    serde_json::json!({}),
                )
                .with_duration(Duration::from_millis(ms));
                prop_assert_eq!(trans.duration_ms, Some(ms));
            }

            #[test]
            fn state_transition_timestamp_preserved(ts in 0u64..u64::MAX) {
                let trans = StateTransition::new(
                    serde_json::json!({}),
                    "event".to_string(),
                    serde_json::json!({}),
                )
                .with_timestamp(ts);
                prop_assert_eq!(trans.timestamp_ms, Some(ts));
            }

            // ExecutionTrace property tests

            #[test]
            fn execution_trace_len_equals_transitions(n in 0usize..50) {
                let mut trace = ExecutionTrace::new(serde_json::json!({}));
                for i in 0..n {
                    trace.add_transition(StateTransition::new(
                        serde_json::json!({"step": i}),
                        format!("event_{}", i),
                        serde_json::json!({"step": i + 1}),
                    ));
                }
                prop_assert_eq!(trace.len(), n);
                prop_assert_eq!(trace.is_empty(), n == 0);
            }

            #[test]
            fn execution_trace_events_order_preserved(events in prop::collection::vec("[a-z]{1,10}", 0..20)) {
                let mut trace = ExecutionTrace::new(serde_json::json!({}));
                for event in &events {
                    trace.add_transition(StateTransition::new(
                        serde_json::json!({}),
                        event.clone(),
                        serde_json::json!({}),
                    ));
                }
                let extracted: Vec<String> = trace.events().iter().map(|s| s.to_string()).collect();
                prop_assert_eq!(extracted, events);
            }

            #[test]
            fn execution_trace_final_state_updated(n in 1usize..20) {
                let mut trace = ExecutionTrace::new(serde_json::json!({"value": 0}));
                for i in 0..n {
                    trace.add_transition(StateTransition::new(
                        serde_json::json!({"value": i}),
                        "step".to_string(),
                        serde_json::json!({"value": i + 1}),
                    ));
                }
                prop_assert_eq!(trace.final_state, serde_json::json!({"value": n}));
            }

            // ConcurrentOperation property tests

            #[test]
            fn concurrent_operation_preserves_fields(name in "[a-z_]{1,20}", thread_id in 0usize..100) {
                let op = ConcurrentOperation::new(name.clone(), thread_id, OperationType::Read);
                prop_assert_eq!(op.name, name);
                prop_assert_eq!(op.thread_id, thread_id);
                prop_assert_eq!(op.operation_type, OperationType::Read);
            }

            #[test]
            fn concurrent_operation_with_data_preserved(value in 0i64..1000) {
                let op = ConcurrentOperation::new("op", 0, OperationType::Write)
                    .with_data(serde_json::json!({"value": value}));
                prop_assert_eq!(op.data, serde_json::json!({"value": value}));
            }

            #[test]
            fn same_thread_never_conflicts(thread_id in 0usize..100) {
                let op1 = ConcurrentOperation::new("op1", thread_id, OperationType::Write);
                let op2 = ConcurrentOperation::new("op2", thread_id, OperationType::Write);
                prop_assert!(!op1.can_conflict_with(&op2));
            }

            #[test]
            fn write_write_different_threads_conflicts(t1 in 0usize..50, t2 in 51usize..100) {
                let op1 = ConcurrentOperation::new("op1", t1, OperationType::Write);
                let op2 = ConcurrentOperation::new("op2", t2, OperationType::Write);
                prop_assert!(op1.can_conflict_with(&op2));
            }

            #[test]
            fn read_write_different_threads_conflicts(t1 in 0usize..50, t2 in 51usize..100) {
                let op1 = ConcurrentOperation::new("op1", t1, OperationType::Read);
                let op2 = ConcurrentOperation::new("op2", t2, OperationType::Write);
                prop_assert!(op1.can_conflict_with(&op2));
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves that StateTransition::new preserves the event field.
    #[kani::proof]
    fn verify_state_transition_event_preserved() {
        let event = String::from("test_event");
        let trans = StateTransition::new(
            serde_json::Value::Null,
            event.clone(),
            serde_json::Value::Null,
        );
        kani::assert(trans.event == event, "Event should be preserved");
    }

    /// Proves that StateTransition::with_duration correctly sets duration_ms.
    #[kani::proof]
    fn verify_state_transition_with_duration() {
        let ms: u64 = kani::any();
        kani::assume(ms <= 1_000_000); // Reasonable bound
        let trans = StateTransition::new(
            serde_json::Value::Null,
            String::from("event"),
            serde_json::Value::Null,
        )
        .with_duration(Duration::from_millis(ms));
        kani::assert(trans.duration_ms == Some(ms), "Duration should be set");
    }

    /// Proves that StateTransition::with_timestamp correctly sets timestamp_ms.
    #[kani::proof]
    fn verify_state_transition_with_timestamp() {
        let ts: u64 = kani::any();
        let trans = StateTransition::new(
            serde_json::Value::Null,
            String::from("event"),
            serde_json::Value::Null,
        )
        .with_timestamp(ts);
        kani::assert(trans.timestamp_ms == Some(ts), "Timestamp should be set");
    }

    /// Proves that ExecutionTrace::new initializes correctly.
    #[kani::proof]
    fn verify_execution_trace_new_empty() {
        let trace = ExecutionTrace::new(serde_json::Value::Null);
        kani::assert(
            trace.transitions.is_empty(),
            "New trace should have no transitions",
        );
        kani::assert(trace.len() == 0, "Length should be 0");
        kani::assert(trace.is_empty(), "Should be empty");
    }

    /// Proves that ExecutionTrace::len matches transitions.len().
    #[kani::proof]
    fn verify_execution_trace_len_matches_transitions() {
        let trace = ExecutionTrace::new(serde_json::Value::Null);
        kani::assert(
            trace.len() == trace.transitions.len(),
            "len() should match transitions.len()",
        );
    }

    /// Proves that ExecutionTrace::is_empty is consistent with len() == 0.
    #[kani::proof]
    fn verify_execution_trace_is_empty_consistent() {
        let trace = ExecutionTrace::new(serde_json::Value::Null);
        kani::assert(
            trace.is_empty() == (trace.len() == 0),
            "is_empty should be consistent with len",
        );
    }

    /// Proves that ConcurrentOperation::new preserves all fields.
    #[kani::proof]
    fn verify_concurrent_operation_new_fields() {
        let thread_id: usize = kani::any();
        kani::assume(thread_id < 1000); // Reasonable bound
        let op = ConcurrentOperation::new("op_name", thread_id, OperationType::Read);
        kani::assert(op.name == "op_name", "Name should be preserved");
        kani::assert(op.thread_id == thread_id, "Thread ID should be preserved");
        kani::assert(
            op.operation_type == OperationType::Read,
            "Operation type should be preserved",
        );
        kani::assert(
            op.data == serde_json::Value::Null,
            "Data should default to Null",
        );
    }

    /// Proves that same-thread operations never conflict.
    #[kani::proof]
    fn verify_same_thread_no_conflict() {
        let thread_id: usize = kani::any();
        kani::assume(thread_id < 1000);
        let op1 = ConcurrentOperation::new("op1", thread_id, OperationType::Write);
        let op2 = ConcurrentOperation::new("op2", thread_id, OperationType::Write);
        kani::assert(
            !op1.can_conflict_with(&op2),
            "Same thread operations should never conflict",
        );
    }

    /// Proves that write-write on different threads conflicts.
    #[kani::proof]
    fn verify_write_write_conflict() {
        let t1: usize = kani::any();
        let t2: usize = kani::any();
        kani::assume(t1 < 1000);
        kani::assume(t2 < 1000);
        kani::assume(t1 != t2);
        let op1 = ConcurrentOperation::new("op1", t1, OperationType::Write);
        let op2 = ConcurrentOperation::new("op2", t2, OperationType::Write);
        kani::assert(
            op1.can_conflict_with(&op2),
            "Write-write on different threads should conflict",
        );
    }

    /// Proves that read-write on different threads conflicts.
    #[kani::proof]
    fn verify_read_write_conflict() {
        let t1: usize = kani::any();
        let t2: usize = kani::any();
        kani::assume(t1 < 1000);
        kani::assume(t2 < 1000);
        kani::assume(t1 != t2);
        let op1 = ConcurrentOperation::new("op1", t1, OperationType::Read);
        let op2 = ConcurrentOperation::new("op2", t2, OperationType::Write);
        kani::assert(
            op1.can_conflict_with(&op2),
            "Read-write on different threads should conflict",
        );
    }

    /// Proves that read-read on different threads does NOT conflict.
    #[kani::proof]
    fn verify_read_read_no_conflict() {
        let t1: usize = kani::any();
        let t2: usize = kani::any();
        kani::assume(t1 < 1000);
        kani::assume(t2 < 1000);
        kani::assume(t1 != t2);
        let op1 = ConcurrentOperation::new("op1", t1, OperationType::Read);
        let op2 = ConcurrentOperation::new("op2", t2, OperationType::Read);
        kani::assert(
            !op1.can_conflict_with(&op2),
            "Read-read on different threads should not conflict",
        );
    }

    /// Proves that OperationType variants are distinct.
    #[kani::proof]
    fn verify_operation_type_variants_distinct() {
        kani::assert(OperationType::Read != OperationType::Write, "Read != Write");
        kani::assert(
            OperationType::LockAcquire != OperationType::LockRelease,
            "LockAcquire != LockRelease",
        );
        kani::assert(
            OperationType::Send != OperationType::Receive,
            "Send != Receive",
        );
        kani::assert(OperationType::Spawn != OperationType::Join, "Spawn != Join");
    }
}
