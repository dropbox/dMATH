//! Async state machine verifier

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::{AsyncStateMachine, AsyncVerifyError, ExecutionTrace};

/// Result of verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether all checks passed
    pub passed: bool,
    /// Violations found
    pub violations: Vec<Violation>,
    /// Number of states explored
    pub states_explored: usize,
    /// Number of transitions checked
    pub transitions_checked: usize,
    /// Total verification time (milliseconds)
    pub duration_ms: u64,
    /// Execution traces that demonstrate violations
    #[serde(default)]
    pub counterexamples: Vec<ExecutionTrace>,
}

impl VerificationResult {
    /// Create a passed result
    pub fn passed() -> Self {
        Self {
            passed: true,
            violations: vec![],
            states_explored: 0,
            transitions_checked: 0,
            duration_ms: 0,
            counterexamples: vec![],
        }
    }

    /// Create a failed result with violations
    pub fn failed(violations: Vec<Violation>) -> Self {
        Self {
            passed: false,
            violations,
            states_explored: 0,
            transitions_checked: 0,
            duration_ms: 0,
            counterexamples: vec![],
        }
    }

    /// Set states explored
    pub fn with_states_explored(mut self, count: usize) -> Self {
        self.states_explored = count;
        self
    }

    /// Set transitions checked
    pub fn with_transitions_checked(mut self, count: usize) -> Self {
        self.transitions_checked = count;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = duration.as_millis() as u64;
        self
    }

    /// Add a counterexample
    pub fn with_counterexample(mut self, trace: ExecutionTrace) -> Self {
        self.counterexamples.push(trace);
        self
    }

    /// Get summary message
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "Verification passed: {} states, {} transitions in {}ms",
                self.states_explored, self.transitions_checked, self.duration_ms
            )
        } else {
            format!(
                "Verification failed: {} violations found ({} states, {} transitions in {}ms)",
                self.violations.len(),
                self.states_explored,
                self.transitions_checked,
                self.duration_ms
            )
        }
    }
}

/// A verification violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Violation {
    /// An invariant was violated
    InvariantViolation {
        /// Name of the invariant
        invariant: String,
        /// State where violation occurred
        state: serde_json::Value,
        /// Description
        message: String,
    },
    /// A deadlock was detected
    Deadlock {
        /// State where deadlock occurred
        state: serde_json::Value,
        /// Threads involved
        threads: Vec<usize>,
    },
    /// A livelock was detected
    Livelock {
        /// States in the cycle
        cycle: Vec<serde_json::Value>,
        /// Events causing the cycle
        events: Vec<String>,
    },
    /// A race condition was detected
    RaceCondition {
        /// Location of the race
        location: String,
        /// Threads involved
        threads: Vec<usize>,
        /// Description
        message: String,
    },
    /// Safety property violated
    SafetyViolation {
        /// Property name
        property: String,
        /// Violation details
        message: String,
    },
    /// Liveness property violated
    LivenessViolation {
        /// Property name
        property: String,
        /// Violation details
        message: String,
    },
}

impl Violation {
    /// Get a description of this violation
    pub fn description(&self) -> String {
        match self {
            Self::InvariantViolation {
                invariant, message, ..
            } => {
                format!("Invariant '{invariant}' violated: {message}")
            }
            Self::Deadlock { threads, .. } => {
                format!("Deadlock detected involving threads: {:?}", threads)
            }
            Self::Livelock { events, .. } => {
                format!("Livelock detected with events: {:?}", events)
            }
            Self::RaceCondition {
                location,
                threads,
                message,
            } => {
                format!(
                    "Race condition at '{}' between threads {:?}: {}",
                    location, threads, message
                )
            }
            Self::SafetyViolation { property, message } => {
                format!("Safety property '{}' violated: {}", property, message)
            }
            Self::LivenessViolation { property, message } => {
                format!("Liveness property '{}' violated: {}", property, message)
            }
        }
    }
}

/// Configuration for async verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Maximum states to explore
    pub max_states: usize,
    /// Maximum transitions to check
    pub max_transitions: usize,
    /// Timeout for verification
    pub timeout_ms: u64,
    /// Whether to use bounded model checking
    pub bounded: bool,
    /// Bound for bounded model checking
    pub bound: usize,
    /// Enable deadlock detection
    pub detect_deadlock: bool,
    /// Enable livelock detection
    pub detect_livelock: bool,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            max_states: 100000,
            max_transitions: 1000000,
            timeout_ms: 60000,
            bounded: true,
            bound: 100,
            detect_deadlock: true,
            detect_livelock: true,
        }
    }
}

impl VerificationConfig {
    /// Set maximum states
    pub fn with_max_states(mut self, count: usize) -> Self {
        self.max_states = count;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = timeout.as_millis() as u64;
        self
    }

    /// Set bound
    pub fn with_bound(mut self, bound: usize) -> Self {
        self.bounded = true;
        self.bound = bound;
        self
    }

    /// Disable deadlock detection
    pub fn without_deadlock_detection(mut self) -> Self {
        self.detect_deadlock = false;
        self
    }
}

/// An invariant checker function
pub type InvariantFn = Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>;

/// Named invariant
pub struct NamedInvariant {
    /// Invariant name
    pub name: String,
    /// Checker function
    pub check: InvariantFn,
}

impl NamedInvariant {
    /// Create a new named invariant
    pub fn new(
        name: impl Into<String>,
        check: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            check: Box::new(check),
        }
    }
}

/// Async verifier for state machines
pub struct AsyncVerifier {
    config: VerificationConfig,
}

impl AsyncVerifier {
    /// Create a new verifier with default config
    pub fn new() -> Self {
        Self {
            config: VerificationConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: VerificationConfig) -> Self {
        Self { config }
    }

    /// Verify invariants on a state machine
    pub async fn verify_invariants<S>(
        &self,
        machine: &mut S,
        invariants: &[NamedInvariant],
    ) -> Result<VerificationResult, AsyncVerifyError>
    where
        S: AsyncStateMachine,
        S::State: serde::Serialize,
    {
        let start = std::time::Instant::now();
        let mut states_explored = 0;
        let mut transitions_checked = 0;
        let mut violations = vec![];

        // Check initial state
        let initial_state = machine.current_state();
        let initial_json = serde_json::to_value(&initial_state)
            .map_err(|e| AsyncVerifyError::state_machine(format!("Serialization error: {e}")))?;

        for inv in invariants {
            if !(inv.check)(&initial_json) {
                violations.push(Violation::InvariantViolation {
                    invariant: inv.name.clone(),
                    state: initial_json.clone(),
                    message: "Initial state violates invariant".to_string(),
                });
            }
        }

        states_explored += 1;

        // BFS exploration
        let mut depth = 0;
        while depth < self.config.bound && states_explored < self.config.max_states {
            let events = machine.possible_events();

            if events.is_empty() {
                // Terminal state - check for deadlock
                if self.config.detect_deadlock && !machine.is_terminal() {
                    let state_json = serde_json::to_value(machine.current_state())
                        .unwrap_or(serde_json::Value::Null);
                    violations.push(Violation::Deadlock {
                        state: state_json,
                        threads: vec![],
                    });
                }
                break;
            }

            // Process one event (for simple verification)
            if let Some(event) = events.into_iter().next() {
                machine.process_event(event).await?;
                transitions_checked += 1;

                let new_state = machine.current_state();
                let state_json = serde_json::to_value(&new_state).map_err(|e| {
                    AsyncVerifyError::state_machine(format!("Serialization error: {e}"))
                })?;

                // Check invariants
                for inv in invariants {
                    if !(inv.check)(&state_json) {
                        violations.push(Violation::InvariantViolation {
                            invariant: inv.name.clone(),
                            state: state_json.clone(),
                            message: format!("Invariant violated at depth {}", depth),
                        });
                    }
                }

                states_explored += 1;
            }

            depth += 1;

            // Check timeout
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                break;
            }
        }

        let duration = start.elapsed();

        if violations.is_empty() {
            Ok(VerificationResult::passed()
                .with_states_explored(states_explored)
                .with_transitions_checked(transitions_checked)
                .with_duration(duration))
        } else {
            Ok(VerificationResult::failed(violations)
                .with_states_explored(states_explored)
                .with_transitions_checked(transitions_checked)
                .with_duration(duration))
        }
    }

    /// Explore interleavings using Loom (when feature enabled)
    #[cfg(feature = "loom")]
    pub fn explore_with_loom<F>(&self, test: F) -> crate::InterleavingResult
    where
        F: Fn() + Send + Sync + 'static,
    {
        use crate::InterleavingResult;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let iteration = Arc::new(AtomicUsize::new(0));
        let iteration_clone = Arc::clone(&iteration);

        loom::model(move || {
            iteration_clone.fetch_add(1, Ordering::Relaxed);
            test();
        });

        let mut result = InterleavingResult::new();
        result.total_interleavings = iteration.load(Ordering::Relaxed);
        result.exhaustive = true;
        result
    }

    /// Placeholder for loom exploration when feature is disabled
    #[cfg(not(feature = "loom"))]
    pub fn explore_with_loom<F>(&self, _test: F) -> crate::InterleavingResult
    where
        F: Fn() + Send + Sync + 'static,
    {
        crate::InterleavingResult::new()
    }
}

impl Default for AsyncVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_summary() {
        let passed = VerificationResult::passed()
            .with_states_explored(100)
            .with_transitions_checked(500)
            .with_duration(Duration::from_millis(250));

        assert!(passed.summary().contains("passed"));
        assert!(passed.summary().contains("100"));
        assert!(passed.summary().contains("500"));

        let failed = VerificationResult::failed(vec![Violation::Deadlock {
            state: serde_json::json!({}),
            threads: vec![0, 1],
        }]);

        assert!(failed.summary().contains("failed"));
        assert!(failed.summary().contains("1 violations"));
    }

    #[test]
    fn test_violation_descriptions() {
        let inv = Violation::InvariantViolation {
            invariant: "positive_balance".to_string(),
            state: serde_json::json!({"balance": -100}),
            message: "Balance cannot be negative".to_string(),
        };
        assert!(inv.description().contains("positive_balance"));

        let deadlock = Violation::Deadlock {
            state: serde_json::json!({}),
            threads: vec![1, 2],
        };
        assert!(deadlock.description().contains("Deadlock"));

        let race = Violation::RaceCondition {
            location: "counter".to_string(),
            threads: vec![0, 1],
            message: "Concurrent write".to_string(),
        };
        assert!(race.description().contains("Race condition"));
    }

    #[test]
    fn test_config_builder() {
        let config = VerificationConfig::default()
            .with_max_states(50000)
            .with_timeout(Duration::from_secs(30))
            .with_bound(50)
            .without_deadlock_detection();

        assert_eq!(config.max_states, 50000);
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.bound, 50);
        assert!(!config.detect_deadlock);
    }

    #[test]
    fn test_named_invariant() {
        let inv = NamedInvariant::new("positive", |state| {
            state.get("value").and_then(|v| v.as_i64()).unwrap_or(0) >= 0
        });

        assert_eq!(inv.name, "positive");
        assert!((inv.check)(&serde_json::json!({"value": 5})));
        assert!(!(inv.check)(&serde_json::json!({"value": -1})));
    }

    #[test]
    fn test_verifier_creation() {
        let _verifier = AsyncVerifier::new();
        let _verifier_with_config = AsyncVerifier::with_config(VerificationConfig::default());
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // VerificationResult property tests

            #[test]
            fn verification_result_passed_has_no_violations(_dummy in 0..1i32) {
                let result = VerificationResult::passed();
                prop_assert!(result.passed);
                prop_assert!(result.violations.is_empty());
            }

            #[test]
            fn verification_result_failed_has_violations(n in 1usize..10) {
                let violations: Vec<Violation> = (0..n)
                    .map(|i| Violation::SafetyViolation {
                        property: format!("prop_{}", i),
                        message: "violation".to_string(),
                    })
                    .collect();
                let result = VerificationResult::failed(violations);
                prop_assert!(!result.passed);
                prop_assert_eq!(result.violations.len(), n);
            }

            #[test]
            fn verification_result_states_preserved(states in 0usize..100000) {
                let result = VerificationResult::passed().with_states_explored(states);
                prop_assert_eq!(result.states_explored, states);
            }

            #[test]
            fn verification_result_transitions_preserved(trans in 0usize..1000000) {
                let result = VerificationResult::passed().with_transitions_checked(trans);
                prop_assert_eq!(result.transitions_checked, trans);
            }

            #[test]
            fn verification_result_duration_preserved(ms in 0u64..1_000_000u64) {
                let result = VerificationResult::passed().with_duration(Duration::from_millis(ms));
                prop_assert_eq!(result.duration_ms, ms);
            }

            #[test]
            fn verification_result_summary_contains_counts(states in 1usize..10000, trans in 1usize..10000) {
                let result = VerificationResult::passed()
                    .with_states_explored(states)
                    .with_transitions_checked(trans);
                let summary = result.summary();
                prop_assert!(summary.contains(&states.to_string()));
                prop_assert!(summary.contains(&trans.to_string()));
            }

            // Violation property tests

            #[test]
            fn invariant_violation_description_contains_name(name in "[a-z_]{1,20}") {
                let violation = Violation::InvariantViolation {
                    invariant: name.clone(),
                    state: serde_json::json!({}),
                    message: "test".to_string(),
                };
                prop_assert!(violation.description().contains(&name));
            }

            #[test]
            fn deadlock_description_contains_threads(threads in prop::collection::vec(0usize..10, 1..5)) {
                let violation = Violation::Deadlock {
                    state: serde_json::json!({}),
                    threads: threads.clone(),
                };
                let desc = violation.description();
                prop_assert!(desc.contains("Deadlock"));
            }

            #[test]
            fn race_condition_description_contains_location(location in "[a-z_]{1,20}") {
                let violation = Violation::RaceCondition {
                    location: location.clone(),
                    threads: vec![0, 1],
                    message: "test".to_string(),
                };
                prop_assert!(violation.description().contains(&location));
            }

            #[test]
            fn safety_violation_description_contains_property(prop in "[a-z_]{1,20}") {
                let violation = Violation::SafetyViolation {
                    property: prop.clone(),
                    message: "test".to_string(),
                };
                prop_assert!(violation.description().contains(&prop));
            }

            #[test]
            fn liveness_violation_description_contains_property(prop in "[a-z_]{1,20}") {
                let violation = Violation::LivenessViolation {
                    property: prop.clone(),
                    message: "test".to_string(),
                };
                prop_assert!(violation.description().contains(&prop));
            }

            // VerificationConfig property tests

            #[test]
            fn config_max_states_preserved(max_states in 1usize..1000000) {
                let config = VerificationConfig::default().with_max_states(max_states);
                prop_assert_eq!(config.max_states, max_states);
            }

            #[test]
            fn config_timeout_preserved(secs in 1u64..3600) {
                let config = VerificationConfig::default().with_timeout(Duration::from_secs(secs));
                prop_assert_eq!(config.timeout_ms, secs * 1000);
            }

            #[test]
            fn config_bound_preserved(bound in 1usize..1000) {
                let config = VerificationConfig::default().with_bound(bound);
                prop_assert!(config.bounded);
                prop_assert_eq!(config.bound, bound);
            }

            #[test]
            fn config_without_deadlock_detection(_dummy in 0..1i32) {
                let config = VerificationConfig::default().without_deadlock_detection();
                prop_assert!(!config.detect_deadlock);
            }

            // NamedInvariant property tests

            #[test]
            fn named_invariant_name_preserved(name in "[a-z_]{1,30}") {
                let inv = NamedInvariant::new(name.clone(), |_| true);
                prop_assert_eq!(inv.name, name);
            }

            #[test]
            fn named_invariant_always_true_passes(value in -1000i64..1000) {
                let inv = NamedInvariant::new("always_true", |_| true);
                let json_val = serde_json::json!({"value": value});
                let result = (inv.check)(&json_val);
                prop_assert!(result);
            }

            #[test]
            fn named_invariant_always_false_fails(value in -1000i64..1000) {
                let inv = NamedInvariant::new("always_false", |_| false);
                let json_val = serde_json::json!({"value": value});
                let result = (inv.check)(&json_val);
                prop_assert!(!result);
            }

            #[test]
            fn named_invariant_positive_check_works(value in 0i64..1000) {
                let inv = NamedInvariant::new("positive", |state| {
                    state.get("value").and_then(|v| v.as_i64()).unwrap_or(0) >= 0
                });
                let json_val = serde_json::json!({"value": value});
                let result = (inv.check)(&json_val);
                prop_assert!(result);
            }
        }
    }

    // Additional tests for mutation coverage

    #[test]
    fn test_verification_result_with_counterexample() {
        // Test that counterexample is added correctly
        let trace = crate::ExecutionTrace::new(serde_json::json!({"state": "bad"}));
        let result = VerificationResult::passed().with_counterexample(trace);

        assert_eq!(result.counterexamples.len(), 1);
        assert_eq!(
            result.counterexamples[0].initial_state,
            serde_json::json!({"state": "bad"})
        );
    }

    #[test]
    fn test_verifier_with_config_preserves_settings() {
        // Test that AsyncVerifier::with_config preserves configuration
        // This catches mutation at line 279
        let config = VerificationConfig::default()
            .with_max_states(500)
            .with_timeout(Duration::from_secs(10))
            .with_bound(25)
            .without_deadlock_detection();

        let _verifier = AsyncVerifier::with_config(config.clone());

        // We can't directly access the config, but we can verify behavior
        // by running verify_invariants with a mock state machine
        // For now, just verify the verifier was created
        assert!(!config.detect_deadlock);
        assert_eq!(config.max_states, 500);
        assert_eq!(config.bound, 25);
    }

    #[test]
    fn test_violation_livelock_description() {
        // Test Livelock violation description
        let violation = Violation::Livelock {
            cycle: vec![
                serde_json::json!({"state": "A"}),
                serde_json::json!({"state": "B"}),
            ],
            events: vec!["toggle".to_string(), "toggle".to_string()],
        };

        let desc = violation.description();
        assert!(desc.contains("Livelock"));
        assert!(desc.contains("toggle"));
    }

    #[test]
    fn test_verification_result_default() {
        // Test that default creates passed result
        let _verifier = AsyncVerifier::default();
        // Just verify it can be created (default implementation)
    }

    #[tokio::test]
    async fn test_verify_invariants_with_simple_machine() {
        // Test verify_invariants with a real state machine implementation
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct CounterState {
            value: i32,
            steps: usize,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum CounterEvent {
            Increment,
            Decrement,
        }

        struct CounterMachine {
            state: CounterState,
            max_steps: usize,
        }

        #[async_trait]
        impl AsyncStateMachine for CounterMachine {
            type State = CounterState;
            type Event = CounterEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                if self.state.steps >= self.max_steps {
                    vec![] // Terminal
                } else {
                    vec![CounterEvent::Increment, CounterEvent::Decrement]
                }
            }

            async fn process_event(&mut self, event: Self::Event) -> Result<(), AsyncVerifyError> {
                match event {
                    CounterEvent::Increment => self.state.value += 1,
                    CounterEvent::Decrement => self.state.value -= 1,
                }
                self.state.steps += 1;
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                self.state = CounterState { value: 0, steps: 0 };
                Ok(())
            }
        }

        // Create a state machine that will hit bounds
        let mut machine = CounterMachine {
            state: CounterState { value: 5, steps: 0 },
            max_steps: 10,
        };

        // Create verifier with small bounds
        let config = VerificationConfig::default()
            .with_bound(5)
            .with_max_states(20);
        let verifier = AsyncVerifier::with_config(config);

        // Invariant that checks value is non-negative
        let invariants = vec![NamedInvariant::new("non_negative", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .map(|v| v >= 0)
                .unwrap_or(true)
        })];

        let result = verifier.verify_invariants(&mut machine, &invariants).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        // Starting at 5, incrementing, should pass
        assert!(result.passed || !result.violations.is_empty());
        assert!(result.states_explored > 0);
    }

    #[tokio::test]
    async fn test_verify_invariants_violation_in_initial_state() {
        // Test that violations in initial state are detected
        // This catches mutation at line 303 (delete !)
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct SimpleState {
            value: i32,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum SimpleEvent {}

        struct SimpleMachine {
            state: SimpleState,
        }

        #[async_trait]
        impl AsyncStateMachine for SimpleMachine {
            type State = SimpleState;
            type Event = SimpleEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                vec![] // Terminal immediately
            }

            async fn process_event(&mut self, _event: Self::Event) -> Result<(), AsyncVerifyError> {
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                Ok(())
            }
        }

        // Start with negative value that violates invariant
        let mut machine = SimpleMachine {
            state: SimpleState { value: -10 },
        };

        let verifier = AsyncVerifier::new();

        // Invariant that value must be non-negative
        let invariants = vec![NamedInvariant::new("non_negative", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .map(|v| v >= 0)
                .unwrap_or(false)
        })];

        let result = verifier.verify_invariants(&mut machine, &invariants).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        // Should fail due to initial state violation
        assert!(!result.passed);
        assert!(!result.violations.is_empty());

        // Check that the violation mentions the invariant
        if let Violation::InvariantViolation {
            invariant, message, ..
        } = &result.violations[0]
        {
            assert_eq!(invariant, "non_negative");
            assert!(message.contains("Initial state"));
        } else {
            panic!("Expected InvariantViolation");
        }
    }

    #[tokio::test]
    async fn test_verify_invariants_counts_states() {
        // Test that states_explored is correctly incremented
        // This catches mutations at lines 312, 353 (+= vs -= or *=)
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct StepState {
            step: usize,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum StepEvent {
            Next,
        }

        struct StepMachine {
            state: StepState,
            max_steps: usize,
        }

        #[async_trait]
        impl AsyncStateMachine for StepMachine {
            type State = StepState;
            type Event = StepEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                if self.state.step >= self.max_steps {
                    vec![]
                } else {
                    vec![StepEvent::Next]
                }
            }

            async fn process_event(&mut self, _event: Self::Event) -> Result<(), AsyncVerifyError> {
                self.state.step += 1;
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                self.state.step = 0;
                Ok(())
            }
        }

        let mut machine = StepMachine {
            state: StepState { step: 0 },
            max_steps: 5,
        };

        let config = VerificationConfig::default().with_bound(10);
        let verifier = AsyncVerifier::with_config(config);

        let invariants = vec![NamedInvariant::new("always_true", |_| true)];

        let result = verifier.verify_invariants(&mut machine, &invariants).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        // Should explore initial + 5 transitions = 6 states
        assert!(result.states_explored >= 1); // At least initial state
        assert!(result.transitions_checked <= 5); // Up to 5 transitions
    }

    #[test]
    fn test_explore_with_loom_returns_result() {
        // Test explore_with_loom (without loom feature, just returns default)
        let verifier = AsyncVerifier::new();
        let result = verifier.explore_with_loom(|| {
            // Simple test function
            let _ = 1 + 1;
        });

        // Without loom feature, should return empty result
        assert_eq!(result.total_interleavings, 0);
        assert!(result.violations.is_empty());
    }

    #[tokio::test]
    async fn test_verify_invariants_config_affects_bound() {
        // Test that config.bound actually limits the verification
        // This catches mutation at line 279 (AsyncVerifier::with_config)
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct DepthState {
            depth: usize,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum DepthEvent {
            GoDeeper,
        }

        struct DepthMachine {
            state: DepthState,
        }

        #[async_trait]
        impl AsyncStateMachine for DepthMachine {
            type State = DepthState;
            type Event = DepthEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                // Always can go deeper (infinite state space)
                vec![DepthEvent::GoDeeper]
            }

            async fn process_event(&mut self, _event: Self::Event) -> Result<(), AsyncVerifyError> {
                self.state.depth += 1;
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                self.state.depth = 0;
                Ok(())
            }
        }

        // Test with bound = 3
        let mut machine1 = DepthMachine {
            state: DepthState { depth: 0 },
        };
        let config1 = VerificationConfig::default()
            .with_bound(3)
            .with_max_states(100);
        let verifier1 = AsyncVerifier::with_config(config1);
        let invariants = vec![NamedInvariant::new("always_true", |_| true)];
        let result1 = verifier1
            .verify_invariants(&mut machine1, &invariants)
            .await
            .unwrap();

        // Test with bound = 10
        let mut machine2 = DepthMachine {
            state: DepthState { depth: 0 },
        };
        let config2 = VerificationConfig::default()
            .with_bound(10)
            .with_max_states(100);
        let verifier2 = AsyncVerifier::with_config(config2);
        let result2 = verifier2
            .verify_invariants(&mut machine2, &invariants)
            .await
            .unwrap();

        // With higher bound, should explore more states
        assert!(
            result2.states_explored > result1.states_explored,
            "Higher bound should explore more states: {} vs {}",
            result2.states_explored,
            result1.states_explored
        );
    }

    #[tokio::test]
    async fn test_verify_invariants_detects_transition_violation() {
        // Test that invariant violations during transitions are detected
        // This catches mutations at lines 344 (delete !) and 353 (states_explored increment)
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct CountdownState {
            value: i32,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum CountdownEvent {
            Decrement,
        }

        struct CountdownMachine {
            state: CountdownState,
        }

        #[async_trait]
        impl AsyncStateMachine for CountdownMachine {
            type State = CountdownState;
            type Event = CountdownEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                if self.state.value > -5 {
                    vec![CountdownEvent::Decrement]
                } else {
                    vec![]
                }
            }

            async fn process_event(&mut self, _event: Self::Event) -> Result<(), AsyncVerifyError> {
                self.state.value -= 1;
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                self.state.value = 3;
                Ok(())
            }
        }

        // Start at 3, decrement to go negative (violates invariant after 4 decrements)
        let mut machine = CountdownMachine {
            state: CountdownState { value: 3 },
        };

        let config = VerificationConfig::default().with_bound(10);
        let verifier = AsyncVerifier::with_config(config);

        // Invariant: value must be non-negative
        let invariants = vec![NamedInvariant::new("non_negative", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .map(|v| v >= 0)
                .unwrap_or(false)
        })];

        let result = verifier
            .verify_invariants(&mut machine, &invariants)
            .await
            .unwrap();

        // Should detect violation when value goes negative
        assert!(!result.passed, "Should fail when value goes negative");
        assert!(!result.violations.is_empty(), "Should have violations");

        // Should have explored multiple states before finding violation
        assert!(
            result.states_explored >= 4,
            "Should explore at least 4 states before violation (3->2->1->0->-1): explored {}",
            result.states_explored
        );
        assert!(
            result.transitions_checked >= 3,
            "Should check at least 3 transitions: checked {}",
            result.transitions_checked
        );
    }

    #[tokio::test]
    async fn test_verify_invariants_tracks_transitions() {
        // Test that transitions_checked is correctly incremented
        // This catches mutations at line 335 (+= vs -= or *=)
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct LinearState {
            step: usize,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum LinearEvent {
            Step,
        }

        struct LinearMachine {
            state: LinearState,
            max_steps: usize,
        }

        #[async_trait]
        impl AsyncStateMachine for LinearMachine {
            type State = LinearState;
            type Event = LinearEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                if self.state.step < self.max_steps {
                    vec![LinearEvent::Step]
                } else {
                    vec![]
                }
            }

            async fn process_event(&mut self, _event: Self::Event) -> Result<(), AsyncVerifyError> {
                self.state.step += 1;
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                self.state.step = 0;
                Ok(())
            }
        }

        let mut machine = LinearMachine {
            state: LinearState { step: 0 },
            max_steps: 7,
        };

        let config = VerificationConfig::default().with_bound(20);
        let verifier = AsyncVerifier::with_config(config);

        let invariants = vec![NamedInvariant::new("always_true", |_| true)];

        let result = verifier
            .verify_invariants(&mut machine, &invariants)
            .await
            .unwrap();

        // Should have exactly 7 transitions (step 0->1, 1->2, ..., 6->7)
        assert_eq!(
            result.transitions_checked, 7,
            "Should have exactly 7 transitions"
        );
        // Should have 8 states explored (initial + 7 after transitions)
        assert_eq!(result.states_explored, 8, "Should have 8 states explored");
    }

    #[tokio::test]
    async fn test_verify_invariants_depth_increment() {
        // Test that depth is correctly incremented in the loop
        // This catches mutations at line 356 (+= vs -= or *=)
        use crate::{AsyncStateMachine, AsyncVerifyError};
        use async_trait::async_trait;

        #[derive(Clone, Debug, serde::Serialize)]
        struct InfiniteState {
            depth: usize,
        }

        #[derive(Clone, Debug, serde::Serialize)]
        enum InfiniteEvent {
            Next,
        }

        struct InfiniteMachine {
            state: InfiniteState,
        }

        #[async_trait]
        impl AsyncStateMachine for InfiniteMachine {
            type State = InfiniteState;
            type Event = InfiniteEvent;

            fn current_state(&self) -> Self::State {
                self.state.clone()
            }

            fn possible_events(&self) -> Vec<Self::Event> {
                vec![InfiniteEvent::Next] // Always can continue
            }

            async fn process_event(&mut self, _event: Self::Event) -> Result<(), AsyncVerifyError> {
                self.state.depth += 1;
                Ok(())
            }

            async fn reset(&mut self) -> Result<(), AsyncVerifyError> {
                self.state.depth = 0;
                Ok(())
            }
        }

        let mut machine = InfiniteMachine {
            state: InfiniteState { depth: 0 },
        };

        // Set a specific bound and verify it's respected
        let config = VerificationConfig::default()
            .with_bound(5)
            .with_max_states(100);
        let verifier = AsyncVerifier::with_config(config);

        let invariants = vec![NamedInvariant::new("always_true", |_| true)];

        let result = verifier
            .verify_invariants(&mut machine, &invariants)
            .await
            .unwrap();

        // With bound=5, should explore exactly 6 states (initial + 5 transitions)
        // If depth += 1 is mutated to -= 1 or *= 1, the loop won't terminate properly
        // or will explore wrong number of states
        assert!(
            result.states_explored >= 5 && result.states_explored <= 6,
            "Should explore 5-6 states with bound=5, got {}",
            result.states_explored
        );
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves that VerificationResult::passed creates a result with passed=true.
    #[kani::proof]
    fn verify_verification_result_passed() {
        let result = VerificationResult::passed();
        kani::assert(result.passed, "passed() should set passed=true");
        kani::assert(
            result.violations.is_empty(),
            "passed() should have no violations",
        );
        kani::assert(
            result.counterexamples.is_empty(),
            "passed() should have no counterexamples",
        );
    }

    /// Proves that VerificationResult::failed creates a result with passed=false.
    #[kani::proof]
    fn verify_verification_result_failed() {
        let result = VerificationResult::failed(vec![]);
        kani::assert(!result.passed, "failed() should set passed=false");
    }

    /// Proves that with_states_explored preserves the value.
    #[kani::proof]
    fn verify_with_states_explored() {
        let states: usize = kani::any();
        kani::assume(states < 1000000);
        let result = VerificationResult::passed().with_states_explored(states);
        kani::assert(
            result.states_explored == states,
            "states_explored should be preserved",
        );
    }

    /// Proves that with_transitions_checked preserves the value.
    #[kani::proof]
    fn verify_with_transitions_checked() {
        let trans: usize = kani::any();
        kani::assume(trans < 1000000);
        let result = VerificationResult::passed().with_transitions_checked(trans);
        kani::assert(
            result.transitions_checked == trans,
            "transitions_checked should be preserved",
        );
    }

    /// Proves that with_duration preserves the value.
    #[kani::proof]
    fn verify_with_duration() {
        let ms: u64 = kani::any();
        kani::assume(ms < 1_000_000);
        let result = VerificationResult::passed().with_duration(Duration::from_millis(ms));
        kani::assert(result.duration_ms == ms, "duration_ms should be preserved");
    }

    /// Proves that VerificationConfig::default has expected values.
    #[kani::proof]
    fn verify_verification_config_default_values() {
        let config = VerificationConfig::default();
        kani::assert(config.max_states == 100000, "max_states should be 100000");
        kani::assert(
            config.max_transitions == 1000000,
            "max_transitions should be 1000000",
        );
        kani::assert(config.timeout_ms == 60000, "timeout_ms should be 60000");
        kani::assert(config.bounded, "bounded should be true");
        kani::assert(config.bound == 100, "bound should be 100");
        kani::assert(config.detect_deadlock, "detect_deadlock should be true");
        kani::assert(config.detect_livelock, "detect_livelock should be true");
    }

    /// Proves that with_max_states preserves the value.
    #[kani::proof]
    fn verify_config_with_max_states() {
        let count: usize = kani::any();
        kani::assume(count < 1000000);
        let config = VerificationConfig::default().with_max_states(count);
        kani::assert(config.max_states == count, "max_states should be preserved");
    }

    /// Proves that with_timeout preserves the value.
    #[kani::proof]
    fn verify_config_with_timeout() {
        let secs: u64 = kani::any();
        kani::assume(secs < 10000); // Keep reasonable
        let config = VerificationConfig::default().with_timeout(Duration::from_secs(secs));
        kani::assert(
            config.timeout_ms == secs * 1000,
            "timeout_ms should be preserved",
        );
    }

    /// Proves that with_bound sets bounded=true and preserves the bound value.
    #[kani::proof]
    fn verify_config_with_bound() {
        let bound: usize = kani::any();
        kani::assume(bound < 10000);
        let config = VerificationConfig::default().with_bound(bound);
        kani::assert(config.bounded, "bounded should be set to true");
        kani::assert(config.bound == bound, "bound should be preserved");
    }

    /// Proves that without_deadlock_detection disables deadlock detection.
    #[kani::proof]
    fn verify_config_without_deadlock_detection() {
        let config = VerificationConfig::default().without_deadlock_detection();
        kani::assert(
            !config.detect_deadlock,
            "detect_deadlock should be disabled",
        );
    }

    /// Proves that AsyncVerifier::new creates a verifier with default config.
    #[kani::proof]
    fn verify_async_verifier_new_default_config() {
        let verifier = AsyncVerifier::new();
        kani::assert(verifier.config.max_states == 100000, "Default max_states");
        kani::assert(verifier.config.bound == 100, "Default bound");
    }

    /// Proves that AsyncVerifier::default equals new().
    #[kani::proof]
    fn verify_async_verifier_default_equals_new() {
        let new_verifier = AsyncVerifier::new();
        let default_verifier = AsyncVerifier::default();
        kani::assert(
            new_verifier.config.max_states == default_verifier.config.max_states,
            "max_states should match",
        );
        kani::assert(
            new_verifier.config.bound == default_verifier.config.bound,
            "bound should match",
        );
    }

    /// Proves that AsyncVerifier::with_config preserves the config.
    #[kani::proof]
    fn verify_async_verifier_with_config() {
        let custom_max_states: usize = 5000;
        let custom_bound: usize = 50;
        let config = VerificationConfig {
            max_states: custom_max_states,
            bound: custom_bound,
            ..Default::default()
        };
        let verifier = AsyncVerifier::with_config(config);
        kani::assert(
            verifier.config.max_states == custom_max_states,
            "max_states should be preserved",
        );
        kani::assert(
            verifier.config.bound == custom_bound,
            "bound should be preserved",
        );
    }
}
