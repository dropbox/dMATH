//! Runtime monitor for checking invariants at runtime
//!
//! This module provides the `RuntimeMonitor` type which can be used to
//! check invariants at runtime as a program executes. It supports both
//! synchronous and asynchronous operation modes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::error::{MonitorError, MonitorResult};
use crate::trace::{MonitoredTrace, SourceLocation};
use crate::traceable::Traceable;

/// A compiled invariant that can be checked at runtime
pub struct CompiledInvariant {
    /// Invariant name
    pub name: String,
    /// The checker function
    pub check: Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>,
    /// Error message template
    pub error_message: String,
    /// Whether violations should halt execution
    pub halt_on_violation: bool,
    /// Priority (higher = checked first)
    pub priority: u32,
}

impl CompiledInvariant {
    /// Create a new compiled invariant
    pub fn new(
        name: impl Into<String>,
        check: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        let name = name.into();
        Self {
            error_message: format!("Invariant '{}' violated", name),
            name,
            check: Box::new(check),
            halt_on_violation: false,
            priority: 0,
        }
    }

    /// Set error message
    pub fn with_error_message(mut self, msg: impl Into<String>) -> Self {
        self.error_message = msg.into();
        self
    }

    /// Set halt on violation
    pub fn halt_on_violation(mut self) -> Self {
        self.halt_on_violation = true;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Check the invariant against a state
    pub fn check(&self, state: &serde_json::Value) -> bool {
        (self.check)(state)
    }
}

/// A violation detected by the runtime monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeViolation {
    /// Name of the violated invariant
    pub invariant_name: String,
    /// Error message
    pub message: String,
    /// State at time of violation
    pub state: serde_json::Value,
    /// When the violation was detected
    pub timestamp_ms: u64,
    /// Source location where violation was detected
    #[serde(default)]
    pub source_location: Option<SourceLocation>,
    /// Stack trace (if available)
    #[serde(default)]
    pub stack_trace: Option<String>,
}

impl RuntimeViolation {
    /// Create a new violation
    pub fn new(
        invariant_name: impl Into<String>,
        message: impl Into<String>,
        state: serde_json::Value,
    ) -> Self {
        Self {
            invariant_name: invariant_name.into(),
            message: message.into(),
            state,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            source_location: None,
            stack_trace: None,
        }
    }

    /// Set source location
    pub fn with_source_location(mut self, location: SourceLocation) -> Self {
        self.source_location = Some(location);
        self
    }
}

/// Statistics about monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitorStatistics {
    /// Total number of checks performed
    pub total_checks: u64,
    /// Number of violations detected
    pub violations_detected: u64,
    /// Total time spent checking (microseconds)
    pub total_check_time_us: u64,
    /// Per-invariant check counts
    pub per_invariant_checks: HashMap<String, u64>,
    /// Per-invariant violation counts
    pub per_invariant_violations: HashMap<String, u64>,
}

impl MonitorStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a check
    pub fn record_check(&mut self, invariant: &str, duration: Duration, violated: bool) {
        self.total_checks += 1;
        self.total_check_time_us += duration.as_micros() as u64;

        *self
            .per_invariant_checks
            .entry(invariant.to_string())
            .or_insert(0) += 1;

        if violated {
            self.violations_detected += 1;
            *self
                .per_invariant_violations
                .entry(invariant.to_string())
                .or_insert(0) += 1;
        }
    }

    /// Get average check time in microseconds
    pub fn average_check_time_us(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            self.total_check_time_us as f64 / self.total_checks as f64
        }
    }

    /// Get violation rate (0.0 to 1.0)
    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            self.violations_detected as f64 / self.total_checks as f64
        }
    }
}

/// Configuration for the runtime monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Whether monitoring is enabled
    pub enabled: bool,
    /// Maximum violations before stopping
    pub max_violations: Option<u64>,
    /// Whether to record state on each check
    pub record_states: bool,
    /// Sampling rate (1.0 = check every time, 0.5 = check half the time)
    pub sampling_rate: f64,
    /// Whether to collect statistics
    pub collect_statistics: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_violations: None,
            record_states: true,
            sampling_rate: 1.0,
            collect_statistics: true,
        }
    }
}

impl MonitorConfig {
    /// Disable monitoring
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set max violations
    pub fn with_max_violations(mut self, max: u64) -> Self {
        self.max_violations = Some(max);
        self
    }

    /// Set sampling rate
    pub fn with_sampling_rate(mut self, rate: f64) -> Self {
        self.sampling_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Disable state recording (for performance)
    pub fn without_state_recording(mut self) -> Self {
        self.record_states = false;
        self
    }
}

/// Runtime monitor that checks invariants during program execution
pub struct RuntimeMonitor {
    config: MonitorConfig,
    invariants: Vec<CompiledInvariant>,
    violations: Arc<RwLock<Vec<RuntimeViolation>>>,
    statistics: Arc<RwLock<MonitorStatistics>>,
    trace: Option<Arc<RwLock<MonitoredTrace>>>,
}

impl RuntimeMonitor {
    /// Create a new runtime monitor
    pub fn new() -> Self {
        Self {
            config: MonitorConfig::default(),
            invariants: vec![],
            violations: Arc::new(RwLock::new(vec![])),
            statistics: Arc::new(RwLock::new(MonitorStatistics::new())),
            trace: None,
        }
    }

    /// Create with custom config
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            config,
            invariants: vec![],
            violations: Arc::new(RwLock::new(vec![])),
            statistics: Arc::new(RwLock::new(MonitorStatistics::new())),
            trace: None,
        }
    }

    /// Add an invariant
    pub fn add_invariant(&mut self, invariant: CompiledInvariant) {
        self.invariants.push(invariant);
        // Sort by priority (highest first)
        self.invariants.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Add a simple invariant from a closure
    pub fn add_simple_invariant(
        &mut self,
        name: impl Into<String>,
        check: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) {
        self.add_invariant(CompiledInvariant::new(name, check));
    }

    /// Enable trace recording
    pub fn enable_tracing(&mut self, name: impl Into<String>, initial_state: serde_json::Value) {
        self.trace = Some(Arc::new(RwLock::new(MonitoredTrace::new(
            name,
            initial_state,
        ))));
    }

    /// Check all invariants against a state
    pub fn check(&self, state: &serde_json::Value) -> MonitorResult<Vec<RuntimeViolation>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        // Sampling
        if self.config.sampling_rate < 1.0 {
            let random: f64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| ((d.as_nanos() % 1000) as f64) / 1000.0)
                .unwrap_or(0.0);
            if random > self.config.sampling_rate {
                return Ok(vec![]);
            }
        }

        let mut new_violations = vec![];

        for invariant in &self.invariants {
            let start = Instant::now();
            let passed = invariant.check(state);
            let duration = start.elapsed();

            // Record statistics
            if self.config.collect_statistics {
                if let Ok(mut stats) = self.statistics.write() {
                    stats.record_check(&invariant.name, duration, !passed);
                }
            }

            if !passed {
                let violation = RuntimeViolation::new(
                    &invariant.name,
                    &invariant.error_message,
                    if self.config.record_states {
                        state.clone()
                    } else {
                        serde_json::Value::Null
                    },
                );

                new_violations.push(violation.clone());

                // Store violation
                if let Ok(mut violations) = self.violations.write() {
                    violations.push(violation);

                    // Check max violations
                    if let Some(max) = self.config.max_violations {
                        if violations.len() as u64 >= max {
                            return Err(MonitorError::invariant_evaluation(format!(
                                "Maximum violations ({}) exceeded",
                                max
                            )));
                        }
                    }
                }

                if invariant.halt_on_violation {
                    return Err(MonitorError::invariant_evaluation(format!(
                        "Invariant '{}' violated (halt_on_violation set)",
                        invariant.name
                    )));
                }
            }
        }

        Ok(new_violations)
    }

    /// Check invariants on a traceable object
    pub fn check_traceable<T: Traceable>(
        &self,
        traceable: &T,
    ) -> MonitorResult<Vec<RuntimeViolation>> {
        let state = traceable.capture_state();
        self.check(&state)
    }

    /// Record a state transition and check invariants
    pub fn record_and_check(
        &self,
        action: impl Into<String>,
        from_state: serde_json::Value,
        to_state: serde_json::Value,
    ) -> MonitorResult<Vec<RuntimeViolation>> {
        // Record transition
        if let Some(ref trace) = self.trace {
            if let Ok(mut trace) = trace.write() {
                trace.record_transition(action, from_state, to_state.clone());
            }
        }

        // Check invariants on new state
        self.check(&to_state)
    }

    /// Get all violations
    pub fn violations(&self) -> Vec<RuntimeViolation> {
        self.violations
            .read()
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get violation count
    pub fn violation_count(&self) -> usize {
        self.violations.read().map(|v| v.len()).unwrap_or(0)
    }

    /// Check if any violations have occurred
    pub fn has_violations(&self) -> bool {
        self.violation_count() > 0
    }

    /// Clear all recorded violations
    pub fn clear_violations(&self) {
        if let Ok(mut violations) = self.violations.write() {
            violations.clear();
        }
    }

    /// Get statistics
    pub fn statistics(&self) -> MonitorStatistics {
        self.statistics
            .read()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Get the recorded trace (if tracing is enabled)
    pub fn trace(&self) -> Option<MonitoredTrace> {
        self.trace
            .as_ref()
            .and_then(|t| t.read().ok().map(|t| t.clone()))
    }

    /// Complete tracing and return the trace
    pub fn complete_trace(&self) -> Option<MonitoredTrace> {
        self.trace.as_ref().and_then(|t| {
            t.write().ok().map(|mut trace| {
                trace.complete();
                trace.clone()
            })
        })
    }

    /// Get number of invariants
    pub fn invariant_count(&self) -> usize {
        self.invariants.len()
    }

    /// Get invariant names
    pub fn invariant_names(&self) -> Vec<&str> {
        self.invariants.iter().map(|i| i.name.as_str()).collect()
    }

    /// Check if monitoring is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enable monitoring
    pub fn enable(&mut self) {
        self.config.enabled = true;
    }

    /// Disable monitoring
    pub fn disable(&mut self) {
        self.config.enabled = false;
    }
}

impl Default for RuntimeMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// A scoped monitor that automatically checks on drop
pub struct ScopedMonitor<'a, T: Traceable> {
    monitor: &'a RuntimeMonitor,
    traceable: &'a T,
    action: String,
    initial_state: serde_json::Value,
}

impl<'a, T: Traceable> ScopedMonitor<'a, T> {
    /// Create a new scoped monitor
    pub fn new(monitor: &'a RuntimeMonitor, traceable: &'a T, action: impl Into<String>) -> Self {
        Self {
            monitor,
            traceable,
            action: action.into(),
            initial_state: traceable.capture_state(),
        }
    }
}

impl<'a, T: Traceable> Drop for ScopedMonitor<'a, T> {
    fn drop(&mut self) {
        let final_state = self.traceable.capture_state();
        let _ =
            self.monitor
                .record_and_check(&self.action, self.initial_state.clone(), final_state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_invariant() {
        let inv = CompiledInvariant::new("positive", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n >= 0)
        })
        .with_priority(10)
        .with_error_message("Value must be non-negative");

        assert_eq!(inv.name, "positive");
        assert_eq!(inv.priority, 10);
        assert!(inv.check(&serde_json::json!({"value": 5})));
        assert!(!inv.check(&serde_json::json!({"value": -1})));
    }

    #[test]
    fn test_runtime_violation() {
        let violation =
            RuntimeViolation::new("test_inv", "Test failed", serde_json::json!({"x": 1}));

        assert_eq!(violation.invariant_name, "test_inv");
        assert_eq!(violation.message, "Test failed");
    }

    #[test]
    fn test_monitor_statistics() {
        let mut stats = MonitorStatistics::new();

        stats.record_check("inv1", Duration::from_micros(100), false);
        stats.record_check("inv1", Duration::from_micros(200), true);
        stats.record_check("inv2", Duration::from_micros(50), false);

        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.violations_detected, 1);
        assert_eq!(stats.total_check_time_us, 350);
        assert_eq!(stats.per_invariant_checks.get("inv1"), Some(&2));
        assert_eq!(stats.per_invariant_violations.get("inv1"), Some(&1));
    }

    #[test]
    fn test_runtime_monitor_basic() {
        let mut monitor = RuntimeMonitor::new();

        monitor.add_simple_invariant("positive", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n >= 0)
        });

        // Should pass
        let violations = monitor.check(&serde_json::json!({"value": 10})).unwrap();
        assert!(violations.is_empty());

        // Should fail
        let violations = monitor.check(&serde_json::json!({"value": -5})).unwrap();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].invariant_name, "positive");
    }

    #[test]
    fn test_runtime_monitor_multiple_invariants() {
        let mut monitor = RuntimeMonitor::new();

        monitor.add_simple_invariant("positive", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n >= 0)
        });

        monitor.add_simple_invariant("bounded", |state| {
            state
                .get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n <= 100)
        });

        // Both pass
        let violations = monitor.check(&serde_json::json!({"value": 50})).unwrap();
        assert!(violations.is_empty());

        // One fails
        let violations = monitor.check(&serde_json::json!({"value": 150})).unwrap();
        assert_eq!(violations.len(), 1);

        // Both fail
        let violations = monitor.check(&serde_json::json!({"value": -1})).unwrap();
        // Only positive fails here since -1 < 100
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_runtime_monitor_disabled() {
        let mut monitor = RuntimeMonitor::with_config(MonitorConfig::disabled());

        monitor.add_simple_invariant("always_fails", |_| false);

        // Should not check when disabled
        let violations = monitor.check(&serde_json::json!({})).unwrap();
        assert!(violations.is_empty());

        // Enable and check again
        monitor.enable();
        let violations = monitor.check(&serde_json::json!({})).unwrap();
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_runtime_monitor_statistics() {
        let mut monitor = RuntimeMonitor::new();

        monitor.add_simple_invariant("test", |state| {
            state.get("pass").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        monitor.check(&serde_json::json!({"pass": true})).unwrap();
        monitor.check(&serde_json::json!({"pass": false})).unwrap();
        monitor.check(&serde_json::json!({"pass": true})).unwrap();

        let stats = monitor.statistics();
        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.violations_detected, 1);
    }

    #[test]
    fn test_runtime_monitor_tracing() {
        let mut monitor = RuntimeMonitor::new();
        monitor.enable_tracing("test_trace", serde_json::json!({"x": 0}));

        monitor
            .record_and_check(
                "increment",
                serde_json::json!({"x": 0}),
                serde_json::json!({"x": 1}),
            )
            .unwrap();

        let trace = monitor.complete_trace().unwrap();
        assert!(trace.is_complete());
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_runtime_monitor_halt_on_violation() {
        let mut monitor = RuntimeMonitor::new();

        monitor.add_invariant(CompiledInvariant::new("critical", |_| false).halt_on_violation());

        let result = monitor.check(&serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_monitor_max_violations() {
        let config = MonitorConfig::default().with_max_violations(2);
        let mut monitor = RuntimeMonitor::with_config(config);

        monitor.add_simple_invariant("always_fails", |_| false);

        // First two should succeed (accumulate violations)
        assert!(monitor.check(&serde_json::json!({})).is_ok());

        // Third should fail because we hit max
        let result = monitor.check(&serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_monitor_clear_violations() {
        let mut monitor = RuntimeMonitor::new();
        monitor.add_simple_invariant("always_fails", |_| false);

        monitor.check(&serde_json::json!({})).unwrap();
        assert!(monitor.has_violations());

        monitor.clear_violations();
        assert!(!monitor.has_violations());
    }

    #[test]
    fn test_monitor_config() {
        let config = MonitorConfig::default()
            .with_max_violations(10)
            .with_sampling_rate(0.5)
            .without_state_recording();

        assert_eq!(config.max_violations, Some(10));
        assert!((config.sampling_rate - 0.5).abs() < f64::EPSILON);
        assert!(!config.record_states);
    }

    #[test]
    fn test_invariant_priority() {
        let mut monitor = RuntimeMonitor::new();

        monitor.add_invariant(CompiledInvariant::new("low", |_| true).with_priority(1));
        monitor.add_invariant(CompiledInvariant::new("high", |_| true).with_priority(10));
        monitor.add_invariant(CompiledInvariant::new("medium", |_| true).with_priority(5));

        // Should be sorted by priority (highest first)
        let names = monitor.invariant_names();
        assert_eq!(names, vec!["high", "medium", "low"]);
    }

    // Mutation-killing tests for MonitorStatistics::average_check_time_us (lines 157-162)
    #[test]
    fn test_monitor_statistics_average_check_time() {
        let mut stats = MonitorStatistics::new();
        // Empty stats should return 0.0
        assert!((stats.average_check_time_us() - 0.0).abs() < f64::EPSILON);

        // Add some checks
        stats.record_check("inv1", Duration::from_micros(100), false);
        stats.record_check("inv1", Duration::from_micros(200), false);

        // Average should be (100 + 200) / 2 = 150
        assert!((stats.average_check_time_us() - 150.0).abs() < f64::EPSILON);
    }

    // Mutation-killing tests for MonitorStatistics::violation_rate (lines 166-171)
    #[test]
    fn test_monitor_statistics_violation_rate() {
        let mut stats = MonitorStatistics::new();
        // Empty stats should return 0.0
        assert!((stats.violation_rate() - 0.0).abs() < f64::EPSILON);

        // 1 violation out of 4 checks = 0.25
        stats.record_check("inv1", Duration::from_micros(10), false);
        stats.record_check("inv1", Duration::from_micros(10), false);
        stats.record_check("inv1", Duration::from_micros(10), true); // violation
        stats.record_check("inv1", Duration::from_micros(10), false);

        assert!((stats.violation_rate() - 0.25).abs() < f64::EPSILON);
    }

    // Mutation-killing tests for RuntimeMonitor::check sampling (lines 293-300)
    #[test]
    fn test_runtime_monitor_full_sampling() {
        let config = MonitorConfig::default().with_sampling_rate(1.0);
        let mut monitor = RuntimeMonitor::with_config(config);

        monitor.add_simple_invariant("always_fails", |_| false);

        // With 1.0 sampling rate, should always check and always get violation
        let violations = monitor.check(&serde_json::json!({})).unwrap();
        assert_eq!(violations.len(), 1);
    }

    // Mutation-killing tests for RuntimeMonitor::violations returning vec![] (line 386)
    #[test]
    fn test_runtime_monitor_violations_returns_correct_data() {
        let mut monitor = RuntimeMonitor::new();
        monitor.add_simple_invariant("test_inv", |_| false);

        monitor.check(&serde_json::json!({})).unwrap();

        let violations = monitor.violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].invariant_name, "test_inv");
    }

    // Mutation-killing tests for RuntimeMonitor::trace returning None (line 419)
    #[test]
    fn test_runtime_monitor_trace_returns_some() {
        let mut monitor = RuntimeMonitor::new();
        monitor.enable_tracing("test", serde_json::json!({"x": 0}));

        monitor
            .record_and_check(
                "action",
                serde_json::json!({"x": 0}),
                serde_json::json!({"x": 1}),
            )
            .unwrap();

        let trace = monitor.trace();
        assert!(trace.is_some());
        let trace = trace.unwrap();
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_runtime_monitor_trace_returns_none_when_not_enabled() {
        let monitor = RuntimeMonitor::new();
        assert!(monitor.trace().is_none());
    }

    // Mutation-killing tests for RuntimeMonitor::invariant_count (lines 435-437)
    #[test]
    fn test_runtime_monitor_invariant_count() {
        let mut monitor = RuntimeMonitor::new();
        assert_eq!(monitor.invariant_count(), 0);

        monitor.add_simple_invariant("inv1", |_| true);
        assert_eq!(monitor.invariant_count(), 1);

        monitor.add_simple_invariant("inv2", |_| true);
        assert_eq!(monitor.invariant_count(), 2);
    }

    // Mutation-killing tests for RuntimeMonitor::is_enabled (lines 445-447)
    #[test]
    fn test_runtime_monitor_is_enabled() {
        let mut monitor = RuntimeMonitor::new();
        assert!(monitor.is_enabled());

        monitor.disable();
        assert!(!monitor.is_enabled());

        monitor.enable();
        assert!(monitor.is_enabled());
    }

    // Mutation-killing tests for RuntimeMonitor::disable (lines 455-457)
    #[test]
    fn test_runtime_monitor_disable_actually_disables() {
        let mut monitor = RuntimeMonitor::new();
        monitor.add_simple_invariant("always_fails", |_| false);

        // Should get violations when enabled
        let violations_enabled = monitor.check(&serde_json::json!({})).unwrap();
        assert!(!violations_enabled.is_empty());

        // After disabling, should get no violations
        monitor.disable();
        let violations_disabled = monitor.check(&serde_json::json!({})).unwrap();
        assert!(violations_disabled.is_empty());
    }

    // Mutation-killing tests for ScopedMonitor::drop (line 488)
    #[test]
    fn test_scoped_monitor_records_on_drop() {
        use std::sync::atomic::{AtomicI32, Ordering};

        #[derive(Debug)]
        struct AtomicCounter {
            value: AtomicI32,
        }

        impl crate::traceable::Traceable for AtomicCounter {
            fn trace_name(&self) -> &str {
                "AtomicCounter"
            }

            fn capture_state(&self) -> serde_json::Value {
                serde_json::json!({"value": self.value.load(Ordering::SeqCst)})
            }
        }

        let mut monitor = RuntimeMonitor::new();
        monitor.enable_tracing("test", serde_json::json!({"value": 0}));

        let counter = AtomicCounter {
            value: AtomicI32::new(0),
        };

        {
            let _guard = ScopedMonitor::new(&monitor, &counter, "increment");
            counter.value.store(1, Ordering::SeqCst);
            // ScopedMonitor will record on drop here
        }

        let trace = monitor.trace().unwrap();
        assert_eq!(trace.len(), 1);
    }

    // Test that complete_trace actually completes and returns trace
    #[test]
    fn test_complete_trace() {
        let mut monitor = RuntimeMonitor::new();
        monitor.enable_tracing("test", serde_json::json!({}));

        let trace = monitor.complete_trace();
        assert!(trace.is_some());
        assert!(trace.unwrap().is_complete());
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves that MonitorConfig::default has expected values.
    #[kani::proof]
    fn verify_monitor_config_default_values() {
        let config = MonitorConfig::default();
        kani::assert(config.enabled, "enabled should be true by default");
        kani::assert(
            config.max_violations.is_none(),
            "max_violations should be None by default",
        );
        kani::assert(
            config.record_states,
            "record_states should be true by default",
        );
        kani::assert(
            config.collect_statistics,
            "collect_statistics should be true by default",
        );
        kani::assert(
            (config.sampling_rate - 1.0).abs() < f64::EPSILON,
            "sampling_rate should be 1.0",
        );
    }

    /// Proves that MonitorConfig::disabled disables monitoring.
    #[kani::proof]
    fn verify_monitor_config_disabled() {
        let config = MonitorConfig::disabled();
        kani::assert(!config.enabled, "disabled() should set enabled=false");
    }

    /// Proves that with_max_violations preserves the value.
    #[kani::proof]
    fn verify_config_with_max_violations() {
        let max: u64 = kani::any();
        kani::assume(max < 1000000);
        let config = MonitorConfig::default().with_max_violations(max);
        kani::assert(
            config.max_violations == Some(max),
            "max_violations should be preserved",
        );
    }

    /// Proves that with_sampling_rate clamps to [0.0, 1.0].
    #[kani::proof]
    fn verify_config_sampling_rate_clamped() {
        let rate: f64 = kani::any();
        kani::assume(!rate.is_nan());
        let config = MonitorConfig::default().with_sampling_rate(rate);
        kani::assert(
            config.sampling_rate >= 0.0,
            "sampling_rate should be >= 0.0",
        );
        kani::assert(
            config.sampling_rate <= 1.0,
            "sampling_rate should be <= 1.0",
        );
    }

    /// Proves that without_state_recording disables state recording.
    #[kani::proof]
    fn verify_config_without_state_recording() {
        let config = MonitorConfig::default().without_state_recording();
        kani::assert(!config.record_states, "record_states should be false");
    }

    /// Proves that MonitorStatistics::new creates empty statistics.
    #[kani::proof]
    fn verify_monitor_statistics_new_empty() {
        let stats = MonitorStatistics::new();
        kani::assert(stats.total_checks == 0, "total_checks should be 0");
        kani::assert(
            stats.violations_detected == 0,
            "violations_detected should be 0",
        );
        kani::assert(
            stats.total_check_time_us == 0,
            "total_check_time_us should be 0",
        );
    }

    /// Proves that average_check_time_us returns 0.0 for empty stats.
    #[kani::proof]
    fn verify_statistics_average_check_time_empty() {
        let stats = MonitorStatistics::new();
        let avg = stats.average_check_time_us();
        kani::assert(avg == 0.0, "average should be 0.0 for empty stats");
    }

    /// Proves that violation_rate returns 0.0 for empty stats.
    #[kani::proof]
    fn verify_statistics_violation_rate_empty() {
        let stats = MonitorStatistics::new();
        let rate = stats.violation_rate();
        kani::assert(rate == 0.0, "violation_rate should be 0.0 for empty stats");
    }

    /// Proves that CompiledInvariant::new preserves the name.
    #[kani::proof]
    fn verify_compiled_invariant_name_preserved() {
        let inv = CompiledInvariant::new("test_invariant", |_| true);
        kani::assert(inv.name == "test_invariant", "name should be preserved");
        kani::assert(
            !inv.halt_on_violation,
            "halt_on_violation should be false by default",
        );
        kani::assert(inv.priority == 0, "priority should be 0 by default");
    }

    /// Proves that halt_on_violation sets the flag.
    #[kani::proof]
    fn verify_compiled_invariant_halt_on_violation() {
        let inv = CompiledInvariant::new("test", |_| true).halt_on_violation();
        kani::assert(inv.halt_on_violation, "halt_on_violation should be true");
    }

    /// Proves that with_priority preserves the value.
    #[kani::proof]
    fn verify_compiled_invariant_with_priority() {
        let priority: u32 = kani::any();
        kani::assume(priority < 1000);
        let inv = CompiledInvariant::new("test", |_| true).with_priority(priority);
        kani::assert(inv.priority == priority, "priority should be preserved");
    }

    /// Proves that RuntimeMonitor::new creates a monitor with default config.
    #[kani::proof]
    fn verify_runtime_monitor_new() {
        let monitor = RuntimeMonitor::new();
        kani::assert(
            monitor.invariant_count() == 0,
            "Should have no invariants initially",
        );
        kani::assert(monitor.is_enabled(), "Should be enabled by default");
        kani::assert(
            !monitor.has_violations(),
            "Should have no violations initially",
        );
    }

    /// Proves that RuntimeMonitor::default equals new().
    #[kani::proof]
    fn verify_runtime_monitor_default_equals_new() {
        let new_monitor = RuntimeMonitor::new();
        let default_monitor = RuntimeMonitor::default();
        kani::assert(
            new_monitor.invariant_count() == default_monitor.invariant_count(),
            "invariant_count should match",
        );
        kani::assert(
            new_monitor.is_enabled() == default_monitor.is_enabled(),
            "is_enabled should match",
        );
    }

    /// Proves that disable sets enabled to false.
    #[kani::proof]
    fn verify_runtime_monitor_disable() {
        let mut monitor = RuntimeMonitor::new();
        kani::assert(monitor.is_enabled(), "Should be enabled initially");
        monitor.disable();
        kani::assert(!monitor.is_enabled(), "Should be disabled after disable()");
    }

    /// Proves that enable sets enabled to true.
    #[kani::proof]
    fn verify_runtime_monitor_enable() {
        let mut monitor = RuntimeMonitor::with_config(MonitorConfig::disabled());
        kani::assert(!monitor.is_enabled(), "Should be disabled initially");
        monitor.enable();
        kani::assert(monitor.is_enabled(), "Should be enabled after enable()");
    }
}
