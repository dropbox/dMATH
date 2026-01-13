//! Integration tests for the Monitored derive macro

use dashprove_monitor::{MonitoredType, Traceable};
use dashprove_monitor_macros::Monitored;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
struct SimpleCounter {
    value: i32,
}

#[test]
fn test_simple_counter_traceable() {
    let counter = SimpleCounter { value: 42 };

    assert_eq!(counter.trace_name(), "SimpleCounter");

    let state = counter.capture_state();
    assert_eq!(state.get("value").and_then(|v| v.as_i64()), Some(42));

    let vars = counter.tracked_variables();
    assert_eq!(vars, vec!["value"]);
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
#[monitored(name = "MyCustomCounter")]
struct CounterWithCustomName {
    count: i64,
}

#[test]
fn test_custom_name() {
    let counter = CounterWithCustomName { count: 100 };
    assert_eq!(counter.trace_name(), "MyCustomCounter");
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
struct CounterWithSkip {
    #[monitored(track)]
    public_value: i32,

    #[monitored(skip)]
    internal_cache: String,
}

#[test]
fn test_skip_field() {
    let counter = CounterWithSkip {
        public_value: 10,
        internal_cache: "cached".to_string(),
    };

    let state = counter.capture_state();
    assert!(state.get("public_value").is_some());
    assert!(state.get("internal_cache").is_none());

    let vars = counter.tracked_variables();
    assert_eq!(vars, vec!["public_value"]);
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
struct CounterWithRename {
    #[monitored(rename = "total")]
    sum: i32,
}

#[test]
fn test_rename_field() {
    let counter = CounterWithRename { sum: 50 };

    let state = counter.capture_state();
    assert!(state.get("sum").is_none());
    assert_eq!(state.get("total").and_then(|v| v.as_i64()), Some(50));

    let vars = counter.tracked_variables();
    assert_eq!(vars, vec!["total"]);
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
#[monitored(invariant = "value >= 0", name = "non_negative")]
struct ConstrainedCounter {
    value: i32,
}

#[test]
fn test_invariant() {
    let counter = ConstrainedCounter { value: 10 };

    // Check MonitoredType trait
    let invariant_names = counter.invariant_names();
    assert_eq!(invariant_names, vec!["non_negative"]);

    // Check invariant passes
    let violations = counter.check_invariants();
    assert!(violations.is_empty());

    // Check invariant fails
    let bad_counter = ConstrainedCounter { value: -5 };
    let violations = bad_counter.check_invariants();
    assert_eq!(violations, vec!["non_negative"]);
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
#[monitored(invariant = "value >= 0", name = "non_negative")]
#[monitored(invariant = "value <= max", name = "bounded")]
struct BoundedCounter {
    value: i32,
    max: i32,
}

#[test]
fn test_multiple_invariants() {
    let counter = BoundedCounter {
        value: 50,
        max: 100,
    };

    let invariant_names = counter.invariant_names();
    assert_eq!(invariant_names, vec!["non_negative", "bounded"]);

    // All pass
    let violations = counter.check_invariants();
    assert!(violations.is_empty());

    // bounded fails
    let over_counter = BoundedCounter {
        value: 150,
        max: 100,
    };
    let violations = over_counter.check_invariants();
    assert_eq!(violations, vec!["bounded"]);

    // non_negative fails
    let neg_counter = BoundedCounter {
        value: -10,
        max: 100,
    };
    let violations = neg_counter.check_invariants();
    assert_eq!(violations, vec!["non_negative"]);
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
struct MultiFieldState {
    x: i32,
    y: i32,
    label: String,
}

#[test]
fn test_multi_field_capture() {
    let state = MultiFieldState {
        x: 10,
        y: 20,
        label: "point".to_string(),
    };

    let captured = state.capture_state();
    assert_eq!(captured.get("x").and_then(|v| v.as_i64()), Some(10));
    assert_eq!(captured.get("y").and_then(|v| v.as_i64()), Some(20));
    assert_eq!(
        captured.get("label").and_then(|v| v.as_str()),
        Some("point")
    );
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
struct UnitStruct;

#[test]
fn test_unit_struct() {
    let s = UnitStruct;
    assert_eq!(s.trace_name(), "UnitStruct");
    let state = s.capture_state();
    assert!(state.is_object());
    assert!(state.as_object().unwrap().is_empty());
}

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
struct TupleStruct(i32, String);

#[test]
fn test_tuple_struct() {
    let s = TupleStruct(42, "hello".to_string());
    assert_eq!(s.trace_name(), "TupleStruct");

    let state = s.capture_state();
    assert_eq!(state.get("field_0").and_then(|v| v.as_i64()), Some(42));
    assert_eq!(state.get("field_1").and_then(|v| v.as_str()), Some("hello"));
}

// Integration test with RuntimeMonitor
use dashprove_monitor::RuntimeMonitor;

#[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
#[monitored(name = "Account")]
struct Account {
    balance: i64,
    #[monitored(skip)]
    account_number: String,
}

#[test]
fn test_integration_with_runtime_monitor() {
    let account = Account {
        balance: 1000,
        account_number: "12345".to_string(),
    };

    // Create a monitor and add invariants
    let mut monitor = RuntimeMonitor::new();
    monitor.add_simple_invariant("positive_balance", |state| {
        state
            .get("balance")
            .and_then(|v| v.as_i64())
            .is_some_and(|b| b >= 0)
    });

    // Check via Traceable trait
    let violations = monitor.check_traceable(&account).unwrap();
    assert!(violations.is_empty());

    // Verify account_number is not in state (skipped)
    let state = account.capture_state();
    assert!(state.get("account_number").is_none());
    assert!(state.get("balance").is_some());

    // Check with negative balance
    let overdraft = Account {
        balance: -100,
        account_number: "12345".to_string(),
    };
    let violations = monitor.check_traceable(&overdraft).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].invariant_name, "positive_balance");
}

#[test]
fn test_snapshot() {
    let counter = SimpleCounter { value: 42 };
    let snapshot = counter.snapshot();

    assert_eq!(
        snapshot.state.get("value").and_then(|v| v.as_i64()),
        Some(42)
    );
}

#[test]
fn test_snapshot_with_label() {
    let counter = SimpleCounter { value: 100 };
    let snapshot = counter.snapshot_with_label("initial_state");

    assert_eq!(snapshot.label, Some("initial_state".to_string()));
}
