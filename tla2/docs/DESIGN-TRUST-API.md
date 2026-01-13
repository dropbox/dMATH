# TLA2 Programmatic API Design for tRust Integration

**Created**: 2025-12-31
**Status**: Design Phase
**Depends On**: Phase 1 (Correctness) completion

---

## Overview

This document specifies the programmatic API that TLA2 will expose for tRust integration. tRust is a verified Rust compiler that dispatches temporal verification conditions to TLA2.

**Key Principle**: TLA2 is the verification engine. tRust handles MIR extraction and source mapping.

---

## API Design

### Core Types

```rust
// crates/tla-wire/src/lib.rs - Wire format for external callers

use std::time::Duration;

/// Unique identifier for tracking VCs and actions
pub type VCId = String;
pub type ActionId = String;

/// Verification condition for temporal properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVC {
    /// Unique identifier for this VC (for caching, tracking)
    pub id: VCId,

    /// State machine extracted from async code
    pub state_machine: StateMachine,

    /// Temporal properties to verify
    pub properties: Vec<TemporalProperty>,

    /// Fairness constraints
    pub fairness: Vec<Fairness>,

    /// Failure model (optional)
    pub fault_model: Option<FaultModel>,

    /// Bounds for model checking
    pub bounds: Bounds,
}

/// State machine representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    /// State variables
    pub variables: Vec<Variable>,

    /// Initial state predicate
    pub init: Predicate,

    /// Next-state transitions
    pub next: Vec<Transition>,
}

/// A single transition in the state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Human-readable name
    pub name: String,

    /// Guard condition (when is this transition enabled?)
    pub guard: Predicate,

    /// Variable assignments (what changes?)
    pub assignments: Vec<Assignment>,

    /// Caller-provided ID for source mapping back to Rust
    pub action_id: ActionId,
}

/// Variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub typ: Type,
}

/// Assignment: variable' = expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub variable: String,
    pub expr: Expr,
}

/// Types supported in state machines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Type {
    Bool,
    Int,
    String,
    Set(Box<Type>),
    Seq(Box<Type>),
    Record(Vec<(String, Type)>),
    Enum(Vec<String>),
}

/// Expression in TLA2 format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Var(String),
    Const(Value),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnOp(UnOp, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Set(Vec<Expr>),
    Seq(Vec<Expr>),
    Record(Vec<(String, Expr)>),
    FuncApply(Box<Expr>, Box<Expr>),
    Quantifier(Quantifier, String, Box<Expr>, Box<Expr>),
}

/// Temporal properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalProperty {
    /// []P - P holds in all states
    Always(Predicate),

    /// <>P - P holds eventually
    Eventually(Predicate),

    /// P ~> Q - P leads to Q
    LeadsTo(Predicate, Predicate),

    /// P U Q - P until Q
    Until(Predicate, Predicate),

    /// Weak until
    WeakUntil(Predicate, Predicate),
}

/// Predicate is just an expression that evaluates to Bool
pub type Predicate = Expr;

/// Fairness constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Fairness {
    /// No fairness - stuttering allowed
    None,

    /// WF_vars(Action) - if continuously enabled, eventually happens
    Weak(ActionId),

    /// SF_vars(Action) - if repeatedly enabled, eventually happens
    Strong(ActionId),
}

/// Failure model for distributed systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultModel {
    /// Max simultaneous node crashes (fail-stop)
    pub node_crashes: Option<u32>,

    /// Allow network partitions
    pub network_partitions: bool,

    /// Message loss probability (0.0-1.0)
    pub message_loss: Option<f64>,

    /// Allow message reordering
    pub message_reorder: bool,

    /// Allow message duplication
    pub message_duplicate: bool,
}

impl Default for FaultModel {
    fn default() -> Self {
        FaultModel {
            node_crashes: None,
            network_partitions: false,
            message_loss: None,
            message_reorder: false,
            message_duplicate: false,
        }
    }
}

/// Bounds for model checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bounds {
    pub max_states: usize,
    pub max_depth: usize,
    pub timeout: Duration,
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            max_states: 1_000_000,
            max_depth: 100,
            timeout: Duration::from_secs(60),
        }
    }
}
```

### Result Types

```rust
/// Result of model checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TLA2Result {
    /// Property verified for all reachable states
    Verified {
        states_explored: u64,
        distinct_states: u64,
        time: Duration,
        max_depth: u32,
    },

    /// Property violated - counterexample found
    Violation {
        property: String,
        trace: CounterexampleTrace,
        time: Duration,
    },

    /// Timeout before completion
    Timeout {
        states_explored: u64,
        coverage_estimate: f64,  // 0.0-1.0
        suggestion: String,
    },

    /// Model error (not a property violation)
    Error {
        message: String,
        location: Option<ActionId>,
    },
}

/// Counterexample trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleTrace {
    /// Sequence of states leading to violation
    pub states: Vec<State>,

    /// Actions taken between states (for source mapping)
    pub actions: Vec<ActionId>,

    /// For liveness: prefix + lasso structure
    pub lasso: Option<LassoInfo>,
}

/// State in the trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub variables: Vec<(String, Value)>,
}

/// Lasso structure for liveness violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LassoInfo {
    /// Number of states before the loop
    pub prefix_len: usize,

    /// Index where the loop begins
    pub loop_start: usize,
}

/// Runtime value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    Int(i64),
    String(String),
    Set(Vec<Value>),
    Seq(Vec<Value>),
    Record(Vec<(String, Value)>),
}
```

### Checker Configuration

```rust
/// Configuration for model checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckConfig {
    /// Number of worker threads
    pub workers: usize,

    /// Maximum states to explore
    pub max_states: usize,

    /// Maximum depth
    pub max_depth: usize,

    /// Timeout
    pub timeout: Duration,

    /// Enable symmetry reduction
    pub symmetry: bool,

    /// Enable partial order reduction
    pub por: bool,
}

impl Default for CheckConfig {
    fn default() -> Self {
        CheckConfig {
            workers: num_cpus::get(),
            max_states: 1_000_000,
            max_depth: 100,
            timeout: Duration::from_secs(60),
            symmetry: true,
            por: false,  // Not yet implemented
        }
    }
}
```

---

## API Entry Points

### Library API (Preferred)

```rust
// crates/tla-check/src/api.rs

/// Check a single temporal VC
pub fn check(vc: &TemporalVC, config: &CheckConfig) -> TLA2Result {
    // Convert VC to internal representation
    let model = convert_vc(vc)?;

    // Run model checker
    let checker = Checker::new(model, config);
    checker.check()
}

/// Check multiple VCs (batch mode)
pub fn batch_check(vcs: &[TemporalVC], config: &CheckConfig) -> Vec<TLA2Result> {
    // Parallel checking of independent VCs
    vcs.par_iter()
        .map(|vc| check(vc, config))
        .collect()
}

/// Check VCs with shared state exploration
pub fn check_with_shared_exploration(
    vcs: &[TemporalVC],
    config: &CheckConfig,
) -> Vec<TLA2Result> {
    // Single state exploration, multiple property checks
    // More efficient when VCs share the same state machine
    todo!()
}
```

### CLI API (Fallback)

```bash
# Check a VC from JSON file
$ tla2 check --format=json model.json > result.json

# Check from stdin
$ cat model.json | tla2 check --format=json --stdin > result.json

# With configuration
$ tla2 check --format=json --workers 8 --max-states 1000000 model.json
```

---

## Source Mapping Contract

TLA2 returns `ActionId` values that tRust provided. tRust maintains the mapping:

```rust
// tRust side (NOT TLA2)
struct SourceMapping {
    action_id_to_span: HashMap<ActionId, Span>,
}

impl SourceMapping {
    fn format_diagnostic(&self, result: &TLA2Result) -> Diagnostic {
        match result {
            TLA2Result::Violation { trace, property, .. } => {
                let mut diag = Diagnostic::new(Level::Error)
                    .with_message(format!("Temporal property violated: {}", property));

                for (i, action_id) in trace.actions.iter().enumerate() {
                    if let Some(span) = self.action_id_to_span.get(action_id) {
                        diag = diag.with_label(
                            Label::secondary(span.file_id, span.range)
                                .with_message(format!("Step {}: {}", i + 1, action_id))
                        );
                    }
                }

                diag
            }
            _ => todo!()
        }
    }
}
```

---

## Implementation Plan

### Phase 1: Basic API (Blocks tRust Integration)

1. Create `tla-wire` crate with type definitions
2. Implement `check()` function accepting `TemporalVC`
3. Return `TLA2Result` with action IDs preserved
4. JSON serialization for CLI mode

### Phase 2: Batch API

1. Implement `batch_check()` for parallel VC checking
2. Implement shared exploration optimization
3. Performance benchmarks

### Phase 3: Fault Injection

1. Implement `FaultModel` support
2. Node crash modeling
3. Network partition modeling
4. Message fault modeling

### Phase 4: Optimizations

1. Incremental checking (only re-explore changed transitions)
2. Symmetry reduction for node sets
3. Partial order reduction for independent actions

---

## Dependencies

| Requirement | Status |
|-------------|--------|
| Correctness (Phase 1) | BLOCKING - false positives must be fixed first |
| Liveness checking | Working (with bugs) |
| Fairness (WF_, SF_) | Working |
| Quantified temporal | Partial (\\A x: P ~> Q not supported) |

---

## Testing

### Unit Tests

```rust
#[test]
fn test_simple_safety_vc() {
    let vc = TemporalVC {
        id: "test_safety".into(),
        state_machine: StateMachine { ... },
        properties: vec![TemporalProperty::Always(...)],
        fairness: vec![],
        fault_model: None,
        bounds: Bounds::default(),
    };

    let result = check(&vc, &CheckConfig::default());
    assert!(matches!(result, TLA2Result::Verified { .. }));
}
```

### Integration Tests with tRust

1. Deadlock detection test cases
2. Race condition test cases
3. Consensus algorithm test cases
4. Performance benchmarks

---

## References

- tRust feature request: `feature-requests/tRust-2025-12-31.md`
- TLA2 response: Pushed to `git@github.com:dropbox/tRust.git`
- lean5 insights: `https://github.com/dropbox/lean5` (batch API, JSON output)
