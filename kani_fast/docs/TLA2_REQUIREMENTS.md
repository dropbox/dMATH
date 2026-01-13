# TLA2 Requirements from Kani Fast

**From:** Kani Fast (Rust verification engine) → eventually tRust (compiler)
**To:** TLA2 (temporal logic model checker)
**Date:** 2025-12-30
**Status:** Requirements for Temporal Property Verification

## Summary

Kani Fast verifies **safety properties** (no overflow, no bounds errors). TLA2 verifies **temporal properties** (eventually, always, fairness) for concurrent/async Rust code.

**Use case:** Verify that async Rust code (tokio, async-std) satisfies liveness and fairness properties.

---

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROPERTY TYPES                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAFETY (Kani Fast)          │  TEMPORAL (TLA2)                │
│  "bad thing never happens"   │  "good thing eventually happens"│
│                              │                                  │
│  • No overflow               │  • Eventually completes          │
│  • No bounds error           │  • Always responds               │
│  • No division by zero       │  • Fairness (no starvation)      │
│  • No panic                  │  • Deadlock freedom              │
│                              │                                  │
│  Backend: Z3/Z4              │  Backend: TLA2                   │
│                              │                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority 1: CRITICAL (Blocks Integration)

### 1.1 CLI Model Checking

**What:** Check a TLA+ spec non-interactively.

**Required interface:**
```bash
tla2 check spec.tla --config spec.cfg
# Exit 0 = all properties verified
# Exit 1 = property violated (with counterexample trace)
```

### 1.2 State Machine Extraction from Async Rust

Kani Fast will extract state machines from async Rust code:

```rust
async fn producer_consumer() {
    let (tx, rx) = channel();

    tokio::spawn(async move {
        loop {
            tx.send(produce()).await;  // State: PRODUCING
        }
    });

    loop {
        let item = rx.recv().await;    // State: CONSUMING
        process(item);
    }
}
```

This becomes TLA+ spec:

```tla
---- MODULE ProducerConsumer ----
VARIABLES state, buffer

Init == state = "idle" /\ buffer = <<>>

Produce == state = "producing" /\ buffer' = Append(buffer, "item")
Consume == state = "consuming" /\ Len(buffer) > 0 /\ buffer' = Tail(buffer)

Next == Produce \/ Consume

Fairness == WF_vars(Produce) /\ WF_vars(Consume)

Liveness == <>[]( Len(buffer) < MAX_SIZE )  \* Buffer never overflows forever

====
```

**Required:** TLA2 must parse this format and verify properties.

### 1.3 Temporal Operators

**Required operators:**

| Operator | Meaning | Example |
|----------|---------|---------|
| `[]P` | Always P | `[](no_deadlock)` |
| `<>P` | Eventually P | `<>(task_complete)` |
| `P ~> Q` | P leads to Q | `request ~> response` |
| `WF_v(A)` | Weak fairness | No starvation |
| `SF_v(A)` | Strong fairness | Must eventually happen |

### 1.4 Counterexample Traces

When property violated, provide execution trace:

```
Property violated: <>(task_complete)
Counterexample trace:
  State 0: state = "idle", buffer = <<>>
  State 1: state = "producing", buffer = <<"item">>
  State 2: state = "producing", buffer = <<"item", "item">>
  ... (loops forever without consuming)
```

---

## Priority 2: HIGH (Async Rust Integration)

### 2.1 Tokio State Machine Patterns

Common async patterns that TLA2 must verify:

```rust
// Pattern 1: Request-Response
async fn handle_request(req: Request) -> Response {
    // TLA2 property: request ~> response
}

// Pattern 2: Producer-Consumer
async fn producer_consumer(tx: Sender, rx: Receiver) {
    // TLA2 property: WF(produce) /\ WF(consume) => <>[](balanced)
}

// Pattern 3: Mutex Lock
async fn critical_section(mutex: Mutex<T>) {
    // TLA2 property: [](at_most_one_in_critical_section)
}
```

### 2.2 Channel Semantics

Rust channels (mpsc, broadcast, watch) have specific semantics:

```tla
\* mpsc: Multiple producers, single consumer
mpsc_send == /\ Len(buffer) < capacity
             /\ buffer' = Append(buffer, msg)

mpsc_recv == /\ Len(buffer) > 0
             /\ buffer' = Tail(buffer)
             /\ received' = Head(buffer)
```

**Required:** TLA2 standard library with channel primitives.

### 2.3 Timeout Handling

```bash
tla2 check spec.tla --timeout=300
```

Bounded model checking should complete or report progress.

---

## Priority 3: MEDIUM (Advanced Features)

### 3.1 JSON Output Mode

```bash
tla2 check spec.tla --output=json
```

```json
{
  "status": "violated",
  "property": "Liveness",
  "trace": [
    {"state": 0, "variables": {"x": 0, "y": 1}},
    {"state": 1, "variables": {"x": 1, "y": 1}},
    ...
  ]
}
```

### 3.2 Symmetry Reduction

For concurrent code with multiple identical processes:

```tla
CONSTANTS Procs
ASSUME Procs = {p1, p2, p3}  \* Symmetric
```

**Required:** Symmetry reduction to avoid state explosion.

### 3.3 Partial Order Reduction

For concurrent transitions that commute:

```
If A and B are independent, don't explore both A→B and B→A
```

---

## Integration with tRust

When integrated into tRust:

```rust
#[tla2::verify(
    property = "request ~> response",
    fairness = "weak"
)]
async fn handle_request(req: Request) -> Response {
    // ...
}
```

tRust compiler:
1. Extracts state machine from async code
2. Generates TLA+ spec
3. Calls `tla2 check`
4. Reports violation as compiler error

---

## Rust Async → TLA+ Translation

| Rust Async | TLA+ |
|------------|------|
| `async fn` | Process with states |
| `.await` | State transition |
| `tokio::spawn` | New process |
| `channel.send()` | Append to buffer |
| `channel.recv()` | Pop from buffer |
| `Mutex::lock()` | Acquire exclusive access |
| `select!` | Non-deterministic choice |

---

## Test Cases

Kani Fast will provide test specs:

1. `simple_liveness.tla` - Basic <> property
2. `producer_consumer.tla` - Channel communication
3. `mutex_safety.tla` - Mutual exclusion
4. `deadlock_free.tla` - No deadlock in dining philosophers
5. `starvation.tla` - Should FAIL (tests fairness violation detection)

---

## Current TLA+ Usage

We would use TLC (TLA+ model checker) via:
```bash
java -jar tla2tools.jar -config spec.cfg spec.tla
```

TLA2 should be a faster, Rust-native replacement.

---

## Contact

- **Kani Fast repo:** https://github.com/dropbox/dMATH/kani_fast
- **tRust repo:** https://github.com/dropbox/tRust
- **TLA integration code:** Future `crates/kani-fast-temporal/` (not yet started)
