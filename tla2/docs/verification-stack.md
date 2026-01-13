# TLA2 Verification Stack: Kani and Verus

This document explains the two-level verification architecture and how Kani and Verus fit into the TLA2 pipeline.

## The Two-Level Problem

**Level 1 - Design**: "Is my TLA+ specification correct?"
**Level 2 - Code**: "Does my Rust code match my specification?"

Traditional approaches leave a gap between spec and code that's filled with hope. TLA2 machine-checks that gap.

```
Traditional:
  Spec (paper) → Manual coding → Hope it's right → Production bugs

TLA2:
  TLA+ Spec → Model check → Generate code → Verify code → Monitor production
       ✓           ✓              ✓              ✓              ✓
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: SPECIFICATION (TLA+)                                              │
│                                                                             │
│  "What should the system do?"                                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ---- MODULE TwoPhaseCommit ----                                     │   │
│  │  VARIABLE rmState, tmState                                           │   │
│  │                                                                      │   │
│  │  Init == rmState = [r ∈ RM ↦ "working"] ∧ tmState = "init"          │   │
│  │  TMCommit == tmState = "init" ∧ tmState' = "committed" ∧ ...        │   │
│  │                                                                      │   │
│  │  Consistency == ¬(rmState[r1]="committed" ∧ rmState[r2]="aborted")  │   │
│  │  ====                                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│                        ┌─────────────┐                                      │
│                        │  tla-check  │  ← Explores ALL states               │
│                        │             │    Finds: "Consistency holds         │
│                        │  (TLA2)     │    in all 10,847 reachable states"   │
│                        └─────────────┘                                      │
│                                                                             │
│  Result: Design is correct ✓                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                │  tla-codegen
                                │  (generates Rust from TLA+ spec)
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LEVEL 2: IMPLEMENTATION (Rust)                                             │
│                                                                             │
│  "Does the code match the spec?"                                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  // Generated from TwoPhaseCommit.tla                                │   │
│  │  impl StateMachine for TwoPhaseCommit {                              │   │
│  │      fn init() -> HashSet<State> { ... }                             │   │
│  │      fn next(state: &State, action: &Action) -> Option<State> { ... }│   │
│  │      fn invariant(state: &State) -> bool {                           │   │
│  │          // Consistency predicate                                    │   │
│  │          !(state.rm[r1] == Committed && state.rm[r2] == Aborted)     │   │
│  │      }                                                               │   │
│  │  }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│           ┌───────────────────┼───────────────────┐                        │
│           ▼                   ▼                   ▼                        │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                 │
│    │   proptest  │     │    KANI     │     │   VERUS     │                 │
│    │             │     │             │     │             │                 │
│    │  Random     │     │  Bounded    │     │   Full      │                 │
│    │  testing    │     │  model      │     │   proofs    │                 │
│    │             │     │  checking   │     │             │                 │
│    └─────────────┘     └─────────────┘     └─────────────┘                 │
│                                                                             │
│  Result: Code correctly implements spec ✓                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tool Comparison

| Tool | Level | Method | Proves | Confidence | Effort |
|------|-------|--------|--------|------------|--------|
| **tla-check** | Design | Explicit state enumeration | Spec is correct | 100% of spec | Low |
| **proptest** | Code | Random testing | No bugs found in N runs | ~80% | Zero |
| **Kani** | Code | Bounded model checking | Invariant holds up to depth N | ~95% | Low |
| **Verus** | Code | SMT deductive verification | Invariant holds ALWAYS | 100% | Medium |
| **Runtime** | Production | Monitoring | System follows spec live | Reactive | Low |

---

## Kani: Bounded Model Checking

### What Kani Does

Kani translates Rust to C via CBMC and exhaustively checks all paths up to a bound:

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(20)]  // ← BOUND: check up to 20 loop iterations
    fn verify_invariant_inductive() {
        // Create ANY possible state
        let state: State = kani::any();

        // Assume it satisfies the invariant
        kani::assume(invariant(&state));

        // Pick ANY action
        let action: Action = kani::any();

        // If action produces a next state...
        if let Some(next) = step(&state, &action) {
            // ...then invariant MUST still hold
            kani::assert(invariant(&next), "Invariant preserved");
        }
    }
}
```

### What Kani Proves

This harness proves: **If invariant(s) holds, then for ANY action a, invariant(step(s, a)) holds.**

Combined with proving `invariant(init())`, this is an **inductive invariant proof** - the invariant holds in all reachable states (up to the bound).

### Kani Limitations

- **Bounded**: Only checks up to `unwind(N)` steps
- A bug at step N+1 would be missed
- Increasing the bound increases verification time exponentially

### When to Use Kani

- Always (it's automatic and catches most bugs)
- Default verification layer for generated code
- Sufficient for most applications

---

## Verus: Full Deductive Verification

### What Verus Does

Verus extends Rust with specification annotations and proves them via SMT solver:

```rust
use verus::*;

verus! {

// Specification function (ghost code, not compiled)
pub open spec fn invariant(s: State) -> bool {
    s.x >= 0 && s.y >= 0 && s.x + s.y <= 100
}

// Implementation with pre/post conditions
pub fn step(state: State, action: Action) -> (next: State)
    requires
        invariant(state),           // Precondition: caller must provide valid state
    ensures
        invariant(next),            // Postcondition: WE PROVE THIS
{
    match action {
        Action::Increment => State { x: state.x + 1, y: state.y },
        Action::Decrement => State { x: state.x - 1, y: state.y },
    }
}

// Verus PROVES: ∀ state, action.
//   invariant(state) → invariant(step(state, action))
// This is a MATHEMATICAL PROOF, not a test.

}
```

### What Verus Proves

- **Unbounded**: Proof holds for ALL inputs, not just up to a bound
- **Mathematical certainty**: SMT solver provides formal proof
- Equivalent to a pen-and-paper proof, but machine-checked

### Verus Requirements

- Must annotate code with `requires`/`ensures`
- Sometimes need proof hints for complex cases
- Separate toolchain installation

### When to Use Verus

- **Consensus protocols** (Raft, Paxos) - must be correct
- **Cryptographic code** - subtle bugs are catastrophic
- **Financial systems** - bugs cost money
- **Safety-critical** - medical devices, aerospace

---

## Verification Confidence Ladder

```
Confidence
    ▲
    │
100%│  ┌─────────────────────────────────────────────────────────────┐
    │  │  VERUS: Mathematical proof                                   │
    │  │  "Proved for ALL inputs, ALL executions, FOREVER"           │
    │  │                                                              │
    │  │  verus! {                                                    │
    │  │      fn step(s: State) -> (n: State)                        │
    │  │          requires invariant(s)                               │
    │  │          ensures invariant(n)                                │
    │  │  }                                                           │
    │  └─────────────────────────────────────────────────────────────┘
    │
 95%│  ┌─────────────────────────────────────────────────────────────┐
    │  │  KANI: Bounded model checking                                │
    │  │  "Proved for all inputs up to depth N"                       │
    │  │                                                              │
    │  │  #[kani::proof]                                              │
    │  │  #[kani::unwind(20)]                                         │
    │  │  fn verify() { kani::assert(invariant(next)); }             │
    │  └─────────────────────────────────────────────────────────────┘
    │
 80%│  ┌─────────────────────────────────────────────────────────────┐
    │  │  PROPTEST: Property-based random testing                     │
    │  │  "No bugs found in 100,000 random test cases"               │
    │  │                                                              │
    │  │  proptest! {                                                 │
    │  │      #[test]                                                 │
    │  │      fn test(actions in vec(any::<Action>(), 0..100)) { }   │
    │  │  }                                                           │
    │  └─────────────────────────────────────────────────────────────┘
    │
 50%│  ┌─────────────────────────────────────────────────────────────┐
    │  │  UNIT TESTS: Example-based testing                           │
    │  │  "Works on the cases I thought of"                          │
    │  │                                                              │
    │  │  #[test]                                                     │
    │  │  fn test_increment() { assert!(invariant(inc(init()))); }   │
    │  └─────────────────────────────────────────────────────────────┘
    │
    └──────────────────────────────────────────────────────────────────▶ Effort
                    Zero        Low         Medium        High
```

---

## Generated Code Structure

tla-codegen produces Rust with all verification layers built-in:

```rust
// ═══════════════════════════════════════════════════════════════════════════
// GENERATED FROM MySpec.tla by tla-codegen
// ═══════════════════════════════════════════════════════════════════════════

use tla_runtime::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// STATE AND ACTIONS (from VARIABLES and Next)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MySpecState {
    pub x: Int,
    pub y: Int,
}

#[derive(Clone, Debug)]
pub enum MySpecAction {
    Increment,
    Decrement,
}

// ─────────────────────────────────────────────────────────────────────────────
// STATE MACHINE IMPLEMENTATION
// ─────────────────────────────────────────────────────────────────────────────

impl StateMachine for MySpec {
    type State = MySpecState;
    type Action = MySpecAction;

    fn init() -> HashSet<Self::State> {
        hashset! { MySpecState { x: 0.into(), y: 0.into() } }
    }

    fn next(state: &Self::State, action: &Self::Action) -> Option<Self::State> {
        match action {
            MySpecAction::Increment => {
                Some(MySpecState { x: &state.x + 1, y: state.y.clone() })
            }
            MySpecAction::Decrement => {
                if state.x > Int::from(0) {
                    Some(MySpecState { x: &state.x - 1, y: state.y.clone() })
                } else {
                    None  // Action not enabled
                }
            }
        }
    }

    fn invariant(state: &Self::State) -> bool {
        state.x >= Int::from(0)  // From TLA+ invariant
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAYER 1: PROPERTY-BASED TESTS (proptest)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod proptest_verification {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn invariant_holds_on_random_traces(
            actions in prop::collection::vec(any::<MySpecAction>(), 0..100)
        ) {
            for init_state in MySpec::init() {
                let mut state = init_state;
                prop_assert!(MySpec::invariant(&state), "Init must satisfy invariant");

                for action in &actions {
                    if let Some(next) = MySpec::next(&state, action) {
                        state = next;
                        prop_assert!(MySpec::invariant(&state), "Invariant violated!");
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAYER 2: BOUNDED MODEL CHECKING (Kani)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(kani)]
mod kani_verification {
    use super::*;

    /// Proves: Init states satisfy invariant
    #[kani::proof]
    fn init_satisfies_invariant() {
        for state in MySpec::init() {
            kani::assert(
                MySpec::invariant(&state),
                "All initial states must satisfy invariant"
            );
        }
    }

    /// Proves: Invariant is inductive (preserved by all transitions)
    #[kani::proof]
    #[kani::unwind(10)]
    fn invariant_is_inductive() {
        // For ANY state satisfying invariant...
        let state: MySpecState = kani::any();
        kani::assume(MySpec::invariant(&state));

        // ...and ANY action...
        let action: MySpecAction = kani::any();

        // ...if the action is enabled...
        if let Some(next_state) = MySpec::next(&state, &action) {
            // ...then the next state MUST satisfy invariant
            kani::assert(
                MySpec::invariant(&next_state),
                "Invariant must be preserved by all transitions"
            );
        }
    }

    /// Proves: No deadlocks (some action is always enabled)
    #[kani::proof]
    #[kani::unwind(10)]
    fn no_deadlock() {
        let state: MySpecState = kani::any();
        kani::assume(MySpec::invariant(&state));

        let has_enabled_action = MySpec::actions(&state)
            .iter()
            .any(|a| MySpec::next(&state, a).is_some());

        kani::assert(has_enabled_action, "Must have at least one enabled action");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAYER 3: FULL PROOFS (Verus) - Optional
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(verus)]
mod verus_verification {
    use super::*;
    use verus::*;

    verus! {

    // Spec function: invariant as a logical predicate
    pub open spec fn spec_invariant(state: MySpecState) -> bool {
        state.x >= 0
    }

    // Proof: init satisfies invariant
    proof fn init_valid()
        ensures forall |s| MySpec::init().contains(s) ==> spec_invariant(s)
    {
        // Automatic from SMT
    }

    // Proof: invariant is preserved
    proof fn invariant_preserved(state: MySpecState, action: MySpecAction)
        requires spec_invariant(state)
        ensures MySpec::next(&state, &action).is_some() ==>
                spec_invariant(MySpec::next(&state, &action).unwrap())
    {
        // Case analysis on action
        match action {
            MySpecAction::Increment => {
                // x + 1 >= 0 when x >= 0 ✓
            }
            MySpecAction::Decrement => {
                // Guard ensures x > 0, so x - 1 >= 0 ✓
            }
        }
    }

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAYER 4: RUNTIME MONITORING
// ─────────────────────────────────────────────────────────────────────────────

pub struct MonitoredMySpec {
    state: MySpecState,
    check_invariant: bool,
    violation_count: u64,
}

impl MonitoredMySpec {
    pub fn new(check_invariant: bool) -> Self {
        let state = MySpec::init().into_iter().next().unwrap();
        Self { state, check_invariant, violation_count: 0 }
    }

    pub fn apply(&mut self, action: &MySpecAction) -> Result<(), SpecViolation> {
        match MySpec::next(&self.state, action) {
            Some(next) => {
                if self.check_invariant && !MySpec::invariant(&next) {
                    self.violation_count += 1;
                    return Err(SpecViolation::InvariantViolated {
                        state: format!("{:?}", next),
                        action: format!("{:?}", action),
                    });
                }
                self.state = next;
                Ok(())
            }
            None => Err(SpecViolation::ActionNotEnabled {
                action: format!("{:?}", action),
            }),
        }
    }

    pub fn state(&self) -> &MySpecState {
        &self.state
    }
}

#[derive(Debug)]
pub enum SpecViolation {
    InvariantViolated { state: String, action: String },
    ActionNotEnabled { action: String },
}
```

---

## Running Verification

```bash
# Layer 1: Property-based tests (fast, catches obvious bugs)
cargo test

# Layer 2: Bounded model checking (thorough, proves up to bound)
cargo kani

# Layer 3: Full proofs (complete, requires annotations)
cargo verus  # if Verus annotations are present

# All layers
cargo test && cargo kani && cargo verus
```

---

## Choosing Your Verification Level

| Application Type | Recommended Layers |
|------------------|-------------------|
| Prototype / MVP | proptest |
| Production service | proptest + Kani |
| Financial / payments | proptest + Kani + Verus |
| Safety-critical | proptest + Kani + Verus + Runtime monitoring |
| Consensus protocol | All layers mandatory |

---

## Summary

1. **tla-check** verifies your DESIGN is correct (Level 1)
2. **tla-codegen** generates Rust with verification harnesses (the bridge)
3. **proptest/Kani/Verus** verify your CODE matches the spec (Level 2)
4. **Runtime monitoring** catches violations in production (safety net)

The gap between spec and code is **machine-checked**, not hoped.
