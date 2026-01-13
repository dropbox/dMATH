# Refinement Design: TLA+ to Verified Rust

This document describes the code generation pipeline from TLA+ specifications to verified Rust implementations.

## Overview

The refinement pipeline transforms high-level TLA+ specifications into production Rust code with multiple layers of verification:

```
TLA+ Specification
       │
       ▼
┌─────────────────┐
│   tla-codegen   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Generated Rust Module                   │
├─────────────────────────────────────────────────────┤
│  • State struct                                      │
│  • Action enum                                       │
│  • StateMachine trait impl                          │
│  • Invariant checks                                 │
│  • Kani harnesses (#[cfg(kani)])                    │
│  • Property tests (proptest)                        │
│  • Runtime monitoring hooks                          │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Verification Layers                     │
├─────────────────────────────────────────────────────┤
│  1. cargo test      - Property-based tests          │
│  2. cargo kani      - Bounded model checking        │
│  3. verus (opt)     - Full proofs                   │
│  4. runtime         - Production monitoring         │
└─────────────────────────────────────────────────────┘
```

## Core Abstraction: StateMachine Trait

All generated code implements this trait:

```rust
pub trait StateMachine: Sized {
    /// The state type containing all specification variables
    type State: Clone + Eq + Hash + Debug;

    /// The action type representing all possible transitions
    type Action: Clone + Debug;

    /// Initial states (Init predicate)
    fn init() -> HashSet<Self::State>;

    /// Next-state relation (Next action)
    /// Returns None if action is not enabled in state
    fn next(state: &Self::State, action: &Self::Action) -> Option<Self::State>;

    /// Safety invariant
    fn invariant(state: &Self::State) -> bool;

    /// All possible actions (for exhaustive checking)
    fn actions(state: &Self::State) -> Vec<Self::Action>;

    /// Check if state is reachable from init
    fn is_reachable(state: &Self::State) -> bool {
        // Default: BFS exploration (can be overridden)
    }
}
```

## Code Generation Strategy

### Variable Mapping

| TLA+ Type | Rust Type |
|-----------|-----------|
| BOOLEAN | `bool` |
| Int, Nat | `num_bigint::BigInt` |
| STRING | `String` |
| Set(T) | `im::HashSet<T>` |
| Seq(T) | `im::Vector<T>` |
| [S -> T] | `im::HashMap<S, T>` |
| Record | Named struct |
| Tuple | Tuple |

### Action Extraction

TLA+ Next predicates are disjunctions of actions:

```tla
Next == Action1 \/ Action2 \/ Action3
```

Each disjunct becomes an enum variant:

```rust
pub enum MySpecAction {
    Action1 { /* parameters */ },
    Action2 { /* parameters */ },
    Action3 { /* parameters */ },
}
```

### Invariant Translation

TLA+ invariants become Rust predicates:

```tla
TypeInvariant == /\ x \in Nat
                 /\ y \in {1, 2, 3}

SafetyInvariant == x + y <= 10
```

```rust
impl MySpec {
    fn type_invariant(state: &State) -> bool {
        state.x >= Int::ZERO
            && [1, 2, 3].contains(&state.y)
    }

    fn safety_invariant(state: &State) -> bool {
        &state.x + &state.y <= Int::from(10)
    }

    fn invariant(state: &State) -> bool {
        Self::type_invariant(state) && Self::safety_invariant(state)
    }
}
```

## Verification Layers

### Layer 1: Property-Based Tests

Generated proptest tests exercise the state machine:

```rust
#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn invariant_preserved(
            actions in prop::collection::vec(any::<Action>(), 0..100)
        ) {
            for init_state in MySpec::init() {
                let mut state = init_state;
                prop_assert!(MySpec::invariant(&state));

                for action in &actions {
                    if let Some(next) = MySpec::next(&state, action) {
                        state = next;
                        prop_assert!(MySpec::invariant(&state));
                    }
                }
            }
        }
    }
}
```

### Layer 2: Bounded Model Checking (Kani)

Kani harnesses verify invariant preservation:

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_invariant_inductive() {
        // Assume we're in an invariant-satisfying state
        let state: State = kani::any();
        kani::assume(MySpec::invariant(&state));

        // For any action
        let action: Action = kani::any();

        // If action is enabled
        if let Some(next_state) = MySpec::next(&state, &action) {
            // Invariant is preserved
            kani::assert(
                MySpec::invariant(&next_state),
                "Invariant must be preserved"
            );
        }
    }

    #[kani::proof]
    fn verify_init_satisfies_invariant() {
        for state in MySpec::init() {
            kani::assert(
                MySpec::invariant(&state),
                "Init states must satisfy invariant"
            );
        }
    }
}
```

### Layer 3: Full Proofs (Verus)

For full verification, generate Verus annotations:

```rust
verus! {
    impl StateMachine for MySpec {
        proof fn invariant_inductive(state: State, action: Action)
            requires
                Self::invariant(state),
            ensures
                Self::next(state, action).is_some() ==>
                    Self::invariant(Self::next(state, action).unwrap())
        {
            // Proof by cases on action
        }
    }
}
```

### Layer 4: Runtime Monitoring

Monitor invariants in production:

```rust
pub struct MonitoredStateMachine<S: StateMachine> {
    state: S::State,
    config: MonitorConfig,
}

impl<S: StateMachine> MonitoredStateMachine<S> {
    pub fn apply(&mut self, action: &S::Action) -> Result<(), ViolationError> {
        if let Some(next) = S::next(&self.state, action) {
            if self.config.check_invariant && !S::invariant(&next) {
                return Err(ViolationError::InvariantViolated {
                    state: format!("{:?}", next),
                    action: format!("{:?}", action),
                });
            }
            self.state = next;
            Ok(())
        } else {
            Err(ViolationError::ActionNotEnabled {
                action: format!("{:?}", action),
            })
        }
    }
}
```

## Example: Two-Phase Commit

### TLA+ Specification

```tla
---- MODULE TwoPhaseCommit ----
CONSTANT RM  \* Set of resource managers

VARIABLE rmState, tmState, tmPrepared, msgs

Init ==
    /\ rmState = [r \in RM |-> "working"]
    /\ tmState = "init"
    /\ tmPrepared = {}
    /\ msgs = {}

TMRcvPrepared(r) ==
    /\ tmState = "init"
    /\ [type |-> "Prepared", rm |-> r] \in msgs
    /\ tmPrepared' = tmPrepared \cup {r}
    /\ UNCHANGED <<rmState, tmState, msgs>>

TMCommit ==
    /\ tmState = "init"
    /\ tmPrepared = RM
    /\ tmState' = "committed"
    /\ msgs' = msgs \cup {[type |-> "Commit"]}
    /\ UNCHANGED <<rmState, tmPrepared>>

\* ... more actions ...

Consistency ==
    \A r1, r2 \in RM:
        ~(rmState[r1] = "committed" /\ rmState[r2] = "aborted")
====
```

### Generated Rust

```rust
use tla_runtime::prelude::*;
use std::collections::{HashSet, HashMap};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TwoPhaseCommitState {
    pub rm_state: HashMap<ResourceManager, RmState>,
    pub tm_state: TmState,
    pub tm_prepared: HashSet<ResourceManager>,
    pub msgs: HashSet<Message>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RmState { Working, Prepared, Committed, Aborted }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TmState { Init, Committed, Aborted }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Message {
    Prepared { rm: ResourceManager },
    Commit,
    Abort,
}

#[derive(Clone, Debug)]
pub enum TwoPhaseCommitAction {
    TMRcvPrepared { r: ResourceManager },
    TMCommit,
    TMAbort,
    RMPrepare { r: ResourceManager },
    RMChooseToAbort { r: ResourceManager },
    RMRcvCommitMsg { r: ResourceManager },
    RMRcvAbortMsg { r: ResourceManager },
}

impl StateMachine for TwoPhaseCommit {
    type State = TwoPhaseCommitState;
    type Action = TwoPhaseCommitAction;

    fn init() -> HashSet<Self::State> {
        let rm_set: HashSet<ResourceManager> = RM.iter().cloned().collect();
        hashset! {
            TwoPhaseCommitState {
                rm_state: rm_set.iter()
                    .map(|r| (r.clone(), RmState::Working))
                    .collect(),
                tm_state: TmState::Init,
                tm_prepared: HashSet::new(),
                msgs: HashSet::new(),
            }
        }
    }

    fn next(state: &Self::State, action: &Self::Action) -> Option<Self::State> {
        match action {
            TwoPhaseCommitAction::TMCommit => {
                // Guard: tmState = "init" /\ tmPrepared = RM
                if state.tm_state != TmState::Init {
                    return None;
                }
                if state.tm_prepared != RM {
                    return None;
                }

                // Effect
                let mut next = state.clone();
                next.tm_state = TmState::Committed;
                next.msgs.insert(Message::Commit);
                Some(next)
            }
            // ... other actions
        }
    }

    fn invariant(state: &Self::State) -> bool {
        // Consistency: no RM committed while another aborted
        for r1 in RM.iter() {
            for r2 in RM.iter() {
                if state.rm_state[r1] == RmState::Committed
                    && state.rm_state[r2] == RmState::Aborted {
                    return false;
                }
            }
        }
        true
    }
}
```

## Integration with Production Code

The generated state machine is the core logic. Users implement the I/O layer:

```rust
struct TwoPhaseCoordinator {
    spec: MonitoredStateMachine<TwoPhaseCommit>,
    network: NetworkLayer,
}

impl TwoPhaseCoordinator {
    async fn handle_prepared(&mut self, from: ResourceManager) -> Result<()> {
        // Apply spec action
        self.spec.apply(&TwoPhaseCommitAction::TMRcvPrepared { r: from })?;

        // Check if we can commit
        if self.spec.state().tm_prepared == RM {
            self.spec.apply(&TwoPhaseCommitAction::TMCommit)?;

            // Send commit to all RMs
            for rm in RM.iter() {
                self.network.send(rm, CommitMessage).await?;
            }
        }

        Ok(())
    }
}
```

## Benefits of This Approach

1. **Design-level verification**: TLA+ model checking explores all states
2. **Code-level verification**: Kani proves Rust implementation preserves invariants
3. **Runtime safety**: Monitoring catches violations in production
4. **Single source of truth**: Spec is the specification, generated code is the implementation
5. **Incremental verification**: Start with tests, add Kani, optionally add Verus
