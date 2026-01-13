//! Bug reproduction tests - each test reproduces a specific catalog bug
//!
//! These tests capture the minimal pattern that triggers each bug.
//! The test should FAIL until the bug is fixed, then pass.

use tla_check::{check_module, CheckResult, Config, ConstantValue};
use tla_core::{lower, parse_to_syntax_tree, FileId};

/// Helper to parse a module from source
fn parse_module(src: &str) -> tla_core::ast::Module {
    let tree = parse_to_syntax_tree(src);
    let result = lower(FileId(0), &tree);
    result.module.expect("Failed to parse module")
}

// ============================================================================
// Bug #3: MCInnerSerial - Prime guard in operator body not enforced
// ============================================================================

/// Bug #3: MCInnerSerial prime guard in operator body
///
/// The MCInnerSerial spec has:
/// - `opOrder' \in SUBSET(opId' \X opId')` which can generate self-loops like <<a, a>>
/// - `UpdateOpOrder` operator containing `Serializable'` prime guard
/// - `Serializable'` checks that opOrder' has no self-loops
///
/// The bug: When the prime guard is inside an operator body (e.g., `UpdateOpOrder`
/// contains `Serializable'`), TLA2 was not validating it because:
/// 1. `needs_next_state_validation(Ident("UpdateOpOrder"))` returned false
/// 2. The operator body containing primed constraints wasn't being evaluated
///
/// This test creates a minimal reproduction where:
/// - Variable `x` is a set of pairs
/// - `NoSelfLoop'` is a prime guard that should filter self-loops
/// - `Update` operator contains the prime guard (not inline in Next)
///
/// Without the fix: TLA2 generates states with self-loops (invariant violated)
/// With the fix: TLA2 correctly filters self-loops (no invariant violation)
#[test]
fn test_bug_prime_guard_in_operator_body() {
    // Minimal spec that reproduces the bug
    let spec = r#"
---- MODULE PrimeGuardInOperatorBody ----
VARIABLE x

\* Prime guard: no self-loops in x'
NoSelfLoop == \A pair \in x : pair[1] # pair[2]

Init == x = {}

\* The constraint is inside an operator body, not inline
Update ==
    /\ x' \in SUBSET({<<1, 1>>, <<1, 2>>, <<2, 1>>})
    /\ NoSelfLoop'  \* This prime guard must be enforced!

Next == Update

\* Invariant: x should never contain self-loops
\* If bug exists, this invariant will be violated
NoSelfLoopInvariant == \A pair \in x : pair[1] # pair[2]

====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NoSelfLoopInvariant".to_string()],
        ..Default::default()
    };

    let result = check_module(&module, &config);

    // With the bug fixed, model checking should complete without violations
    match &result {
        CheckResult::Success(stats) => {
            // Expected: Should find valid states (empty set, sets without self-loops)
            // States: {}, {<<1,2>>}, {<<2,1>>}, {<<1,2>>, <<2,1>>}
            // That's 4 distinct states (2^2 subsets of non-self-loop pairs)
            assert!(
                stats.states_found >= 1,
                "Should find at least 1 state, got {}",
                stats.states_found
            );
        }
        CheckResult::InvariantViolation { invariant, .. } => {
            panic!(
                "Bug exists! Invariant {} violated - TLA2 is generating states with self-loops",
                invariant
            );
        }
        CheckResult::Deadlock { .. } => {
            panic!("Unexpected deadlock");
        }
        _ => {
            panic!("Unexpected result: {:?}", result);
        }
    }
}

/// Variant: Prime guard directly in Next (baseline - should always work)
///
/// This test verifies that inline prime guards work correctly.
/// The bug only affected prime guards inside operator bodies.
#[test]
fn test_inline_prime_guard_works() {
    let spec = r#"
---- MODULE InlinePrimeGuard ----
VARIABLE x

NoSelfLoop == \A pair \in x : pair[1] # pair[2]

Init == x = {}

\* Prime guard is inline in Next, not inside an operator
Next ==
    /\ x' \in SUBSET({<<1, 1>>, <<1, 2>>, <<2, 1>>})
    /\ NoSelfLoop'

NoSelfLoopInvariant == \A pair \in x : pair[1] # pair[2]

====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NoSelfLoopInvariant".to_string()],
        ..Default::default()
    };

    let result = check_module(&module, &config);

    match &result {
        CheckResult::Success { .. } => {
            // Expected: inline prime guards should work
        }
        CheckResult::InvariantViolation { invariant, .. } => {
            panic!(
                "Inline prime guard failed! Invariant {} violated",
                invariant
            );
        }
        _ => {
            panic!("Unexpected result: {:?}", result);
        }
    }
}

// ============================================================================
// Bug #12: bosco regression - hidden primed vars in value-op guards (inline EXISTS path)
// ============================================================================

/// Bug #12: Hidden primed variables in value-operator guards must be validated.
///
/// Pattern (from `bosco`):
/// - A value operator references primed state (e.g., `MsgCount(self) == Cardinality(rcvd'[self])`)
/// - The Next relation uses the value operator in a guard position (e.g., `MsgCount(self) >= 1`)
/// - The primed variable is not syntactically visible at the call site, so naive guard evaluation
///   errors (no next-state context) and must be deferred and validated against each successor.
///
/// Regression root cause: the inline EXISTS fast path treated action-level guard eval errors as
/// "guard passed" but did not re-validate them against the computed next-state, allowing bogus
/// transitions (e.g., deciding with `rcvd = {}`).
#[test]
fn test_hidden_prime_in_value_operator_guard_is_validated() {
    let spec = r#"
---- MODULE HiddenPrimeInValueGuard ----
EXTENDS Naturals, FiniteSets

CONSTANT N
VARIABLE pc, rcvd

Corr == 1 .. N
vars == <<pc, rcvd>>

Init ==
  /\ pc = [i \in Corr |-> "S"]
  /\ rcvd = [i \in Corr |-> {}]

\* Nondeterministically "receive" a set for this process (can be empty).
Receive(self) ==
  \E r \in SUBSET {1}:
    /\ rcvd[self] \subseteq r
    /\ rcvd' = [rcvd EXCEPT ![self] = r]

\* Value operator that depends on next-state (hidden prime at call sites).
MsgCount(self) == Cardinality(rcvd'[self])

Step(self) ==
  /\ Receive(self)
  /\ \/
      /\ pc[self] = "S"
      /\ MsgCount(self) >= 1
      /\ pc' = [pc EXCEPT ![self] = "D"]
     \/ pc' = pc

Next == \E self \in Corr: Step(self)

\* Invariant: if a process decides, it must have received at least one message.
DecideHasMsg ==
  \A i \in Corr: (pc[i] = "D") => (Cardinality(rcvd[i]) >= 1)

====
"#;

    let module = parse_module(spec);
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["DecideHasMsg".to_string()],
        ..Default::default()
    };
    config
        .constants
        .insert("N".to_string(), ConstantValue::Value("1".to_string()));

    let result = check_module(&module, &config);
    match &result {
        CheckResult::Success(_) => {}
        CheckResult::InvariantViolation { invariant, .. } => {
            panic!(
                "Invariant {} violated - hidden prime guard was not enforced",
                invariant
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}

/// Variant: Nested operator with prime guard (deeper nesting)
///
/// Tests that prime guards are enforced even when nested multiple levels deep.
#[test]
fn test_nested_operator_prime_guard() {
    let spec = r#"
---- MODULE NestedPrimeGuard ----
VARIABLE x

NoSelfLoop == \A pair \in x : pair[1] # pair[2]

\* Inner operator contains the prime guard
InnerUpdate ==
    /\ x' \in SUBSET({<<1, 1>>, <<1, 2>>})
    /\ NoSelfLoop'

\* Outer operator calls inner
OuterUpdate == InnerUpdate

Init == x = {}
Next == OuterUpdate

NoSelfLoopInvariant == \A pair \in x : pair[1] # pair[2]

====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["NoSelfLoopInvariant".to_string()],
        ..Default::default()
    };

    let result = check_module(&module, &config);

    match &result {
        CheckResult::Success { .. } => {
            // Good: nested prime guards work
        }
        CheckResult::InvariantViolation { invariant, .. } => {
            panic!(
                "Nested operator prime guard failed! Invariant {} violated",
                invariant
            );
        }
        _ => {
            panic!("Unexpected result: {:?}", result);
        }
    }
}

// ============================================================================
// Bug: Compiled assignments must allow primed RHS + correlated copies
// ============================================================================

/// Regression test: action-level assignments must be able to reference other primed vars.
///
/// Minimal pattern:
///   /\ lo' = 1
///   /\ buff' = lo'
///
/// The preprocessed/compiled Next enumerator must treat `lo'` as a next-state read
/// when compiling `buff'`'s RHS; otherwise it will either fail to enumerate the
/// transition or compute the wrong successor.
#[test]
fn test_action_assignment_can_reference_primed_var() {
    let spec = r#"
---- MODULE PrimedRhsAssignment ----
VARIABLES lo, buff

Init == /\ lo = 0 /\ buff = 0

Next ==
  /\ lo' = 1
  /\ buff' = lo'

Inv == lo = buff

Spec == Init /\ [][Next]_<<lo, buff>>
====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        ..Default::default()
    };

    let result = check_module(&module, &config);
    match &result {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 2,
                "Expected 2 states (0,0) and (1,1), got {}",
                stats.states_found
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}

/// Regression test: correlated copies x' = y' must preserve nondeterministic choices.
///
/// Minimal pattern:
///   /\ y' \in {0, 1}
///   /\ x' = y'
///
/// The successor set should contain exactly the states where x=y.
#[test]
fn test_action_assignment_copy_from_inset_prime() {
    let spec = r#"
---- MODULE PrimedCopyFromInSet ----
VARIABLES x, y

Init == /\ x = 0 /\ y = 0

Next ==
  /\ y' \in {0, 1}
  /\ x' = y'

Inv == x = y

Spec == Init /\ [][Next]_<<x, y>>
====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Inv".to_string()],
        ..Default::default()
    };

    let result = check_module(&module, &config);
    match &result {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 2,
                "Expected 2 states where x=y, got {}",
                stats.states_found
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}

// ============================================================================
// Bug: LET bindings with primed variables (BufferedRandomAccessFile)
// ============================================================================

/// Regression test: LET bindings that reference primed variables must be handled
/// by substitution, not evaluation.
///
/// The BufferedRandomAccessFile spec uses patterns like:
///   LET diskPosA == lo' IN buff' = MkArray(...diskPosA...)
///
/// When we try to evaluate `lo'` without a next-state context, evaluation fails.
/// The fix substitutes the primed expression directly into the body.
#[test]
fn test_let_binding_with_primed_variable() {
    let spec = r#"
---- MODULE LetPrimeBinding ----
EXTENDS Naturals

VARIABLES lo, marker

Init == lo = 0 /\ marker = 0

\* LET x == lo' pattern - must use substitution
ComputeMarker ==
  LET x == lo' IN
  marker' = x * 10

ChangeLo(newLo) ==
  /\ lo' = newLo
  /\ ComputeMarker

Next == \E nl \in 0..3: ChangeLo(nl)

Spec == Init /\ [][Next]_<<lo, marker>>
====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };

    let result = check_module(&module, &config);
    match &result {
        CheckResult::Success(stats) => {
            // 4 states: lo \in {0, 1, 2, 3}, marker = lo * 10
            assert_eq!(
                stats.states_found, 4,
                "Expected 4 states (lo=0,1,2,3 with marker=lo*10), got {}",
                stats.states_found
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}

// ============================================================================
// Bug #472: CHOOSE variable shadowing when LET-bound name exists
// ============================================================================

/// Bug #472: CHOOSE variable shadowing with LET-defined name
///
/// The MultiPaxos_MC spec had a pattern like:
/// ```
/// FirstEmptySlot(insts) == CHOOSE s \in Slots: insts[s].status = "Empty" /\ ...
///
/// TakeNewRequest(self) ==
///     LET s == FirstEmptySlot(node[self].insts)
///     IN  /\ node' = [node EXCEPT ![self].insts[s].status = "Accepting"]
///         /\ ...
/// ```
///
/// The bug: When CHOOSE binds `s`, the value was added to `env`, but lookup order is:
///   local_stack → local_ops → env
///
/// If a LET expression defines `s` in the enclosing scope, the lookup of `s`
/// inside CHOOSE would find the LET definition in `local_ops` instead of the
/// CHOOSE-bound value from `env`. This caused infinite recursion when `FirstEmptySlot`
/// was re-evaluated repeatedly.
///
/// The fix: In `bind()`, check if the name exists in `local_ops` and if so,
/// add the binding to `local_stack` to properly shadow the LET definition.
#[test]
fn test_bug_choose_variable_shadowing_with_let() {
    let spec = r#"
---- MODULE ChooseVariableShadowing ----
EXTENDS Naturals

VARIABLE x

\* This operator uses CHOOSE to find a value
FindValue(set) == CHOOSE v \in set: v > 0

Init == x = 0

\* The key pattern: LET defines `v`, then the body uses FindValue which
\* internally binds `v` in CHOOSE. The CHOOSE-bound `v` must shadow the LET-defined `v`.
Next ==
    LET v == FindValue({1, 2, 3})
    IN  x' = v

Spec == Init /\ [][Next]_x

====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };

    let result = check_module(&module, &config);

    match &result {
        CheckResult::Success(stats) => {
            // Should find at least 2 states: x=0, x=1 (or x=2 or x=3 depending on CHOOSE)
            assert!(
                stats.states_found >= 2,
                "Should find at least 2 states, got {}",
                stats.states_found
            );
        }
        CheckResult::Deadlock { .. } => {
            // Deadlock is acceptable - the important thing is we don't get infinite recursion
        }
        _ => {
            panic!(
                "Bug exists! Got unexpected result (likely infinite recursion): {:?}",
                result
            );
        }
    }
}

/// More complex test: CHOOSE inside LET where CHOOSE variable name matches LET name
///
/// This directly tests the case where the bound variable in CHOOSE has the same
/// name as the LET-defined operator that contains the CHOOSE.
#[test]
fn test_bug_choose_shadows_containing_let_name() {
    let spec = r#"
---- MODULE ChooseShadowsLet ----
EXTENDS Naturals

VARIABLE result

Init == result = 0

\* The LET defines `s`, and the CHOOSE inside also binds `s`.
\* The CHOOSE's `s` must shadow the LET's `s` within the CHOOSE body.
Next ==
    LET s == CHOOSE s \in {1, 2, 3}: s > 1
    IN  result' = s

Spec == Init /\ [][Next]_result

====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };

    let result = check_module(&module, &config);

    match &result {
        CheckResult::Success(stats) => {
            // Should complete without infinite recursion
            // We expect states: result=0 (init), result=2 or result=3 (after transition)
            assert!(
                stats.states_found >= 2,
                "Should find at least 2 states, got {}",
                stats.states_found
            );
        }
        CheckResult::Deadlock { .. } => {
            // Deadlock is acceptable - the important thing is no infinite recursion
        }
        _ => {
            panic!(
                "Bug exists! Got unexpected result (likely infinite recursion): {:?}",
                result
            );
        }
    }
}

// ============================================================================
// Bug #499: Variable capture in action preprocessing operator inlining
// ============================================================================

/// Bug #499: Variable capture in preprocessing operator inlining
///
/// The MCCheckpointCoordination spec has:
/// ```
/// SendReplicatedRequest(prospect) ==
///     ...
///     ReplicatedLog' = [n \in Node |-> ... WriteLog(n, index, prospect) ...]
/// ```
///
/// When preprocessing `\E n \in Node: SendReplicatedRequest(n)`, the inliner was:
/// 1. Creating substitution: `prospect` -> `n` (the argument expression)
/// 2. Substituting in the body, producing: `[n \in Node |-> ... WriteLog(n, index, n) ...]`
/// 3. The function comprehension's `n` variable captured both the original `n` and
///    what was meant to be `prospect`!
///
/// This caused each node to write its OWN name to the log instead of the `prospect` value.
///
/// The fix: Check `inlining_substitution_is_capture_safe` before inlining operators
/// during action preprocessing, similar to how `compiled_guard.rs` does.
#[test]
fn test_bug_preprocessing_variable_capture() {
    // Minimal reproduction of the MCCheckpointCoordination pattern
    let spec = r#"
---- MODULE PreprocessingVariableCapture ----
EXTENDS Naturals

CONSTANTS Node

VARIABLE log

LogIndex == {1, 2, 3}

\* WriteLog is called with `prospect` parameter - the key is that
\* the inner comprehension uses `i` not `n`, but the outer comprehension
\* in Send uses `n` which could capture `prospect` if inlining is unsafe.
WriteLog(node, idx, val) ==
    [i \in LogIndex |-> IF i = idx THEN val ELSE log[node][i]]

\* The bug pattern: EXISTS binds `n`, operator parameter is `prospect`,
\* body has `[n \in Node |-> ... WriteLog(n, index, prospect) ...]`
Send(prospect) ==
    log' = [n \in Node |-> WriteLog(n, 1, prospect)]

Init == log = [n \in Node |-> <<0, 0, 0>>]

Next == \E n \in Node: Send(n)

\* INVARIANT: All nodes should have the SAME value at position 1
\* (since they all received the same `prospect` argument)
AllSame == \A x, y \in Node: log[x][1] = log[y][1]

====
"#;

    let module = parse_module(spec);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["AllSame".to_string()],
        constants: vec![("Node".into(), ConstantValue::Value("{1, 2, 3}".into()))]
            .into_iter()
            .collect(),
        ..Default::default()
    };

    let result = check_module(&module, &config);

    match &result {
        CheckResult::Success(stats) => {
            // Should complete without invariant violation
            // Expected states: 1 initial + 3 successors (one for each prospect)
            // But successors are equivalent under symmetry, so may be fewer
            assert!(
                stats.states_found >= 2,
                "Should find at least 2 states, got {}",
                stats.states_found
            );
        }
        CheckResult::InvariantViolation { invariant, .. } => {
            panic!(
                "Bug exists! Invariant {} violated - variable capture during preprocessing. \
                 Each node is getting its own name instead of the prospect value.",
                invariant
            );
        }
        _ => {
            panic!("Unexpected result: {:?}", result);
        }
    }
}

// ============================================================================
// Bug #12: bosco regression - Cardinality of primed set comprehension in value operator
// ============================================================================

/// Bug #12: Value operators with primed set comprehensions must be validated.
///
/// Pattern (from `bosco`):
/// - A value operator returns Cardinality of a set comprehension over primed state
///   e.g., `MsgCount(self) == Cardinality({m \in rcvd'[self] : ...})`
/// - The Next relation combines an EXISTS that binds primed vars with guards using the value op
///   e.g., `Step(self) == Receive(self) /\ (... MsgCount(self) >= threshold ...)`
/// - The Receive action contains `\E r \in SUBSET ...: ... rcvd' = [rcvd EXCEPT ![self] = r]`
///
/// This is more complex than the simple `MsgCount(self) == Cardinality(rcvd'[self])` pattern
/// because the primed variable is inside a set comprehension filter, not just directly accessed.
///
/// Root cause: When the value operator body contains a set comprehension over primed state,
/// evaluation may fail or produce incorrect results if not properly handled against next-state.
#[test]
fn test_bosco_style_cardinality_primed_setcomp() {
    let spec = r#"
---- MODULE BoscoStylePrimedSetComp ----
EXTENDS Naturals, FiniteSets

CONSTANT N
VARIABLE pc, rcvd

Corr == 1 .. N
M == {"ECHO0", "ECHO1"}
vars == <<pc, rcvd>>

\* Value operator: counts messages of a specific type in NEXT state
\* This is the bosco pattern: Cardinality of set comprehension over rcvd'[self]
MsgCount0(self) == Cardinality({m \in rcvd'[self] : m[2] = "ECHO0"})

\* Value operator: counts distinct senders in NEXT state
MsgSenders(self) == Cardinality({p \in Corr : (\E m \in rcvd'[self] : m[1] = p)})

Init ==
  /\ pc = [i \in Corr |-> "S"]
  /\ rcvd = [i \in Corr |-> {}]

\* Receive: nondeterministically receive a subset of possible messages
Receive(self) ==
  \E r \in SUBSET (Corr \X M):
    /\ rcvd[self] \subseteq r
    /\ rcvd' = [rcvd EXCEPT ![self] = r]

\* Decide: can only decide if received enough messages
\* Guard requires at least N-1 messages from distinct senders AND at least 1 ECHO0
UponDecide(self) ==
  /\ pc[self] = "S"
  /\ MsgSenders(self) >= N - 1   \* Need msgs from N-1 senders (hidden primed var)
  /\ MsgCount0(self) >= 1        \* Need at least 1 ECHO0 (hidden primed var)
  /\ pc' = [pc EXCEPT ![self] = "D"]

\* Stay: don't decide, just update rcvd
UponStay(self) ==
  /\ pc' = pc

Step(self) ==
  /\ Receive(self)
  /\ (UponDecide(self) \/ UponStay(self))

Next == \E self \in Corr: Step(self)

\* INVARIANT: If a process decides, it must have received at least 1 message.
\* This checks that the guards were properly enforced.
DecideHasMsg ==
  \A i \in Corr: (pc[i] = "D") => (Cardinality(rcvd[i]) >= 1)

====
"#;

    let module = parse_module(spec);
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["DecideHasMsg".to_string()],
        ..Default::default()
    };
    config
        .constants
        .insert("N".to_string(), ConstantValue::Value("2".to_string()));

    let result = check_module(&module, &config);
    match &result {
        CheckResult::Success(_) => {}
        CheckResult::InvariantViolation { invariant, .. } => {
            panic!(
                "Invariant {} violated - bosco-style primed setcomp guard was not enforced. \
                 Process decided with rcvd=empty, meaning MsgCount0/MsgSenders guards \
                 containing primed set comprehensions were not validated.",
                invariant
            );
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}
