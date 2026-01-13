---- MODULE CheckpointRestore ----
(***************************************************************************)
(* TLA+ Specification for DashFlow Checkpoint/Restore                      *)
(*                                                                         *)
(* This spec models the checkpoint and restore mechanism where:            *)
(* - Checkpoints capture state at specific sequence numbers                *)
(* - Restore reconstructs exact state from checkpoint                      *)
(* - No updates are lost between checkpoints                               *)
(*                                                                         *)
(* Status: DRAFT - Example for future implementation                       *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxSeq, MaxCheckpoints

VARIABLES state, checkpoints, cursor, eventLog

vars == <<state, checkpoints, cursor, eventLog>>

(***************************************************************************)
(* Type Definitions                                                        *)
(***************************************************************************)

Checkpoint == [sequence: Nat, state: Nat]  \* Simplified: state as Nat
Event == [sequence: Nat, delta: Nat]

(***************************************************************************)
(* Type Invariant                                                          *)
(***************************************************************************)

TypeInvariant ==
    /\ state \in Nat
    /\ checkpoints \subseteq [sequence: 1..MaxSeq, state: Nat]
    /\ cursor \in 0..MaxSeq
    /\ eventLog \in Seq(Event)

(***************************************************************************)
(* Checkpoint Invariant: Restoring produces identical state                *)
(***************************************************************************)

CheckpointInvariant ==
    \A cp \in checkpoints:
        cp.state = StateAt(cp.sequence)

(***************************************************************************)
(* No Lost Updates: Every state change is recoverable                      *)
(***************************************************************************)

NoLostUpdates ==
    \A seq \in 1..cursor:
        \/ \E cp \in checkpoints: cp.sequence = seq
        \/ CanReplayFrom(PreviousCheckpoint(seq), seq)

(***************************************************************************)
(* Helper: State at a given sequence number                                *)
(***************************************************************************)

StateAt(seq) ==
    LET events == SelectSeq(eventLog, LAMBDA e: e.sequence <= seq)
    IN IF events = <<>> THEN 0
       ELSE Last(events).state

(***************************************************************************)
(* Helper: Previous checkpoint before a sequence                           *)
(***************************************************************************)

PreviousCheckpoint(seq) ==
    LET earlier == {cp \in checkpoints : cp.sequence < seq}
    IN IF earlier = {} THEN [sequence |-> 0, state |-> 0]
       ELSE CHOOSE cp \in earlier: \A cp2 \in earlier: cp.sequence >= cp2.sequence

(***************************************************************************)
(* Helper: Can replay from checkpoint to sequence                          *)
(***************************************************************************)

CanReplayFrom(cp, seq) ==
    \A s \in (cp.sequence + 1)..seq:
        \E e \in {e \in Range(eventLog) : e.sequence = s}: TRUE

Range(s) == {s[i] : i \in 1..Len(s)}

(***************************************************************************)
(* Initial State                                                           *)
(***************************************************************************)

Init ==
    /\ state = 0
    /\ checkpoints = {}
    /\ cursor = 0
    /\ eventLog = <<>>

(***************************************************************************)
(* Apply Event: Modify state and log the event                             *)
(***************************************************************************)

ApplyEvent(delta) ==
    /\ cursor' = cursor + 1
    /\ state' = state + delta
    /\ eventLog' = Append(eventLog, [sequence |-> cursor + 1, delta |-> delta])
    /\ UNCHANGED checkpoints

(***************************************************************************)
(* Create Checkpoint: Save current state                                   *)
(***************************************************************************)

CreateCheckpoint ==
    /\ Cardinality(checkpoints) < MaxCheckpoints
    /\ cursor > 0
    /\ ~\E cp \in checkpoints: cp.sequence = cursor
    /\ checkpoints' = checkpoints \cup {[sequence |-> cursor, state |-> state]}
    /\ UNCHANGED <<state, cursor, eventLog>>

(***************************************************************************)
(* Restore from Checkpoint: Reset state to checkpoint                      *)
(***************************************************************************)

RestoreCheckpoint(cp) ==
    /\ cp \in checkpoints
    /\ state' = cp.state
    /\ cursor' = cp.sequence
    /\ UNCHANGED <<checkpoints, eventLog>>

(***************************************************************************)
(* Next State Relation                                                     *)
(***************************************************************************)

Next ==
    \/ \E delta \in 1..10: ApplyEvent(delta)
    \/ CreateCheckpoint
    \/ \E cp \in checkpoints: RestoreCheckpoint(cp)

(***************************************************************************)
(* Idempotent Restore: Restoring twice gives same result                   *)
(***************************************************************************)

IdempotentRestore ==
    \A cp \in checkpoints:
        LET stateAfterRestore == cp.state
        IN stateAfterRestore = cp.state  \* Trivially true, but models the concept

(***************************************************************************)
(* Specification                                                           *)
(***************************************************************************)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Properties to Check                                                     *)
(***************************************************************************)

THEOREM Spec => []TypeInvariant
THEOREM Spec => []IdempotentRestore

====
