----------------------------- MODULE MinimalFail -----------------------------
(* Minimal TLA+ spec that should FAIL model checking.
   This spec has a counter that can exceed the invariant bound. *)

EXTENDS Naturals

VARIABLE counter

Init == counter = 0

Next ==
    \/ counter < 5 /\ counter' = counter + 1
    \/ counter = 5 /\ UNCHANGED counter

Spec == Init /\ [][Next]_counter

TypeOK == counter \in 0..5

(* This invariant will be VIOLATED because counter can reach 4 and 5 *)
Safety == counter >= 0 /\ counter <= 3

=============================================================================
