----------------------------- MODULE MinimalPass -----------------------------
(* Minimal TLA+ spec that should pass model checking.
   This spec models a simple counter from 0 to 3. *)

EXTENDS Naturals

VARIABLE counter

Init == counter = 0

Next ==
    \/ counter < 3 /\ counter' = counter + 1
    \/ counter = 3 /\ UNCHANGED counter

Spec == Init /\ [][Next]_counter

TypeOK == counter \in 0..3

Safety == counter >= 0 /\ counter <= 3

=============================================================================
