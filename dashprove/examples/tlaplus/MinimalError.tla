----------------------------- MODULE MinimalError -----------------------------
(* TLA+ spec with intentional syntax error. *)

EXTENDS Naturals

VARIABLE counter

Init == counter = 0

(* Syntax error: missing /\ between conditions *)
Next ==
    \/ counter < 3 counter' = counter + 1
    \/ counter = 3 /\ UNCHANGED counter

Spec == Init /\ [][Next]_counter

=============================================================================
