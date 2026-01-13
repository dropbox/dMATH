(* Minimal Coq example with provable theorems *)

(* Trivially true *)
Theorem trivial_true : True.
Proof. trivial. Qed.

(* Excluded middle - requires classical logic *)
Require Import Classical.
Theorem excluded_middle : forall P : Prop, P \/ ~P.
Proof. intros. apply classic. Qed.

(* Simple arithmetic *)
Require Import Arith.
Theorem add_comm : forall n m : nat, n + m = m + n.
Proof. intros. apply Nat.add_comm. Qed.
