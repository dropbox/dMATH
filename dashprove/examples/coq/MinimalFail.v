(* Minimal Coq example with unprovable theorem *)

(* This is false and cannot be proven *)
Theorem false_theorem : False.
Proof.
  (* No way to prove False without contradiction *)
Admitted. (* Proof abandoned *)
