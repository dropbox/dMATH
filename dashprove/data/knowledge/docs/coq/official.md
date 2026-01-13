# [Rocq logo] [Rocq logo]

A trustworthy, industrial-strength interactive theorem prover and dependently-typed programming
language for mechanised reasoning in mathematics, computer science and more.

[Install][1] [ About The Rocq Prover ][2] [ Why Rocq? ][3]
[Latest Rocq Prover release: 9.1.0][4]
[Latest Rocq Platform release: 2025.01.0][5]
The Rocq Prover was formerly known as the Coq Proof Assistant (see more on [the name evolution][6]).

## A short introduction to the Rocq Prover

The Rocq Prover is an interactive theorem prover, or proof assistant. This means that it is designed
to develop mathematical proofs, and especially to write formal specifications: programs and proofs
that programs comply to their specifications. An interesting additional feature of Rocq is that it
can automatically extract executable programs from specifications, as either OCaml or Haskell source
code.
From Stdlib Require Import Lia.

> [Loading ML file ring_plugin.cmxs (using legacy method) ... done]

> [Loading ML file zify_plugin.cmxs (using legacy method) ... done]

> [Loading ML file micromega_core_plugin.cmxs (using legacy method) ... done]

> [Loading ML file micromega_plugin.cmxs (using legacy method) ... done]



Fixpoint fac n :=
  match n with
  | 0 => 1
  | S n' => n * fac n'
  end.

Compute fac 5.

> = 120 : nat



Lemma fac_succ n : fac (S n) = S n * fac n.

> n: nat
> fac (S n) = S n * fac n


Proof.

> n: nat
> fac (S n) = S n * fac n

 reflexivity. Qed.

Lemma fac_pos n : fac n > 0.

> n: nat
> fac n > 0


Proof.

> n: nat
> fac n > 0


  induction n.

> fac 0 > 0

> n: nat
> IHn: fac n > 0
> fac (S n) > 0


  -

> fac 0 > 0

 simpl.

> 1 > 0

 lia.
  -

> n: nat
> IHn: fac n > 0
> fac (S n) > 0

 rewrite fac_succ.

> n: nat
> IHn: fac n > 0
> S n * fac n > 0

 lia.
Qed.

From Corelib Require Import Extraction.

> [Loading ML file extraction_plugin.cmxs (using legacy method) ... done]


Extraction Language OCaml.
Extraction fac.

> (** val fac : nat -> nat **) let rec fac n = match n with | O -> S O | S n' -> mul n (fac n')

## Trusted in Academia and Industry

These organisations and companies rely on Rocq every day — along with hundreds of other researchers
and engineers.
[ [Inria] [Inria] ][7]
[ [Université de Paris] [Université de Paris] ][8]
[ [Université de Nantes] [Université de Nantes] ][9]
[ [Polytechnic Institute of Paris] [Polytechnic Institute of Paris] ][10]
[ [Collège de France] [Collège de France] ][11]
[ [University of Pennsylvania] [University of Pennsylvania] ][12]
[ [MIT Massachusetts Institute of Technology] [MIT Massachusetts Institute of Technology] ][13]
[ [Max Planck Institute for Software Systems] [Max Planck Institute for Software Systems] ][14]
[ [Arhus University] [Arhus University] ][15]
[ [AbsInt] [AbsInt] ][16]
[ [BlueRock] [BlueRock] ][17]
[ [Formal Vindication] [Formal Vindication] ][18]
[ [Google] [Google] ][19]

## The **Rocq Prover** has been **recognised** by the **ACM** for their prestigious **Software
## System Award**.

The recipients of the awards are thanking all developers who contributed to the success of the Rocq
Prover, the users who illustrated how to use Rocq for so many impressive projects in formal
certification, programming, logic, formalization of mathematics and teaching as well as the whole
surrounding scientific community in proof theory, type theory, programming languages, interactive
theorem proving.
[Read more][20]

### RELIABILITY
### 
### A Foundationally Sound, Trustworthy Formal Language and Implementation

Rocq's highly expressive type system and proof language enable fully mechanised verification of
programs with respect to strong specifications in a wide variety of languages. Through the
Curry-Howard lens, it simultaneously provides a rich logic and foundational computational theory for
mathematical developments, supported by a flexible proof environment. Its well-studied core type
theory, resulting from over 40 years of research, is implemented in a well-delimited kernel using
the performant and safe OCaml programming language, providing the highest possible guarantees on
mechanised artifacts. The core type theory is itself formalised in Rocq in the MetaRocq project, a
verified reference checker is proven correct and complete with respect to this specification and can
be extracted to reduce the trusted code base of any formalization to the specification of Rocq's
theory and the user specification.

[trustworthy]

### DIVERSITY OF APPLICATIONS
### From Low-Level Verification to Homotopy Type Theory

The Rocq Prover enables a very wide variety of developments to coexist in the same system, ranging
from end-to-end verified software and hardware models to the development of higher-dimensional
mathematics. Flagship projects using Rocq include the Mathematical Components library and its
derived proofs of the Four-Color and Feith-Thompson theorems; the verified CompCert C compiler and
the associated Verified Software Toolchain for proofs of C-like programs, or the development of
Homotopy Type Theory and Univalent Foundations of mathematics. Rocq is also a great vehicle for
teaching logic and computer science as exemplified by the thousands of students that have gone
through the [Software Foundations][21] series of books. Rocq's development is entirely open-source
and a large and diverse community of users participate in its continued evolution.

COMPCERT
Mathematical Components
(* (c) Copyright 2006-2018 Microsoft Corporation and Inria.                  *)
(* Distributed under the terms of CeCILL-B.                                  *)
From fourcolor Require Import real realplane.
From fourcolor Require combinatorial4ct discretize finitize.

> [Loading ML file ssrmatching_plugin.cmxs (using legacy method) ... done]

> [Loading ML file ssreflect_plugin.cmxs (using legacy method) ... done]

> [Loading ML file ring_plugin.cmxs (using legacy method) ... done]

> Serlib plugin: coq-elpi.elpi is not available: serlib support is missing. Incremental checking for
> commands in this plugin will be impacted.

> [Loading ML file coq-elpi.elpi ... done]



(******************************************************************************)
(*   This files contains the proof of the high-level statement of the Four    *)
(* Color Theorem, whose statement uses only the elementary real topology      *)
(* defined in libraries real and realplane. The theorem is stated for an      *)
(* arbitrary model of the real line, which we show in separate libraries      *)
(* (dedekind and realcategorical) is equivalent to assuming the classical     *)
(* excluded middle axiom.                                                     *)
(*   We only import the real and realplane libraries, which do not introduce  *)
(* any extra-logical context, in particular no new notation, so that the      *)
(* interpretation of the text below is as transparent as possible.            *)
(*   Accordingly we use qualified names refer to the supporting result in the *)
(* finitize, discretize and combinatorial4ct libraries, and do not rely on    *)
(* the ssreflect extensions in the formulation of the final arguments.        *)
(******************************************************************************)
Section FourColorTheorem.

Variable Rmodel : Real.model.
Let R := Real.model_structure Rmodel.
Implicit Type m : map R.

Theorem four_color_finite m : finite_simple_map m -> colorable_with 4 m.

> Rmodel: Real.model
> R:= Real.model_structure Rmodel: Real.structure
> m: map R
> finite_simple_map (R:=R) m -> colorable_with (R:=R) 4 m


Proof.

> Rmodel: Real.model
> R:= Real.model_structure Rmodel: Real.structure
> m: map R
> finite_simple_map (R:=R) m -> colorable_with (R:=R) 4 m


intros fin_m.

> Rmodel: Real.model
> R:= Real.model_structure Rmodel: Real.structure
> m: map R
> fin_m: finite_simple_map (R:=R) m
> colorable_with (R:=R) 4 m


pose proof (discretize.discretize_to_hypermap fin_m) as [G planarG colG].

> Rmodel: Real.model
> R:= Real.model_structure Rmodel: Real.structure
> m: map R
> fin_m: finite_simple_map (R:=R) m
> G: hypermap.hypermap
> planarG: geometry.planar_bridgeless G
> colG: coloring.four_colorable G -> colorable_with (R:=Real.model_structure Rmodel) 4 m
> colorable_with (R:=R) 4 m


exact (colG (combinatorial4ct.four_color_hypermap planarG)).
Qed.

Theorem four_color m : simple_map m -> colorable_with 4 m.

> Rmodel: Real.model
> R:= Real.model_structure Rmodel: Real.structure
> m: map R
> simple_map (R:=R) m -> colorable_with (R:=R) 4 m


Proof.

> Rmodel: Real.model
> R:= Real.model_structure Rmodel: Real.structure
> m: map R
> simple_map (R:=R) m -> colorable_with (R:=R) 4 m

 revert m; exact (finitize.compactness_extension four_color_finite). Qed.

End FourColorTheorem.
Homotopy Type Theory

Set Universe Polymorphism.

(* Equivalences *)

Class IsEquiv {A : Type} {B : Type} (f : A -> B) := BuildIsEquiv {
  e_inv : B -> A ;
  e_sect : forall x, e_inv (f x) = x;
  e_retr : forall y, f (e_inv y) = y;
  e_adj : forall x : A, e_retr (f x) = ap f (e_sect x);
}.


(** A class that includes all the data of an adjoint equivalence. *)
Class Equiv A B := BuildEquiv {
  e_fun : A -> B ;
  e_isequiv :> IsEquiv e_fun
}.

Definition is_adjoint' {A B : Type} (f : A -> B) (g : B -> A)
           (issect : g∘ f == id) (isretr : f  ∘ g == id) (a : A) :
  isretr (f a) = ap f (issect' f g issect isretr a).

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> isretr (f a) = ap f (issect' f g issect isretr a)


Proof.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> isretr (f a) = ap f (issect' f g issect isretr a)


  unfold issect'.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> isretr (f a) = ap f ((ap g (ap f (issect a)^) @ ap g (isretr (f a))) @ issect a)


  apply moveL_M1.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> eq_refl = ap f ((ap g (ap f (issect a)^) @ ap g (isretr (f a))) @ issect a) @ (isretr (f a))^


  repeat rewrite ap_pp.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> eq_refl = ((ap f (ap g (ap f (issect a)^)) @ ap f (ap g (isretr (f a)))) @ ap f (issect a)) @
> (isretr (f a))^

 rewrite <- concat_p_pp; rewrite <- ap_compose.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> eq_refl = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ (ap f
> (issect a) @ (isretr (f a))^)


  pose  (concat_pA1 (fun b => (isretr b)^) (ap f (issect a))).

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> eq_refl = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ (ap f
> (issect a) @ (isretr (f a))^)


  eapply concat.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> eq_refl = ?y

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> ?y = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ (ap f (issect
> a) @ (isretr (f a))^)

 2: {

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> ?y = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ (ap f (issect
> a) @ (isretr (f a))^)

 apply ap2.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> ?x = ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> ?x' = ap f (issect a) @ (isretr (f a))^

 reflexivity.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> ?x' = ap f (issect a) @ (isretr (f a))^

 exact e. }

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> eq_refl = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ ((fun b :
> B => (isretr b)^) (f (g (f a))) @ ap (fun b : B => f (g b)) (ap f (issect a)))


  cbn.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (ap f (issect a)): (fun b : B => (isretr b)^) (f (g (f
> a))) @ ap (fun b : B => f (g b)) (ap f (issect a)) = ap f (issect a) @ (fun b : B => (isretr b)^)
> (f a)
> eq_refl = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ ((isretr
> (f (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a)))

 clear e.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> eq_refl = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ ((isretr
> (f (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a)))


  pose (concat_pA1 (fun b => (isretr b)^) (isretr (f a))).

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> eq_refl = (ap (fun x : B => f (g x)) (ap f (issect a)^) @ ap f (ap g (isretr (f a)))) @ ((isretr
> (f (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a)))


  rewrite <- concat_p_pp.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  pose (concat_A1p (fun b => (isretr b)) (isretr (f a))).

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0:= concat_A1p (fun b : B => isretr b) (isretr (f a)): ap (fun x : B => f (g x)) (isretr (f a)) @
> (fun b : B => isretr b) (f a) = (fun b : B => isretr b) (f (g (f a))) @ isretr (f a)
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  apply moveL_Vp in e0.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap (fun x : B => f (g x)) (isretr (f a)) = (isretr (f (g (f a))) @ isretr (f a)) @ (isretr (f
> a))^
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  rewrite <- concat_p_pp in e0.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap (fun x : B => f (g x)) (isretr (f a)) = isretr (f (g (f a))) @ (isretr (f a) @ (isretr (f
> a))^)
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))

 rewrite inv_inv' in e0.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap (fun x : B => f (g x)) (isretr (f a)) = isretr (f (g (f a))) @ eq_refl
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  rewrite concat_refl in e0.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap (fun x : B => f (g x)) (isretr (f a)) = isretr (f (g (f a)))
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  rewrite ap_compose in e0.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f
> (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  eapply concat.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> eq_refl = ?y

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f (g
> (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))


  2: {

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (ap f (ap g (isretr (f a))) @ ((isretr (f (g
> (f a))))^ @ ap (fun b : B => f (g b)) (ap f (issect a))))

 apply ap2.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x = ap (fun x : B => f (g x)) (ap f (issect a)^)

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x' = ap f (ap g (isretr (f a))) @ ((isretr (f (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f
> (issect a)))

 reflexivity.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x' = ap f (ap g (isretr (f a))) @ ((isretr (f (g (f a))))^ @ ap (fun b : B => f (g b)) (ap f
> (issect a)))

 rewrite concat_p_pp.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x' = (ap f (ap g (isretr (f a))) @ (isretr (f (g (f a))))^) @ ap (fun b : B => f (g b)) (ap f
> (issect a))

 eapply concat.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x' = ?y

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = (ap f (ap g (isretr (f a))) @ (isretr (f (g (f a))))^) @ ap (fun b : B => f (g b)) (ap f
> (issect a))


       2: {

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = (ap f (ap g (isretr (f a))) @ (isretr (f (g (f a))))^) @ ap (fun b : B => f (g b)) (ap f
> (issect a))

 apply ap2.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x = ap f (ap g (isretr (f a))) @ (isretr (f (g (f a))))^

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'0 = ap (fun b : B => f (g b)) (ap f (issect a))

 eapply concat.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x = ?y

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = ap f (ap g (isretr (f a))) @ (isretr (f (g (f a))))^

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'0 = ap (fun b : B => f (g b)) (ap f (issect a))


            2:{

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = ap f (ap g (isretr (f a))) @ (isretr (f (g (f a))))^

 apply ap2.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x0 = ap f (ap g (isretr (f a)))

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'1 = (isretr (f (g (f a))))^

 symmetry.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ap f (ap g (isretr (f a))) = ?x0

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'1 = (isretr (f (g (f a))))^

 apply e0.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'1 = (isretr (f (g (f a))))^

 reflexivity. }

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x = isretr (f (g (f a))) @ (isretr (f (g (f a))))^

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'0 = ap (fun b : B => f (g b)) (ap f (issect a))


              symmetry.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> isretr (f (g (f a))) @ (isretr (f (g (f a))))^ = ?x

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'0 = ap (fun b : B => f (g b)) (ap f (issect a))

 apply inv_inv'.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x'0 = ap (fun b : B => f (g b)) (ap f (issect a))

 reflexivity. }

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?x' = eq_refl @ ap (fun b : B => f (g b)) (ap f (issect a))


              reflexivity. }

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> eq_refl = ap (fun x : B => f (g x)) (ap f (issect a)^) @ (eq_refl @ ap (fun b : B => f (g b)) (ap
> f (issect a)))


  repeat rewrite <- ap_compose.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> eq_refl = ap (fun x : A => f (g (f x))) (issect a)^ @ (eq_refl @ ap (fun x : A => f (g (f x)))
> (issect a))


  cbn.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> eq_refl = ap (fun x : A => f (g (f x))) (issect a)^ @ ap (fun x : A => f (g (f x))) (issect a)

 symmetry.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ap (fun x : A => f (g (f x))) (issect a)^ @ ap (fun x : A => f (g (f x))) (issect a) = eq_refl

 eapply concat.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ap (fun x : A => f (g (f x))) (issect a)^ @ ap (fun x : A => f (g (f x))) (issect a) = ?y

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ?y = eq_refl

 refine (ap_pp ((f ∘ g) ∘f) _ _)^.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ap (f ∘ g) ∘ f ((issect a)^ @ issect a) = eq_refl


  rewrite inv_inv.

> A, B: Type
> f: A -> B
> g: B -> A
> issect: g ∘ f == id
> isretr: f ∘ g == id
> a: A
> e:= concat_pA1 (fun b : B => (isretr b)^) (isretr (f a)): (fun b : B => (isretr b)^) (f (g (f a)))
> @ ap (fun b : B => f (g b)) (isretr (f a)) = isretr (f a) @ (fun b : B => (isretr b)^) (f a)
> e0: ap f (ap g (isretr (f a))) = isretr (f (g (f a)))
> ap (fun x : A => f (g (f x))) eq_refl = eq_refl

 reflexivity.
Defined.

Definition isequiv_adjointify {A B : Type} (f : A -> B) (g : B -> A)
           (issect : g∘ f == id) (isretr : f  ∘ g == id)  : IsEquiv f
  := BuildIsEquiv A B f g (issect' f g issect isretr) isretr
                  (is_adjoint' f g issect isretr).
[Iris]
[Iris Proof Script]
[Iris Proof Goal]

### EXTENSIBILITY AND CUSTOMIZABILITY
### Elaboration, Metaprogramming and Embedded Domain-Specific Logics and Languages

Developing formal proofs and verified programs often requires extending the core language with
domain-specific notations, proof strategies and application-specific structures. The Rocq Prover
provides many mechanisms to tailor the environment to one's requirements and structure developments.
Rocq comes with a built-in lightweight scoped notation system, a coercion system and typeclass
systems that allow user-defined extension of the elaboration phases. This support is essential for
developing embedded domain-specific logics and languages, as exemplified in the [Iris][22] project
which allows to reason about effectful programs in languages like [Rust][23] using sophisticated
variants of separation logic.

### PERFORMANCE
### Fast Proof Checker

The Rocq Prover offers a finely-tuned proof engine and kernel implementation allowing large-scale
formalization, with efficient bytecode and native conversion checkers relying on the OCaml runtime.
It can also interoperate with code written in other languages thanks to its unique extraction
facilities.

[perf]
[profiler]

### Releases

Recent Releases
[

### Coq Platform 2025.01.0 (2025-02-06)

][24]

* For Coq 8.20.1
* Coq 8.12.2-8.19.0 available
[

### Coq Platform 2024.10.1 (2024-12-02)

][25]

* For Coq 8.19.2
* Coq 8.12.2-8.18.0 available
* Compatibility with opam 2.3.0
[

### Rocq Prover 9.1.0 (2025-09-15)

][26]

* Fixed incorrect guard checking leading to inconsistencies
* Sort polymorphic universe instances should now be written as `@{s ; u}` instead of `@{s | u}`
* Fixed handling of notation variables for ltac2 in notations (i.e. `Notation "'foo' x" :=
  ltac2:(...)`)
* Support for `refine` attribute in `Definition`
* Rocq can be compile-time configured to be relocatable
* Extraction handles sort polymorphic definitions
[ See All Releases ][27]

### Changelog

[
Releases & Updates

### Rocq Prover and Rocq Platform

][28]
[Release of Rocq 9.0][29] 12 Mar 2025

We have the pleasure of announcing the first release of Rocq 9. The main changes are: - "The Roc...

[ See full changelog ][30]
[Release of Coq 8.20.1][31] 20 Jan 2025

We have the pleasure of announcing the release of Coq 8.20.1. The full list of changes is
availabl...

[ See full changelog ][32]
[Preview of rocq-prover.org, open for contributions][33] 25 Dec 2024

See full changelog

[ See Full Changelog ][34]

## Users of Rocq

Rocq is used by hundreds of developers, companies, research labs, teachers, and more. Learn how it
fits your use case.

### For Educators

With its mathematical roots, the Rocq Prover has always had strong ties to academia. It is taught in
universities around the world, and has accrued an ever-growing body of research. Learn more about
the academic rigor that defines the culture of Rocq.

[Learn more][35]

### For Industrial Users

Rocq's strong correctness guarantees and high performance empower companies to provide reliable
services and products. Learn more about how Rocq is used in the industry.

[Learn more][36]

## Curated Resources

Get up to speed quickly to enjoy the benefits of the Rocq Prover across your projects.

[

Getting Started

Install the Rocq Prover, set up your favorite text editor and start your first project.

][37] [

Reference Manual

The authoritative reference for the Rocq Prover (not learning oriented).

][38] [

Books

Discover books on Rocq for both computer science and mathematics - from complete beginner level to
advanced topics.

][39] [

Standard Library

The documentation of the standard library.

][40] [

Exercises

Learn Rocq by solving problems on a variety of topics, from easy to challenging.

][41] [

Papers

Explore papers that have influenced or used the Rocq Prover.

][42]

## Rocq Packages

Explore hundreds of open-source Rocq packages with their documentation.

[ Explore packages ][43]

[1]: /install
[2]: /about
[3]: /why
[4]: /releases/9.1.0
[5]: /releases/2025.01.0
[6]: about#Name
[7]: https://www.inria.fr/
[8]: https://www.irif.fr/
[9]: https://www.ls2n.fr/
[10]: https://www.telecom-paris.fr/en/research/labs/information-processing-ltci
[11]: https://www.college-de-france.fr
[12]: https://www.upenn.edu/
[13]: https://web.mit.edu/
[14]: https://www.mpi-sws.org/
[15]: https://cs.au.dk/
[16]: https://absint.com/
[17]: https://bluerock.io
[18]: https://formalv.com
[19]: https://google.com
[20]: about#awards
[21]: /books
[22]: iris-project.org
[23]: https://www.rust-lang.org
[24]: /releases/2025.01.0
[25]: /releases/2024.10.1
[26]: /releases/9.1.0
[27]: /releases
[28]: /changelog
[29]: /changelog/2025-03-12-rocq-9.0
[30]: /changelog
[31]: /changelog/2025-01-20-coq-8.20.1
[32]: /changelog
[33]: /changelog/2024-12-25-rocq-prover-org-is-launched
[34]: /changelog
[35]: /academic-users
[36]: /industrial-users
[37]: /install
[38]: /refman
[39]: /books
[40]: /stdlib
[41]: /exercises
[42]: /papers
[43]: /packages
