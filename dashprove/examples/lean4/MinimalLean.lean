/-
  Minimal LEAN 4 examples demonstrating theorem proving.
  These examples are designed to show different verification outcomes.
-/

-- Theorem that should PASS: basic logical implication
theorem pass_simple : ∀ (a b : Prop), a → a := by
  intro a _ ha
  exact ha

-- Theorem that should PASS: conjunction elimination
theorem pass_and_elim : ∀ (a b : Prop), a ∧ b → a := by
  intro a b hab
  exact hab.left

-- Theorem that should PASS: natural number property
theorem pass_nat : ∀ n : Nat, n + 0 = n := by
  intro n
  rfl

-- Theorem that should PASS: arithmetic
theorem pass_arith : 2 + 2 = 4 := by
  rfl
