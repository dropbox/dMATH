/-
  LEAN 4 examples that should FAIL compilation.
  These demonstrate different error types.
-/

-- Theorem that CANNOT be proven: false implication
theorem fail_false_impl : ∀ (a : Prop), a := by
  intro a
  -- This is stuck - we cannot prove an arbitrary proposition
  sorry

-- Theorem with WRONG proof attempt
theorem fail_wrong_proof : 2 + 2 = 5 := by
  rfl  -- This will fail: rfl cannot prove 4 = 5

-- Theorem with type mismatch
theorem fail_type_mismatch : ∀ n : Nat, n = n + 1 := by
  intro n
  rfl  -- This will fail: n ≠ n + 1
