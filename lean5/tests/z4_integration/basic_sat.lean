/-
  Z4 Integration Tests: Basic SAT

  Tests simple satisfiability problems and proof extraction.
-/

import Lean5.Tactic.Z4  -- Will be implemented in lean5-elab

-- =============================================================================
-- SAT Tests: Should find satisfying models
-- =============================================================================

-- Simple satisfiable formula
-- Expected: SAT with model {p := true}
#check_sat p ∨ ¬p

-- Conjunction that is satisfiable
-- Expected: SAT with model {p := true, q := true}
#check_sat (p ∧ q) ∨ (¬p ∧ ¬q)

-- Implication that is satisfiable
-- Expected: SAT with model {p := false} or {p := true, q := true}
#check_sat p → q

-- =============================================================================
-- UNSAT Tests: Should return proofs
-- =============================================================================

-- Direct contradiction
-- Expected: UNSAT with trivial proof
theorem unsat_contradiction (p : Prop) : ¬(p ∧ ¬p) := by
  z4_decide

-- Transitivity leads to contradiction
-- Expected: UNSAT
theorem unsat_transitivity (p q r : Prop) :
    ¬((p → q) ∧ (q → r) ∧ p ∧ ¬r) := by
  z4_decide

-- Resolution example
-- Expected: UNSAT with resolution proof
theorem unsat_resolution (p q : Prop) :
    ¬((p ∨ q) ∧ (¬p ∨ q) ∧ (p ∨ ¬q) ∧ (¬p ∨ ¬q)) := by
  z4_decide

-- =============================================================================
-- Proof Verification Tests
-- =============================================================================

-- This should generate a DRAT proof and verify it
theorem drat_proof_test (a b c : Prop) :
    ((a ∨ b) ∧ (¬a ∨ c) ∧ (¬b ∨ c) ∧ ¬c) → False := by
  z4_decide_with_proof

-- =============================================================================
-- Performance Baseline Tests
-- =============================================================================

-- Small problem (should complete in < 1ms)
theorem perf_small : ∀ (p : Prop), p ∨ ¬p := by
  intro p
  z4_decide

-- Medium problem - pigeonhole principle for 3 pigeons, 2 holes
-- (should complete in < 10ms)
theorem perf_pigeonhole_3_2 :
    ¬(∃ (p1 p2 p3 : Fin 2), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) := by
  z4_decide
