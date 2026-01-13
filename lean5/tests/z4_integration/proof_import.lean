/-
  Z4 Integration Tests: DRAT/LRAT Proof Import

  Tests verification of SAT solver proof certificates.
-/

import Lean5.Tactic.Z4

-- =============================================================================
-- DRAT Proof Verification
-- =============================================================================

-- Simple UNSAT with explicit DRAT proof
-- CNF: (a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b) ∧ (¬a ∨ ¬b)
-- This is unsatisfiable
theorem drat_simple : ∀ (a b : Prop), ¬((a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b) ∧ (¬a ∨ ¬b)) := by
  z4_decide_with_proof
  /-
  Expected DRAT proof:
  b 0        -- Derive unit clause b by resolving clauses 1 and 3
  0          -- Empty clause (contradiction)
  -/

-- Unit propagation chain
theorem drat_unit_prop : ∀ (a b c : Prop),
    (a) → (¬a ∨ b) → (¬b ∨ c) → c := by
  z4_decide_with_proof
  /-
  This should not need DRAT (SAT with model), but if negated:
  The negation (a ∧ (¬a ∨ b) ∧ (¬b ∨ c) ∧ ¬c) is UNSAT
  -/

-- Resolution proof
theorem drat_resolution (p q r : Prop) :
    ¬((p ∨ q) ∧ (¬p ∨ r) ∧ (¬q) ∧ (¬r)) := by
  z4_decide_with_proof
  /-
  Expected DRAT:
  p 0        -- From (p ∨ q) and (¬q)
  r 0        -- From (¬p ∨ r) and p
  0          -- Contradiction with (¬r)
  -/

-- =============================================================================
-- LRAT Proof Verification (with clause IDs)
-- =============================================================================

-- LRAT provides clause IDs for efficient checking
theorem lrat_simple (a b : Prop) :
    ¬((a) ∧ (¬a ∨ b) ∧ (¬b)) := by
  z4_decide_with_lrat_proof
  /-
  Expected LRAT proof:
  4 b 0 1 2 0      -- Derive b from clauses 1, 2
  5 0 3 4 0        -- Empty clause from clauses 3, 4
  -/

-- =============================================================================
-- Proof Format Tests
-- =============================================================================

-- Test that we correctly parse DRAT format
#check_proof_format drat """
1 2 0
-1 2 0
1 -2 0
-1 -2 0
2 0
0
"""

-- Test LRAT format parsing
#check_proof_format lrat """
1 1 2 0 0
2 -1 2 0 0
3 1 -2 0 0
4 -1 -2 0 0
5 2 0 1 3 0
6 0 2 4 5 0
"""

-- =============================================================================
-- Proof Reconstruction
-- =============================================================================

-- From DRAT proof, construct Lean proof term
theorem reconstruct_from_drat (a b c : Prop) :
    ((a ∨ b) ∧ (¬a ∨ c) ∧ (¬b ∨ c) ∧ ¬c) → False := by
  z4_decide_with_proof
  -- The tactic should:
  -- 1. Call Z4, get UNSAT + DRAT proof
  -- 2. Verify DRAT proof
  -- 3. Construct Lean proof term from verified resolution steps

-- =============================================================================
-- Large Proof Tests
-- =============================================================================

-- Pigeonhole principle: 4 pigeons, 3 holes (PHP_4^3)
-- This generates a non-trivial DRAT proof
theorem php_4_3 : ¬∃ (f : Fin 4 → Fin 3), Function.Injective f := by
  z4_decide_with_proof (timeout := 5000)
  -- Expected: UNSAT with proof ~100-500 clauses

-- =============================================================================
-- Proof Certificate Caching
-- =============================================================================

-- Same formula proved twice should use cached proof
theorem cached_proof_1 (p q : Prop) : p ∨ ¬p := by
  z4_decide

theorem cached_proof_2 (p q : Prop) : p ∨ ¬p := by
  z4_decide  -- Should hit cache

-- =============================================================================
-- Incremental Proof Verification
-- =============================================================================

-- Proof verification should be incremental for efficiency
theorem incremental_proof (a b c d e : Prop) :
    let clause1 := a ∨ b
    let clause2 := ¬a ∨ c
    let clause3 := ¬b ∨ d
    let clause4 := ¬c ∨ e
    let clause5 := ¬d ∨ e
    let clause6 := ¬e
    ¬(clause1 ∧ clause2 ∧ clause3 ∧ clause4 ∧ clause5 ∧ clause6) := by
  z4_decide_with_proof

-- =============================================================================
-- Error Handling
-- =============================================================================

-- Invalid DRAT proof should be rejected
-- (This is a meta-test; the actual test checks error handling)
#check_invalid_proof drat """
1 2 0
-1 -2 0
3 0
"""
-- Error expected: clause "3 0" is not RAT with respect to formula

-- =============================================================================
-- Performance Metrics
-- =============================================================================

-- Measure proof verification overhead
-- Verification should add < 10% overhead to solving time
#bench_proof_overhead 100 (a b c : Prop) :
    ¬((a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b) ∧ (¬a ∨ ¬b))
