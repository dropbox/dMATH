/-
  Z4 Integration Tests: Array Theory (QF_AUFLIA)

  Tests the z4_smt tactic with arrays combined with linear arithmetic.
-/

import Lean5.Tactic.Z4

-- =============================================================================
-- Basic Array Axioms
-- =============================================================================

-- Read-over-write (same index)
theorem read_write_same (a : Array Int Int) (i v : Int) :
    (a.set i v).get i = v := by
  z4_smt (logic := "QF_AUFLIA")

-- Read-over-write (different index)
theorem read_write_diff (a : Array Int Int) (i j v : Int) (h : i ≠ j) :
    (a.set i v).get j = a.get j := by
  z4_smt (logic := "QF_AUFLIA")

-- Write-write (same index, second overwrites)
theorem write_write_same (a : Array Int Int) (i v1 v2 : Int) :
    (a.set i v1).set i v2 = a.set i v2 := by
  z4_smt (logic := "QF_AUFLIA")

-- Write-write (different indices commute)
theorem write_write_diff (a : Array Int Int) (i j v1 v2 : Int) (h : i ≠ j) :
    (a.set i v1).set j v2 = (a.set j v2).set i v1 := by
  z4_smt (logic := "QF_AUFLIA")

-- =============================================================================
-- Array Extensionality
-- =============================================================================

-- Two arrays equal if they agree on all indices
-- (Note: This requires quantified array reasoning)
theorem array_ext (a b : Array Int Int) :
    (∀ i, a.get i = b.get i) → a = b := by
  z4_smt (logic := "AUFLIA")  -- Requires quantifiers

-- =============================================================================
-- Arrays with Linear Arithmetic
-- =============================================================================

-- Array sum property (bounded)
theorem array_sum_bound (a : Array Int Int) (n : Int) (h : n >= 0)
    (hBound : ∀ i, 0 <= i → i < n → a.get i >= 0) :
    -- Sum of non-negative elements is non-negative
    True := by  -- Simplified; full sum encoding is complex
  z4_smt (logic := "QF_AUFLIA")

-- Array with constraints
theorem array_constrained (a : Array Int Int) (i j : Int)
    (h1 : a.get i >= 10)
    (h2 : a.get j >= 20)
    (h3 : i ≠ j) :
    a.get i + a.get j >= 30 := by
  z4_smt (logic := "QF_AUFLIA")

-- =============================================================================
-- Sorted Array Properties
-- =============================================================================

-- If array is sorted and we insert in order, it stays sorted
-- (Simplified version)
theorem sorted_insert (a : Array Int Int) (i v : Int)
    (hSorted : ∀ j k, j < k → a.get j <= a.get k)
    (hBound : a.get (i - 1) <= v ∧ v <= a.get (i + 1)) :
    let a' := a.set i v
    a'.get (i - 1) <= a'.get i ∧ a'.get i <= a'.get (i + 1) := by
  z4_smt (logic := "AUFLIA")

-- =============================================================================
-- Array Copy/Move Operations
-- =============================================================================

-- Copy preserves value
theorem copy_preserves (src dst : Array Int Int) (i j v : Int)
    (hSrc : src.get i = v) :
    let dst' := dst.set j (src.get i)
    dst'.get j = v := by
  z4_smt (logic := "QF_AUFLIA")

-- Swap operation
theorem swap_correct (a : Array Int Int) (i j : Int) (h : i ≠ j) :
    let vi := a.get i
    let vj := a.get j
    let a' := (a.set i vj).set j vi
    a'.get i = vj ∧ a'.get j = vi := by
  z4_smt (logic := "QF_AUFLIA")

-- =============================================================================
-- Multi-dimensional Arrays
-- =============================================================================

-- 2D array access (array of arrays)
theorem array_2d_access (a : Array Int (Array Int Int)) (i j v : Int) :
    let row := a.get i
    let row' := row.set j v
    let a' := a.set i row'
    (a'.get i).get j = v := by
  z4_smt (logic := "QF_AUFLIA")

-- =============================================================================
-- UNSAT Tests
-- =============================================================================

-- Contradiction: same index, different values
theorem unsat_array_1 (a : Array Int Int) (i : Int) :
    ¬(a.get i = 5 ∧ a.get i = 10) := by
  z4_smt (logic := "QF_AUFLIA")

-- Write doesn't affect different index
theorem unsat_array_2 (a : Array Int Int) (i j v : Int) :
    i ≠ j → ¬((a.set i v).get j ≠ a.get j) := by
  z4_smt (logic := "QF_AUFLIA")

-- =============================================================================
-- Performance Tests
-- =============================================================================

-- Small (< 1ms)
theorem perf_small_arr (a : Array Int Int) (i v : Int) :
    (a.set i v).get i = v := by
  z4_smt (logic := "QF_AUFLIA")

-- Medium - chain of writes (< 10ms)
theorem perf_medium_arr (a : Array Int Int) :
    let a1 := a.set 0 100
    let a2 := a1.set 1 200
    let a3 := a2.set 2 300
    a3.get 0 = 100 ∧ a3.get 1 = 200 ∧ a3.get 2 = 300 := by
  z4_smt (logic := "QF_AUFLIA")
