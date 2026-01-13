/-
  Z4 Integration Tests: Linear Integer Arithmetic (QF_LIA)

  Tests the z4_omega tactic for linear integer arithmetic goals.
-/

import Lean5.Tactic.Z4

-- =============================================================================
-- Basic Inequalities
-- =============================================================================

-- Non-negativity of sum
theorem add_nonneg (x y : Int) (hx : x >= 0) (hy : y >= 0) : x + y >= 0 := by
  z4_omega

-- Non-negativity of square (encoded as product bounds)
theorem square_nonneg (x : Int) : x * x >= 0 := by
  z4_omega  -- May need nonlinear extension

-- Absolute value bound
theorem abs_bound (x : Int) (h : -5 <= x) (h' : x <= 5) : -10 <= 2 * x := by
  z4_omega

-- =============================================================================
-- Transitivity and Chains
-- =============================================================================

-- Simple transitivity
theorem lt_trans (x y z : Int) (h1 : x < y) (h2 : y < z) : x < z := by
  z4_omega

-- Long chain
theorem chain_4 (a b c d e : Int)
    (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) : a < e := by
  z4_omega

-- Mixed chain with equality
theorem chain_mixed (a b c d : Int)
    (h1 : a <= b) (h2 : b = c) (h3 : c < d) : a < d := by
  z4_omega

-- =============================================================================
-- Linear Combinations
-- =============================================================================

-- Weighted sum
theorem weighted_sum (x y : Int) (hx : x >= 1) (hy : y >= 2) : 3*x + 2*y >= 7 := by
  z4_omega

-- Negative coefficients
theorem neg_coeff (x y : Int) (h1 : x + y >= 10) (h2 : x <= 3) : y >= 7 := by
  z4_omega

-- System of inequalities
theorem system_2x2 (x y : Int)
    (h1 : x + y >= 5)
    (h2 : x - y <= 3)
    (h3 : x >= 0)
    (h4 : y >= 0) : x + 2*y >= 4 := by
  z4_omega

-- =============================================================================
-- Division and Modulo
-- =============================================================================

-- Division-modulo relationship
theorem div_mod_rel (n d : Int) (hd : d > 0) : n = (n / d) * d + n % d := by
  z4_omega

-- Modulo bounds
theorem mod_bounds (n d : Int) (hd : d > 0) : 0 <= n % d ∧ n % d < d := by
  z4_omega

-- =============================================================================
-- Quantifier-Free Fragments
-- =============================================================================

-- Should be in QF_LIA (no quantifiers)
theorem qf_lia_1 (x : Int) (h : 2*x + 1 = 7) : x = 3 := by
  z4_omega

-- Disjunctive constraint (still decidable)
theorem disj_lia (x : Int) (h : x = 1 ∨ x = 2) : x >= 1 ∧ x <= 2 := by
  z4_omega

-- =============================================================================
-- Unsatisfiable Systems (UNSAT expected)
-- =============================================================================

-- No solution exists
theorem unsat_system (x : Int) : ¬(x > 5 ∧ x < 3) := by
  z4_omega

-- Contradiction from linear combination
theorem unsat_combo (x y : Int) : ¬(x + y >= 10 ∧ x <= 2 ∧ y <= 3) := by
  z4_omega

-- =============================================================================
-- Performance Tests
-- =============================================================================

-- Small system (< 1ms)
theorem perf_small_lia (x y : Int) (h : x + y = 10) (hx : x >= 0) : y <= 10 := by
  z4_omega

-- Medium system with 5 variables (< 10ms)
theorem perf_medium_lia (a b c d e : Int)
    (h1 : a + b + c + d + e = 100)
    (h2 : a >= 10) (h3 : b >= 15) (h4 : c >= 20) (h5 : d >= 25)
    : e <= 30 := by
  z4_omega
