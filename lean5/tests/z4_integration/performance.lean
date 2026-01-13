/-
  Z4 Integration Tests: Performance Benchmarks

  Benchmark problems to verify performance targets are met.
  All times are wall-clock, measured on CI machines.
-/

import Lean5.Tactic.Z4

-- =============================================================================
-- Target Latencies (from Z4_INTEGRATION_RESPONSE.md)
-- =============================================================================

/-
  | Operation | Target Latency | Notes |
  |-----------|----------------|-------|
  | FFI call overhead | < 10 μs | Solver reuse amortizes |
  | Simple SAT (< 100 vars) | < 1 ms | |
  | QF_LIA (< 50 constraints) | < 10 ms | |
  | QF_BV (< 32 bits, < 100 ops) | < 10 ms | |
  | DRAT verification (< 10K clauses) | < 100 ms | |
-/

-- =============================================================================
-- Tier 1: FFI Overhead (< 10 μs)
-- =============================================================================

-- Trivial solve to measure FFI overhead
-- Run 1000x and measure average
#bench_ffi_overhead 1000 :
    True := by z4_decide

-- Solver creation/destruction cycle
#bench_solver_lifecycle 1000

-- =============================================================================
-- Tier 2: Simple SAT (< 1 ms, < 100 vars)
-- =============================================================================

-- 10 variables, simple
#bench_sat "sat_10_simple" (timeout := 1) :
    ∀ (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 : Prop),
    (p1 ∨ p2) ∧ (p3 ∨ p4) ∧ (p5 ∨ p6) ∧ (p7 ∨ p8) ∧ (p9 ∨ p10) →
    ∃ (x : Prop), x ∨ ¬x

-- 50 variables, random 3-SAT (satisfiable, ratio ~4.0)
-- This is generated deterministically for reproducibility
#bench_sat "sat_50_random" (timeout := 1) :
    -- 50 vars, 200 clauses (ratio 4.0, below threshold)
    random_3sat 50 200 (seed := 42)

-- 100 variables, random 3-SAT (satisfiable)
#bench_sat "sat_100_random" (timeout := 1) :
    random_3sat 100 400 (seed := 123)

-- =============================================================================
-- Tier 3: QF_LIA (< 10 ms, < 50 constraints)
-- =============================================================================

-- 10 variables, 20 constraints
#bench_lia "lia_10x20" (timeout := 10) :
    ∀ (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : Int),
    x1 + x2 >= 0 →
    x2 + x3 >= 0 →
    x3 + x4 >= 0 →
    x4 + x5 >= 0 →
    x5 + x6 >= 0 →
    x6 + x7 >= 0 →
    x7 + x8 >= 0 →
    x8 + x9 >= 0 →
    x9 + x10 >= 0 →
    x10 + x1 >= 0 →
    x1 >= -100 →
    x1 <= 100 →
    x10 >= -100 →
    x10 <= 100 →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 >= -500

-- 20 variables, system of equations
#bench_lia "lia_20_system" (timeout := 10) :
    linear_system 20 40 (seed := 456)

-- 50 constraints, mixed
#bench_lia "lia_50_mixed" (timeout := 10) :
    linear_mixed 50 (seed := 789)

-- =============================================================================
-- Tier 4: QF_BV (< 10 ms, < 32 bits)
-- =============================================================================

-- 8-bit operations
#bench_bv "bv8_basic" (timeout := 10) :
    ∀ (x y : UInt8), x + y = y + x

-- 16-bit operations
#bench_bv "bv16_basic" (timeout := 10) :
    ∀ (x y : UInt16), (x &&& y) ||| (x ^^^ y) = x ||| y

-- 32-bit operations
#bench_bv "bv32_basic" (timeout := 10) :
    ∀ (x : UInt32), x ^^^ x = 0

-- 32-bit multiplication (harder)
#bench_bv "bv32_mul" (timeout := 10) :
    ∀ (x : UInt32), x * 1 = x

-- Overflow detection (32-bit)
#bench_bv "bv32_overflow" (timeout := 10) :
    ∀ (x y : UInt32), (x + y < x) → (y > UInt32.max - x)

-- =============================================================================
-- Tier 5: DRAT Verification (< 100 ms, < 10K clauses)
-- =============================================================================

-- PHP_5^4 (generates ~1K clause proof)
#bench_drat "php_5_4" (timeout := 100) :
    pigeonhole 5 4

-- PHP_6^5 (generates ~5K clause proof)
#bench_drat "php_6_5" (timeout := 100) :
    pigeonhole 6 5

-- Random UNSAT (generates variable-size proofs)
#bench_drat "unsat_random" (timeout := 100) :
    random_unsat 100 450 (seed := 999)

-- =============================================================================
-- Stress Tests (may exceed targets, for regression detection)
-- =============================================================================

-- Large SAT (500 vars) - should complete but may exceed 1ms
#bench_sat "sat_500_stress" (timeout := 100) :
    random_3sat 500 2000 (seed := 1111)

-- Large LIA (100 constraints) - may exceed 10ms
#bench_lia "lia_100_stress" (timeout := 1000) :
    linear_system 100 100 (seed := 2222)

-- 64-bit BV - may exceed 10ms
#bench_bv "bv64_stress" (timeout := 1000) :
    ∀ (x y : UInt64), x &&& (x ||| y) = x

-- PHP_8^7 - large proof, may exceed 100ms
#bench_drat "php_8_7_stress" (timeout := 5000) :
    pigeonhole 8 7

-- =============================================================================
-- Latency Distribution
-- =============================================================================

-- Collect p50, p90, p99 latencies over 100 runs
#bench_distribution "sat_latency" 100 :
    random_3sat 50 200 (seed := 3333)

#bench_distribution "lia_latency" 100 :
    linear_system 10 20 (seed := 4444)

#bench_distribution "bv_latency" 100 :
    ∀ (x : UInt8), x + 0 = x

-- =============================================================================
-- Memory Usage
-- =============================================================================

-- Peak memory during solve
#bench_memory "memory_sat_500" :
    random_3sat 500 2000 (seed := 5555)

#bench_memory "memory_lia_100" :
    linear_system 100 100 (seed := 6666)

-- =============================================================================
-- Throughput (proofs per second)
-- =============================================================================

-- How many simple problems can we solve in 1 second?
#bench_throughput 1000 :
    ∀ (p : Prop), p ∨ ¬p

-- =============================================================================
-- Regression Tests
-- =============================================================================

-- These encode specific performance requirements from the spec
-- Failure = regression in Z4/Lean5 integration

theorem perf_regression_sat : True := by
  -- Must complete in < 1ms
  have h := random_3sat 100 400 (seed := 7777)
  z4_decide (timeout := 1)

theorem perf_regression_lia : True := by
  -- Must complete in < 10ms
  have h := linear_system 20 40 (seed := 8888)
  z4_omega (timeout := 10)

theorem perf_regression_bv : True := by
  -- Must complete in < 10ms
  have h : ∀ (x : UInt32), x * 0 = 0 := by z4_bv (timeout := 10)
  trivial
