# Kani Fast Feature Request: Z4 CHC Solver Requirements

**Date:** 2026-01-02
**From:** Kani Fast Team
**Priority:** CRITICAL - Blocks Production Use
**Status:** FEATURE REQUEST

## Executive Summary

Z4's CHC/PDR solver works for simple cases but **fails on real-world MIR verification patterns**. We ran 8 benchmarks representing actual Kani Fast use cases. Results:

| Benchmark | Z3 Spacer | Z4 PDR | Status |
|-----------|-----------|--------|--------|
| B1: Two Counter | sat | sat | **PASS** |
| B2: Bounded Loop | sat | sat | **PASS** |
| B3: Nested Loop | sat | (empty) | **FAIL** |
| B4: Conditional Branch | sat | unknown | **FAIL** |
| B5: Array Bounds | sat | parse error | **CRITICAL** |
| B6: Overflow Check | sat | sat | **PASS** |
| B7: Fibonacci | sat | unknown | **FAIL** |
| B8: Mutex Protocol | sat | (empty) | **FAIL** |

**Pass Rate: 3/8 (37.5%)**

## Critical Issues

### 1. Array Theory Not Supported (CRITICAL)

**File:** `/tmp/bench_5_array_bounds.smt2`

Z4 fails to parse basic Array theory:
```
(error "Parse error: parse error: Expected '_', found 'A'")
```

**Impact:** Cannot verify any Rust code with arrays, slices, or Vec.

**Required:** Full SMT-LIB Array theory support:
- `(Array Int Int)` sort
- `(select arr idx)`
- `(store arr idx val)`

### 2. Nested Loop Handling (HIGH)

**File:** `/tmp/bench_3_nested_loop.smt2`

Z4 returns empty output (timeout or crash) on nested loop patterns.

**Pattern:**
```smt2
; Outer loop: i < 10
; Inner loop: j < 10
; Property: bounds always hold
```

**Impact:** Cannot verify any Rust code with nested `for`/`while` loops.

### 3. Conditional Branch Analysis (HIGH)

**File:** `/tmp/bench_4_conditional_branch.smt2`

Z4 returns `unknown` on path-sensitive conditional branching.

**Pattern:**
```smt2
; Branch on x > 5, different updates in then/else
; Merge and loop back
```

**Impact:** Cannot verify `if`/`else` inside loops.

### 4. Multi-Variable State (MEDIUM)

**File:** `/tmp/bench_7_fibonacci_bounded.smt2`

Z4 returns `unknown` when invariant involves 4+ state variables.

**Impact:** Cannot verify functions with multiple local variables.

### 5. Concurrent Protocol Verification (MEDIUM)

**File:** `/tmp/bench_8_mutex_protocol.smt2`

Z4 returns empty on mutual exclusion protocols.

**Impact:** Cannot verify concurrent Rust code (mutex, locks, channels).

## Benchmark Files

All benchmark files are available in this repo under `/tmp/bench_*.smt2`:

### B1: Two Counter (PASS)
```smt2
(set-logic HORN)
(declare-fun Inv (Int Int) Bool)
(assert (forall ((x Int) (y Int)) (=> (and (= x 0) (= y 0)) (Inv x y))))
(assert (forall ((x Int) (y Int) (x_next Int) (y_next Int))
  (=> (and (Inv x y) (= x_next (+ x 1)) (or (= y_next (+ y 1)) (= y_next y)))
      (Inv x_next y_next))))
(assert (forall ((x Int) (y Int)) (=> (and (Inv x y) (not (>= x y))) false)))
(check-sat)
(get-model)
```

### B3: Nested Loop (FAIL - Critical for Rust)
```smt2
(set-logic HORN)
(declare-fun Inv (Int Int Int) Bool)
; pc=0: outer loop entry, pc=1: inner loop, pc=2: exit
(assert (forall ((pc Int) (i Int) (j Int))
  (=> (and (= pc 0) (= i 0) (= j 0)) (Inv pc i j))))
; Outer loop: i < 10
(assert (forall ((pc Int) (i Int) (j Int) (pc_next Int) (i_next Int) (j_next Int))
  (=> (and (Inv pc i j) (= pc 0) (< i 10) (= pc_next 1) (= i_next i) (= j_next 0))
      (Inv pc_next i_next j_next))))
; Inner loop: j < 10
(assert (forall ((pc Int) (i Int) (j Int) (pc_next Int) (i_next Int) (j_next Int))
  (=> (and (Inv pc i j) (= pc 1) (< j 10) (= j_next (+ j 1)) (= pc_next 1) (= i_next i))
      (Inv pc_next i_next j_next))))
; Inner loop exit -> outer loop continue
(assert (forall ((pc Int) (i Int) (j Int) (pc_next Int) (i_next Int) (j_next Int))
  (=> (and (Inv pc i j) (= pc 1) (>= j 10) (= i_next (+ i 1)) (= pc_next 0) (= j_next j))
      (Inv pc_next i_next j_next))))
; Property: bounds always hold
(assert (forall ((pc Int) (i Int) (j Int))
  (=> (and (Inv pc i j) (not (and (>= i 0) (<= i 10) (>= j 0) (<= j 10)))) false)))
(check-sat)
(get-model)
```

### B5: Array Bounds (FAIL - Parse Error)
```smt2
(set-logic HORN)
(declare-fun Inv (Int (Array Int Int)) Bool)
; Initialize: arr[0..9] = 0, i = 0
(assert (forall ((i Int) (arr (Array Int Int)))
  (=> (and (= i 0) (= (select arr 0) 0))
      (Inv i arr))))
; Loop: arr[i] := i, i++, while i < 10
(assert (forall ((i Int) (arr (Array Int Int)) (i_next Int) (arr_next (Array Int Int)))
  (=> (and (Inv i arr) (< i 10)
           (= arr_next (store arr i i))
           (= i_next (+ i 1)))
      (Inv i_next arr_next))))
; Property: after loop, arr[5] >= 0
(assert (forall ((i Int) (arr (Array Int Int)))
  (=> (and (Inv i arr) (>= i 10) (< (select arr 5) 0)) false)))
(check-sat)
(get-model)
```

## Required Capabilities for Kani Fast

### Must Have (Blocks Integration)
1. **Array Theory** - SMT-LIB `(Array K V)` with select/store
2. **Nested Loop Invariants** - 3+ variable predicates with nested iteration
3. **Conditional Branch** - Path-sensitive `if/else` inside loops

### Should Have (Production Quality)
4. **4+ Variable State** - Predicates with many state variables
5. **Timeout Handling** - Return `unknown` with reason, not empty/crash
6. **Counterexample Output** - When UNSAT, provide trace

### Nice to Have (Future)
7. **Concurrent State** - Mutex/lock protocol verification
8. **Bitvector Theory** - `(_ BitVec 32)` for precise overflow checking

## Performance Targets

| Benchmark Type | Z3 Time | Z4 Target |
|----------------|---------|-----------|
| Simple counter | 0.01s | <0.05s |
| Bounded loop | 0.02s | <0.1s |
| Nested loop | 0.05s | <0.5s |
| Array bounds | 0.03s | <0.3s |
| 4-var state | 0.04s | <0.4s |
| Mutex protocol | 0.02s | <0.2s |

## Integration Notes

Kani Fast currently falls back to Z3 for:
- MIR encoding tests
- Any CHC with Array theory
- Complex control flow patterns

We want to **use Z4 as the primary solver** but need these gaps closed first.

## Contact

Please add these benchmarks to Z4's test suite. They represent real verification patterns from Rust MIR analysis.

---
*Generated by Kani Fast integration testing*
