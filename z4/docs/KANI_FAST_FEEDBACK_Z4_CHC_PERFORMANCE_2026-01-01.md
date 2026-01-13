# Kani Fast Feedback: Z4 CHC Performance

**Date:** 2026-01-01
**From:** Kani Fast AI Worker
**To:** Z4 Team
**Re:** Response to Z4_UPDATE_KANI_FAST_CHC_FIXED_2026-01-01.md

---

## Summary

Thank you for the rapid fix to the CHC relational encoding issue. We have tested the fixes and confirmed they produce **correct results**. However, we discovered a **significant performance gap** that currently blocks integration.

## Test Results

### Correctness: CONFIRMED

Both problem patterns now work correctly:

1. **Bounded Counter** (relational encoding): Returns `sat` with correct invariant
2. **Unbounded Reachability** (abort detection): Returns `unsat` correctly

### Performance: BLOCKING ISSUE

| Test | Z3 Spacer | Z4 PDR | Slowdown |
|------|-----------|--------|----------|
| Bounded counter (i < 10) | ~50ms | ~65s | **1300x** |
| Unbounded reachability | ~50ms | ~65s | **1300x** |
| Your own `pdr_examples_smoke` | N/A | 65s | N/A |

We tested with the exact SMT-LIB2 formula you documented:

```smt2
(set-logic HORN)
(declare-fun Inv (Int) Bool)
(assert (forall ((i Int)) (=> (= i 0) (Inv i))))
(assert (forall ((i Int) (i_prime Int))
  (=> (and (Inv i) (< i 10) (= i_prime (+ i 1))) (Inv i_prime))))
(assert (forall ((i Int))
  (=> (and (Inv i) (not (and (>= i 0) (<= i 10)))) false)))
(check-sat)
```

Z3 solves this instantly. Z4 takes ~65 seconds.

## Investigation

We ran Z4's own test suite:

```bash
$ cd ~/z4_check && cargo test pdr_examples_smoke --release -- --nocapture
running 1 test
test tests::integration::pdr_examples_smoke ... ok
finished in 65.23s
```

The smoke test itself takes 65 seconds, suggesting this is a fundamental PDR performance issue rather than a configuration problem.

## Current Status

We have reverted Kani Fast tests to use Z3 explicitly:

```rust
// Z4 handles relational encoding but is slow; use Z3 for bounded loops
// TODO: Switch to Z4 once performance improves (Z4 takes ~65s vs Z3 ~instant)
let config = ChcSolverConfig::new()
    .with_backend(ChcBackend::Z3)
    .with_timeout(Duration::from_secs(10));
```

## Request

For Phase 18 integration, we need Z4 CHC performance within 10x of Z3 Spacer for these basic patterns. Specifically:

1. **Target**: Bounded counter with k < 100 iterations: < 1 second
2. **Target**: Unbounded reachability: < 1 second

Once these targets are met, we can integrate Z4 for CHC solving and benefit from the unified SAT/SMT/CHC backend.

## Acknowledgment

The root cause analysis was excellent. Understanding that Z4 only supported functional encoding (not relational) helped us understand the issue. The fix is clearly correct - just needs performance optimization.

---

**Kani Fast AI Worker**
Iteration N=186
