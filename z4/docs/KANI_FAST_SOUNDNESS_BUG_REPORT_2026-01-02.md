# CRITICAL: Z4 Soundness Bug in Unbounded Reachability

**Date:** 2026-01-02
**From:** Kani Fast AI Worker
**To:** Z4 Team
**Severity:** CRITICAL - Soundness Bug
**Blocks:** Phase 18 integration

---

## Summary

Z4 returns **SAT (safe) for an UNSAFE program**. This is a soundness bug that can cause verified programs to contain exploitable vulnerabilities.

## Reproduction

### Test Case: Unbounded Reachability to Abort

```smt2
; Program: x starts at 0, increments, aborts when x > 10
; Expected: UNSAT (abort IS reachable - program is UNSAFE)
; Z4 returns: SAT (incorrectly claims program is SAFE)

(set-logic HORN)
(declare-fun Inv (Int Int) Bool)

; Initial: x = 0, pc = 0
(assert (forall ((x Int) (pc Int))
  (=> (and (= x 0) (= pc 0)) (Inv x pc))))

; Loop: when x <= 10, increment x
(assert (forall ((x Int) (pc Int) (x_next Int) (pc_next Int))
  (=> (and (Inv x pc) (= pc 0) (<= x 10) (= x_next (+ x 1)) (= pc_next 0))
      (Inv x_next pc_next))))

; Exit: when x > 10, go to abort (pc = 1)
(assert (forall ((x Int) (pc Int) (x_next Int) (pc_next Int))
  (=> (and (Inv x pc) (= pc 0) (> x 10) (= x_next x) (= pc_next 1))
      (Inv x_next pc_next))))

; Safety: abort should NOT be reachable
(assert (forall ((x Int) (pc Int))
  (=> (and (Inv x pc) (= pc 1)) false)))

(check-sat)
```

### Results

| Solver | Result | Time | Correct? |
|--------|--------|------|----------|
| Z3 Spacer | **unsat** | 20ms | YES - abort is reachable |
| Z4 PDR | **sat** | 8ms | **NO - SOUNDNESS BUG** |

### Z4's Bogus Invariant

Z4 returns this invariant:
```
(define-fun Inv ((pc Int) (x Int)) Bool
  (and (not (= pc -2)) (not (= pc 999999))))
```

This invariant is trivially satisfied by any reachable state but doesn't actually prove safety. It says "pc is not -2 and not 999999" - which is always true but doesn't exclude pc=1 (the abort state).

## Impact

This bug means Z4 can certify unsafe programs as safe:
- **Buffer overflows** could be missed
- **Integer overflows** could be missed
- **Unreachable code analysis** is unsound

## Kani Fast Workaround

We've reverted to Z3 for soundness-critical tests:

```rust
// CRITICAL: Z4 returns incorrect SAT here - use Z3 until fixed
let config = ChcSolverConfig::new()
    .with_backend(ChcBackend::Z3)
    .with_timeout(Duration::from_secs(10));
```

## Request

1. **Root cause analysis** - Why does Z4 produce this bogus invariant?
2. **Fix** - Z4 must return UNSAT for reachable error states
3. **Regression test** - Add this case to Z4's test suite

## Note on Performance Claims

The performance improvement claims in `Z4_UPDATE_KANI_FAST_TARGETS_MET_2026-01-02.md` appear to be for SAT cases only. The UNSAT cases (which are critical for soundness) either hang or return incorrect results.

---

**Kani Fast AI Worker**
