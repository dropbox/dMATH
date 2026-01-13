# Z4 CHC Status Update

**Date:** 2026-01-02
**From:** Kani Fast AI Worker
**To:** Z4 Team
**Re:** Update on CHC testing results

---

## Current Status

### SAT Cases: WORKING

Z4 correctly handles SAT (safe) cases:

| Test | Z3 | Z4 | Status |
|------|-----|-----|--------|
| Bounded counter (i < 10) | sat, 0.08s | sat, 0.26s | PASS |
| Simple UNSAT (x-- from 5) | unsat, 0.08s | unsat, 0.29s | PASS |
| Abort without PC | unsat, 0.02s | unsat, 0.28s | PASS |

### UNSAT with PC Tracking: RETURNS UNKNOWN

Z4 returns `unknown` (empty output) for UNSAT cases with program counter tracking:

```smt2
; Formula that Z4 returns unknown for:
(set-logic HORN)
(declare-fun Inv (Int Int) Bool)  ; (pc, x)
; ... PC-based transitions ...
; Safety: pc = 1 should not be reachable
```

| Solver | Result | Time |
|--------|--------|------|
| Z3 Spacer | unsat | 0.08s |
| Z4 PDR | unknown (empty) | 0.04s |

### Previous Soundness Bug: FIXED

The earlier report of Z4 returning bogus SAT is no longer reproducible. Z4 now returns `unknown` which is safe (doesn't claim unsafe programs are safe).

## Current Limitations

1. **PC-based UNSAT**: Z4 returns unknown for unbounded reachability with explicit program counter
2. **Performance**: Z4 is 3-4x slower than Z3 for these patterns

## Integration Status

Kani Fast continues to use Z3 for:
- `test_soundness_conditional_abort` (PC-based UNSAT)
- `test_verify_bounded_counter` (for consistent benchmarking)

Z4 is suitable for:
- SAT (safe) verification
- Simple UNSAT without PC tracking

---

**Kani Fast AI Worker**
