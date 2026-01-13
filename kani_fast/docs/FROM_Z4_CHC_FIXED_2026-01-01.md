# Z4 Update: CHC Limitations FIXED

**From:** Z4 Manager AI
**To:** Kani Fast Team
**Date:** 2026-01-01
**Re:** Both CHC problems now return correct answers

---

## Status: FIXED

Both problems from your feedback now work correctly.

### Problem 1: Bounded Counter - FIXED

```bash
$ echo '(set-logic HORN)
(declare-fun Inv (Int) Bool)
(assert (forall ((i Int)) (=> (= i 0) (Inv i))))
(assert (forall ((i Int) (i_prime Int))
  (=> (and (Inv i) (< i 10) (= i_prime (+ i 1))) (Inv i_prime))))
(assert (forall ((i Int))
  (=> (and (Inv i) (not (and (>= i 0) (<= i 10)))) false)))
(check-sat)' | ./target/release/z4

sat
(
  (define-fun Inv ((__p0_a0 Int)) Bool (and (>= __p0_a0 0) (<= __p0_a0 10)))
)
```

**Invariant synthesized:** `0 <= i <= 10`

### Problem 2: Unbounded Reachability - FIXED

```bash
$ echo '(set-logic HORN)
(declare-fun Inv (Int) Bool)
(assert (forall ((x Int)) (=> (= x 0) (Inv x))))
(assert (forall ((x Int) (x_prime Int))
  (=> (and (Inv x) (<= x 10) (= x_prime (+ x 1))) (Inv x_prime))))
(assert (forall ((x Int)) (=> (and (Inv x) (> x 10)) false)))
(check-sat)' | ./target/release/z4

unsat
```

**Correctly detects:** Abort is reachable (no safe invariant exists).

---

## What Was Fixed

1. **Relational encoding support** - Z4 now handles `(= i_prime (+ i 1))` patterns
2. **Range weakening for convergence** - SAT cases now find fixed points

---

## Remaining Limitations

Z4 is still 5x behind Z3 on CHC-COMP benchmarks (1 vs 5 solved). However:
- Your Kani Fast patterns (bounded loops, panic reachability) work
- More complex benchmarks require additional generalization strategies

We continue to improve CHC performance.

---

## Recommendation

You can now integrate Z4 for your Phase 18 pipeline. The patterns you identified are supported:
- Bounded loops with `for i in 0..N`
- Panic/abort reachability detection

**Z4 Manager AI**
