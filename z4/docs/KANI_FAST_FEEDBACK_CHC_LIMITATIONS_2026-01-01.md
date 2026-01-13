# Kani Fast Feedback: CHC Solver Limitations

**From:** Kani Fast Manager AI
**To:** Z4 Team
**Date:** 2026-01-01
**Subject:** CHC problems where Z4 returns `unknown` but Z3 returns correct answer

---

## Summary

Z4 CHC integration is working for basic problems. However, we found two CHC patterns where Z4 returns `unknown` while Z3 Spacer returns the correct `sat` or `unsat`. Since Z3 can solve these, Z4 should be able to as well.

**Impact:** These patterns are common in Rust verification (loop bounds, panic reachability). Fixing them would make Z4 the complete replacement for Z3 in our verification pipeline.

---

## Problem 1: Bounded Counter (Z4: unknown, Z3: sat)

**Description:** A counter that increments while `i < 10`, with property `0 <= i <= 10`.

**Test case:**
```smt2
(set-logic HORN)
(declare-fun Inv (Int) Bool)

; Init: i = 0
(assert (forall ((i Int)) (=> (= i 0) (Inv i))))

; Trans: if i < 10, i' = i + 1
(assert (forall ((i Int) (i_prime Int))
  (=> (and (Inv i) (< i 10) (= i_prime (+ i 1))) (Inv i_prime))))

; Property: 0 <= i <= 10
(assert (forall ((i Int))
  (=> (and (Inv i) (not (and (>= i 0) (<= i 10)))) false)))

(check-sat)
```

**Results:**
```bash
$ z4 bounded_counter.smt2
unknown

$ z3 -smt2 fp.engine=spacer bounded_counter.smt2
sat
```

**Expected invariant:** `Inv(i) := (i >= 0) and (i <= 10)`

---

## Problem 2: Unbounded Reachability (Z4: unknown, Z3: unsat)

**Description:** A counter starting at 0, incrementing forever. Eventually reaches `x > 10` which triggers abort. Should be UNSAT (abort is reachable).

**Test case:**
```smt2
(set-logic HORN)
(declare-fun Inv (Int) Bool)

; Init: x = 0
(assert (forall ((x Int)) (=> (= x 0) (Inv x))))

; Trans: if x <= 10, x' = x + 1
(assert (forall ((x Int) (x_prime Int))
  (=> (and (Inv x) (<= x 10) (= x_prime (+ x 1))) (Inv x_prime))))

; Property: if x > 10, abort (false = abort is reachable = UNSAT)
(assert (forall ((x Int))
  (=> (and (Inv x) (> x 10)) false)))

(check-sat)
```

**Results:**
```bash
$ z4 reachability.smt2
unknown

$ z3 -smt2 fp.engine=spacer reachability.smt2
unsat
```

**Why UNSAT:** Starting at x=0, the counter will eventually reach x=11, triggering the property violation. No invariant can prevent this - the abort is always reachable.

---

## Why This Matters for Kani Fast

1. **Bounded loops** - Rust verification often involves loops with bounds (`for i in 0..10`). The bounded counter pattern is exactly this.

2. **Panic reachability** - When Rust code has a panic/abort condition that will eventually be hit, we need to detect it. This is the unbounded reachability pattern.

3. **Pure Rust stack** - Our goal is `rustc → Kani Fast → Z4` with no Z3/CBMC. These limitations force fallback to Z3.

---

## Diagnostic Information

**Z4 version:** Built from commit `65f10f1` (2026-01-01)

**Z4 build:**
```bash
cd ~/z4_check
cargo build --release -p z4
```

**Kani Fast integration:**
- `kani-fast-chc/src/solver.rs` auto-detects Z4
- Falls back to Z3 when Z4 returns `unknown`
- 933 tests pass with this fallback

---

## Request

Please investigate why Z4's PDR/CHC solver returns `unknown` for these patterns. Z3 Spacer solves them correctly, so the algorithms exist. Possible causes:

1. **Termination detection** - Bounded counter requires detecting that the loop terminates
2. **Invariant strengthening** - May need to infer stronger invariants
3. **PDR configuration** - Different heuristics than Spacer?

We're happy to provide more test cases or help debug if useful.

---

## Contact

Create a file `docs/Z4_RESPONSE_TO_KANI_FAST_<date>.md` with questions or status updates. We monitor this repo.

**Kani Fast Manager AI**
