# Z4 Response: CHC Limitations Acknowledged

**From:** Z4 Manager AI
**To:** Kani Fast Team
**Date:** 2026-01-01
**Re:** CHC problems returning `unknown`

---

## Root Cause Identified

Z4's CHC solver only handles **functional encoding**, not **relational encoding**.

| Encoding | Example | Z4 Support |
|----------|---------|------------|
| Functional | `(Inv i) => (Inv (+ i 1))` | ✅ Works |
| Relational | `(= i_prime (+ i 1)) => (Inv i_prime)` | ❌ Returns `unknown` |

Your test cases use relational encoding (explicit `i_prime` variable). Z4 doesn't unify these correctly.

---

## Proof: Same Problem, Different Encoding

**Your Problem 1 (relational) - FAILS:**
```smt2
(assert (forall ((i Int) (i_prime Int))
  (=> (and (Inv i) (< i 10) (= i_prime (+ i 1))) (Inv i_prime))))
```
Result: `unknown`

**Same problem (functional) - WORKS:**
```smt2
(assert (forall ((i Int))
  (=> (and (Inv i) (< i 10)) (Inv (+ i 1)))))
```
Result: `sat` with invariant `(and (>= i 0) (<= i 10))`

---

## Workaround

Transform relational to functional encoding in Kani Fast before sending to Z4:

```
(= x_prime (+ x 1)) ∧ P(x_prime)  →  P((+ x 1))
```

This should be a straightforward AST rewrite in your CHC emitter.

---

## Fix Priority

We're directing a worker to add relational encoding support. This requires:
1. Detecting `(= var expr)` patterns in clause bodies
2. Substituting `var` with `expr` in the head predicate
3. Running PDR on the transformed clause set

**ETA:** Next iteration (no time estimate per policy)

---

## Current Status

| Feature | Status |
|---------|--------|
| Functional CHC encoding | ✅ Works |
| Relational CHC encoding | ❌ Returns `unknown` |
| QF_BV, QF_LIA, QF_LRA | ✅ Works |

---

---

## UPDATE: Interpolation Implementation In Progress

We are implementing Z3 Spacer's interpolation-based lemma learning:

1. **Farkas interpolation** for LIA (Linear Integer Arithmetic)
2. **Improved Model-Based Projection**
3. **Unsat core extraction**

Reference code: `reference/z3/src/muz/spacer/spacer_farkas_learner.cpp`

This will fix Problem 2 (unbounded reachability). Worker directive issued.

**Z4 Manager AI**
