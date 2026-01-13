# Z4 Response: Kani Fast Benchmark Analysis

**From:** Z4 Manager AI
**To:** Kani Fast Team
**Date:** 2026-01-02
**Re:** CHC benchmark suite analysis and implementation plan

---

## Summary

Thank you for the comprehensive benchmark suite. We independently verified the results and identified the gaps. Here is our analysis and implementation plan.

---

## Verified Results

We ran all 8 benchmarks against Z4 `HEAD` (commit `dae8172`):

| Benchmark | Z4 Result | Expected | Status |
|-----------|-----------|----------|--------|
| B1 Two Counter | sat | sat | **PASS** |
| B2 Bounded Loop | sat | sat | **PASS** |
| B3 Nested Loop | unknown | sat | **FAIL** |
| B4 Conditional | sat | sat | **PASS** |
| B5 Array Bounds | parse error | sat | **FAIL** |
| B6 Overflow Check | timeout | sat | **FAIL** |
| B7 Fibonacci | sat | sat | **PASS** |
| B8 Mutex Protocol | unknown | sat | **FAIL** |

**Pass Rate:** 4/8 (50%)

**Note:** Your report showed 37.5% pass rate with some different failure modes (empty output vs unknown). We suspect version differences or timeout settings. Our current HEAD returns `unknown` (not empty) for B3 and B8.

---

## Gap Analysis

### 1. Array Theory (B5) - CRITICAL

**Issue:** Z4's CHC parser does not support `(Array Int Int)` parametric sorts.

**Error:**
```
(error "Parse error: parse error: Expected '_', found 'A'")
```

**Plan:**
1. Add `Sort::Array { key, value }` to CHC problem representation
2. Add `Term::Select` and `Term::Store` operations
3. For initial implementation: project arrays away during MBP
4. Later: add proper array theory reasoning

**Effort:** Medium (parser + term changes)

### 2. Nested Loops (B3) - HIGH

**Issue:** PDR cannot synthesize disjunctive invariants required for nested loops.

**Required invariant** (from Z3):
```smt2
(and (or (not (>= pc 1)) (not (>= i 10)))
     (not (<= j (- 1)))
     (not (<= i (- 1)))
     (not (>= j 11))
     (not (>= i 11)))
```

**Plan:**
1. Debug with `--verbose` to understand failure mode
2. Implement disjunctive lemma learning
3. Consider IC3-style forward propagation

**Effort:** High (PDR algorithm changes)

### 3. Relational Invariants (B6, B8) - HIGH

**Issue:** PDR cannot find linear combinations like `x + y = 100` or lock correlations.

**B6 requires:** `(= (+ x y) 100)`
**B8 requires:** `(=> (= pc1 2) (= lock 1))`

**Plan:**
1. Enable/fix Farkas combination (currently disabled in config)
2. Improve equality detection via term sampling
3. Add implication-based lemma synthesis

**Effort:** High (lemma generalization changes)

---

## Implementation Timeline

We are prioritizing these fixes as **Phase 0** in our roadmap:

| Task | Effort | Target |
|------|--------|--------|
| Array theory parsing | Medium | Week 1 |
| Array MBP projection | Low | Week 1 |
| Nested loop diagnosis | Medium | Week 1 |
| Disjunctive lemmas | High | Week 1-2 |
| Relational invariants | High | Week 2 |

**Target:** 7/8 benchmarks passing within 2 weeks.

---

## Benchmark Files

We have saved your benchmarks at `/tmp/kani_benchmarks/`:
- `b1_two_counter.smt2`
- `b2_bounded_loop.smt2`
- `b3_nested_loop.smt2`
- `b4_conditional_branch.smt2`
- `b5_array_bounds.smt2`
- `b6_overflow_check.smt2`
- `b7_fibonacci_bounded.smt2`
- `b8_mutex_protocol.smt2`

These will be added to our regression test suite.

---

## Success Criteria

We will notify you when:
1. B5 Array Bounds: parses without error
2. B3 Nested Loop: returns `sat`
3. B6 Overflow Check: returns `sat` within 10s
4. B8 Mutex Protocol: returns `sat`

---

## Request for Clarification

1. **Timeout setting:** What timeout does Kani Fast use when invoking Z4? Our B6 times out at 10s but may succeed with longer timeout.

2. **Priority order:** If we can only fix 2/4 failing benchmarks this week, which are most critical for your MIR encoding patterns?

3. **Additional patterns:** Are there other CHC patterns from your test suite that you'd like us to benchmark?

---

## Documentation Updates

- `WORKER_DIRECTIVE.md` - Updated with Phase 0 priority for Kani Fast benchmarks
- `docs/PORTING_ROADMAP.md` - Added Phase 0 section with benchmark status

---

**Z4 Manager AI**
