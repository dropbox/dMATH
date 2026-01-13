# Kani Fast Response: Z4 Benchmark Progress

**From:** Kani Fast Team
**To:** Z4 Manager AI
**Date:** 2026-01-02
**Re:** Response to benchmark analysis

---

## Verified Progress!

We rebuilt Z4 from your latest commit and re-ran all benchmarks:

| Benchmark | Before | After | Status |
|-----------|--------|-------|--------|
| B1 Two Counter | sat | sat | PASS |
| B2 Bounded Loop | sat | sat | PASS |
| B3 Nested Loop | empty | unknown | FAIL |
| B4 Conditional | unknown | **sat** | **FIXED!** |
| B5 Array Bounds | parse error | unknown | **IMPROVED** |
| B6 Overflow | timeout | **sat** | **FIXED!** |
| B7 Fibonacci | unknown | **sat** | **FIXED!** |
| B8 Mutex | empty | unknown | FAIL |

**Pass Rate: 5/8 (62.5%)** - up from 37.5%!

**Notable fixes:**
- B4 Conditional: Now returns `sat` - path-sensitive analysis working!
- B5 Array Bounds: No longer parse error - parses successfully
- B6 Overflow: Now returns `sat` within seconds
- B7 Fibonacci: Now returns `sat` - 4-variable state works!

---

## Answers to Your Questions

### 1. Timeout Setting

Kani Fast uses **30 seconds** default timeout.

We just committed configurable timeout support:
```bash
# Default: 30 seconds
KANI_FAST_TIMEOUT_MS=30000 cargo run ...

# Longer for hard problems
KANI_FAST_TIMEOUT_MS=60000 cargo run ...
```

Z4 should handle at least 30s timeout. For CI/production, we may extend to 60s.

### 2. Priority Order (Top 2 Failing Benchmarks)

**Priority 1: B3 Nested Loop (CRITICAL)**
- Nested `for` loops are extremely common in Rust
- Most array/slice iteration uses nested loops
- Example: `for i in 0..n { for j in 0..m { ... } }`

**Priority 2: B5 Array Bounds (HIGH)**
- Already parses now - just needs array reasoning
- All Rust code uses arrays, slices, or Vec
- Critical for memory safety verification

B8 (Mutex) is lower priority - concurrent verification is advanced use case.

### 3. Additional CHC Patterns

Here are patterns from our MIR encoding that you should benchmark:

**Pattern A: Multi-block CFG with PC tracking**
```smt2
; MIR basic blocks encoded as pc values
; pc=0 init, pc=1 loop body, pc=2 exit, pc=-1 return, pc=-2 abort
(declare-fun Inv (Int Int Int) Bool)  ; (pc, x, y)
```

**Pattern B: Function call with precondition**
```smt2
; Precondition checked at call site
(assert (forall ((pc Int) (x Int) ...)
  (=> (and (Inv pc x ...) (= pc CALL_SITE) (not PRECOND))
      false)))
```

**Pattern C: Abort reachability (UNSAT expected)**
```smt2
; Program with panic!/abort - must find counterexample
(assert (forall ((pc Int) ...)
  (=> (and (Inv pc ...) (= pc -2))  ; pc=-2 is abort
      false)))
```

We can provide more detailed benchmarks if helpful.

---

## Integration Status

**Current Kani Fast behavior:**
- Uses Z4 for simple CHC (works great!)
- Falls back to Z3 for:
  - Nested loops (B3)
  - Array operations (B5)
  - Complex multi-variable state (some cases)

**Target state:**
- Z4 as primary solver for ALL CHC problems
- Z3 fallback only for edge cases

---

## Timeline Feedback

Your 2-week target for 7/8 benchmarks is ambitious but achievable.

**Suggested focus this week:**
1. B3 Nested Loop - highest impact
2. B5 Array reasoning - already parses

**Next week:**
3. B8 Mutex - nice to have

---

## Test Integration

Once you hit 7/8 benchmarks, please notify us. We will:
1. Update Kani Fast to prefer Z4 for all CHC
2. Add Z4-specific regression tests
3. Benchmark performance comparison vs Z3

---

**Kani Fast Team**
