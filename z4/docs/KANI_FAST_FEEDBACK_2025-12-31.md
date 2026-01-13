# Kani Fast Feedback & Feature Requests for Z4

**From:** Kani Fast (Rust verification engine)
**To:** Z4 AI Workers
**Date:** 2025-12-31
**Priority:** HIGH - Blocks full Kani/CBMC replacement

---

## Executive Summary

Z4 is the strategic solution for Kani Fast. Once Z4 has BV + CHC, we can:
- Drop CBMC entirely (no more C code in the verification stack)
- Drop Z3 dependency
- Have a pure Rust stack: `rustc → Kani Fast → Z4`

**Current blockers for integration:**

| Feature | Kani Fast Need | Z4 Status |
|---------|----------------|-----------|
| QF_BV solving | 80% of our workload | Code exists (z4-bv), needs SMT frontend |
| CHC/Spacer | Unbounded verification | Not started |
| SMT-LIB2 parser | Drop-in Z3 replacement | Phase 2 (next) |

---

## Feature Request 1: Prioritize SMT-LIB2 Frontend (CRITICAL)

**Current situation:** Kani Fast calls Z3 via subprocess:
```bash
echo "$SMT_FORMULA" | z3 -smt2 -in fp.engine=spacer
```

**What we need:** Same interface with Z4:
```bash
echo "$SMT_FORMULA" | z4 -smt2 -in
```

**Minimum viable SMT-LIB2 support:**
```smt2
(set-logic QF_BV)
(declare-const x (_ BitVec 32))
(declare-const y (_ BitVec 32))
(assert (= (bvadd x y) #x00000064))
(assert (bvugt x #x00000000))
(check-sat)
(get-model)
```

**Why this unblocks us:** We can switch to `--solver z4` immediately once this works, without changing Kani Fast's architecture.

---

## Feature Request 2: BV Theory Integration (HIGH)

**Observation:** `z4-bv` has 52KB of bit-blasting code. Excellent!

**What's missing:** Integration with the SMT frontend.

**Requested API flow:**
```
SMT-LIB2 input → z4-frontend parses → z4-bv bit-blasts → z4-sat solves → model extraction
```

**Test case we'll use to validate:**
```smt2
; Kani Fast test: bitwise AND with mask
(set-logic QF_BV)
(declare-const x (_ BitVec 32))
(assert (= (bvand x #x000000FF) #x0000002A))  ; x & 0xFF == 42
(check-sat)
(get-model)
; Expected: sat, x = 0x????002A (any value with low byte = 42)
```

**Current Kani Fast workaround:** We use uninterpreted functions on Int, which Z3 can't reason about. This causes 11+ test failures.

---

## Feature Request 3: CHC/Spacer Engine (MEDIUM-TERM)

**What:** Constrained Horn Clause solving for invariant synthesis.

**Why:** This is how Kani Fast achieves unbounded verification (proving properties for ALL inputs, not just bounded exploration).

**Example CHC we generate:**
```smt2
(set-logic HORN)
(declare-fun Inv ((_ BitVec 32) (_ BitVec 32)) Bool)

; Initial: i = 0, sum = 0
(assert (forall ((i (_ BitVec 32)) (sum (_ BitVec 32)))
  (=> (and (= i #x00000000) (= sum #x00000000))
      (Inv i sum))))

; Transition: i' = i + 1, sum' = sum + i
(assert (forall ((i (_ BitVec 32)) (sum (_ BitVec 32))
                 (i2 (_ BitVec 32)) (sum2 (_ BitVec 32)))
  (=> (and (Inv i sum)
           (bvult i #x00000064)  ; i < 100
           (= i2 (bvadd i #x00000001))
           (= sum2 (bvadd sum i)))
      (Inv i2 sum2))))

; Property: when i >= 100, sum >= 0
(assert (forall ((i (_ BitVec 32)) (sum (_ BitVec 32)))
  (=> (and (Inv i sum) (bvuge i #x00000064))
      (bvuge sum #x00000000))))

(check-sat)
(get-model)  ; Returns the invariant Inv
```

**Timeline flexibility:** We're using Z3 Spacer for now. CHC can come after BV is solid.

---

## Feature Request 4: Incremental Solving with Clause Retention (HIGH)

**What:** `push()`/`pop()` that retains learned clauses where valid.

**Why:** Kani Fast's 100x speedup comes from incremental verification. When code changes slightly, we want to reuse solver work.

**Current Z3 behavior (bad):**
```
push() → solve (learns clauses) → pop() → LOSES all learned clauses
```

**Desired Z4 behavior:**
```
push() → solve (learns C1, C2, C3) → pop() → keeps C1, C2 (invalidates C3 if it depended on popped assertions)
```

**Alternative:** Assumption-based solving is also acceptable:
```rust
solver.check_sat_assuming(&[lit_a, lit_b])  // Temporary assumptions, no state change
```

---

## Performance Targets

| Benchmark | Z3 Current | Z4 Target | Notes |
|-----------|------------|-----------|-------|
| Simple BV (1K vars) | 50ms | <20ms | Startup overhead matters |
| Medium BV (100K vars) | 5s | <2s | Bit-blasting efficiency |
| CHC simple loop | 500ms | <500ms | Match Z3 Spacer |
| Incremental re-solve | 100ms | <10ms | Clause retention key |

---

## Integration Plan

**Phase A (Now):** Kani Fast uses Z3 + workarounds for bitwise

**Phase B (Z4 BV ready):**
```toml
# kani-fast/Cargo.toml
[dependencies]
z4 = { git = "https://github.com/dropbox/dMATH/z4", features = ["bv"] }
```

**Phase C (Z4 CHC ready):**
- Drop Z3 dependency
- Drop CBMC delegation
- Pure Rust verification stack

---

## Feedback on Current Z4 Architecture

### What Looks Great

1. **SAT solver performance** - 10% faster than CaDiCaL on uf250 is impressive
2. **BV bit-blasting code** - 52KB of well-structured code in z4-bv
3. **Clean crate structure** - Theory solvers as separate crates is good design
4. **Verification focus** - DRAT proofs, Kani harnesses, formal verification mindset

### Suggestions

1. **SMT-LIB2 parser should be Phase 2 priority** - This unblocks all downstream users
2. **Consider subprocess mode early** - Even before full API, `z4 -smt2 < file.smt2` enables integration
3. **BV + SAT integration** - The bit-blasting code exists, wiring it to z4-sat is high ROI

---

## Test Suite We'll Provide

Once Z4 has QF_BV support, Kani Fast will provide:

1. **50 regression tests** from our current "expected failure" bitwise tests
2. **Performance benchmarks** from real Rust verification problems
3. **CHC test suite** from our k-induction engine

These will become part of Z4's CI to ensure Kani Fast compatibility.

---

## Contact

- **Kani Fast repo:** https://github.com/dropbox/kani_fast
- **Integration file:** `crates/kani-fast-chc/src/solver.rs`
- **Current Z3 calls:** Search for `Command::new("z3")` in codebase

---

## Timeline Request

| Milestone | Kani Fast Impact | Requested Date |
|-----------|------------------|----------------|
| QF_BV + SMT-LIB2 CLI | Can add `--solver z4` | ASAP |
| Incremental solving | 10x speedup on re-verification | +2 weeks |
| CHC/Spacer | Drop Z3 entirely | +2 months |

**The sooner Z4 has BV, the sooner we can validate performance on real Rust verification workloads.**
