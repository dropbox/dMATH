# Z4 Requirements from Kani Fast

**From:** Kani Fast (Rust verification engine)
**To:** Z4 (SMT solver)
**Date:** 2025-12-30
**Status:** Requirements for Integration

## Summary

Kani Fast needs Z4 to be a drop-in replacement for Z3, specifically for CHC/Spacer solving. We currently use Z3 via subprocess calls.

---

## Priority 1: CRITICAL (Blocks Integration)

### 1.1 CHC/Spacer Engine

**What:** Constrained Horn Clause solving for invariant synthesis.

**Current Z3 usage:**
```bash
z3 -smt2 -in fp.engine=spacer -t:60000 < formula.smt2
```

**Required:** Z4 must support the same interface:
- Parse SMT-LIB2 with `(set-logic HORN)`
- Support `fp.engine=spacer` or equivalent
- Return `sat` with model (invariant) or `unsat` with counterexample

**Example CHC we generate:**
```smt2
(set-logic HORN)
(declare-fun Inv (Int Int Int) Bool)

; Initial state
(assert (forall ((x Int)) (=> (= x 0) (Inv x 0 0))))

; Transition
(assert (forall ((x Int) (y Int) (x2 Int) (y2 Int))
  (=> (and (Inv x y 0) (= x2 (+ x 1)) (= y2 (+ y x)))
      (Inv x2 y2 0))))

; Property: y >= 0 always
(assert (forall ((x Int) (y Int))
  (=> (and (Inv x y 0) (< y 0)) false)))

(check-sat)
(get-model)
```

### 1.2 Bitvector Theory (QF_BV)

**What:** Rust integers are fixed-width bitvectors.

**Required:**
- `(_ BitVec 8)` for u8/i8
- `(_ BitVec 32)` for u32/i32
- `(_ BitVec 64)` for u64/i64
- All bitvector operations: `bvadd`, `bvmul`, `bvudiv`, `bvslt`, etc.
- Overflow detection: `bvadd_overflow`, `bvmul_overflow`

### 1.3 Model Extraction

**What:** When SAT, we need the satisfying assignment.

**Required:**
```
(check-sat)    ; returns sat
(get-model)    ; returns (define-fun Inv (...) ...)
```

The invariant model tells us what property holds.

---

## Priority 2: HIGH (Performance)

### 2.1 Speed Target

| Problem Size | Z3 Time | Z4 Target |
|--------------|---------|-----------|
| Simple CHC (3 vars) | 10ms | <10ms |
| Medium CHC (10 vars) | 100ms | <50ms |
| Complex CHC (30 vars) | 1s | <500ms |

### 2.2 Incremental Solving

**What:** We call Z4 repeatedly with slight variations.

**Required:** `push`/`pop` that retains learned clauses where valid.

```smt2
(push)
(assert (= x 5))
(check-sat)  ; learns clauses
(pop)
(push)
(assert (= x 6))
(check-sat)  ; should reuse some learned clauses
```

### 2.3 Timeout Handling

**Required:** `-t:MILLISECONDS` flag that gracefully returns `unknown` on timeout.

---

## Priority 3: MEDIUM (Nice to Have)

### 3.1 Counterexample for UNSAT CHC

When the property is violated, Z3 Spacer can sometimes provide a counterexample trace. This helps debugging.

### 3.2 Proof Production

For high-assurance mode, we want checkable proofs:
```smt2
(set-option :produce-proofs true)
(check-sat)
(get-proof)
```

### 3.3 Statistics Output

```smt2
(check-sat)
(get-info :all-statistics)
```

Useful for performance tuning.

---

## Integration Interface

### Current (Z3)

```bash
# How Kani Fast calls Z3 today:
echo "$SMT_FORMULA" | z3 -smt2 -in fp.engine=spacer -t:60000
```

### Required (Z4)

Same interface, drop-in replacement:
```bash
echo "$SMT_FORMULA" | z4 -smt2 -in fp.engine=spacer -t:60000
```

Or if flags differ, document the equivalent.

---

## Test Cases

Kani Fast will provide test cases:

1. `simple_counter.smt2` - Basic CHC, should be SAT in <10ms
2. `overflow_check.smt2` - Bitvector overflow property
3. `array_bounds.smt2` - Array access bounds checking
4. `nested_loop.smt2` - Two nested loops, harder invariant

Z4 should produce same results as Z3 on all test cases.

---

## Contact

- **Kani Fast repo:** https://github.com/dropbox/dMATH/kani_fast
- **Integration file:** `crates/kani-fast-chc/src/solver.rs`
- **Current Z3 calls:** Search for `Command::new("z3")` in codebase
