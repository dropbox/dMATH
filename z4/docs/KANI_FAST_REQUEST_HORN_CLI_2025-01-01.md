# Feature Request: HORN Logic CLI Support

**From:** Kani Fast (Rust verification engine)
**To:** Z4 Team
**Date:** 2025-01-01
**Priority:** HIGH - Blocks Phase 18 (Z4 CHC Integration)
**Status:** Kani Fast is READY to integrate, waiting on Z4

---

## Summary

Kani Fast needs Z4's CLI to support `(set-logic HORN)` for CHC solving via subprocess.

**Current state:**
- Z4 has `z4-chc` crate with full PDR/IC3 implementation (82 tests passing)
- Z4 CLI works for QF_BV, QF_LIA, QF_LRA
- Z4 CLI returns `(error "unsupported logic: HORN")` for CHC problems

**What we need:**
```bash
echo '(set-logic HORN)
(declare-fun Inv (Int) Bool)
(assert (forall ((x Int)) (=> (= x 0) (Inv x))))
(assert (forall ((x Int)) (=> (and (Inv x) (< x 10)) (Inv (+ x 1)))))
(query Inv)
(check-sat)' | z4

# Expected: sat (with invariant model)
# Current: (error "unsupported logic: HORN")
```

---

## Context: Kani Fast Status

Kani Fast has completed:
- Phase 16: Algebraic rewriting for bitwise ops
- Phase 17: BitVec encoding (QF_BV) - **3 bitwise tests now PASS**

Phase 18 (Z4 CHC Integration) is **BLOCKED** on this feature.

**Current workaround:** We use Z3 Spacer for CHC solving:
```bash
echo "$CHC_FORMULA" | z3 -smt2 -in fp.engine=spacer
```

**Goal:** Replace Z3 with Z4:
```bash
echo "$CHC_FORMULA" | z4 -smt2 -in
```

---

## Technical Details

### What Z4 Already Has

The `z4-chc` crate is complete:

```
crates/z4-chc/src/
├── clause.rs      # Horn clause representation
├── expr.rs        # CHC expressions
├── mbp.rs         # Model-Based Projection (39KB)
├── parser.rs      # CHC parser (39KB)
├── pdr.rs         # PDR/IC3 algorithm (142KB!)
├── predicate.rs   # Predicate handling
├── problem.rs     # CHC problem representation
└── smt.rs         # SMT integration (36KB)
```

**Test evidence:**
```
$ cargo test -p z4-chc
test result: ok. 73 passed; 0 failed
test result: ok. 9 passed (integration tests)
```

### What's Missing

Wire `z4-chc` to the CLI frontend:

1. In `crates/z4-frontend/src/elaborate.rs`: Add HORN to supported logics
2. In `crates/z4/src/main.rs`: Route HORN problems to `z4-chc::PdrSolver`

### Suggested Implementation

```rust
// In CLI main loop
match logic.as_str() {
    "QF_BV" | "QF_LIA" | "QF_LRA" => {
        // Existing DPLL(T) path
        dpll_solver.solve(commands)
    }
    "HORN" => {
        // New CHC path
        let problem = z4_chc::ChcParser::parse(&input)?;
        let solver = z4_chc::PdrSolver::new(config);
        solver.solve(&problem)
    }
    _ => Err("unsupported logic")
}
```

---

## Integration Test

When HORN is supported, this should work:

```bash
# Test file: test_horn.smt2
(set-logic HORN)
(declare-fun Inv (Int) Bool)

; Base case: Inv(0)
(assert (forall ((x Int)) (=> (= x 0) (Inv x))))

; Inductive step: Inv(x) ∧ x < 10 → Inv(x+1)
(assert (forall ((x Int)) (=> (and (Inv x) (< x 10)) (Inv (+ x 1)))))

; Safety: Inv(x) ∧ x ≥ 10 → x = 10
(assert (forall ((x Int)) (=> (and (Inv x) (>= x 10)) (= x 10))))

(check-sat)
(get-model)
```

**Expected output:**
```
sat
(model
  (define-fun Inv ((x Int)) Bool (<= 0 x 10))
)
```

---

## Impact

Once Z4 supports HORN via CLI:

1. **Kani Fast Phase 18 unblocked** - Z4 becomes CHC backend
2. **Pure Rust verification stack** - No Z3 dependency
3. **Performance gains** - Z4 PDR may be faster than Z3 Spacer
4. **Unified solver** - One solver for QF_BV + CHC

---

## Timeline Request

| Milestone | Impact |
|-----------|--------|
| HORN CLI support | Unblocks Kani Fast Phase 18 |
| Invariant model output | Full integration |
| Benchmarks vs Z3 Spacer | Performance validation |

**We're ready to test as soon as HORN is wired up.**

---

## Contact

- **Kani Fast repo:** https://github.com/dropbox/kani_fast
- **Integration file:** `crates/kani-fast-chc/src/solver.rs`
- **Current Z3 calls:** `ChcBackend::Z3` subprocess

When HORN support lands, we'll add `ChcBackend::Z4` and benchmark immediately.
