# Z4 SMT Solver Roadmap

**Last Updated:** 2026-01-01
**Iteration:** 256

---

## Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| z4-sat | ✅ STABLE | CDCL solver, all tests pass |
| z4-core | ✅ STABLE | Terms, sorts, theory trait |
| z4-dpll | ✅ STABLE | DPLL(T) integration |
| z4-frontend | ✅ STABLE | SMT-LIB parser |
| z4-lia | ✅ WORKING | 0.96x vs Z3 |
| z4-lra | ✅ WORKING | 1.38x vs Z3 |
| z4-bv | ✅ WORKING | 1.30x vs Z3 |
| z4-euf | ✅ WORKING | Congruence closure |
| z4-chc | ⚠️ BUGS | PDR solver, 2 critical bugs |
| z4-arrays | ✅ WORKING | Weak equivalence |

---

## 🚨 IMMEDIATE: Critical Bugs

### Bug 1: CHC Soundness (BLOCKING)
- **Issue:** Wrong answers for non-false conclusions
- **Impact:** Kani Fast Phase 18 partially blocked
- **Status:** Worker directive issued
- **ETA:** Next iteration

### Bug 2: CHC Unicode Parser (BLOCKING)
- **Issue:** Crash on Unicode in comments
- **Impact:** Breaks some SMT-LIB files
- **Status:** Worker directive issued
- **ETA:** Next iteration

---

## Integration Status

### Kani Fast (Rust Verification)

| Phase | Feature | Status |
|-------|---------|--------|
| 17 | QF_BV encoding | ✅ COMPLETE |
| 18 | CHC integration | ⚠️ PARTIAL (bugs above) |

**Unblocked when:** CHC soundness bug fixed

### tRust (Verified Rust Compiler)

| Feature | Status |
|---------|--------|
| QF_BV | ✅ WORKS |
| QF_LIA | ✅ WORKS |
| Incremental solving | ❌ NOT IMPLEMENTED |
| Minimal counterexamples | ❌ NOT IMPLEMENTED |

### Lean 5 Bridge

| Feature | Status |
|---------|--------|
| Basic integration | ⚠️ PARTIAL |
| Proof export | ❌ NOT IMPLEMENTED |

---

## Performance vs Z3

| Logic | Z4/Z3 Ratio | Target | Status |
|-------|-------------|--------|--------|
| QF_BV | 1.30x | >1.0x | ✅ ACHIEVED |
| QF_LRA | 1.38x | >1.0x | ✅ ACHIEVED |
| QF_LIA | 0.96x | >1.0x | ❌ NEEDS WORK |
| QF_UF | ~1.0x | >1.0x | ✅ OK |
| CHC | Unknown | >1.0x | ❌ NOT BENCHMARKED |

---

## Roadmap

### Phase 1: Bug Fixes (NOW)
1. ~~HORN CLI auto-detection~~ ✅ DONE
2. CHC soundness bug ⚠️ IN PROGRESS
3. Unicode parser crash ⚠️ IN PROGRESS

### Phase 2: Kani Fast Full Integration
1. Verify all CHC test cases work
2. Benchmark CHC vs Z3 Spacer
3. Document integration API

### Phase 3: Performance
1. QF_LIA optimization (study OpenSMT VSDIS)
2. CHC optimization (study Golem techniques)

### Phase 4: tRust Features
1. Incremental solving (push/pop)
2. Minimal counterexamples
3. Proof certificates

### Phase 5: Advanced
1. QF_FP (floating point)
2. Quantifiers
3. Strings theory

---

## Code Statistics

| Crate | Lines | Tests |
|-------|-------|-------|
| z4-sat | 17,390 | 271 |
| z4-dpll | 16,468 | 419 |
| z4-core | 10,017 | 93 |
| z4-chc | 9,365 | 104 |
| z4-lia | 6,400 | 177 |
| z4-frontend | ~5,000 | 75 |
| **Total** | **~65,000** | **1,200+** |

---

## Reference Implementations

| Solver | Purpose | Location |
|--------|---------|----------|
| Golem | CHC (SOTA) | `reference/golem/` |
| OpenSMT | QF_LIA (SOTA) | `reference/opensmt/` |
| CaDiCaL | SAT | `reference/cadical/` |
| Z3 | Reference | `reference/z3/` |

---

## Success Criteria

1. All tests pass (1,200+)
2. CHC bugs fixed
3. Kani Fast Phase 18 unblocked
4. QF_LIA >= 1.0x vs Z3
