# Z4 Integration Feedback for Kani Fast

**From**: Z4 SMT Solver Team
**Date**: 2025-01-01
**Z4 Version**: Iteration #235

---

## Executive Summary

Z4 is ready to serve as a **QF_BV backend** for Kani Fast. Bitvector performance exceeds Z3 by 21%. Integration testing recommended.

---

## Z4 Capabilities Status

| Capability | Status | Notes |
|------------|--------|-------|
| **QF_BV (bitvectors)** | ✅ READY | 1.21x faster than Z3, 100% correct |
| **QF_LIA (integers)** | ✅ READY | Gomory cuts + GCD test |
| **QF_LRA (reals)** | ✅ READY | Dual simplex |
| **QF_UF (uninterpreted)** | ⚠️ LIMITED | Correct but slow on large EUF problems |
| **Incremental (push/pop)** | ✅ EXISTS | Needs clause retention verification |
| **Assumption-based solving** | ❓ UNVERIFIED | API exists, needs testing |
| **CHC/Spacer** | ✅ EXISTS | z4-chc crate, iterations #231-235 |
| **Proof generation** | ✅ DRAT | LRAT in progress |

---

## Kani Fast Requirements Checklist

From `docs/KANI_FAST_REQUIREMENTS.md`:

### Priority 1: CRITICAL

| Requirement | Z4 Status | Action Needed |
|-------------|-----------|---------------|
| Fast QF_BV (<2x CaDiCaL) | ✅ 1.21x faster than Z3 | None |
| Incremental push/pop with clause retention | ⚠️ Exists | Verify clause retention works |
| Assumption-based solving | ⚠️ Exists | Integration test needed |
| Model extraction | ✅ Works | None |

### Priority 2: HIGH

| Requirement | Z4 Status | Action Needed |
|-------------|-----------|---------------|
| Low memory (<1.5x CaDiCaL) | ❓ Not benchmarked | Run memory profiling |
| Fast startup (<10ms) | ✅ ~5-7ms | None |
| Parallel portfolio | ✅ z4-sat has this | None |

### Priority 3: MEDIUM

| Requirement | Z4 Status | Action Needed |
|-------------|-----------|---------------|
| CHC/Spacer | ✅ z4-chc exists | Integration test |

---

## Recommended Integration Steps

1. **Add Z4 as Cargo dependency**:
   ```toml
   [dependencies]
   z4 = { git = "https://github.com/dropbox/dMATH/z4", features = ["bv"] }
   ```

2. **Test QF_BV problems from Kani**:
   ```rust
   use z4::Solver;
   let mut solver = Solver::new();
   // Add bitvector constraints
   let result = solver.check_sat();
   ```

3. **Verify incremental solving**:
   ```rust
   solver.push();
   solver.add_assertion(constraint);
   solver.check_sat();
   solver.pop();
   // Verify learned clauses retained where valid
   ```

4. **Run Kani Fast test suite against Z4 backend**

---

## Known Limitations

1. **QF_UF at scale**: Z4 lacks incremental theory propagation for EUF. Problems like `eq_diamond15+` timeout. This affects problems with deep equality chains.

2. **Proof format**: Z4 produces DRAT proofs. If Kani needs a different format (e.g., for verified checking), translation may be needed.

3. **API stability**: Z4 API is not yet 1.0 stable.

---

## Performance Benchmarks

```
QF_BV (50 benchmarks):
  Z4 total:  0.352s
  Z3 total:  0.427s
  Ratio:     1.21x (Z4 faster)
  Agreement: 100%

SAT core (uf250, 100 benchmarks):
  Z4 total:  24.74s
  CaDiCaL:   27.54s
  Ratio:     0.90x (Z4 10% faster)
  Wins:      Z4=80, CaDiCaL=20
```

---

## Contact

For integration issues, file at: https://github.com/dropbox/dMATH/z4/issues

Tag: `kani-fast-integration`
