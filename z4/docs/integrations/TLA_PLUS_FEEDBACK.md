# Z4 Integration Feedback for TLA+

**From**: Z4 SMT Solver Team
**Date**: 2025-01-01
**Z4 Version**: Iteration #235

---

## Executive Summary

The `z4-tla-bridge` crate is a **TLC wrapper**, not a Z4-TLA+ integration. Z4 is not used as a solver for TLA+ specifications.

**Current State**: Z4 cannot serve as a TLA+ backend.
**Effort to Ready**: Significant (needs set theory solver for TLAPS).

---

## What Exists

### z4-tla-bridge (841 lines)

| Feature | Status | Notes |
|---------|--------|-------|
| TLC execution | ✅ Works | Runs TLC model checker |
| Output parsing | ✅ Works | Parses TLC stdout to structured results |
| Error classification | ✅ Works | Deadlock, invariant violation, liveness, etc. |
| Counterexample extraction | ✅ Works | Extracts traces, suggests fixes |
| Integration tests | ✅ Pass | Tests with cdcl.tla spec |

### What z4-tla-bridge Actually Does

```
TLA+ Spec → TLC Model Checker → Parsed Results
                ↑
         z4-tla-bridge wraps this

Z4 is NOT involved in solving.
```

The bridge is useful for **testing TLA+ specs of Z4's algorithms** (e.g., verifying cdcl.tla), not for using Z4 as a TLA+ solver.

---

## What's Missing for TLA+ Integration

### 1. No TLC Constraint Solver Integration

TLC has its own built-in constraint solving. There's no mechanism to use Z4 instead.

**Would require**: Implementing TLC's internal constraint format in Z4.

### 2. No TLAPS Integration

TLAPS (TLA+ Proof System) uses SMT solvers via SMT-LIB. To support TLAPS, Z4 needs:

| Requirement | Z4 Status | Notes |
|-------------|-----------|-------|
| Full SMT-LIB 2.6 | ⚠️ Partial | Parser exists, some commands missing |
| Integer arithmetic | ✅ QF_LIA | Works |
| Set theory | ❌ Missing | Critical for TLA+ |
| Sequence operations | ❌ Missing | TLA+ uses sequences heavily |
| Function application | ⚠️ Partial | EUF exists |
| `get-proof` command | ⚠️ Partial | DRAT, not SMT-LIB proof format |

### 3. No Set Theory Solver

TLA+ heavily uses finite sets:
- `\in` (membership)
- `\cup`, `\cap`, `\setminus` (operations)
- `Cardinality(S)` (size)
- `SUBSET S` (powerset)

Z4 has no set theory implementation.

### 4. No Bidirectional Translation

There's no mechanism to:
- Translate TLA+ formulas to SMT-LIB
- Use Z4 to discharge TLAPS proof obligations
- Encode TLC state predicates as Z4 constraints

---

## What Z4 Provides That Might Help

| Z4 Feature | TLA+ Use Case | Status |
|------------|---------------|--------|
| QF_LIA | Integer constraints in specs | ✅ Ready |
| QF_BV | Bit-level verification | ✅ Ready |
| CHC/Spacer | Invariant inference | ✅ z4-chc exists |
| PDR | Property-directed reachability | ✅ In z4-chc |

The **CHC solver** (z4-chc) might be relevant for TLA+ invariant checking, but integration work is needed.

---

## Recommended Path Forward

### If Goal is TLAPS Backend

1. **Implement set theory solver** (z4-sets crate)
   - Finite set membership
   - Union, intersection, difference
   - Cardinality constraints
   - Effort: 4-6 weeks

2. **Complete SMT-LIB proof production**
   - Map DRAT to SMT-LIB proof format
   - Or implement direct proof term generation
   - Effort: 2-3 weeks

3. **Test against TLAPS benchmarks**
   - TLAPS generates SMT-LIB files
   - Run Z4 on these to identify gaps
   - Effort: 1 week

### If Goal is TLC Enhancement

This would require deep integration with TLC internals. Not recommended unless TLC team is involved.

### If Goal is Spec Verification (Current Use)

The z4-tla-bridge already works for this. It can:
- Run TLA+ specs that model Z4's algorithms
- Parse results for CI integration
- Verify algorithm correctness via TLC

---

## Current TLA+ Specs in Z4

Located in `/Users/ayates/z4/specs/`:

| File | Purpose |
|------|---------|
| `cdcl.tla` | Full CDCL algorithm model |
| `cdcl_test.tla` | Test instance with specific formula |
| `*.cfg` | TLC configuration files |

These verify Z4's **algorithm design**, not its implementation.

---

## Contact

For integration issues, file at: https://github.com/dropbox/dMATH/z4/issues

Tag: `tla-integration`
