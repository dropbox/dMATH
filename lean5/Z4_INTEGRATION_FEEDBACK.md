# Z4 Integration Feedback for Lean 5

**From**: Z4 SMT Solver Team
**Date**: 2025-01-01
**Z4 Version**: Iteration #235

---

## Executive Summary

Z4 has a **foundation for Lean 5 integration** but critical components are missing. The `z4-lean-bridge` crate can export Z4 terms to Lean syntax, but there is **no FFI binding** and **no proof import mechanism**.

**Current State**: Not ready for production use.
**Effort to Ready**: 2-4 weeks of focused work.

---

## What Exists

### z4-lean-bridge (1,100 lines)

| Feature | Status | Notes |
|---------|--------|-------|
| Term export to Lean syntax | ✅ Works | All SMT-LIB sorts and operations |
| Sort mapping | ✅ Works | Bool, Int, Real, BitVec, Array, String |
| SAT model verification codegen | ✅ Works | Generates `native_decide` proofs |
| Backend discovery | ✅ Works | Finds `lean` or `lake env lean` |
| Unit tests | ✅ 9 pass | Core export functionality |

### Example Output

Z4 can generate:
```lean
def x : Bool := true
def y : Bool := true
theorem sat_verification : x && y = true := by native_decide
```

---

## What's Missing (Critical)

### 1. No C FFI Bindings

The `z4-lean-bridge` crate has:
- NO `#[no_mangle]` exports
- NO `extern "C"` functions
- NO `crate-type = ["cdylib"]` in Cargo.toml

**Required for**:
```lean
@[extern "z4_sat_solve"]
opaque z4Solve : Array Clause → IO SolveResult
```

**Work needed**: Add FFI layer with proper C ABI exports.

### 2. No DRAT→Lean Proof Translation

From z4-lean-bridge code:
> "Full UNSAT verification requires checking the DRAT proof, which is beyond simple Lean type-checking."

Z4 produces DRAT proofs. These need to be translated to Lean proof terms for verified UNSAT results.

**Work needed**: DRAT certificate parser + Lean proof term generator.

### 3. Tactics Are Placeholders

The Lean tactics (`z4_decide`, `z4_smt`, `z4_bv`) just call `native_decide`:
```lean
macro_rules
| `(tactic| z4_decide) => `(tactic| native_decide)
```

They do NOT invoke Z4.

**Work needed**: Connect tactics to actual Z4 invocation.

### 4. No Integration Tests

No tests actually invoke Lean. The doc-test is marked `ignore`.

---

## What Z4 Needs from Lean Team

1. **Lean FFI specification**: What C functions should Z4 export? What types?

2. **Proof format requirements**:
   - Is DRAT acceptable?
   - Do you need Lean-native proof terms?
   - What about LRAT (with clause IDs)?

3. **Tactic interface design**:
   - What should `z4_omega` look like?
   - What theories are needed? (LIA for omega, BV for decide?)

4. **Testing infrastructure**: Sample Lean files that would use Z4.

---

## Z4 Capabilities for Lean

| Theory | Z4 Status | Lean Use Case |
|--------|-----------|---------------|
| QF_LIA | ✅ 1.33x faster than Z3 | `omega` tactic |
| QF_BV | ✅ 1.21x faster than Z3 | `decide` on fixed-width |
| QF_LRA | ✅ 1.37x faster than Z3 | Real arithmetic |
| Proofs | ✅ DRAT, LRAT in progress | Verified UNSAT |

---

## Recommended Path Forward

### Phase 1: FFI Layer (1 week)
1. Add `crate-type = ["cdylib"]` to z4-lean-bridge
2. Export `extern "C"` functions for:
   - `z4_create_solver()`
   - `z4_add_assertion()`
   - `z4_check_sat()`
   - `z4_get_model()`
   - `z4_get_proof()`
3. Test with basic Lean `@[extern]` calls

### Phase 2: Proof Translation (2 weeks)
1. Parse DRAT proofs from Z4
2. Generate Lean proof terms
3. Verify with Lean kernel

### Phase 3: Tactics (1 week)
1. Implement `z4_omega` for LIA
2. Implement `z4_bv` for bitvectors
3. Add test suite

---

## Contact

For integration issues, file at: https://github.com/dropbox/z4/issues

Tag: `lean-integration`
