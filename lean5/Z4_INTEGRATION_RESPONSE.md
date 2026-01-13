# Lean5 Integration Response to Z4 Team

**From**: Lean5 Kernel Team
**To**: Z4 SMT Solver Team & Archimedes Platform
**Date**: 2026-01-02
**RE**: Integration Specifications for FFI, Proofs, and Tactics

---

## Executive Summary

We have reviewed the Z4 Integration Feedback and the Archimedes Feature Request. This document provides the specifications needed for Z4 to implement the FFI layer and integrate with Lean5.

**Key decisions:**
- FFI via C ABI (`extern "C"` + `cdylib`)
- DRAT proofs acceptable for initial integration; LRAT preferred long-term
- Three initial tactics: `z4_omega`, `z4_bv`, `z4_smt`
- Test files provided in `tests/z4_integration/`

---

## 1. FFI Requirements

### 1.1 C ABI Exports

Z4 should add the following to `z4-lean-bridge/Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

### 1.2 Required Function Exports

```c
// Solver lifecycle
z4_solver_t* z4_create_solver(void);
void z4_destroy_solver(z4_solver_t* solver);

// Configuration
void z4_set_logic(z4_solver_t* solver, const char* logic);
void z4_set_timeout_ms(z4_solver_t* solver, uint64_t timeout_ms);

// Assertions (SMT-LIB2 format)
int z4_assert(z4_solver_t* solver, const char* smtlib2_assertion);
int z4_assert_soft(z4_solver_t* solver, const char* assertion, uint64_t weight);

// Core operations
z4_result_t z4_check_sat(z4_solver_t* solver);
z4_result_t z4_check_sat_assuming(z4_solver_t* solver, const char** assumptions, size_t count);

// Results
const char* z4_get_model(z4_solver_t* solver);  // JSON format
const char* z4_get_proof(z4_solver_t* solver);  // DRAT/LRAT format
const char* z4_get_unsat_core(z4_solver_t* solver);  // List of assumption names

// Incremental solving
void z4_push(z4_solver_t* solver);
void z4_pop(z4_solver_t* solver, uint32_t n);

// Memory management
void z4_free_string(const char* str);
```

### 1.3 Return Types

```c
typedef enum {
    Z4_SAT = 0,
    Z4_UNSAT = 1,
    Z4_UNKNOWN = 2,
    Z4_ERROR = 3
} z4_result_t;
```

### 1.4 Thread Safety

All functions must be thread-safe. Each `z4_solver_t*` is independent; no global state. Multiple solvers may run concurrently.

---

## 2. Proof Format Requirements

### 2.1 Initial Integration: DRAT

For the initial integration, DRAT proofs are acceptable:

```
# DRAT proof format (text, one clause per line)
# Positive literal = variable ID
# Negative literal = -(variable ID)
# Each line is a learned clause or deletion
1 -2 3 0
d 1 -2 0
```

**API:**
```c
// Returns NULL if SAT or unknown
// Returns DRAT proof string if UNSAT
const char* z4_get_proof(z4_solver_t* solver);
```

### 2.2 Long-term: LRAT Preferred

LRAT format includes clause IDs, enabling efficient verification:

```
# LRAT format: clause_id literals 0 hints 0
1 1 0 0
2 -1 0 0
3 0 1 2 0
```

When LRAT is available in Z4, we will prefer it over DRAT for efficiency.

### 2.3 SMT Proofs (Future)

For SMT proofs (beyond propositional), we need theory-specific justifications:

```json
{
  "type": "smt_proof",
  "clauses": [...],
  "theory_lemmas": [
    {"theory": "QF_LIA", "lemma": "x + y >= 0", "justification": "bound_propagation"}
  ]
}
```

This will be specified in a future iteration once DRAT integration is working.

---

## 3. Tactic Interface

### 3.1 z4_omega (Linear Integer Arithmetic)

**Logic:** QF_LIA

**Use case:** Prove linear integer goals automatically.

```lean
-- Example: ∀ x y : Int, x ≥ 0 → y ≥ 0 → x + y ≥ 0
example (x y : Int) (hx : x ≥ 0) (hy : y ≥ 0) : x + y ≥ 0 := by
  z4_omega
```

**Translation to SMT-LIB2:**
```smt2
(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (>= x 0))
(assert (>= y 0))
(assert (not (>= (+ x y) 0)))
(check-sat)
; Expected: unsat
```

### 3.2 z4_bv (Bitvector Decision)

**Logic:** QF_BV

**Use case:** Prove bitvector goals (overflow, masking, etc.).

```lean
-- Example: ∀ x : UInt8, x &&& 0xFF = x
example (x : UInt8) : x &&& 0xFF = x := by
  z4_bv
```

**Translation:**
```smt2
(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(assert (not (= (bvand x #xFF) x)))
(check-sat)
; Expected: unsat
```

### 3.3 z4_smt (General SMT)

**Logic:** Auto-detect or specified

**Use case:** General decidable goals with theory combination.

```lean
-- Example with arrays
example (a : Array Int Int) (i j : Int) (h : i ≠ j) :
    (a.set i 42).get j = a.get j := by
  z4_smt  -- Uses QF_AUFLIA
```

### 3.4 Common Options

All tactics should support:

```lean
z4_omega (timeout := 5000)  -- milliseconds
z4_bv (verbose := true)     -- print SMT-LIB2 and result
z4_smt (logic := "QF_LRA")  -- override logic detection
```

---

## 4. Goal State Export (for Archimedes)

Lean5 will provide JSON export of goal states:

```json
{
  "goal_id": "mv_42",
  "target": {
    "type": "forall",
    "binder": {"name": "x", "type": "Int"},
    "body": {"type": "app", "fn": ">=", "args": ["x", "0"]}
  },
  "hypotheses": [
    {"name": "h1", "type": "Nat", "value": "5"},
    {"name": "h2", "type": "x > 0"}
  ],
  "metavars": ["mv_43", "mv_44"]
}
```

API endpoint (lean5-server):
```
POST /api/v1/goal/export
{
  "file": "path/to/file.lean",
  "line": 42,
  "column": 10
}
```

---

## 5. Test Files

We will provide sample Lean5 files for integration testing:

```
tests/z4_integration/
├── basic_sat.lean          # Simple SAT problems
├── linear_arith.lean       # QF_LIA test cases
├── bitvector.lean          # QF_BV test cases
├── arrays.lean             # QF_AUFLIA test cases
├── proof_import.lean       # DRAT proof verification
└── performance.lean        # Benchmark problems
```

### 5.1 Sample: basic_sat.lean

```lean
-- Test: Z4 SAT solving with model extraction
import Lean5.Tactic.Z4

-- Simple SAT (should find model)
#check_sat (p ∨ ¬p) ∧ (q ∨ ¬q)

-- Simple UNSAT (should return proof)
theorem unsat_example : ¬((p ∧ ¬p) ∧ q) := by
  z4_decide
```

### 5.2 Sample: linear_arith.lean

```lean
-- Test: Z4 linear integer arithmetic
import Lean5.Tactic.Z4

-- Basic inequality
theorem add_nonneg (x y : Int) (hx : x ≥ 0) (hy : y ≥ 0) : x + y ≥ 0 := by
  z4_omega

-- Transitivity
theorem trans_lt (x y z : Int) (h1 : x < y) (h2 : y < z) : x < z := by
  z4_omega

-- Division (requires careful encoding)
theorem div_mod (n d : Int) (hd : d > 0) : n = (n / d) * d + (n % d) := by
  z4_omega
```

---

## 6. Performance Targets

For AI-assisted proof search (Archimedes), we need:

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| FFI call overhead | < 10 μs | Solver reuse amortizes |
| Simple SAT (< 100 vars) | < 1 ms | |
| QF_LIA (< 50 constraints) | < 10 ms | |
| QF_BV (< 32 bits, < 100 ops) | < 10 ms | |
| DRAT verification (< 10K clauses) | < 100 ms | |

These match Lean5's sub-millisecond verification goal for simple properties.

---

## 7. Integration Phases

### Phase 1: FFI + Basic SAT (Ready Now)
- Implement C ABI exports per §1
- Test with `basic_sat.lean`
- Lean5 will provide: FFI bindings, `z4_decide` tactic

### Phase 2: DRAT Proofs (2 Weeks)
- Z4 returns DRAT proofs for UNSAT
- Lean5 implements DRAT verifier
- Test with `proof_import.lean`

### Phase 3: Theory Tactics (1 Week After Phase 2)
- `z4_omega` for QF_LIA
- `z4_bv` for QF_BV
- Test with `linear_arith.lean`, `bitvector.lean`

### Phase 4: Archimedes Integration (Ongoing)
- Goal state export API
- Proof trace export for ML training
- Parallel tactic execution (`race`)

---

## 8. Contact

For integration coordination:
- Lean5 repo: github.com/dropbox/dMATH/lean5
- Use mail system: `./mail.sh new lean5 <subject>`

We will create a shared `tests/z4_integration/` directory and push sample files within 1 commit.

---

## Appendix A: Lean5 Automation Capabilities

Lean5 already has native automation in `lean5-auto`:

| Component | Description |
|-----------|-------------|
| `cdcl.rs` | CDCL SAT solver |
| `smt.rs` | SMT solver core with DPLL(T) |
| `egraph.rs` | E-graph for equality reasoning |
| `superposition.rs` | First-order theorem prover |
| `premise.rs` | Premise selection (ML-based) |
| `bridge/` | SMT-Kernel translation |

Z4 integration complements this with faster specialized solvers for:
- Complex bitvector arithmetic
- Large linear arithmetic problems
- Theory combination (arrays, datatypes)

## Appendix B: SMT-LIB2 Theory Support Matrix

| Theory | Z4 Status | Lean5 Use |
|--------|-----------|-----------|
| QF_SAT | Yes | `decide` |
| QF_LIA | Yes | `omega` |
| QF_LRA | Yes | Real arithmetic |
| QF_BV | Yes | Bitvectors |
| QF_AUFLIA | Planned | Arrays + LIA |
| QF_UFDT | Planned | Datatypes |
