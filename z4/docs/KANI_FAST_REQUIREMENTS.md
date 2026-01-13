# Z4 Requirements for Kani Fast Integration

**Author**: Kani Fast AI (via DashProve)
**Date**: 2024-12-29
**Status**: Requirements Specification

## Executive Summary

[Kani Fast](https://github.com/dropbox/kani_fast) is a next-generation Rust verification tool that aims to be 10-100x faster than Kani. Currently, verification tools use multiple backends:

- **SAT solvers** (CaDiCaL, Kissat, MiniSat) for bounded model checking
- **Z3** for SMT queries, k-induction, and CHC solving

**Goal**: Make Z4 fast enough to be the **single backend** for all Kani Fast needs, eliminating the complexity of multiple solver integrations.

This document specifies exactly what Z4 needs to achieve this goal.

---

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KANI FAST                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Bounded   │  │ K-Induction │  │     CHC     │                 │
│  │     BMC     │  │  (unbounded)│  │   Solving   │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          │                                          │
│                          ▼                                          │
│                   ┌─────────────┐                                   │
│                   │     Z4      │  ← Single unified backend         │
│                   │  (this is   │                                   │
│                   │    you!)    │                                   │
│                   └─────────────┘                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Requirement Categories

### Priority 1: CRITICAL (Blocks Integration)

These features are required for Z4 to replace current backends.

### Priority 2: HIGH (Required for Performance Parity)

These features are needed to match current multi-backend performance.

### Priority 3: MEDIUM (Enables Advanced Features)

These features unlock Kani Fast's advanced verification modes.

### Priority 4: NICE-TO-HAVE (Polish)

These features improve usability but aren't blockers.

---

## Priority 1: CRITICAL

### 1.1 Fast QF_BV (Quantifier-Free Bitvectors)

**What**: Kani produces 100% bitvector problems. All Rust integers (u8, i32, u64, etc.) become fixed-width bitvectors.

**Requirement**: Z4's QF_BV performance must be within **2x of CaDiCaL** on equivalent SAT problems.

**Why**: This is 80% of Kani's workload. If QF_BV is slow, everything is slow.

**Benchmark**: SAT Competition 2023 bitvector-encoded benchmarks.

**Implementation Notes**:
- Aggressive bit-blasting to CNF
- Use z4-sat directly for the resulting SAT problem
- Avoid unnecessary abstraction layers
- Consider lazy bit-blasting for large problems

```
Performance Target:
- Small problems (<10k vars): <100ms
- Medium problems (10k-100k vars): <10s
- Large problems (100k-1M vars): <5min
```

### 1.2 Incremental Solving with Clause Retention

**What**: Kani Fast's 100x speedup comes from incremental verification. When code changes slightly, we want to reuse previous solver work.

**Requirement**: `push()`/`pop()` that **retains learned clauses** where valid.

**Current Z3 Behavior** (what to improve):
```
push()           // Save state
add_assertion(A)
check_sat()      // Learns clauses C1, C2, C3
pop()            // LOSES all learned clauses! ← Bad
push()
add_assertion(A')
check_sat()      // Must re-learn everything ← Slow
```

**Required Z4 Behavior**:
```
push()
add_assertion(A)
check_sat()      // Learns clauses C1, C2, C3
pop()            // Keeps C1, C2 (invalidates C3 which depended on A)
push()
add_assertion(A')
check_sat()      // Reuses C1, C2! ← Fast
```

**Implementation Notes**:
- Track clause provenance (which assertions each clause depends on)
- On pop(), only invalidate clauses that depend on popped assertions
- Consider "assumption-based" solving as alternative

### 1.3 Assumption-Based Solving

**What**: Alternative to push/pop for incremental solving.

**Requirement**: `check_sat_assuming(assumptions: &[Literal])` that:
1. Adds assumptions as temporary unit clauses
2. Returns SAT/UNSAT
3. On UNSAT, provides unsat core of assumptions
4. Does NOT modify solver state permanently

**API**:
```rust
// Check with temporary assumptions
let result = solver.check_sat_assuming(&[lit_a, lit_b, lit_c]);

match result {
    Sat(model) => { /* ... */ }
    Unsat(core) => {
        // core is subset of {lit_a, lit_b, lit_c} that caused UNSAT
        // Solver state unchanged, can immediately call again
    }
}
```

**Why**: This is how modern SAT solvers (CaDiCaL, Kissat) handle incrementality. Very efficient.

---

## Priority 2: HIGH

### 2.1 Low Memory Footprint

**What**: Z3 uses 2-5x more memory than pure SAT solvers.

**Requirement**: Memory usage within **1.5x of CaDiCaL** for equivalent problems.

**Why**: Verification problems can be huge. OOM = verification failure.

**Techniques**:
- Clause database garbage collection
- Compact literal representation (32-bit, not 64-bit pointers)
- Memory-mapped clause storage for very large problems
- Hard memory limits with graceful degradation

### 2.2 Fast Startup

**What**: Z3 takes ~100ms just to initialize.

**Requirement**: Cold start to first `check_sat()` in **<10ms**.

**Why**: Incremental verification may spawn many solver instances. Startup overhead kills throughput.

**Techniques**:
- Lazy initialization of theory solvers
- Pre-compiled clause database formats
- No unnecessary allocations at startup

### 2.3 Parallel Check-Sat (Internal Portfolio)

**What**: Run multiple solving strategies in parallel, return first result.

**Requirement**: Use all available cores during `check_sat()`.

**Why**: Different strategies win on different problems. Portfolio approach is robust.

**Strategies to parallelize**:
- Different restart policies
- Different variable ordering heuristics (VSIDS vs CHB vs LRB)
- Different phase saving policies
- Preprocessing vs no preprocessing

**API**:
```rust
let config = SolverConfig {
    parallel: true,
    num_threads: 8,  // or auto-detect
    portfolio: vec![
        Strategy::VSIDS_Luby,
        Strategy::CHB_Geometric,
        Strategy::LRB_Glucose,
    ],
};
let solver = Solver::new(config);
```

---

## Priority 3: MEDIUM (For Unbounded Verification)

### 3.1 CHC/Spacer (Constrained Horn Clauses)

**What**: Z3's Spacer engine solves Constrained Horn Clauses to find inductive invariants.

**Requirement**: CHC solver that can find invariants for simple loops in **<30 seconds**.

**Example CHC Problem**:
```smt2
; Find invariant Inv(x) such that:
; 1. x = 0 → Inv(x)              (base case)
; 2. Inv(x) ∧ x < 100 → Inv(x+1) (inductive step)
; 3. Inv(x) ∧ x >= 100 → x = 100 (property)

(declare-rel Inv (Int))
(declare-var x Int)

(rule (=> (= x 0) (Inv x)))
(rule (=> (and (Inv x) (< x 100)) (Inv (+ x 1))))
(query (=> (and (Inv x) (>= x 100)) (not (= x 100))))
```

**Why**: This is how Kani Fast achieves unbounded verification. K-induction alone isn't enough for complex loops; we need invariant synthesis.

**Implementation Options**:
1. Port Z3's Spacer algorithm
2. Implement Property-Directed Reachability (PDR/IC3)
3. Use counterexample-guided invariant synthesis (CEGIS)

### 3.2 Quantifiers (forall/exists)

**What**: K-induction requires checking `∀x. P(x) ∧ T(x,x') → P(x')`.

**Requirement**: Handle quantifiers in QF_BV extended with quantifiers.

**Why**: Inductive step verification requires universal quantification.

**Techniques**:
- E-matching for simple patterns
- Model-based quantifier instantiation (MBQI)
- Quantifier elimination for bitvectors

### 3.3 Interpolation

**What**: Craig interpolants for counterexample-guided abstraction refinement (CEGAR).

**Requirement**: Given `A ∧ B = UNSAT`, produce interpolant `I` where:
- `A → I`
- `I ∧ B = UNSAT`

**Why**: CEGAR is key to scaling verification. Interpolants tell us what to track.

**Current Z3 Status**: Interpolation exists but is slow/buggy.

---

## Priority 4: NICE-TO-HAVE

### 4.1 Proof Production

**What**: Generate checkable proofs for SAT/UNSAT results.

**Formats**:
- **DRAT/LRAT** for SAT (can verify with drat-trim)
- **Alethe** for SMT (emerging standard)

**Why**: High-assurance verification needs proof checking. Don't trust the solver, verify it.

### 4.2 Rust-Native API

**What**: Current Z3 bindings go through C FFI.

**Requirement**: Native Rust API that's ergonomic and zero-copy.

```rust
// Dream API
let solver = z4::Solver::new();
let x = solver.bv_const("x", 32);
let y = solver.bv_const("y", 32);

solver.assert(x.bvadd(y).eq(solver.bv_val(100u32)));
solver.assert(x.bvugt(solver.bv_val(0u32)));

match solver.check() {
    Sat => {
        let x_val: u32 = solver.eval(&x).unwrap();
        let y_val: u32 = solver.eval(&y).unwrap();
    }
    Unsat => { /* ... */ }
}
```

### 4.3 Configuration Presets

**What**: Pre-tuned configurations for specific problem classes.

```rust
let solver = Solver::with_preset(Preset::BitvectorBmc);  // For Kani
let solver = Solver::with_preset(Preset::LiaOptimization);  // For scheduling
let solver = Solver::with_preset(Preset::ChcInvariant);  // For invariant inference
```

---

## Benchmark Targets

### QF_BV (Primary - 80% of workload)

| Benchmark Set | Target | Comparison |
|---------------|--------|------------|
| SAT Competition BV-encoded | Within 2x of CaDiCaL | Pure SAT baseline |
| SMT-LIB QF_BV | Within 1.5x of Z3 | Direct comparison |
| Kani benchmarks (s2n-quic) | Within 1.2x of current | Real-world |

### Incrementality

| Scenario | Target |
|----------|--------|
| Small change, re-verify | <10% of cold start time |
| 1000 incremental queries | <5min total (not 1000x cold start) |

### CHC

| Benchmark | Target |
|-----------|--------|
| Simple loop invariants | <30s |
| Nested loops (depth 2) | <2min |
| Recursive functions | <5min |

---

## Integration Points

### Rust API Location

Z4 should expose a clean Rust API that Kani Fast can depend on:

```toml
# In kani-fast/Cargo.toml
[dependencies]
z4 = { git = "https://github.com/dropbox/dMATH/z4", features = ["bv", "incremental", "chc"] }
```

### Feature Flags

```toml
# z4/Cargo.toml
[features]
default = ["sat", "bv"]
sat = []           # Pure SAT solving
bv = ["sat"]       # Bitvector theory (bit-blasts to SAT)
incremental = []   # push/pop/assumptions
chc = []          # CHC/Spacer
quantifiers = []  # forall/exists
proofs = []       # Proof production
parallel = []     # Multi-threaded portfolio
```

---

## Development Priority Order

Recommended order of implementation for maximum impact:

1. **z4-sat**: CDCL SAT solver (already in progress)
2. **QF_BV**: Bit-blasting bitvector solver
3. **Incrementality**: push/pop with clause retention + assumptions
4. **Performance tuning**: Match CaDiCaL on benchmarks
5. **CHC/Spacer**: For unbounded verification
6. **Parallel portfolio**: Use all cores
7. **Quantifiers**: For k-induction
8. **Proofs**: For high assurance

---

## Questions for Z4 Team

1. Is the current z4-sat architecture amenable to clause retention on pop()?
2. What's the plan for bit-blasting? Eager (full blast) or lazy (on-demand)?
3. Any plans for DRAT proof output from z4-sat?
4. Interest in sharing benchmarks/test suites with Kani Fast?

---

## Contact

- **Kani Fast repo**: https://github.com/dropbox/kani_fast
- **DashProve repo**: https://github.com/dropbox/dashprove
- **This requirements doc**: Lives in Z4 repo at `docs/KANI_FAST_REQUIREMENTS.md`

The Kani Fast and Z4 projects should be developed in parallel, with Z4 providing the solver foundation and Kani Fast providing real-world verification benchmarks to drive Z4's optimization.
