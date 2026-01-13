# Z4 Formal Verification Strategy

## Objective

Build Z4 as BOTH:
1. **The most performant SMT solver** — competitive with Z3/CVC5/Yices
2. **A fully formally verified implementation** — like IsaSAT/CreuSAT but faster

This is not either/or. IsaSAT solves 40% as many problems as the best unverified solvers. The gap comes from missing advanced techniques, NOT from verification overhead. We can close that gap.

**Bootstrap Vision:** Use Z3 (unverified but battle-tested) to verify Z4, then use Z4 to verify the next generation.

This document specifies a multi-layered verification strategy that combines proof production, bounded model checking, deductive verification, and rigorous testing.

---

## Critical Properties

An SMT solver must satisfy:

| Property | Definition | Consequence of Violation |
|----------|------------|-------------------------|
| **Soundness** | SAT → satisfying assignment exists; UNSAT → no assignment exists | Silent wrong answers; catastrophic for users |
| **Completeness** | Solver terminates with definite answer (for decidable theories) | Hangs, timeouts |
| **Memory Safety** | No undefined behavior, no use-after-free, no data races | Crashes, security vulnerabilities |
| **Termination** | Solver eventually terminates | Resource exhaustion |

**Soundness is paramount.** A solver that says "UNSAT" when the formula is satisfiable can cause downstream tools to accept buggy programs, miss security vulnerabilities, or produce incorrect proofs.

---

## Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 0: RUST TYPE SYSTEM                        │
│         Memory safety, ownership, lifetimes, enums for FSMs         │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: PROOF PRODUCTION                        │
│    DRAT (SAT), Alethe/LFSC (SMT) — externally verifiable proofs     │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 2: BOUNDED MODEL CHECKING                  │
│         Kani for unsafe code, critical invariants, panic-freedom    │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 3: ALGORITHM SPECIFICATION                 │
│         TLA+ models for CDCL, DPLL(T), theory propagation           │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 4: DEDUCTIVE VERIFICATION (SELECTIVE)      │
│       Verus/Creusot for union-find, watched literals, simplex       │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 5: PROPERTY-BASED TESTING                  │
│         proptest for all APIs, random formula generation            │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 6: DIFFERENTIAL TESTING                    │
│              Compare against Z3, CVC5 on all inputs                 │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 7: FUZZING                                 │
│         Grammar-based SMT-LIB fuzzing, AFL, cargo-fuzz              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 0: Rust Type System

**Effort: Built-in**
**Guarantee: Memory safety for safe Rust**

Rust's type system already provides significant guarantees. Maximize these by:

### Guidelines

1. **Minimize `unsafe`**: Every `unsafe` block requires:
   - A `// SAFETY:` comment explaining the invariant
   - A Kani proof harness (Layer 2)
   - Review by another AI worker or human

2. **Use newtypes for disambiguation**:
   ```rust
   // Bad: easy to confuse
   fn assign(var: u32, value: bool);

   // Good: type-safe
   struct Variable(u32);
   struct Literal(u32);
   fn assign(var: Variable, value: bool);
   ```

3. **Use enums for state machines**:
   ```rust
   enum SolverState {
       Idle,
       Propagating,
       Deciding,
       Conflicting,
       Satisfied,
       Unsatisfied,
   }
   ```

4. **Leverage the borrow checker**: Design data structures so invalid states are unrepresentable.

---

## Layer 1: Proof Production (CRITICAL)

**Effort: High (but essential)**
**Guarantee: Externally verifiable correctness**

This is the industry standard. SAT competitions have required DRAT proofs since 2013. Any production-quality solver must support proof generation.

### SAT: DRAT Proofs

**DRAT** (Delete Resolution Asymmetric Tautologies) certificates prove UNSAT results.

```
Format: sequence of clause additions and deletions
- "d 1 2 3 0" — delete clause (1 ∨ 2 ∨ 3)
- "1 -2 0"    — add clause (1 ∨ ¬2)
- Final empty clause proves UNSAT
```

**Implementation requirements:**
- Add proof logging infrastructure to `z4-sat`
- Log every learned clause with its derivation
- Log clause deletions
- Support binary DRAT for efficiency (3x smaller)

**Verification:**
- Use `drat-trim` (reference checker) in CI
- Use `cake_lpr` (verified checker in CakeML) for high-assurance

### SMT: Alethe Proofs

**Alethe** is the emerging standard for SMT proofs (used by CVC5, veriT).

**Implementation requirements:**
- Add `z4-proof` crate (already in design)
- Record theory lemmas with justifications
- Record resolution steps
- Export in Alethe format

**Verification:**
- Use `carcara` (Alethe proof checker)
- Use `SMTCoq` for Coq-verified checking

### Alternative: LFSC

LFSC (Logical Framework with Side Conditions) is older but well-supported:
- CVC5 supports it
- `LFSC` checker is available

### Metrics

| Metric | Target |
|--------|--------|
| Proof overhead | < 2x runtime |
| Proof checking | < 10x solving time |
| Proof coverage | 100% of UNSAT answers |

---

## Layer 2: Bounded Model Checking (Kani)

**Effort: Medium**
**Guarantee: Absence of bugs within bounds**

Kani uses CBMC to verify Rust code up to a bounded execution depth.

### Target Areas

1. **All unsafe blocks**: Every `unsafe` block gets a Kani harness.

2. **Watched literal invariants**:
   ```rust
   #[kani::proof]
   #[kani::unwind(100)]
   fn watched_literals_invariant() {
       let mut solver = Solver::new();
       let clause: Vec<Literal> = kani::any();
       kani::assume(clause.len() >= 2);
       solver.add_clause(&clause);

       // Invariant: first two literals are watched
       let watches = solver.get_watches(&clause);
       kani::assert!(watches.contains(&clause[0]) || watches.contains(&clause[1]));
   }
   ```

3. **Unit propagation correctness**:
   ```rust
   #[kani::proof]
   fn unit_propagation_sound() {
       let mut solver = Solver::new();
       // ... setup with symbolic clauses
       let result = solver.propagate();
       if result == PropagationResult::Conflict {
           // If conflict detected, assignment violates some clause
           kani::assert!(solver.has_violated_clause());
       }
   }
   ```

4. **Panic freedom**: Prove that core functions don't panic.

### Integration

```toml
# Cargo.toml
[dev-dependencies]
kani = "0.66"

[package.metadata.kani]
default-unwind = 10
```

```bash
# Run Kani proofs
cargo kani --tests
```

### Limitations

- Bounded: can only verify up to N iterations/elements
- Slow: may take minutes per proof
- No concurrency support yet

---

## Layer 3: Algorithm Specification (TLA+)

**Effort: Medium**
**Guarantee: Algorithm correctness before implementation**

Model algorithms in TLA+ BEFORE implementing them. Find design bugs early.

### Target Specifications

1. **CDCL State Machine** (`specs/cdcl.tla`):
   ```tla
   VARIABLES
       assignment,     \* Current partial assignment
       decision_level, \* Current decision level
       trail,          \* Stack of assignments with reasons
       clauses,        \* Original + learned clauses
       state           \* PROPAGATING | DECIDING | CONFLICTING | SAT | UNSAT

   TypeInvariant ==
       /\ assignment \in [Variables -> {TRUE, FALSE, UNKNOWN}]
       /\ decision_level \in Nat
       /\ state \in {"PROPAGATING", "DECIDING", "CONFLICTING", "SAT", "UNSAT"}

   Soundness ==
       state = "SAT" => AllClausesSatisfied(assignment)
       state = "UNSAT" => \A a \in AllAssignments : ~AllClausesSatisfied(a)
   ```

2. **Theory Propagation Protocol** (`specs/dpll_t.tla`):
   - Model interaction between SAT core and theory solvers
   - Verify no propagation cycles
   - Verify conflict detection

3. **Simplex Algorithm** (`specs/simplex.tla`):
   - Model pivoting operations
   - Verify termination (Bland's rule)
   - Verify Farkas lemma for conflicts

4. **Congruence Closure** (`specs/cc.tla`):
   - Model union-find operations
   - Verify equivalence class invariants
   - Verify explanation correctness

### Workflow

1. Write TLA+ spec for algorithm
2. Run TLC model checker with small bounds
3. Fix any invariant violations
4. Implement in Rust following the spec
5. Keep spec updated as implementation evolves

### Tools

- **TLC**: Explicit-state model checker (default)
- **Apalache**: Symbolic model checker (for larger state spaces)

---

## Layer 4: Deductive Verification (Selective)

**Effort: Very High**
**Guarantee: Mathematical proof of correctness**

Full deductive verification of the entire codebase is impractical. However, we can verify critical components.

### Candidate: Verus

Verus is a deductive verifier for Rust using Z3 as backend.

**Irony noted**: Using Z3 to verify Z4. However:
- We're verifying implementation, not the algorithm
- Z3 is trusted, battle-tested
- Once Z4 is mature, could self-host

### Target Components

1. **Union-Find** (z4-euf):
   ```rust
   // Verus specification
   #[requires(self.valid())]
   #[ensures(result == old(self).find_pure(x))]
   #[ensures(self.valid())]
   fn find(&mut self, x: Node) -> Node {
       // path compression implementation
   }
   ```

2. **Watched Literal Management** (z4-sat):
   - Verify two-watched invariant maintained
   - Verify no clause is missed during propagation

3. **Simplex Tableau** (z4-lra):
   - Verify pivoting maintains feasibility
   - Verify Bland's rule termination

### Alternative: Creusot

Creusot translates Rust to Why3, enabling use of multiple provers.

**Note**: CreuSAT is a fully verified SAT solver built with Creusot. We could:
- Study its verification approach
- Adapt its specifications
- Or even use CreuSAT as a reference implementation

### Decision

Start with Kani (Layer 2). Graduate critical components to Verus/Creusot as they stabilize. Don't let verification block development.

---

## Layer 5: Property-Based Testing

**Effort: Low-Medium**
**Guarantee: High confidence in implementation**

Property-based testing generates random inputs and checks invariants.

### Implementation

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn sat_result_valid(formula in arbitrary_cnf(100, 500)) {
        let mut solver = Solver::new();
        for clause in &formula {
            solver.add_clause(clause);
        }

        match solver.solve() {
            SolveResult::Sat(model) => {
                // Verify model satisfies all clauses
                for clause in &formula {
                    prop_assert!(clause.iter().any(|lit| model.value(*lit)));
                }
            }
            SolveResult::Unsat => {
                // Can't directly verify UNSAT without proof
                // But we can cross-check with other solvers (Layer 6)
            }
        }
    }

    #[test]
    fn incremental_consistency(ops in vec(solver_op(), 1..100)) {
        let mut solver = Solver::new();
        for op in ops {
            match op {
                Op::AddClause(c) => solver.add_clause(&c),
                Op::Push => solver.push(),
                Op::Pop => { let _ = solver.pop(); }
                Op::Solve => { let _ = solver.solve(); }
            }
        }
        // Should not panic
    }
}
```

### Coverage Targets

| Component | Property |
|-----------|----------|
| SAT solver | SAT models satisfy all clauses |
| SAT solver | UNSAT + SAT = consistent across incremental |
| EUF | Equivalence relation properties |
| LRA | Feasible models satisfy all constraints |
| LRA | Infeasible has valid Farkas certificate |

---

## Layer 6: Differential Testing

**Effort: Low**
**Guarantee: Agreement with reference solvers**

Compare Z4's answers against Z3 and CVC5.

### Implementation

```rust
fn differential_test(smt2: &str) {
    let z4_result = run_z4(smt2);
    let z3_result = run_z3(smt2);
    let cvc5_result = run_cvc5(smt2);

    // All must agree on sat/unsat (ignoring unknown/timeout)
    if z4_result.is_definite() && z3_result.is_definite() {
        assert_eq!(z4_result.is_sat(), z3_result.is_sat(),
            "Z4/Z3 disagreement on:\n{}", smt2);
    }
}
```

### Corpus

- SMT-LIB benchmarks (standard test suite)
- SMT-COMP benchmarks
- Fuzzer-generated formulas
- Regression tests from bugs

### CI Integration

```yaml
differential-tests:
  runs-on: ubuntu-latest
  steps:
    - run: cargo build --release
    - run: ./scripts/differential_test.sh benchmarks/
```

---

## Layer 7: Fuzzing

**Effort: Medium**
**Guarantee: Crash-freedom, robustness**

### Grammar-Based SMT-LIB Fuzzing

Generate syntactically valid but semantically adversarial SMT-LIB inputs.

```rust
// Use afl or cargo-fuzz
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(smt2) = std::str::from_utf8(data) {
        let _ = z4::parse_and_solve(smt2);
    }
});
```

### Mutation-Based Fuzzing

Take valid SMT-LIB files and mutate them:
- Bit flips
- Token substitution
- Formula combination

### Coverage-Guided Fuzzing

Use AFL++ or libfuzzer with coverage instrumentation:

```bash
cargo +nightly fuzz run smt_parser
```

### Targets

- Parser (crash resistance)
- Solver (no panics on malformed input)
- Proof generation (consistency)

---

## Implementation Roadmap

### Phase 1: Foundations (with SAT Core)

1. **Proof infrastructure**: Add DRAT logging to z4-sat from the start
2. **Property tests**: proptest for clause database, watched literals
3. **Differential tests**: Compare against MiniSat on DIMACS files
4. **TLA+ spec**: Model CDCL algorithm

### Phase 2: SMT Infrastructure

1. **Alethe proof logging**: Design proof data structures
2. **Kani harnesses**: For unsafe blocks in z4-core
3. **TLA+ spec**: Model DPLL(T) integration

### Phase 3: Theory Solvers

1. **Per-theory property tests**: Each theory gets proptest coverage
2. **Kani**: Critical invariants in each theory
3. **TLA+**: Simplex, congruence closure algorithms

### Phase 4: Maturity

1. **Verus/Creusot**: Graduate stable components to full verification
2. **Continuous fuzzing**: OSS-Fuzz integration
3. **Proof checking in CI**: Every UNSAT must have valid proof

---

## Tool Summary

| Tool | Purpose | Phase | Priority |
|------|---------|-------|----------|
| **DRAT/drat-trim** | SAT proof checking | 1 | CRITICAL |
| **Alethe/carcara** | SMT proof checking | 2-3 | CRITICAL |
| **Kani** | Bounded model checking | 1+ | HIGH |
| **TLA+/TLC** | Algorithm specification | 1+ | HIGH |
| **proptest** | Property-based testing | 1+ | HIGH |
| **cargo-fuzz** | Fuzzing | 1+ | MEDIUM |
| **Verus** | Deductive verification | 4+ | MEDIUM |
| **Creusot** | Deductive verification | 4+ | MEDIUM |
| **Miri** | UB detection | 1+ | MEDIUM |
| **ASAN/MSAN** | Sanitizers | 1+ | LOW |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Unsafe blocks with Kani proofs | 100% |
| Test coverage | > 80% |
| Differential test pass rate | 100% |
| Proof generation for UNSAT | 100% |
| Known soundness bugs | 0 |

---

## References

- CreuSAT: Verified SAT Solver (Ekici et al.) — https://github.com/sarsko/CreuSAT
- IsaSAT: Verified SAT Solver in Isabelle — Lammich, TUM
- DRAT-trim: Proof checker — https://github.com/marijnheule/drat-trim
- Alethe: Proof format — https://verit.loria.fr/documentation/alethe.html
- Kani: Rust model checker — https://model-checking.github.io/kani/
- Verus: Rust verifier — https://github.com/verus-lang/verus
- Creusot: Rust to Why3 — https://github.com/creusot-rs/creusot
