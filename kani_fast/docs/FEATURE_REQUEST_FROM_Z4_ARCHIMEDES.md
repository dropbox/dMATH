# Feature Request: Z4/Archimedes Integration

**From**: Z4 SMT Solver & Archimedes AI Math Research Platform
**To**: Kani Fast Team
**Date**: 2025-12-31
**Priority**: Medium

---

## Context

**Z4** is a high-performance SMT solver in Rust (github.com/dropbox/z4).

**Archimedes** is an AI-assisted mathematical research platform. We're building infrastructure to let AI explore mathematical conjectures with formal verification.

Kani Fast is critical for verifying Z4's correctness. We also see potential for Kani to verify properties relevant to complexity theory and algorithm correctness.

---

## Feature Requests

### P0: Z4 Verification Support

#### 1. Better SAT/SMT Algorithm Verification

**Request**: Harnesses optimized for verifying CDCL invariants.

Z4 needs to verify:
- Watch literal invariant (each clause has 2 watched literals)
- Propagation completeness (no missed unit clauses)
- Conflict analysis correctness (learned clause is implied)
- Backtrack consistency (assignment matches decision level)

```rust
// Example: verify propagation doesn't miss units
#[kani::proof]
#[kani::unwind(10)]
fn verify_propagation_complete() {
    let mut solver = Solver::new();
    // Add small formula
    let clauses: Vec<Vec<i32>> = kani::any();
    kani::assume(clauses.len() <= 5);
    kani::assume(clauses.iter().all(|c| c.len() <= 3));

    for clause in &clauses {
        solver.add_clause(clause);
    }

    solver.propagate();

    // Verify: no unit clause is unsatisfied
    for clause in &clauses {
        let satisfied = clause.iter().any(|&lit| solver.value(lit) == Some(true));
        let unit = clause.iter().filter(|&&lit| solver.value(lit).is_none()).count() == 1;
        kani::assert!(!unit || satisfied, "Missed unit propagation");
    }
}
```

**Why**: Z4 must be correct. Kani can catch subtle bugs in core algorithms.

#### 2. Incremental Verification

**Request**: Verify that incremental solving preserves correctness.

```rust
#[kani::proof]
fn verify_incremental_consistency() {
    let mut solver = Solver::new();

    // Add clauses incrementally
    solver.add_clause(&[1, 2]);
    let result1 = solver.solve();

    solver.add_clause(&[-1, 3]);
    let result2 = solver.solve();

    // If SAT, model must satisfy ALL clauses
    if result2 == SolveResult::Sat {
        let model = solver.model();
        kani::assert!(satisfies(&model, &[[1, 2], [-1, 3]]));
    }
}
```

---

### P1: Proof Certificate Verification

#### 3. DRAT Proof Correctness

**Request**: Verify DRAT proof generation produces valid proofs.

```rust
#[kani::proof]
fn verify_drat_generation() {
    let formula: Formula = kani::any();
    kani::assume(formula.num_vars() <= 5);
    kani::assume(formula.num_clauses() <= 8);

    let mut solver = Solver::new_with_proof();
    solver.add_formula(&formula);

    let (result, proof) = solver.solve_with_proof();

    if result == SolveResult::Unsat {
        // Verify DRAT proof is valid
        kani::assert!(verify_drat(&formula, &proof.unwrap()));
    }
}
```

**Why**: DRAT proofs are our certificate of correctness. Must be valid.

---

### P2: Algorithm Correctness for Archimedes

#### 4. Circuit Analysis Verification

**Request**: Support for verifying circuit complexity code.

Archimedes includes `z4-circuits` for Boolean circuit analysis. Need to verify:
- Circuit evaluation is correct
- Circuit equivalence checking is sound
- Minimum circuit synthesis is complete

```rust
// In z4-circuits
#[kani::proof]
fn verify_circuit_eval() {
    let circuit: Circuit = kani::any();
    kani::assume(circuit.num_inputs() <= 4);
    kani::assume(circuit.num_gates() <= 8);

    let inputs: Vec<bool> = kani::any();
    kani::assume(inputs.len() == circuit.num_inputs());

    // Evaluate two ways and compare
    let result1 = circuit.eval(&inputs);
    let result2 = circuit.eval_naive(&inputs);

    kani::assert!(result1 == result2);
}
```

#### 5. QBF Solver Verification

**Request**: Verify QBF solver correctness.

```rust
// In z4-qbf
#[kani::proof]
fn verify_qbf_solver() {
    let qbf: QBF = kani::any();
    kani::assume(qbf.num_vars() <= 4);

    let result = solve_qbf(&qbf);

    match result {
        QbfResult::True => {
            // Verify Skolem functions witness truth
            let skolem = result.certificate();
            kani::assert!(qbf.eval_with_skolem(&skolem));
        }
        QbfResult::False => {
            // Verify Herbrand functions witness falsity
            let herbrand = result.certificate();
            kani::assert!(!qbf.eval_with_herbrand(&herbrand));
        }
    }
}
```

---

### P3: Performance for Larger Verification

#### 6. Better Unwind Bounds

**Current limitation**: Kani unwind bounds limit formula sizes we can verify.

**Request**: Techniques to verify larger instances:
- Symbolic execution optimizations
- Partial verification (verify invariants, not full behavior)
- Compositional verification

#### 7. Parallel Verification

**Request**: Run multiple harnesses in parallel.

```bash
# Current: sequential
cargo kani --harness verify_propagation

# Requested: parallel
cargo kani --parallel --harnesses "verify_*"
```

**Why**: Z4 has many harnesses. Parallel execution saves time.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────┐
│                 Z4 Codebase                      │
├─────────────────────────────────────────────────┤
│  z4-sat    │  z4-qbf    │  z4-circuits          │
│  CDCL      │  QCDCL     │  Circuit Analysis     │
├────────────┴────────────┴───────────────────────┤
│                Kani Harnesses                    │
│  #[kani::proof] verify_*                        │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│                 Kani Fast                        │
│  - Model checking                               │
│  - Symbolic execution                           │
│  - Property verification                        │
└─────────────────────────────────────────────────┘
                      │
                      ▼
              Verification Report
              (all properties hold / counterexample)
```

---

## Z4 Components to Verify

| Component | Priority | Properties |
|-----------|----------|------------|
| z4-sat CDCL | P0 | Propagation, conflict analysis, backtrack |
| z4-sat DRAT | P0 | Proof generation correctness |
| z4-qbf | P1 | QCDCL correctness, certificate validity |
| z4-circuits | P1 | Evaluation, equivalence, synthesis |
| z4-proof-complexity | P2 | Proof system properties |

---

## Contact

**Repo**: github.com/dropbox/z4
**Docs**: z4/docs/ARCHIMEDES_ROADMAP.md

Z4 aims to be the most verified SAT/SMT solver. Kani is essential to that goal.
