# Phase 1 Execution Roadmap: Verified CDCL SAT Solver

**Status**: ✅ PHASE 1 COMPLETE (Iteration 144)
**Performance**: 0.90x vs CaDiCaL on uf250 (Z4 is 10% faster, wins 80/100 instances)
**Next Phase**: Phase 2 - SMT Infrastructure (z4-core, z4-frontend, z4-dpll)
**Target**: `z4-sat` crate with Kani verification from day 1

---

## Philosophy: Verification First

Z4's differentiator is **both fast AND verified**. We don't add verification after the fact—we build it in from day 1.

```
Traditional approach:     Z4 approach:
1. Write code             1. Write TLA+ spec
2. Write tests            2. Write Kani harness
3. Add verification       3. Write code
4. Find it doesn't fit    4. Tests verify both
```

**IsaSAT proves this works**: A verified SAT solver can be competitive. The 60% gap comes from missing techniques (inprocessing), not verification overhead.

---

## Execution Order

### Week 1: Verification Infrastructure + Core Types

#### Day 1-2: Project Setup

```bash
# Add dependencies
cargo add --dev kani-verifier@0.66 proptest@1.5
cargo add --dev --path ../benchmarks criterion

# Create test infrastructure
mkdir -p benchmarks/dimacs
curl -O https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz
tar -xzf uf20-91.tar.gz -C benchmarks/dimacs/
```

**Files to create**:
- `z4-sat/Cargo.toml` - Add kani, proptest, criterion deps
- `z4-sat/src/dimacs.rs` - DIMACS parser for testing
- `z4-sat/tests/integration.rs` - Test harness

#### Day 3: TLA+ Specification

**File**: `specs/cdcl.tla`

Write the CDCL algorithm in TLA+ BEFORE implementing:

```tla
---------------------------- MODULE CDCL ----------------------------
VARIABLES
    assignment,      \* Variable -> {TRUE, FALSE, UNDEF}
    trail,           \* Sequence of (literal, reason) pairs
    level,           \* Variable -> Nat (decision level)
    clauses,         \* Set of clauses
    state            \* {PROPAGATING, DECIDING, CONFLICTING, SAT, UNSAT}

TypeInvariant ==
    /\ assignment \in [Variables -> {TRUE, FALSE, UNDEF}]
    /\ level \in [Variables -> Nat]
    /\ state \in {"PROPAGATING", "DECIDING", "CONFLICTING", "SAT", "UNSAT"}

\* If UNSAT, there exists a resolution proof from original clauses to empty clause
UnsatCorrect ==
    state = "UNSAT" => HasResolutionProof(clauses, EmptyClause)

\* If SAT, assignment satisfies all clauses
SatCorrect ==
    state = "SAT" => \A c \in clauses : Satisfies(assignment, c)

\* Watched literal invariant
WatchedInvariant ==
    \A c \in clauses :
        c.len >= 2 =>
        \/ assignment[c[0]] # FALSE
        \/ assignment[c[1]] # FALSE
        \/ \E i \in 2..c.len : assignment[c[i]] = UNDEF

Soundness == SatCorrect /\ UnsatCorrect
=====================================================================
```

Run TLC model checker:
```bash
cd specs
tlc cdcl.tla -config cdcl.cfg
```

#### Day 4-5: Kani Harnesses for Core Types

**File**: `z4-sat/src/literal.rs` - Add Kani proofs

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn literal_negation_involutive() {
        let lit: Literal = kani::any();
        kani::assume(lit.0 < 1_000_000); // Bound for tractability
        assert_eq!(lit.negated().negated(), lit);
    }

    #[kani::proof]
    fn literal_variable_roundtrip() {
        let var: Variable = kani::any();
        kani::assume(var.0 < 500_000);

        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        assert_eq!(pos.variable(), var);
        assert_eq!(neg.variable(), var);
        assert!(pos.is_positive());
        assert!(!neg.is_positive());
    }

    #[kani::proof]
    fn literal_encoding_unique() {
        let var1: Variable = kani::any();
        let var2: Variable = kani::any();
        kani::assume(var1.0 < 500_000 && var2.0 < 500_000);

        let pos1 = Literal::positive(var1);
        let pos2 = Literal::positive(var2);

        // Same encoding implies same variable
        if pos1.0 == pos2.0 {
            assert_eq!(var1, var2);
        }
    }
}
```

Run Kani:
```bash
cargo kani --tests -p z4-sat
```

### Week 2: Core CDCL with Verification

#### Day 6-7: Watched Literals with Kani

**Key invariant to verify**: After propagation, for every clause of length >= 2, at least one of the first two literals is not false.

```rust
// z4-sat/src/watched.rs

#[cfg(kani)]
mod verification {
    use super::*;

    /// After adding a clause, it has exactly 2 watches
    #[kani::proof]
    #[kani::unwind(10)]
    fn watch_count_after_add() {
        let mut watches = WatchedLists::new(10);
        let clause = Clause::new(vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
            Literal::positive(Variable(2)),
        ], false);

        watches.add_clause(ClauseRef(0), &clause);

        // Count watches for this clause
        let mut count = 0;
        for lit_idx in 0..20 {
            for w in watches.watches[lit_idx].iter() {
                if w.clause == ClauseRef(0) {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 2);
    }
}
```

#### Day 8-9: Unit Propagation with Verification

Implement `propagate()` with property tests:

```rust
// z4-sat/src/solver.rs

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Propagation is sound: if it says conflict, clause is actually violated
        #[test]
        fn propagation_soundness(
            clauses in prop::collection::vec(arbitrary_clause(5, 10), 1..20),
            decisions in prop::collection::vec(0u32..10, 0..5)
        ) {
            let mut solver = Solver::new(10);
            for c in &clauses {
                solver.add_clause(c.clone());
            }

            // Make some decisions
            for &var in &decisions {
                if solver.value(Variable(var)).is_none() {
                    solver.decide(Literal::positive(Variable(var)));
                }
            }

            // Propagate
            if let Some(conflict_ref) = solver.propagate() {
                // Verify conflict: all literals in clause must be false
                let conflict = &solver.clauses[conflict_ref.0 as usize];
                for lit in &conflict.literals {
                    let val = solver.value(lit.variable());
                    let expected = if lit.is_positive() { Some(false) } else { Some(true) };
                    prop_assert_eq!(val, expected, "Conflict clause not fully falsified");
                }
            }
        }
    }
}
```

#### Day 10: 1UIP Conflict Analysis with Verification

```rust
// z4-sat/src/conflict.rs

#[cfg(kani)]
mod verification {
    use super::*;

    /// Learned clause is an asserting clause (exactly one literal at highest level)
    #[kani::proof]
    #[kani::unwind(20)]
    fn learned_clause_is_asserting() {
        // Symbolic conflict analysis scenario
        let mut analyzer = ConflictAnalyzer::new(10);
        // ... setup symbolic clauses and trail ...

        let result = analyzer.analyze(/* ... */);

        // Count literals at the backtrack level
        let at_bt_level = result.learned_clause.iter()
            .filter(|&lit| /* level of lit */ true)
            .count();

        // Asserting clause property: exactly one literal from conflict level
        kani::assert!(at_bt_level == 1, "Learned clause must be asserting");
    }
}
```

### Week 3: Complete CDCL Loop

#### Day 11-12: Main Solve Loop

Implement the full CDCL loop following the TLA+ spec:

```rust
pub fn solve(&mut self) -> SolveResult {
    self.initialize_watches();

    loop {
        // State: PROPAGATING
        if let Some(conflict) = self.propagate() {
            // State: CONFLICTING
            if self.decision_level == 0 {
                // State: UNSAT
                return SolveResult::Unsat;
            }

            let result = self.conflict.analyze(/* ... */);
            let learned_ref = self.add_learned_clause(result.learned_clause, result.lbd);

            self.backtrack(result.backtrack_level);

            // Assert UIP
            let uip = self.clauses[learned_ref.0 as usize].literals[0];
            self.enqueue(uip, Some(learned_ref));

            // Bump VSIDS for conflict variables
            self.vsids.decay();
        } else {
            // State: DECIDING or SAT
            if let Some(var) = self.vsids.pick_branching_variable(&self.assignment) {
                let lit = self.pick_phase(var);
                self.decide(lit);
            } else {
                // State: SAT
                return SolveResult::Sat(self.get_model());
            }
        }
    }
}
```

#### Day 13-14: DRAT Proof Infrastructure

**Critical for verification!** Every learned clause must be logged.

```rust
// z4-sat/src/proof.rs

pub struct DratWriter<W: Write> {
    writer: W,
    binary: bool,
}

impl<W: Write> DratWriter<W> {
    /// Log addition of learned clause
    pub fn add(&mut self, clause: &[Literal]) -> io::Result<()> {
        if self.binary {
            self.write_binary_clause(clause, false)
        } else {
            self.write_text_clause(clause, false)
        }
    }

    /// Log deletion of clause
    pub fn delete(&mut self, clause: &[Literal]) -> io::Result<()> {
        if self.binary {
            self.write_binary_clause(clause, true)
        } else {
            write!(self.writer, "d ")?;
            self.write_text_clause(clause, false)
        }
    }
}

// Integration with solver
impl Solver {
    fn add_learned_clause_with_proof(&mut self, lits: Vec<Literal>, lbd: u32) -> ClauseRef {
        if let Some(ref mut proof) = self.proof_writer {
            proof.add(&lits).unwrap();
        }
        self.add_learned_clause(lits, lbd)
    }
}
```

### Week 4: Testing and Benchmarking

#### Day 15-16: Differential Testing

```rust
// z4-sat/tests/differential.rs

fn compare_with_minisat(cnf: &str) -> bool {
    let z4_result = run_z4(cnf);
    let minisat_result = run_minisat(cnf);

    match (z4_result, minisat_result) {
        (SolveResult::Sat(_), true) => true,
        (SolveResult::Unsat, false) => true,
        (z4, minisat) => {
            eprintln!("Disagreement! Z4: {:?}, MiniSat: {}", z4, minisat);
            false
        }
    }
}

#[test]
fn test_all_benchmarks() {
    for entry in fs::read_dir("benchmarks/dimacs").unwrap() {
        let path = entry.unwrap().path();
        if path.extension() == Some("cnf".as_ref()) {
            let cnf = fs::read_to_string(&path).unwrap();
            assert!(compare_with_minisat(&cnf), "Failed on {}", path.display());
        }
    }
}
```

#### Day 17-18: DRAT Proof Verification

```bash
# Install drat-trim
git clone https://github.com/marijnheule/drat-trim
cd drat-trim && make

# Test proof verification
./z4 benchmarks/dimacs/uf20-01.cnf --proof /tmp/proof.drat
./drat-trim benchmarks/dimacs/uf20-01.cnf /tmp/proof.drat
```

Add CI check:
```yaml
# .github/workflows/verify.yml
verify-proofs:
  runs-on: ubuntu-latest
  steps:
    - run: cargo build --release
    - run: |
        for f in benchmarks/dimacs/*.cnf; do
          ./target/release/z4 "$f" --proof /tmp/proof.drat
          if [ $? -eq 20 ]; then  # UNSAT
            drat-trim "$f" /tmp/proof.drat || exit 1
          fi
        done
```

#### Day 19-20: Performance Benchmarks

```rust
// benches/sat_bench.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_uf20(c: &mut Criterion) {
    let cnf = include_str!("../benchmarks/dimacs/uf20-01.cnf");

    c.bench_function("uf20-01", |b| {
        b.iter(|| {
            let mut solver = Solver::from_dimacs(black_box(cnf));
            solver.solve()
        })
    });
}

criterion_group!(benches, bench_uf20);
criterion_main!(benches);
```

---

## Verification Checklist

Before each PR/commit, verify:

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Kani proofs | `cargo kani -p z4-sat` | All proofs pass |
| Property tests | `cargo test -p z4-sat` | All tests pass |
| Clippy | `cargo clippy -p z4-sat -- -W clippy::all` | No warnings |
| Differential | `cargo test differential` | Agreement with MiniSat |
| DRAT proofs | `./scripts/verify_proofs.sh` | All proofs valid |

---

## Commit Sequence

Expected commits for Phase 1A:

```
# 2: Add Kani and proptest infrastructure to z4-sat
# 3: TLA+ spec for CDCL algorithm (specs/cdcl.tla)
# 4: Kani proofs for Literal and Variable types
# 5: Implement watched literal scheme with verification
# 6: Unit propagation with property tests
# 7: 1UIP conflict analysis with Kani proof
# 8: Complete CDCL solve loop
# 9: DRAT proof generation
# 10: Differential testing against MiniSat
# 11: DIMACS parser and benchmark tests
# 12: Performance benchmarks with criterion
```

---

## Success Criteria for Phase 1A

| Metric | Target | Verification |
|--------|--------|--------------|
| Test pass rate | 100% | `cargo test -p z4-sat` |
| Kani proof coverage | All unsafe + critical invariants | `cargo kani -p z4-sat` |
| DRAT validity | 100% of UNSAT results | `drat-trim` on all |
| MiniSat agreement | 100% on uf20 benchmarks | Differential tests |
| Benchmark coverage | 50+ DIMACS files | `ls benchmarks/dimacs | wc -l` |

---

## References

| Document | Purpose |
|----------|---------|
| `papers/minisat-sat04.pdf` | CDCL algorithm |
| `papers/chaff-dac01.pdf` | VSIDS heuristic |
| `docs/FORMAL_VERIFICATION_STRATEGY.md` | Verification layers |
| `docs/KANI_FAST_REQUIREMENTS.md` | Performance targets |
| `research/BENCHMARKS_AND_TECHNIQUES.md` | Techniques checklist |

---

## WORKER: Start Here

1. Read this document fully
2. Read `papers/minisat-sat04.pdf` (10 pages, explains everything)
3. Run `cargo kani --version` to verify Kani is available
4. Start with Day 1-2 tasks: project setup and DIMACS parser
5. Your first commit should be `# 2: Add Kani and proptest infrastructure to z4-sat`

**Remember**: Verification first. Write the Kani harness, then the code.
