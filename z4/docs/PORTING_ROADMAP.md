# Z4 Feature Porting Roadmap

**Date**: 2026-01-02
**Goal**: Beat Z3 AND CVC5 on all major SMT-COMP categories
**Companion Document**: `docs/SOLVER_ANALYSIS_REPORT.md`

---

## Overview

This roadmap incorporates techniques from CaDiCaL, CryptoMiniSat, and CVC5 into Z4.

**Current Z4 Strengths**:
- 1.5x faster than Z3 on QF_LIA
- 1.79x faster than Z3 on QF_UF
- 1.37x faster than Z3 on QF_LRA
- 1.19x faster than Z3 on QF_BV
- Very fast startup time
- Clean Rust codebase

**Current Z4 Weaknesses**:
- No string theory (CVC5 dominates)
- No quantifiers (CVC5 dominates)
- Limited CHC solving (Z3 Spacer is better)
- Basic theory combination
- No XOR handling (CryptoMiniSat specialty)
- **No Array theory in CHC (blocks Kani Fast integration)**

---

## Phase 0: Kani Fast Integration (CRITICAL - 1 week)

**Goal**: Enable Z4 as primary CHC solver for Kani Fast Rust verification
**Priority**: IMMEDIATE - blocks production use

### Current Status

| Benchmark | Z4 Result | Z3 Result | Status |
|-----------|-----------|-----------|--------|
| B1 Two Counter | sat | sat | PASS |
| B2 Bounded Loop | sat | sat | PASS |
| B3 Nested Loop | **unknown** | sat | **FAIL** |
| B4 Conditional | sat | sat | PASS |
| B5 Array Bounds | **parse error** | sat | **FAIL** |
| B6 Overflow Check | **timeout** | sat | **FAIL** |
| B7 Fibonacci | sat | sat | PASS |
| B8 Mutex Protocol | **unknown** | sat | **FAIL** |

**Pass Rate**: 4/8 (50%) - Target: 7/8 (87.5%)

### 0.1 Array Theory for CHC (CRITICAL)

**Current**: Parse error on `(Array Int Int)` sort
**Required**: Full SMT-LIB Array theory for CHC predicates

**Files to Modify**:
- `crates/z4-chc/src/parser.rs` - Add `(Array K V)` parametric sort
- `crates/z4-chc/src/problem.rs` - Add `Select`/`Store` to Term enum
- `crates/z4-chc/src/pdr.rs` - Project arrays during MBP

```rust
// crates/z4-chc/src/problem.rs
pub enum Sort {
    Int,
    Bool,
    Array { key: Box<Sort>, value: Box<Sort> },  // ADD
}

pub enum Term {
    // ... existing ...
    Select(Box<Term>, Box<Term>),                 // ADD: arr[idx]
    Store(Box<Term>, Box<Term>, Box<Term>),       // ADD: arr[idx := val]
}
```

**Tasks**:
- [ ] Parse `(Array K V)` parametric sort syntax
- [ ] Parse `select` and `store` operations
- [ ] Treat arrays as uninterpreted in MBP (project away)
- [ ] Test: B5 Array Bounds returns `sat`

### 0.2 Nested Loop Invariants (HIGH)

**Current**: B3 returns `unknown` (Z3: `sat`)
**Issue**: PDR cannot synthesize disjunctive invariants

**Required invariant** (from Z3):
```smt2
(and (or (not (>= pc 1)) (not (>= i 10)))
     (not (<= j (- 1)))
     (not (<= i (- 1)))
     (not (>= j 11))
     (not (>= i 11)))
```

**Potential Fixes**:
1. Allow disjunctive lemmas in PDR
2. Implement IC3-style forward propagation
3. Predicate splitting for pc-based control flow

**Tasks**:
- [ ] Debug B3 with `--verbose` to understand failure mode
- [ ] Implement disjunctive lemma learning
- [ ] Test: B3 Nested Loop returns `sat`

### 0.3 Relational Invariants (HIGH)

**Current**: B6 timeout, B8 unknown
**Issue**: PDR cannot find linear combinations like `x + y = 100`

**Required invariant** for B6:
```smt2
(= (+ x y) 100)
```

**Potential Fixes**:
1. Farkas combination lemma synthesis
2. Interval abstract interpretation
3. Equality detection via sampling

**Tasks**:
- [ ] Debug B6 with `--verbose`
- [ ] Improve Farkas combination (currently disabled)
- [ ] Test: B6 Overflow returns `sat` within 10s

### 0.4 Mutex Protocol (HIGH)

**Current**: B8 returns `unknown`
**Issue**: PDR cannot find invariant with lock correlation

**Required invariant** (from Z3):
```smt2
(and (or (not (>= pc1 2)) (= lock 1))
     (or (not (>= pc1 2)) (not (>= pc2 2)))
     (or (not (<= lock 0)) (not (>= pc2 2))))
```

**Tasks**:
- [ ] Debug B8 with `--verbose`
- [ ] Improve implication-based invariant discovery
- [ ] Test: B8 Mutex returns `sat`

### Success Criteria

- [ ] B5 Array Bounds: parses and returns result
- [ ] B3 Nested Loop: returns `sat`
- [ ] B6 Overflow Check: returns `sat` within 10s
- [ ] B8 Mutex Protocol: returns `sat`
- [ ] **7/8 Kani Fast benchmarks pass**
- [ ] No regression on CHC-COMP (maintain 6/20)

---

## Phase 1: SAT Solver Hardening (2 weeks)

**Goal**: Match CaDiCaL on all SAT Competition benchmarks

### 1.1 VMTF Decision Heuristic
**Source**: CaDiCaL `queue.hpp`, `decide.cpp`
**Effort**: Medium
**Files to Modify**: `z4-sat/src/vsids.rs` (add VMTF module)

```rust
// New: crates/z4-sat/src/vmtf.rs
pub struct VMTF {
    links: Vec<Link>,      // prev/next pointers
    queue_head: Variable,  // Most recently bumped
    queue_tail: Variable,  // Oldest
    timestamp: u64,
}
```

**Tasks**:
- [ ] Implement doubly-linked list for VMTF
- [ ] Add bump operation (move to front)
- [ ] Integrate with solver decision loop
- [ ] Add mode switching between VSIDS and VMTF

### 1.2 On-the-Fly Strengthening (OTFS)
**Source**: CaDiCaL `analyze.cpp:281-320`
**Effort**: Low
**Files to Modify**: `z4-sat/src/conflict.rs`

During conflict analysis, if resolvent is smaller than antecedent, strengthen the original clause:
```rust
// In analyze_conflict():
if resolvent.len() < antecedent.len() && antecedent.is_redundant() {
    strengthen_clause(antecedent, resolvent);
}
```

**Tasks**:
- [ ] Add OTFS check in conflict analysis loop
- [ ] Implement clause strengthening
- [ ] Track statistics

### 1.3 Reason-Side Bumping
**Source**: CaDiCaL `analyze.cpp:632-680` (from MapleSAT)
**Effort**: Low
**Files to Modify**: `z4-sat/src/conflict.rs`, `z4-sat/src/vsids.rs`

After learning a clause, bump variables in the reasons of learned clause literals:
```rust
for lit in &learned_clause {
    if let Some(reason) = var_reason[lit.var()] {
        for reason_lit in reason.lits() {
            vsids.bump(reason_lit.var(), small_factor);
        }
    }
}
```

**Tasks**:
- [ ] Add reason-side bumping after clause learning
- [ ] Tune bump factor (CaDiCaL uses 0.5)
- [ ] Measure impact on benchmarks

### 1.4 Mode Switching (Stable/Focused)
**Source**: CaDiCaL `decide.cpp`, `restart.cpp`
**Effort**: Medium
**Files to Modify**: `z4-sat/src/solver.rs`

Alternate between:
- **Focused mode**: VMTF + Glucose restarts (aggressive)
- **Stable mode**: VSIDS + Luby restarts (conservative)

```rust
enum SolverMode {
    Focused { conflicts_until_switch: u64 },
    Stable { conflicts_until_switch: u64 },
}
```

**Tasks**:
- [ ] Add mode tracking to solver
- [ ] Implement geometric mode duration increase
- [ ] Switch heuristics based on mode
- [ ] Tune parameters

### 1.5 Target Phase Saving
**Source**: CaDiCaL `phases.hpp`, `decide.cpp`
**Effort**: Low
**Files to Modify**: `z4-sat/src/solver.rs`

Track the best assignment found (deepest level reached):
```rust
struct PhaseData {
    saved: Vec<bool>,    // Last assignment
    target: Vec<bool>,   // Best assignment
    best: Vec<bool>,     // Overall best
}
```

**Tasks**:
- [ ] Track best assignment per rephase interval
- [ ] Use target phase during stable mode
- [ ] Implement rephasing strategy

### 1.6 Internal LRAT Generation
**Source**: CaDiCaL `lrattracer.cpp`, `analyze.cpp`
**Effort**: Medium
**Files to Modify**: `z4-sat/src/proof.rs`, `z4-sat/src/conflict.rs`

Generate LRAT directly without drat-trim post-processing:
```rust
// During conflict analysis, collect clause IDs
let mut chain: Vec<ClauseId> = vec![];
for antecedent in resolution_chain {
    chain.push(antecedent.id);
}
// Write: <new_id> <literals> 0 <chain> 0
```

**Tasks**:
- [ ] Assign IDs to all clauses
- [ ] Track resolution chain during analysis
- [ ] Output LRAT format
- [ ] Verify with lrat-check

---

## Phase 2: XOR Handling (2 weeks)

**Goal**: Match CryptoMiniSat on XOR-heavy benchmarks (crypto, parity)

### 2.1 XOR Recovery from CNF
**Source**: CryptoMiniSat `xorfinder.cpp`
**Effort**: High
**Files to Create**: `z4-sat/src/xor.rs`

Detect XOR constraints encoded as CNF:
```rust
// x1 XOR x2 XOR x3 = true encoded as:
// (x1 OR x2 OR x3)
// (x1 OR ~x2 OR ~x3)
// (~x1 OR x2 OR ~x3)
// (~x1 OR ~x2 OR x3)

pub struct XorFinder {
    clause_abstraction: HashMap<ClauseId, u64>,
    possible_xors: Vec<PossibleXor>,
}
```

**Tasks**:
- [ ] Implement clause abstraction (bit signature)
- [ ] Detect XOR patterns from clause sets
- [ ] Extract XOR constraints
- [ ] Remove subsumed CNF clauses

### 2.2 Gaussian Elimination
**Source**: CryptoMiniSat `gaussian.cpp`, `packedmatrix.cpp`
**Effort**: High
**Files to Create**: `z4-sat/src/gaussian.rs`

Packed bit matrix for GF(2) operations:
```rust
pub struct PackedMatrix {
    rows: Vec<PackedRow>,
    var_to_col: Vec<Option<usize>>,
    col_to_var: Vec<Variable>,
}

impl PackedMatrix {
    fn eliminate(&mut self, pivot_row: usize, target_row: usize);
    fn propagate(&mut self, assigned_var: Variable) -> GaussResult;
}
```

**Tasks**:
- [ ] Implement packed bit row (64-bit words)
- [ ] Implement Gauss-Jordan elimination
- [ ] Add incremental propagation on assignment
- [ ] Integrate with watch system

### 2.3 XOR Clause Watching
**Source**: CryptoMiniSat `gaussian.cpp:find_truths`
**Effort**: Medium
**Files to Modify**: `z4-sat/src/watched.rs`

Watch XOR clauses like regular clauses:
```rust
enum WatchType {
    Binary { other: Literal },
    Long { clause: ClauseRef },
    Xor { matrix: MatrixId, row: RowId },
}
```

**Tasks**:
- [ ] Add XOR watch type
- [ ] Propagate when watched variable assigned
- [ ] Generate conflict clauses from XOR conflicts

---

## Phase 3: SMT Theory Improvements (4 weeks)

**Goal**: Proper DPLL(T) integration, beat Z3 on all QF_* categories

### 3.1 CDCL(T) Propagator Interface
**Source**: CVC5 `cdclt_propagator.h`
**Effort**: High
**Files to Modify**: `z4-dpll/src/lib.rs`, `z4-sat/src/solver.rs`

Clean interface between SAT and theories:
```rust
pub trait TheoryPropagator {
    /// Called when SAT assigns a literal
    fn notify_assignment(&mut self, lit: Literal, level: u32);

    /// Called after BCP - return theory propagations
    fn propagate(&mut self) -> Vec<TheoryPropagation>;

    /// Called when complete model found
    fn check_model(&mut self, model: &Model) -> TheoryResult;

    /// Get theory-guided decision (optional)
    fn decide(&mut self) -> Option<Literal>;

    /// Explain a theory propagation as clause
    fn explain(&mut self, lit: Literal) -> Vec<Literal>;
}
```

**Tasks**:
- [ ] Define clean trait interface
- [ ] Modify SAT solver to call propagator
- [ ] Implement lazy clause generation
- [ ] Support theory decisions

### 3.2 Theory Combination (Care Graph)
**Source**: CVC5 `combination_care_graph.cpp`
**Effort**: High
**Files to Modify**: `z4-dpll/src/lib.rs`

For theories sharing terms:
```rust
pub struct CareGraph {
    pairs: Vec<(Term, Term, TheoryId)>,
}

impl TheoryEngine {
    fn combine_theories(&mut self) {
        let care_graph = self.compute_care_graph();
        for (t1, t2, theory) in care_graph.pairs {
            // Add splitting lemma: (= t1 t2) OR (not (= t1 t2))
            self.add_lemma(splitting_lemma(t1, t2));
        }
    }
}
```

**Tasks**:
- [ ] Implement care graph computation
- [ ] Add splitting lemma generation
- [ ] Iterate to fixed point
- [ ] Handle shared variables correctly

### 3.3 Array Theory Axioms
**Source**: CVC5 `theory/arrays/theory_arrays.cpp`
**Effort**: Medium
**Files to Modify**: `z4-theories/arrays/src/lib.rs`

Implement array axioms with lazy instantiation:
```rust
// Read-over-write (row1): store(a,i,v)[i] = v
// Read-over-write (row): i != j => store(a,i,v)[j] = a[j]
// Extensionality: a != b => exists k. a[k] != b[k]

impl ArrayTheory {
    fn check_row(&mut self, store: Term, index: Term);
    fn check_extensionality(&mut self, arr1: Term, arr2: Term);
}
```

**Tasks**:
- [ ] Implement row axiom instantiation
- [ ] Implement extensionality on demand
- [ ] Track read/write indices per array
- [ ] Optimize with weak equivalence

---

## Phase 4: String Theory (6 weeks)

**Goal**: Competitive with CVC5 on QF_SLIA

### 4.1 Basic String Constraints
**Source**: CVC5 `theory/strings/base_solver.cpp`
**Effort**: High
**Files to Modify**: `z4-theories/strings/src/lib.rs`

```rust
pub struct StringTheory {
    // String variables and their lengths
    str_vars: Vec<StringVar>,
    // Equality graph
    eq_graph: UnionFind,
    // Length constraints
    arith_interface: ArithInterface,
}
```

**Tasks**:
- [ ] Define string term representation
- [ ] Implement basic equality handling
- [ ] Connect length constraints to arithmetic
- [ ] Handle constant strings

### 4.2 Word Equation Solver
**Source**: CVC5 `theory/strings/core_solver.cpp` (98K LOC)
**Effort**: Very High
**Files to Create**: `z4-theories/strings/src/word_eq.rs`

Word equation solving (x.y = z.w):
```rust
pub struct WordEquationSolver {
    equations: Vec<WordEquation>,
    normal_forms: HashMap<Term, NormalForm>,
}

impl WordEquationSolver {
    fn normalize(&mut self, eq: &WordEquation) -> Result<Vec<WordEquation>>;
    fn check(&mut self) -> TheoryResult;
}
```

**Tasks**:
- [ ] Implement normal form computation
- [ ] Implement splitting heuristics
- [ ] Handle length-based reasoning
- [ ] Integrate with arithmetic theory

### 4.3 Extended String Functions
**Source**: CVC5 `theory/strings/extf_solver.cpp`
**Effort**: High
**Files to Create**: `z4-theories/strings/src/extf.rs`

Support: contains, indexOf, replace, substr
```rust
enum ExtendedFunc {
    Contains(Term, Term),
    IndexOf(Term, Term, Term),
    Replace(Term, Term, Term),
    Substr(Term, Term, Term),
}
```

**Tasks**:
- [ ] Implement reduction to word equations
- [ ] Handle special cases (constants)
- [ ] Implement eager evaluation when possible

### 4.4 Regular Expression Membership
**Source**: CVC5 `theory/strings/regexp_solver.cpp`
**Effort**: High
**Files to Create**: `z4-theories/strings/src/regexp.rs`

```rust
pub struct RegexpSolver {
    memberships: Vec<(Term, Regex)>,
    derivatives: HashMap<(Regex, char), Regex>,
}
```

**Tasks**:
- [ ] Implement regex to automaton conversion
- [ ] Implement derivative-based membership
- [ ] Handle regex intersection
- [ ] Integrate with word equations

---

## Phase 5: Quantifier Handling (4 weeks)

**Goal**: Support quantified formulas for completeness

### 5.1 E-Matching Infrastructure
**Source**: CVC5 `theory/quantifiers/ematching/`
**Effort**: High
**Files to Create**: `z4-theories/quantifiers/src/ematching.rs`

```rust
pub struct EMatcher {
    triggers: HashMap<QuantId, Vec<Trigger>>,
    instances: HashSet<Instantiation>,
}

impl EMatcher {
    fn find_matches(&self, trigger: &Trigger, egraph: &EGraph) -> Vec<Substitution>;
    fn instantiate(&mut self, quant: QuantId, subst: &Substitution);
}
```

**Tasks**:
- [ ] Implement trigger selection
- [ ] Implement pattern matching against E-graph
- [ ] Instantiation loop with fairness
- [ ] Relevance filtering

### 5.2 Model-Based Quantifier Instantiation (MBQI)
**Source**: CVC5 `inst_strategy_mbqi.cpp`
**Effort**: High
**Files to Create**: `z4-theories/quantifiers/src/mbqi.rs`

```rust
pub struct MBQI {
    model: Model,
    candidates: Vec<QuantCandidate>,
}

impl MBQI {
    fn check_quantifier(&mut self, quant: &Quantifier) -> MBQIResult;
    fn repair_model(&mut self, counterexample: &Substitution);
}
```

**Tasks**:
- [ ] Build models for quantifier-free part
- [ ] Check quantifiers against model
- [ ] Find counterexample instantiations
- [ ] Iterate to convergence

---

## Phase 6: Test Suite Extension (Ongoing)

**Goal**: Comprehensive regression testing against all competitors

### 6.1 SAT Test Suite
**Source**: SAT Competition, CaDiCaL, CryptoMiniSat

| Category | Count | Source |
|----------|-------|--------|
| SAT-COMP 2023 | 400 | `benchmarks/sat/satcomp2023/` |
| CaDiCaL tests | 88 | `reference/cadical/test/cnf/` |
| XOR benchmarks | 50 | Create from CryptoMiniSat |
| Random 3-SAT | 1000 | Generate uf100, uf250, uf500 |

**Tasks**:
- [ ] Download SAT-COMP benchmarks
- [ ] Create XOR test generator
- [ ] Add CaDiCaL trace tests
- [ ] Set up CI comparison vs CaDiCaL

### 6.2 SMT Test Suite
**Source**: SMT-COMP, SMT-LIB, CVC5

| Category | Count | Source |
|----------|-------|--------|
| SMT-LIB QF_LIA | 2000+ | `benchmarks/smt/QF_LIA/` |
| SMT-LIB QF_LRA | 1000+ | `benchmarks/smt/QF_LRA/` |
| SMT-LIB QF_BV | 5000+ | `benchmarks/smt/QF_BV/` |
| SMT-LIB QF_UF | 500+ | `benchmarks/smt/QF_UF/` |
| SMT-LIB QF_SLIA | 2000+ | `benchmarks/smt/QF_SLIA/` |
| CVC5 regress | 3994 | `reference/cvc5/test/regress/` |

**Tasks**:
- [ ] Download SMT-LIB benchmark sets
- [ ] Import CVC5 regression tests
- [ ] Set up CI comparison vs Z3, CVC5
- [ ] Track solve rates and times

### 6.3 CHC Test Suite
**Source**: CHC-COMP, Spacer benchmarks

| Category | Count | Source |
|----------|-------|--------|
| CHC-COMP 2023 | 500+ | `benchmarks/chc/chccomp2023/` |
| Spacer tests | 200+ | From Z3 repository |
| Extra-small-lia | 55 | Already in repo |

**Tasks**:
- [ ] Download CHC-COMP benchmarks
- [ ] Port Spacer test cases
- [ ] Track solve rate vs Z3, Golem
- [ ] Profile timeout distribution

### 6.4 Proof Verification Suite
**Source**: drat-trim, lrat-check

**Tasks**:
- [ ] Verify all UNSAT results with drat-trim
- [ ] Create LRAT verification pipeline
- [ ] Add proof checking to CI
- [ ] Track proof size metrics

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 0** | 1 week | **Array theory CHC, nested loops, relational invariants (Kani Fast)** |
| **Phase 1** | 2 weeks | VMTF, OTFS, mode switching, internal LRAT |
| **Phase 2** | 2 weeks | XOR recovery, Gaussian elimination |
| **Phase 3** | 4 weeks | CDCL(T) interface, care graph, array axioms |
| **Phase 4** | 6 weeks | String theory (word equations, regex) |
| **Phase 5** | 4 weeks | E-matching, MBQI |
| **Phase 6** | Ongoing | Test suite expansion, CI |

**Total**: ~19 weeks to feature parity with CVC5 core (Phase 0 is CRITICAL path)

---

## Success Metrics

### SAT Performance
- [ ] Match CaDiCaL on SAT-COMP 2023 (solve rate within 5%)
- [ ] Beat CaDiCaL on XOR benchmarks (with Gaussian)
- [ ] Verify all UNSAT with LRAT

### SMT Performance
- [ ] Beat Z3 by >20% on all QF_* categories
- [ ] Match CVC5 on QF_SLIA (string benchmarks)
- [ ] 100% agreement with Z3/CVC5 on benchmark results

### CHC Performance
- [ ] Solve 15/20 on CHC-COMP small (vs Z3's 5/20)
- [ ] Match Golem solve rate on LIA benchmarks

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| String theory complexity | Start with CVC5's approach, optimize later |
| XOR integration complexity | Port CryptoMiniSat incrementally |
| Performance regression | CI benchmark comparison on every PR |
| Correctness bugs | LRAT proofs, fuzzing, differential testing |

---

## Reference Implementation Locations

| Feature | Primary Source | Secondary Source |
|---------|---------------|------------------|
| VMTF | `reference/cadical/src/queue.hpp` | |
| OTFS | `reference/cadical/src/analyze.cpp` | |
| XOR | `reference/cryptominisat/src/gaussian.cpp` | `xorfinder.cpp` |
| CDCL(T) | `reference/cvc5/src/prop/cadical/cdclt_propagator.h` | |
| Strings | `reference/cvc5/src/theory/strings/` | Z3 `src/smt/theory_str/` |
| Quantifiers | `reference/cvc5/src/theory/quantifiers/` | |
| Care Graph | `reference/cvc5/src/theory/combination_care_graph.cpp` | |
