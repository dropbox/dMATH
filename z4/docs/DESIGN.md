# Z4 Design Document

Z4 is a high-performance SMT solver written in Rust, targeting feature parity with Z3 and superior performance through modern systems design.

**Ultimate Goal**: Become the SMT backend for Lean 5, replacing Z3.

## Related Documents

| Document | Purpose |
|----------|---------|
| `research/BENCHMARKS_AND_TECHNIQUES.md` | Competition benchmarks, phase-by-phase techniques, Lean integration |
| `research/REFERENCES.md` | Academic papers, verified solver references |
| `research/SYNTHESIS.md` | What to take from each existing solver |
| `FORMAL_VERIFICATION_STRATEGY.md` | Verification approach (Kani, TLA+, proofs) |

## Implementation Sources

Each component draws from the best available implementation:

| Component | Primary Source | License | Approach |
|-----------|---------------|---------|----------|
| SAT core | Z3 (`src/sat/`) | MIT | Study and port |
| SAT optimizations | CaDiCaL | MIT | Study and port |
| Arithmetic | Yices 2 papers | GPL | **Re-implement from papers** |
| Bitvectors | Bitwuzla | MIT | Study and port |
| Arrays | Bitwuzla | MIT | Study and port |
| Strings | CVC5 | BSD | Study and port |
| Quantifiers | CVC5 | BSD | Study and port |
| Interpolation | MathSAT papers | Proprietary | **Implement from papers** |
| AllSAT | MathSAT papers | Proprietary | **Implement from papers** |
| API design | Z3 | MIT | Follow conventions |

**License strategy**: For GPL (Yices) and Proprietary (MathSAT), we implement from published papers only. See `research/REFERENCES.md` for the clean-room process.

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           z4 (main binary)          │
                    │   CLI interface, file/interactive   │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         z4-frontend                 │
                    │  SMT-LIB 2.6 parser, preprocessor   │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │           z4-dpll                   │
                    │   DPLL(T) integration framework     │
                    └───────┬─────────────────┬───────────┘
                            │                 │
           ┌────────────────▼───┐   ┌────────▼────────────┐
           │      z4-sat        │   │   Theory Solvers    │
           │  CDCL SAT solver   │   │  (z4-theories/*)    │
           └────────────────────┘   └─────────────────────┘
                            │                 │
                    ┌───────▼─────────────────▼───────────┐
                    │           z4-core                   │
                    │   Terms, sorts, models, proofs      │
                    └─────────────────────────────────────┘
```

## Crate Responsibilities

### z4-core
Foundation types shared across all components:
- **Sort**: SMT-LIB sort system (Bool, Int, Real, BitVec, Array, etc.)
- **Term**: Hash-consed DAG representation for maximal sharing
- **Model**: Satisfying assignments and function interpretations
- **TheorySolver trait**: Interface for all theory solvers
- **Proof**: Resolution proofs and theory lemmas

### z4-sat
Standalone CDCL SAT solver competitive with CaDiCaL/Kissat:
- **Literal**: Compact encoding (2*var for positive, 2*var+1 for negative)
- **Clause**: With LBD tracking and activity scores
- **2-Watched Literals**: Efficient unit propagation
- **VSIDS**: Exponentially decaying variable activities
- **1UIP Learning**: Conflict analysis with clause minimization
- **Restarts**: Luby sequence and glucose-style

### z4-dpll
DPLL(T) integration layer:
- Theory abstraction (boolean variables for theory atoms)
- Theory propagation interface
- Theory conflict clause generation
- Nelson-Oppen theory combination
- Incremental solving (push/pop)

### z4-frontend
SMT-LIB 2.6 compliant parser:
- Lexer using `logos` for speed
- S-expression parser
- Sort checking and inference
- Command execution
- Preprocessing and simplification
- Tseitin transformation to CNF

### z4-proof
Proof generation and export:
- Resolution proof logging
- Theory lemma recording
- LFSC format export
- Alethe format export

### z4-chc
CHC (Constrained Horn Clause) solver using PDR/IC3:
- **Algorithm**: Property-Directed Reachability (PDR/IC3)
- Horn clause representation (predicates, body => head)
- Frame-based over-approximation refinement
- Lemma learning and propagation
- Invariant synthesis for loop verification
- Used by Kani Fast for unbounded verification

### Theory Solvers

#### z4-euf (Equality + Uninterpreted Functions)
**Algorithm**: Congruence Closure (Nieuwenhuis-Oliveras 2007)
- Union-Find with path compression and union by rank
- Pending list for propagation
- Explanation generation for proofs
- O(n log n) amortized time

#### z4-lra (Linear Real Arithmetic)
**Algorithm**: Dual Simplex (Dutertre-de Moura 2006)
- Sparse matrix representation
- Pivoting with Bland's rule for termination
- Farkas coefficient extraction for conflicts
- Delta-rationals for strict inequalities

#### z4-lia (Linear Integer Arithmetic)
**Algorithm**: Simplex + Branch-and-Bound + Cuts
- Inherits from LRA
- GCD test for fast infeasibility
- Gomory cuts for tighter bounds
- Branch-and-bound for completeness

#### z4-bv (Bitvectors)
**Algorithm**: Lazy bit-blasting
- Word-level propagation when possible
- On-demand bit-blasting to SAT
- AIG representation for circuits
- Structural hashing

#### z4-arrays (Array Theory)
**Algorithm**: Weak Equivalence Graphs (de Moura-Bjorner 2009)
- Read-over-write optimization
- Lazy extensionality
- Ackermanization for finite models

#### z4-strings (String Theory)
**Algorithm**: Normal Form (CVC5 approach - Liang et al. CAV 2014)
- **Source**: CVC5 `src/theory/strings/` (BSD)
- Modular sub-solvers: CoreSolver, RegExpSolver, ExtfSolver
- Normal form computation for word equations
- Length-arithmetic tight integration
- Brzozowski derivatives for regex membership
- Sequences rewriter (132KB of rules in CVC5)

#### z4-fp (Floating Point)
**Algorithm**: Reduction to BV + Real
- IEEE 754 semantics
- Bit-level encoding via BV theory
- Real approximations for bounds

#### z4-dt (Algebraic Datatypes)
**Algorithm**: Constructor reasoning
- Injectivity and distinctness
- Selector semantics
- Cycle detection for acyclicity

### Advanced Features

#### Interpolation (z4-proof)
**Algorithm**: Proof-based (MathSAT approach - Griggio 2009)
- **Source**: MathSAT papers (implement from literature)
- Extract interpolants from resolution proofs
- Theory-specific interpolation for LRA, LIA, EUF
- Interpolation sequences for multiple partitions
- **Critical** for IC3/PDR model checking

#### AllSAT Enumeration
**Algorithm**: Blocking-free enumeration (Spallitta et al. 2024)
- **Source**: MathSAT papers (implement from literature)
- Projected model enumeration
- Implicant shrinking
- No blocking clause overhead

#### Optimization Modulo Theories (OMT)
**Algorithm**: Search-based optimization
- **Source**: OptiMathSAT papers
- Linear/binary search on objectives
- MaxSMT support
- Pareto optimization for multi-objective

#### User Propagators
**Algorithm**: Custom theory integration (Z3 approach)
- **Source**: Z3 API design
- User-defined propagation callbacks
- Custom conflict generation
- Enables domain-specific theories

---

## Key Algorithms

### CDCL SAT Solving

```
function CDCL(clauses):
    while true:
        while propagate() == CONFLICT:
            if decision_level == 0:
                return UNSAT
            (learned, level) = analyze_conflict()
            add_clause(learned)
            backtrack(level)
        if all_assigned():
            return SAT
        decide(pick_variable())
```

Key optimizations:
1. **2-Watched Literals**: Only watch 2 literals per clause, update watches lazily
2. **VSIDS**: Bump activity of variables in conflicts, decay periodically
3. **Phase Saving**: Remember last assignment, prefer that polarity
4. **Clause Deletion**: Remove low-LBD learned clauses periodically

### Congruence Closure

```
function merge(a, b):
    ra = find(a)
    rb = find(b)
    if ra == rb: return
    union(ra, rb)
    for each pending congruence involving ra or rb:
        propagate_congruence()
```

### Dual Simplex

```
function check():
    while exists violated bound:
        select pivot (row, col) using Bland's rule
        pivot(row, col)
        if infeasible:
            return UNSAT with Farkas proof
    return SAT
```

---

## Performance Design

### Memory Layout
- Clause literals stored contiguously for cache efficiency
- Small literal encoding (u32) for compact clause storage
- Arena allocation for terms (bumpalo)
- Data-oriented design for cache friendliness

### Parallelism
- Portfolio solving with different random seeds
- Parallel preprocessing
- Theory solver parallelism where applicable
- Work stealing with rayon

### SIMD
- Vectorized literal scanning in watched lists
- Batch operations in simplex tableau
- AVX2/AVX-512 for bulk literal operations

### GPU Acceleration (Phase 6)
- **Parallel SAT solving**: Multiple CDCL instances on GPU cores
- **Batch theory checks**: Bulk constraint evaluation
- **BV operations**: Parallel bit-blasting and evaluation
- **Local search**: GPU-accelerated stochastic local search
- Target: 10x speedup on parallelizable workloads

---

## Testing Strategy

1. **Unit Tests**: Per-module functionality
2. **SMT-LIB Benchmarks**: Regression against SMT-COMP benchmarks
3. **Fuzzing**: Grammar-based input generation
4. **Differential Testing**: Compare against Z3/CVC5
5. **Performance Benchmarks**: Criterion-based benchmarking

---

## References

- Nieuwenhuis & Oliveras (2007) - Fast Congruence Closure
- Dutertre & de Moura (2006) - Simplex for SMT
- de Moura & Bjorner (2008) - Z3 Architecture
- Eén & Sörensson (2003) - MiniSat techniques
- Biere et al. (2021) - CaDiCaL techniques

---

## Roadmap

### Phase 1: SAT Core (Current)

Build CDCL solver competitive with CaDiCaL/Kissat, **fully verified**.

**Competition**: SAT Competition (~400 benchmarks/year)
**Benchmark access**: `benchmark-database.de` (see `research/BENCHMARKS_AND_TECHNIQUES.md`)

#### 1A: Core CDCL (Match IsaSAT baseline)
- [ ] Literal/Variable encoding (u32-based)
- [ ] Clause database with arena allocation
- [ ] 2-Watched literals with blocking literals
- [ ] EVSIDS scoring with decay
- [ ] Phase saving
- [ ] 1-UIP conflict analysis
- [ ] Clause minimization (recursive)
- [ ] Non-chronological backtracking
- [ ] Luby restarts
- [ ] **DRAT proof generation** (critical)

**Verification**: Kani proofs, TLA+ spec, proptest, differential testing vs MiniSat

#### 1B: Performance (Match CaDiCaL)
- [ ] Chronological backtracking + lazy reimplication
- [ ] Glucose-style EMA restarts
- [ ] LBD-based clause management
- [ ] Tier-based clause database (core/mid/local)
- [ ] Reluctant doubling restarts
- [ ] Target phases

#### 1C: Inprocessing (Close the 60% gap vs IsaSAT)
- [ ] Vivification (strengthen clauses)
- [ ] Bounded variable elimination (BVE)
- [ ] Blocked clause elimination (BCE)
- [ ] Subsumption & self-subsumption
- [ ] Failed literal probing
- [ ] Hyper-ternary resolution

#### 1D: Advanced SAT
- [ ] Gate extraction (AND/XOR/ITE)
- [ ] Congruence closure in preprocessing
- [ ] SAT sweeping
- [ ] LRAT proof generation

**Target**: 300/400 problems (75%) on SAT Competition, vs IsaSAT's 141 (35%)
**Sources**: CaDiCaL (MIT), Kissat (MIT), IsaSAT (verified reference), CreuSAT (Rust reference)

### Phase 2: SMT Infrastructure
Add z4-core, z4-frontend, z4-dpll framework.
- Complete SMT-LIB 2.6 parser
- Term DAG with hash-consing
- DPLL(T) integration layer
- Nelson-Oppen theory combination

### Phase 3: Core Theories
EUF, LRA, LIA - covers QF_LRA, QF_LIA, QF_UF.
- **EUF**: Congruence closure from Z3
- **LRA/LIA**: Implement from Yices 2 papers (Dutertre 2006)
- **Target**: Beat Yices 2 on QF_LIA/QF_LRA

### Phase 4: Program Verification Theories
BV, Arrays - covers QF_BV, QF_AUFBV.
- **Source**: Bitwuzla (MIT) for both
- Port 118KB of BV rewrite rules
- Lazy array lemma generation
- **Target**: Beat Bitwuzla on QF_BV/QF_ABV

### Phase 5: Advanced Theories
Strings, FP, Datatypes, Quantifiers.
- **Strings**: Port CVC5 normal form algorithm (BSD)
- **Quantifiers**: CEGQI from CVC5 papers
- **Target**: Competitive across all SMT-COMP categories

### Phase 6: Acceleration
GPU, SIMD, distributed solving.
- GPU-accelerated parallel SAT solving
- SIMD-optimized data structures
- Distributed solving for massive problems
- **Target**: 10x speedup on parallelizable workloads

### Phase 7: Advanced Features
Interpolation, AllSAT, OMT.
- Interpolation from MathSAT papers
- AllSAT enumeration
- Optimization modulo theories
- User propagators

---

## Lean 5 Integration (Long-Term Goal)

Z4's ultimate purpose is to become the SMT backend for Lean 5.

### Bootstrap Strategy

```
Stage 1: Z4 verified by Z3
├── Verus/Dafny proofs about Z4 algorithms (Z3 as backend)
├── Z4 produces DRAT/Alethe proofs
├── Verified checkers (cake_lpr, carcara) validate proofs
└── Z4 becomes trusted through proof production

Stage 2: Z4 integrated with Lean 4
├── Z4 as external solver (like Z3 today)
├── Z4 returns proof certificates
├── Lean kernel checks certificates
└── Soundness preserved even if Z4 has bugs

Stage 3: Z4 self-hosting (Lean 5)
├── Z4 replaces Z3 as Verus backend
├── Z4 verifies itself (or next-gen Z5)
├── Lean 5 uses Z4 natively
└── Full bootstrap: Lean 5 ←→ Z4 mutually verified
```

### Proof Certificate Flow

```
User writes: theorem foo : P := by smt
                    ↓
Lean 5 smt tactic:
1. Encode goal as SMT-LIB
2. Call Z4
3. Z4 returns: (unsat, proof_certificate)
4. Lean kernel checks certificate
5. If valid → theorem proven
   If invalid → tactic fails (soundness preserved)
```

### Why This Matters

1. **Soundness**: Lean's kernel is the trust anchor, not Z4
2. **Performance**: Z4 can be fast without compromising soundness
3. **Dogfooding**: Lean verifies Z4, Z4 powers Lean
4. **Independence**: No reliance on Microsoft (Z3) or Stanford (CVC5)

See `research/BENCHMARKS_AND_TECHNIQUES.md` for detailed Lean integration plan.
