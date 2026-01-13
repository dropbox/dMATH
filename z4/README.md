# Z4: The Last SMT Solver

| Director | Status |
|:--------:|:------:|
| MATH | ACTIVE |

**Z4 is a next-generation SMT solver built to make formal verification fast enough for everyday programming.**

---

## The Problem

Formal verification is too slow. SMT solvers—the engines behind program verification, symbolic execution, and AI reasoning—take seconds or minutes where they should take milliseconds. This makes them impractical for:

- **Compile-time verification**: Catching bugs before code runs
- **AI-assisted programming**: Real-time constraint solving in code generation
- **Continuous verification**: Checking invariants on every commit

The result: formal methods remain academic curiosities instead of standard practice.

## The Vision

**Z4 changes this.**

We are building an SMT solver that is:
- **Faster than all existing solvers** on every category
- **Complete**: All features from Z3, CVC5, Yices, Bitwuzla, and MathSAT
- **Modern**: Pure Rust with formal verification of the solver itself
- **Accelerated**: GPU and SIMD parallelism where it matters

The goal is to make SMT solving so fast that it becomes part of the compiler, part of the IDE, part of the AI programming loop itself.

### Why This Matters

The path to the AI singularity runs through better tools. AI systems today are limited by the tools they use. When AI can generate formally verified code in real-time, when constraints can be solved at the speed of thought, we unlock a new generation of AI capabilities:

```
Better tools → Better AI → Better tools → ...
```

Z4 is a critical piece of that infrastructure.

---

## Competitive Landscape

We have studied every major SMT solver. Here's what exists:

| Solver | License | Language | Strength | Weakness |
|--------|---------|----------|----------|----------|
| **Z3** | MIT | C++ | Comprehensive, best API | Not fastest, complex codebase |
| **CVC5** | BSD | C++ | Strings, quantifiers | Slower on BV/arithmetic |
| **Yices 2** | GPL | C | Fastest on BV/arithmetic | Limited theories, GPL |
| **Bitwuzla** | MIT | C++ | Dominates BV/arrays | No arithmetic, strings |
| **MathSAT 5** | Proprietary | C++ | Best interpolation | Closed source |

**Key insight**: No single solver is best at everything. Each has carved out a niche:
- Yices wins on raw speed but lacks theories
- CVC5 wins on strings but is slower elsewhere
- Bitwuzla dominates BV but has no arithmetic
- Z3 is comprehensive but not the fastest anywhere
- MathSAT has the best interpolation but is closed

**Z4's opportunity**: Be the fastest at everything, in one solver.

---

## Technical Strategy

### Architecture: Take the Best from Each

| Component | Source | Why |
|-----------|--------|-----|
| **SAT core** | Yices 2 | Blocker literals, clause pools, in-processing |
| **Arithmetic** | Yices 2 | Sparse simplex, theory propagation |
| **Bitvectors** | Bitwuzla | 118KB of rewrite rules, abstraction-refinement |
| **Arrays** | Bitwuzla | Lazy lemmas, path conditions |
| **Strings** | CVC5 | Normal form algorithm, modular sub-solvers |
| **Quantifiers** | CVC5 | CEGQI, E-matching |
| **Interpolation** | MathSAT papers | Proof-based approach |
| **API design** | Z3 | Clean, composable tactics |

### License Strategy

| Solver | Can Port Code? | Our Approach |
|--------|----------------|--------------|
| Z3 | Yes (MIT) | Study and port directly |
| CVC5 | Yes (BSD) | Study and port directly |
| Bitwuzla | Yes (MIT) | Study and port directly |
| Yices 2 | **No (GPL)** | Re-implement algorithms from papers |
| MathSAT 5 | **No (Proprietary)** | Implement from published papers |

### Performance Strategy

1. **Rust advantage**: Zero-cost abstractions, no GC pauses, fearless concurrency
2. **Data-oriented design**: Cache-friendly structures, arena allocation
3. **SIMD acceleration**: Vectorized operations where applicable
4. **GPU offload**: Parallel SAT solving, batch theory checks
5. **Formal verification**: Use Kani to verify critical algorithms

### Feature Completeness

Z4 will support **all** SMT-LIB 2.6 theories:

| Theory | Status | Best-in-class Target |
|--------|--------|---------------------|
| Core SAT | Phase 1 | Beat CaDiCaL |
| EUF | Phase 3 | Match Z3 |
| LRA/LIA | Phase 3 | Beat Yices 2 |
| BV | Phase 4 | Beat Bitwuzla |
| Arrays | Phase 4 | Beat Bitwuzla |
| Strings | Phase 5 | Beat CVC5 |
| FP | Phase 5 | Match Z3 |
| Datatypes | Phase 5 | Match CVC5 |
| Quantifiers | Phase 5 | Match CVC5 |

Plus advanced features:
- **Interpolation** (from MathSAT research)
- **AllSAT enumeration**
- **Optimization modulo theories**
- **Proof production** (DRAT, Alethe)
- **User propagators** (custom theory integration)

---

## Development Phases

### Phase 1: SAT Core (Current)
Build a CDCL SAT solver competitive with CaDiCaL/Kissat.
- Watched literals with blocker optimization
- VSIDS with decay and rescaling
- 1UIP conflict analysis with clause minimization
- Luby restarts, clause deletion by LBD
- In-processing: SCC, subsumption, self-subsuming resolution

**Target**: Top-5 on SAT Competition benchmarks.

### Phase 2: SMT Infrastructure
- Term DAG with hash-consing
- Complete SMT-LIB 2.6 parser
- DPLL(T) integration layer
- Nelson-Oppen theory combination

### Phase 3: Core Theories
- **EUF**: E-graph with congruence closure
- **LRA**: Dual simplex with Farkas lemmas
- **LIA**: Branch-and-bound, Gomory cuts

**Target**: Beat Yices 2 on QF_LIA/QF_LRA.

### Phase 4: Program Verification Theories
- **BV**: Bit-blasting + abstraction-refinement
- **Arrays**: Lazy axiom instantiation

**Target**: Beat Bitwuzla on QF_BV/QF_ABV.

### Phase 5: Advanced Theories
- Strings with CVC5-style normal forms
- Floating-point via bit-blasting
- Algebraic datatypes
- Quantifiers with CEGQI

**Target**: Competitive across all SMT-COMP categories.

### Phase 6: Acceleration
- GPU-accelerated parallel SAT solving
- SIMD-optimized data structures
- Distributed solving for massive problems

**Target**: 10x speedup on parallelizable workloads.

---

## Architecture

```
z4/
├── crates/
│   ├── z4/              # Main binary and library facade
│   ├── z4-core/         # Terms, sorts, models, theory trait
│   ├── z4-sat/          # CDCL SAT solver
│   ├── z4-dpll/         # DPLL(T) orchestration
│   ├── z4-frontend/     # SMT-LIB 2.6 parser
│   ├── z4-proof/        # Proof production
│   └── z4-theories/     # Theory solvers
│       ├── euf/         # Equality + Uninterpreted Functions
│       ├── lra/         # Linear Real Arithmetic
│       ├── lia/         # Linear Integer Arithmetic
│       ├── bv/          # Bitvectors
│       ├── arrays/      # Array theory
│       ├── strings/     # String theory
│       ├── fp/          # Floating-point
│       └── dt/          # Algebraic Datatypes
├── docs/
│   └── DESIGN.md        # Detailed algorithm specifications
└── research/            # Competitive analysis
    ├── SYNTHESIS.md     # Cross-solver learnings
    ├── z3_analysis.md
    ├── cvc5_analysis.md
    ├── yices2_analysis.md
    ├── bitwuzla_analysis.md
    └── mathsat5_analysis.md
```

---

## Building

```bash
# Build all crates
cargo build --release

# Run the solver
cargo run --release -- input.smt2

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Usage

```bash
# Interactive mode
z4

# File mode
z4 input.smt2

# With options
z4 --timeout 30 --model input.smt2
```

### Example

```smt2
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (> x 0))
(assert (> y 0))
(assert (= (+ x y) 10))
(check-sat)
(get-model)
```

---

## References

### Key Papers
- MiniSat (Een & Sorensson 2003) - CDCL foundations
- DPLL(T) (de Moura & Bjorner 2008) - SMT architecture
- CaDiCaL (Biere 2021) - Modern SAT techniques
- Simplex for SMT (Dutertre 2006) - Arithmetic solving
- String solving (Liang et al. 2014) - CVC5's approach

### Solver Repositories
- [Z3](https://github.com/Z3Prover/z3) - MIT
- [CVC5](https://github.com/cvc5/cvc5) - BSD
- [Yices 2](https://github.com/SRI-CSL/yices2) - GPL (study only)
- [Bitwuzla](https://github.com/bitwuzla/bitwuzla) - MIT

---

## License

Apache 2.0 - See [LICENSE](LICENSE)

Copyright 2025 Dropbox, Inc. Created by Andrew Yates.

Z4 builds upon research from [Z3](https://github.com/Z3Prover/z3) (Microsoft Research) and other SMT solvers. See [NOTICE](NOTICE) for attributions.

---

## Status

**Phase 1: SAT Core** - Under active development.

The architecture is complete. We are implementing the CDCL SAT solver.
