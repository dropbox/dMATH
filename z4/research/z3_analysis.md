# Z3 SMT Solver - Research Analysis

**Repository:** https://github.com/Z3Prover/z3
**License:** MIT
**Language:** C++ (C++20)
**Developer:** Microsoft Research (Leonardo de Moura, Nikolaj Bjorner)
**First Release:** ~2007

---

## 1. Overview

Z3 is the most widely-used SMT solver, known for its excellent API, wide theory support, and general-purpose applicability. It serves as the reference implementation for many SMT features.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                               │
│              (C, C++, Python, Java, .NET, OCaml, JS)            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                      Tactic Framework                           │
│               (Composable solving strategies)                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼───────┐   ┌─────────────▼───────────┐   ┌────────▼────────┐
│   SAT Solver  │   │    SMT Solver (smt/)    │   │  NL-SAT Solver  │
│    (sat/)     │   │      Classic DPLL(T)    │   │   (nlsat/)      │
└───────────────┘   └─────────────────────────┘   └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │                  Theory Plugins                    │
        │  EUF │ Arith │ BV │ Arrays │ Strings │ FP │ DT   │
        └───────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │                    AST Layer                       │
        │        (Hash-consed terms, sort system)           │
        └───────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │                  Foundation                        │
        │   (Memory mgmt, multi-precision arithmetic)       │
        └───────────────────────────────────────────────────┘
```

### Key Directories

| Directory | Size | Purpose |
|-----------|------|---------|
| `src/sat/` | ~200KB | CDCL SAT solver |
| `src/smt/` | ~600KB | Classic SMT solver with theory plugins |
| `src/sat/smt/` | ~400KB | New SAT-based SMT architecture |
| `src/nlsat/` | ~180KB | Nonlinear arithmetic (CAD) |
| `src/math/lp/` | ~500KB | Linear programming / simplex |
| `src/ast/` | ~400KB | AST infrastructure |
| `src/api/` | ~600KB | Public APIs |

---

## 3. Strengths

### 3.1 Comprehensive Theory Support
Z3 supports more theories than any other solver:
- Linear/Nonlinear Real/Integer Arithmetic
- Bitvectors (all operations)
- Arrays with extensionality
- Strings and sequences with regex
- Floating-point (IEEE 754)
- Algebraic datatypes
- Uninterpreted functions
- Pseudo-boolean constraints
- Finite domains
- Recursive functions

### 3.2 Excellent API
- Clean C API with consistent conventions
- C++ wrapper with operator overloading
- Bindings for 7+ languages
- Comprehensive documentation

### 3.3 Tactic Framework
- Composable solving strategies
- User can define custom solving pipelines
- Enables domain-specific optimizations

### 3.4 Proof Production
- DRAT proofs for SAT
- Theory lemma proofs
- Supports proof checking

### 3.5 Optimization Support
- MaxSMT (optimization modulo theories)
- Weighted objectives
- Pareto optimization

### 3.6 Advanced Features
- Quantifier elimination for specific theories
- Model-based quantifier instantiation (MBQI)
- User propagator API
- Fixed-point engine (Datalog, Spacer)

---

## 4. Weaknesses

### 4.1 Performance on Specific Domains
- Strings: CVC5 often faster on string-heavy benchmarks
- Bitvectors: Bitwuzla/Yices faster on pure BV
- Linear arithmetic: Yices faster on large LIA/LRA

### 4.2 Codebase Complexity
- ~500K lines of C++
- Two parallel SMT architectures (classic and SAT-based)
- Deep inheritance hierarchies
- Difficult to understand and modify

### 4.3 Build Time
- Long compilation (10-30 minutes)
- Many optional features add complexity

### 4.4 Memory Usage
- Can be memory-intensive on large problems
- Reference counting overhead

---

## 5. Key Algorithms

### 5.1 SAT Solver
- CDCL with 2-watched literals
- VSIDS variable selection
- Luby restarts (configurable)
- Clause minimization
- Local search integration
- Parallel solving support

### 5.2 Arithmetic
- Dual simplex for LRA
- Gomory cuts for LIA
- CAD (Cylindrical Algebraic Decomposition) for nonlinear
- Grobner bases for polynomial constraints

### 5.3 Bitvectors
- Lazy bit-blasting with word-level reasoning
- Ackermanization for UF
- AIG representation

### 5.4 E-matching
- Code tree compilation for efficient matching
- Trigger inference
- Relevancy filtering

---

## 6. What Z4 Should Adopt from Z3

### Must Have
1. **Tactic Framework** - Composable solving strategies are powerful for users
2. **Comprehensive API** - Follow Z3's API conventions (they're well-designed)
3. **Theory Plugin Architecture** - Clean separation of theory solvers
4. **Proof Production** - Essential for verification applications

### Should Have
1. **User Propagator** - Allows custom theory integration
2. **Optimization Support** - MaxSMT is valuable
3. **Model-based QI** - For quantifier support
4. **Fixed-point Engine** - For program verification

### Could Adopt
1. **NL-SAT** - CAD-based nonlinear solver
2. **AC Unification** - For associative-commutative theories
3. **Interpolation** - Though MathSAT is better here

---

## 7. Key Files for Study

| Component | File | Why Study |
|-----------|------|-----------|
| SAT Solver | `src/sat/sat_solver.cpp` | State-of-art CDCL implementation |
| Theory Plugin | `src/smt/smt_theory.h` | Clean theory interface |
| E-matching | `src/smt/mam.cpp` | Efficient pattern matching |
| Simplex | `src/math/lp/lar_solver.cpp` | Modern arithmetic solver |
| EUF | `src/ast/euf/euf_egraph.cpp` | E-graph implementation |
| API Design | `src/api/z3_api.h` | API conventions |

---

## 8. Performance Characteristics

Based on SMT-COMP results:

| Category | Z3 Performance | Notes |
|----------|----------------|-------|
| QF_LIA | Good | Competitive but not fastest |
| QF_LRA | Good | Competitive |
| QF_BV | Moderate | Bitwuzla/Yices faster |
| QF_AUFBV | Good | Strong integration |
| QF_S (Strings) | Good | CVC5 often wins |
| QF_NRA | Excellent | CAD-based solver |
| UFLIA | Good | Strong quantifier support |

---

## 9. Integration Notes

Z3's MIT license allows full integration. Key integration points:
- Study `src/sat/sat_solver.h` for SAT solver interface
- Study `src/smt/smt_theory.h` for theory solver interface
- Study `src/api/z3_api.h` for API design patterns
- The newer `src/sat/smt/` architecture is cleaner than classic `src/smt/`
