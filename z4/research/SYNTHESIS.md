# Z4 Design Synthesis - Learning from Existing Solvers

This document synthesizes key learnings from analyzing Z3, CVC5, Yices 2, Bitwuzla, and MathSAT 5 to guide Z4's implementation.

---

## Executive Summary

| Solver | License | Key Strength | Adopt For |
|--------|---------|--------------|-----------|
| **Z3** | MIT | Comprehensive, API | Architecture, tactics, API design |
| **CVC5** | BSD | Strings, quantifiers | String solver, CEGQI |
| **Yices 2** | GPL | Speed (BV, arithmetic) | SAT optimizations, simplex |
| **Bitwuzla** | MIT | BV/Arrays | BV rewriting, array solver |
| **MathSAT 5** | Proprietary | Interpolation | Interpolation algorithm |

---

## Architecture Decisions

### From Z3: Plugin Architecture
```
Z4 should adopt Z3's plugin architecture for theories:
- Clean TheorySolver trait (already designed)
- Tactic framework for composable solving
- AST with pluggable declaration system
```

### From CVC5: Modular Theory Solvers
```
String solver with sub-solvers:
- CoreSolver (word equations)
- RegExpSolver (membership)
- ExtfSolver (contains, replace)
- This decomposition aids maintenance
```

### From Bitwuzla: Hybrid Solving
```
BV solver should support:
- Traditional bitblasting
- Propagation-based local search
- Switch based on problem characteristics
```

---

## Component-by-Component Recommendations

### 1. SAT Solver (z4-sat)

**Primary Source:** Yices 2 (fastest)
**Secondary:** CaDiCaL (via Bitwuzla/CVC5 integration patterns)

**Must Implement:**
| Feature | Source | File Reference |
|---------|--------|----------------|
| Blocker literals | Yices 2 | `new_sat_solver.h:265-300` |
| Binary clause inlining | Yices 2 | `new_sat_solver.h` |
| Clause pool storage | Yices 2 | `new_sat_solver.h:163-254` |
| VSIDS with decay | All | Standard |
| 1UIP conflict analysis | All | Standard |
| Luby restarts | Z3/Yices | Configurable |
| In-processing | Yices 2/CaDiCaL | SCC, subsumption |

### 2. Arithmetic Solver (z4-lra, z4-lia)

**Primary Source:** Yices 2 (fastest)
**Secondary:** Z3 (more complete)

**Must Implement:**
| Feature | Source | Notes |
|---------|--------|-------|
| Sparse simplex | Yices 2 | `simplex.c` |
| Theory propagation | Yices 2 | From tableau bounds |
| Gomory cuts | Yices 2/Z3 | For integers |
| Diophantine solver | Yices 2 | Equality systems |
| Farkas lemmas | Z3/MathSAT | For interpolation |

### 3. Bitvector Solver (z4-bv)

**Primary Source:** Bitwuzla (best)
**Secondary:** Yices 2 (fast paths)

**Must Implement:**
| Feature | Source | Notes |
|---------|--------|-------|
| Rewrite rules (118KB!) | Bitwuzla | `rewrites_bv.cpp` |
| Normalization | Bitwuzla | `rewrites_bv_norm.cpp` |
| Abstraction-refinement | Bitwuzla | For mul/div/rem |
| 64-bit fast paths | Yices 2 | `bv64_*.c` |
| Interval abstraction | Yices 2 | Range analysis |
| AIG bitblasting | Bitwuzla | Efficient CNF |
| Local search | Bitwuzla | SLS for SAT instances |

### 4. Array Solver (z4-arrays)

**Primary Source:** Bitwuzla (best)
**Secondary:** Z3

**Must Implement:**
| Feature | Source | Notes |
|---------|--------|-------|
| Lazy lemma generation | Bitwuzla | On-demand axioms |
| Path condition optimization | Bitwuzla | Minimal conditions |
| Congruence detection | Bitwuzla | Hash-based |
| Weak equivalence | Z3 | For complex cases |

### 5. String Solver (z4-strings)

**Primary Source:** CVC5 (best)
**Secondary:** Z3

**Must Implement:**
| Feature | Source | Notes |
|---------|--------|-------|
| Normal form algorithm | CVC5 | Liang et al. CAV 2014 |
| Modular sub-solvers | CVC5 | Core/Regex/Extf split |
| Length-arithmetic bridge | CVC5 | Critical integration |
| Regex derivatives | CVC5 | Membership checking |
| Sequences rewriter | CVC5 | 132KB of rules |

### 6. Quantifier Handling

**Primary Source:** CVC5
**Secondary:** Z3

**Must Implement:**
| Feature | Source | Notes |
|---------|--------|-------|
| E-matching | CVC5/Z3 | Pattern-based |
| CEGQI | CVC5 | For arithmetic |
| MBQI | Z3 | For finite domains |
| Conflict instantiation | CVC5 | From conflicts |

### 7. Interpolation

**Primary Source:** MathSAT 5 (papers)
**Secondary:** Z3

**Must Implement:**
| Feature | Source | Notes |
|---------|--------|-------|
| Proof-based interpolation | MathSAT papers | Griggio 2009 |
| LRA interpolation | MathSAT papers | Farkas-based |
| EUF interpolation | MathSAT papers | From proofs |
| Interpolation sequences | MathSAT papers | Multiple partitions |

---

## API Design

**Follow Z3's conventions:**
```rust
// Context-based API
let ctx = Context::new();
let solver = Solver::new(&ctx);

// Term construction
let x = ctx.int_const("x");
let y = ctx.int_const("y");
let constraint = x.gt(&y);

// Solving
solver.assert(&constraint);
match solver.check() {
    Sat => { let model = solver.get_model(); }
    Unsat => { /* handle */ }
    Unknown => { /* handle */ }
}
```

**From CVC5:** Term manager separation
**From MathSAT:** Interpolation group API

---

## Performance Targets

Based on SMT-COMP analysis:

| Category | Target | Benchmark Against |
|----------|--------|-------------------|
| QF_BV | Top 3 | Bitwuzla, Yices 2 |
| QF_LIA/LRA | Top 3 | Yices 2, Z3 |
| QF_AUFBV | Top 5 | Bitwuzla, Z3 |
| QF_S (Strings) | Top 3 | CVC5, Z3 |
| UFLIA | Top 5 | CVC5, Z3 |
| Interpolation | Parity | MathSAT 5 |

---

## Implementation Priority

### Phase 1: SAT Core (Current)
Focus: Yices-level SAT performance
- Implement all SAT optimizations from Yices
- Benchmark against MiniSat, CaDiCaL

### Phase 2: Core Theories
Focus: BV and Arithmetic
- Port Bitwuzla's BV rewrite rules
- Port Yices' simplex implementation

### Phase 3: String Theory
Focus: CVC5-level string support
- Implement normal form algorithm
- Build modular sub-solver architecture

### Phase 4: Advanced Features
Focus: Quantifiers and interpolation
- CEGQI from CVC5
- Interpolation from MathSAT papers

### Phase 5: Optimization
Focus: Beat or match all solvers
- Profile against SMT-COMP benchmarks
- Targeted optimizations per category

---

## File Reference Quick Lookup

### SAT Solver
- Yices: `reference/yices2/src/solvers/cdcl/new_sat_solver.c`
- Z3: `reference/z3/src/sat/sat_solver.cpp`

### Arithmetic
- Yices: `reference/yices2/src/solvers/simplex/simplex.c`
- Z3: `reference/z3/src/math/lp/lar_solver.cpp`

### Bitvectors
- Bitwuzla: `reference/bitwuzla/src/rewrite/rewrites_bv.cpp`
- Yices: `reference/yices2/src/solvers/bv/bvsolver.c`

### Arrays
- Bitwuzla: `reference/bitwuzla/src/solver/array/array_solver.cpp`
- Z3: `reference/z3/src/sat/smt/array_solver.cpp`

### Strings
- CVC5: `reference/cvc5/src/theory/strings/core_solver.cpp`
- Z3: `reference/z3/src/smt/theory_seq.cpp`

### Quantifiers
- CVC5: `reference/cvc5/src/theory/quantifiers/cegqi/`
- Z3: `reference/z3/src/smt/mam.cpp`

---

## License Compatibility

| Solver | License | Can Port Code? | Can Study? |
|--------|---------|----------------|------------|
| Z3 | MIT | Yes | Yes |
| CVC5 | BSD | Yes | Yes |
| Yices 2 | GPL | No (viral) | Yes (re-implement) |
| Bitwuzla | MIT | Yes | Yes |
| MathSAT 5 | Proprietary | No | Papers only |

**Strategy for Yices:**
- Study algorithms, re-implement from scratch in Rust
- Do not copy any code
- Use papers as primary reference

**Strategy for MathSAT:**
- Implement from published papers
- Griggio 2009 for interpolation
- No access to source code
