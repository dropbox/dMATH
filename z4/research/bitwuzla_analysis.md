# Bitwuzla SMT Solver - Research Analysis

**Repository:** https://github.com/bitwuzla/bitwuzla
**License:** MIT
**Language:** C++ (modern C++17)
**Developers:** Academic (Aina Niemetz, Mathias Preiner)
**First Release:** ~2020 (newest major solver)

---

## 1. Overview

Bitwuzla is the newest major SMT solver, focused on bitvectors and arrays. It consistently wins SMT-COMP in QF_BV and QF_ABV categories. Written in modern C++ with clean architecture, it's an excellent model for Z4's bitvector and array solvers.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Public API (bitwuzla.h)                      │
│                    (C and C++ interfaces)                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    SolvingContext                               │
│         (Top-level orchestration, preprocessing)                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    SolverEngine                                 │
│              (Theory solver coordination)                       │
└───────┬─────────────────┬───────────────────┬───────────────────┘
        │                 │                   │
┌───────▼───────┐ ┌───────▼───────┐   ┌───────▼───────────────────┐
│   BvSolver    │ │  ArraySolver  │   │   Other Solvers           │
│ ┌───────────┐ │ │               │   │ FpSolver│FunSolver│Quant  │
│ │Bitblast   │ │ │               │   │                           │
│ │Propagation│ │ │               │   │                           │
│ └───────────┘ │ │               │   │                           │
└───────────────┘ └───────────────┘   └───────────────────────────┘
        │                 │
        └────────┬────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    SAT Solver Layer                             │
│         CaDiCaL (default) │ Kissat │ CryptoMiniSat              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/solver/bv/` | Bitvector solver (Bitwuzla's strength) |
| `src/solver/array/` | Array solver (Bitwuzla's strength) |
| `src/solver/abstract/` | Abstraction-refinement module |
| `src/rewrite/` | Rewriting system (massive - 200KB+) |
| `src/lib/ls/` | Local search library |
| `src/lib/bitblast/` | Bit-blasting infrastructure |
| `src/preprocess/` | Preprocessing passes |
| `src/sat/` | SAT solver abstraction |

---

## 3. Strengths

### 3.1 Bitvector Performance (Best in Class)

Bitwuzla dominates QF_BV benchmarks through multiple techniques:

**1. Massive Rewrite Database** (`rewrites_bv.cpp` - 118KB!)

Over 100 specialized BV rewrite rules for:
- Arithmetic: add, sub, mul, udiv, urem, sdiv, srem
- Bitwise: and, or, xor, not
- Shifts: shl, shr, ashr
- Comparisons: ult, slt, eq
- Extraction/Concatenation

**2. Hybrid Solving Strategy**

```
BvSolver
├── BvBitblastSolver  - Traditional bit-blasting to SAT
└── BvPropSolver      - Propagation-based local search
```

The solver can use both approaches:
- Bit-blast for precision
- Local search for speed on satisfiable instances

**3. Operator Abstraction**

For expensive operations (mul, div, rem):
- Abstract with uninterpreted function
- Solve abstracted formula
- Refine with lemmas if needed

**4. AIG-Based Bitblasting**

- Convert BV ops to And-Inverter Graphs
- Apply AIG optimizations
- Convert AIG to CNF efficiently

**5. Constant Bit Propagation**

- Track known bits through operations
- Propagate constants without full blasting
- Inform local search with partial information

### 3.2 Array Performance (Best in Class)

**Lazy Lemma Generation:**

```cpp
enum class LemmaId {
  CONGRUENCE,         // i = j => a[i] = a[j]
  ACCESS_STORE,       // i = j => (store a i v)[j] = v
  ACCESS_CONST_ARRAY, // ((as const v))[i] = v
  DISEQUALITY,        // a != b => exists k. a[k] != b[k]
};
```

Only generates necessary array axiom instances.

**Path Condition Optimization:**

- Collects conditions along store chains
- Minimal lemmas based on current model

**Efficient Congruence Detection:**

- Hash-based access comparison
- Quick detection of equal indices

### 3.3 Modern C++ Codebase

- Clean, readable code
- RAII patterns
- Modern build system (Meson)
- Easy to understand and modify

### 3.4 Preprocessing Pipeline

Extensive preprocessing:
- Normalization (49KB of code)
- Variable substitution (30KB)
- Embedded constraint extraction
- Skeleton preprocessing

---

## 4. Weaknesses

### 4.1 Limited Theory Support

Only supports:
- Bitvectors (excellent)
- Arrays (excellent)
- Floating-point (good)
- Uninterpreted functions (basic)
- Quantifiers (basic)

Missing:
- Strings
- Linear arithmetic (except BV encoding)
- Datatypes
- Sets

### 4.2 Quantifier Handling

- Basic instantiation only
- Not competitive on quantified benchmarks
- Focus is QF (quantifier-free) theories

### 4.3 Arithmetic

- No native integer/real arithmetic
- Must encode as bitvectors
- Not suitable for pure arithmetic problems

### 4.4 Newer Project

- Less battle-tested than Z3/CVC5
- Smaller community
- Fewer language bindings

---

## 5. Key Algorithms

### 5.1 BV Rewriting

**File:** `src/rewrite/rewrites_bv.cpp`

Examples of sophisticated rewrites:

```cpp
// Multiplication by constant
// x * c where c is power of 2 => x << log2(c)

// Division simplification
// x / 1 => x
// x / x => 1 (when x != 0)

// Extraction optimization
// extract[h:l](concat(a, b)) => ... (complex rules)
```

### 5.2 Abstraction-Refinement

**File:** `src/solver/abstract/abstraction_module.cpp`

```cpp
// 1. Replace mul(x, y) with UF f(x, y)
// 2. Solve abstracted formula
// 3. If SAT, check concrete model
// 4. If model invalid, add refinement lemma:
//    f(x, y) = x * y  (for specific x, y values)
// 5. Repeat
```

### 5.3 Local Search for BV

**File:** `src/lib/ls/ls_bv.cpp`

Propagation-based stochastic local search:
1. Start with random assignment
2. Identify essential inputs for unsatisfied constraints
3. Flip bits with probability based on improvement
4. Integrate with constant propagation from bitblasting

### 5.4 Array Solver

**File:** `src/solver/array/array_solver.cpp`

```cpp
// Key method: check()
// 1. For each array access a[i]:
//    - Check for congruent accesses a[j] where i = j
//    - Generate CONGRUENCE lemma if values differ
// 2. For store chains:
//    - Collect path conditions
//    - Generate ACCESS_STORE lemmas as needed
// 3. For disequalities:
//    - Find witness index k where a[k] != b[k]
//    - Generate DISEQUALITY lemma
```

---

## 6. What Z4 Should Adopt from Bitwuzla

### Must Have
1. **BV Rewrite Rules**
   - Study `rewrites_bv.cpp` carefully
   - Implement comprehensive BV simplification
   - Normalization is key

2. **Abstraction-Refinement**
   - For expensive BV operations
   - Lazy precision approach
   - Clean module design

3. **Array Solver Architecture**
   - Lazy lemma generation
   - Path condition optimization
   - Clean lemma types

4. **Hybrid BV Solving**
   - Bitblasting + local search
   - Constant bit propagation

### Should Have
1. **AIG Infrastructure**
   - Efficient bitblasting
   - AIG optimizations before CNF

2. **Preprocessing Pipeline**
   - Normalization passes
   - Variable substitution
   - Skeleton preprocessing

### Could Adopt
1. **Local Search Library**
   - Propagation-based SLS
   - Useful for satisfiable instances

---

## 7. Key Files for Study

| Component | File | Why Study |
|-----------|------|-----------|
| BV Rewrites | `src/rewrite/rewrites_bv.cpp` | 118KB of BV rules |
| BV Normalization | `src/rewrite/rewrites_bv_norm.cpp` | Canonical forms |
| BV Solver | `src/solver/bv/bv_solver.cpp` | Hybrid architecture |
| Abstraction | `src/solver/abstract/abstraction_module.cpp` | Lazy refinement |
| Array Solver | `src/solver/array/array_solver.cpp` | Lemma generation |
| Local Search | `src/lib/ls/ls_bv.cpp` | SLS for BV |
| AIG | `src/lib/bitblast/aig/aig_manager.cpp` | Bitblasting |

---

## 8. Performance Characteristics

Based on SMT-COMP results:

| Category | Bitwuzla Performance | Notes |
|----------|----------------------|-------|
| QF_BV | **Dominant** | Usually wins |
| QF_ABV | **Dominant** | Usually wins |
| QF_AUFBV | **Excellent** | Top performer |
| QF_FP | Good | Competitive |
| QF_BVFP | Good | Combined |
| Quantified | Limited | Not focus |
| Arithmetic | N/A | Not supported |
| Strings | N/A | Not supported |

---

## 9. Integration Notes

Bitwuzla's MIT license allows full integration. Key integration points:

1. **BV Rewriting:**
   - Port the rewrite rules to Rust
   - Maintain the same coverage
   - Add Rust-specific optimizations

2. **Array Solver:**
   - Clean design, easy to port
   - Lemma types are well-defined
   - Path condition algorithm is elegant

3. **Abstraction Module:**
   - Study the lemma schemas
   - Clean refinement loop
   - Good error recovery

---

## 10. Key Papers

1. **Bitwuzla:**
   - Niemetz & Preiner, CAV 2020 - "Bitwuzla at the SMT-COMP 2020"

2. **Bit-Blasting:**
   - Brummayer & Biere, DATE 2006 - "Local Two-Level And-Inverter Graph Minimization without Blowup"

3. **Array Solver:**
   - de Moura & Bjorner, TACAS 2009 - "Generalized, Efficient Array Decision Procedures"

4. **Local Search:**
   - Froehlich et al., FMCAD 2015 - "Stochastic Local Search for Satisfiability Modulo Theories"
