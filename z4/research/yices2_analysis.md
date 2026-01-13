# Yices 2 SMT Solver - Research Analysis

**Repository:** https://github.com/SRI-CSL/yices2
**License:** GPLv3
**Language:** C (pure C99)
**Developer:** SRI International (Bruno Dutertre)
**First Release:** ~2006

---

## 1. Overview

Yices 2 is known for exceptional speed on bitvectors and linear arithmetic. Written in pure C, it's highly optimized and consistently among the fastest solvers for QF_BV and QF_LIA/LRA. Its clean C implementation makes it an excellent study for performance optimization.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Public API (yices.h)                       │
│                     (323KB implementation!)                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                        Context                                  │
│            (Problem management, theory coordination)            │
└───────┬─────────────────────────────────────────────────┬───────┘
        │                                                 │
┌───────▼───────────────────┐         ┌───────────────────▼───────┐
│       SMT Core            │◄───────►│        E-graph            │
│   (DPLL(T) backbone)      │         │   (UF, theory hub)        │
└───────────────────────────┘         └───────────────────────────┘
        │                                       │
        │                    ┌──────────────────┼──────────────────┐
        │                    │                  │                  │
┌───────▼───────┐    ┌───────▼───────┐  ┌──────▼──────┐   ┌───────▼───────┐
│  SAT Solver   │    │    Simplex    │  │  BV Solver  │   │   MC-SAT      │
│(new_sat_solver)│   │  (arithmetic) │  │(bit-blasting)│   │ (nonlinear)   │
└───────────────┘    └───────────────┘  └─────────────┘   └───────────────┘
```

### Key Directories

| Directory | Size | Purpose |
|-----------|------|---------|
| `src/solvers/cdcl/` | ~550KB | SAT solver and SMT core |
| `src/solvers/simplex/` | ~500KB | Arithmetic (Yices' strength) |
| `src/solvers/bv/` | ~400KB | Bitvectors (Yices' strength) |
| `src/solvers/egraph/` | ~290KB | E-graph for UF |
| `src/mcsat/` | ~150KB | MC-SAT for nonlinear |
| `src/terms/` | ~200KB | Term representation |
| `src/api/` | ~330KB | API implementation |

---

## 3. Strengths

### 3.1 Speed (Best in Class for BV/Arithmetic)

Yices consistently wins or places top-3 in:
- QF_BV (bitvectors)
- QF_LIA (linear integer arithmetic)
- QF_LRA (linear real arithmetic)
- QF_IDL/RDL (difference logic)

**Why It's Fast:**

1. **Pure C Implementation**
   - No C++ overhead
   - Direct memory control
   - Cache-friendly data structures

2. **Optimized SAT Solver** (`new_sat_solver.c` - 309KB)
   - Blocker literals (avoid clause visits)
   - Binary clauses inline in watch lists
   - Clause pool with compact storage
   - SCC-based simplification
   - Aggressive in-processing

3. **Efficient Arithmetic** (`simplex.c` - 339KB)
   - Sparse matrix representation
   - Efficient pivoting
   - Gomory cuts for integers
   - Theory propagation from tableau

4. **Fast BV Solver** (`bvsolver.c` - 216KB)
   - Lazy bit-blasting
   - 64-bit specialized paths
   - Interval abstraction for pruning
   - DAG-based polynomial representation

### 3.2 Clean Architecture

Despite being C, Yices has excellent architecture:
- Well-documented interfaces (`doc/smt_architecture.txt`)
- Clear separation between:
  - Control interface (search control)
  - SMT interface (atom handling)
  - Egraph interface (equality)

### 3.3 MC-SAT for Nonlinear

Model-Constructing SAT for nonlinear arithmetic:
- Alternative to CAD
- Plugin-based theory integration
- Uses libpoly for polynomial operations

### 3.4 Low Memory Footprint

- Compact data structures
- No reference counting overhead
- Efficient term representation

---

## 4. Weaknesses

### 4.1 Limited Theory Support

Missing or limited:
- Strings (not supported)
- Floating-point (basic)
- Sequences (not supported)
- Sets/Bags (not supported)

### 4.2 Limited Quantifier Support

- Basic quantifier handling only
- No CEGQI or advanced instantiation
- Not competitive on quantified benchmarks

### 4.3 GPL License

- GPLv3 is more restrictive than MIT/BSD
- Cannot be used in proprietary projects without careful consideration

### 4.4 No Proof Production

- Limited proof output
- Harder to verify results externally

### 4.5 C Codebase

- Harder to extend than C++
- Manual memory management
- No RAII patterns

---

## 5. Key Algorithms

### 5.1 SAT Solver Optimizations

**File:** `src/solvers/cdcl/new_sat_solver.h`

```c
// Key optimizations:
// 1. Blocker literals - skip clause if blocker is true
typedef struct watch_s {
  uint32_t blocker;  // If true, don't visit clause
  uint32_t cid;      // Clause ID
} watch_t;

// 2. Binary clause inlining
// Binary clauses stored directly in watch list

// 3. Clause pool - compact storage
// Problem clauses: indices 0 to learned_base-1
// Learned clauses: indices learned_base to end
```

### 5.2 Simplex Algorithm

**File:** `src/solvers/simplex/simplex.c`

Key features:
- Revised simplex with sparse tableau
- Bland's rule with configurable threshold
- Equality propagation module
- Gomory cut generation

```c
// From simplex_types.h:
typedef struct arith_vartable_s {
  // Bounds
  xrational_t *lower_bound;
  xrational_t *upper_bound;
  // Current value
  xrational_t *value;
  // ...
} arith_vartable_t;
```

### 5.3 BV Solver

**File:** `src/solvers/bv/bvsolver.c`

Key techniques:
1. **Interval Analysis** - Infer value ranges
2. **Lazy Bit-Blasting** - Only blast when needed
3. **64-bit Fast Path** - Specialized for small BVs
4. **Polynomial DAG** - Share subexpressions

```c
// From bvsolver_types.h:
// Variable classification:
// - bitblasted variables
// - pseudo-mapped variables (to 64-bit)
// - mapped to compiler output
```

---

## 6. What Z4 Should Adopt from Yices

### Must Have
1. **SAT Solver Optimizations**
   - Blocker literals
   - Binary clause inlining
   - Clause pool storage
   - In-processing techniques

2. **Arithmetic Optimizations**
   - Sparse matrix representation
   - Efficient pivoting
   - Theory propagation from tableau

3. **BV Optimizations**
   - 64-bit specialized paths
   - Interval abstraction
   - Lazy bit-blasting

### Should Have
1. **Difference Logic Solver**
   - Floyd-Warshall for IDL/RDL
   - Fast path for restricted arithmetic

2. **MC-SAT Architecture**
   - For nonlinear arithmetic
   - Plugin-based design

### Could Adopt
1. **Sparse Data Structures**
   - Study Yices' hash tables
   - Compact representations

---

## 7. Key Files for Study

| Component | File | Why Study |
|-----------|------|-----------|
| SAT Optimizations | `src/solvers/cdcl/new_sat_solver.c` | Best-in-class SAT |
| Simplex | `src/solvers/simplex/simplex.c` | Efficient arithmetic |
| BV Solver | `src/solvers/bv/bvsolver.c` | Fast BV handling |
| Bit-Blaster | `src/solvers/bv/bit_blaster.c` | BV to SAT |
| Intervals | `src/solvers/bv/bv64_intervals.c` | BV abstraction |
| Architecture | `doc/smt_architecture.txt` | Design overview |

---

## 8. Performance Characteristics

Based on SMT-COMP results:

| Category | Yices 2 Performance | Notes |
|----------|---------------------|-------|
| QF_BV | **Excellent** | Fastest or top-3 |
| QF_LIA | **Excellent** | Often fastest |
| QF_LRA | **Excellent** | Very fast |
| QF_IDL/RDL | **Excellent** | Difference logic |
| QF_AUFBV | Good | Strong |
| QF_NRA | Good | With MC-SAT |
| Quantified | Poor | Limited support |
| Strings | N/A | Not supported |

---

## 9. Integration Notes

**License Warning:** Yices is GPLv3. Z4 cannot directly incorporate Yices code without also being GPL. However, we can:

1. **Study algorithms** - The techniques are not copyrightable
2. **Re-implement in Rust** - Clean-room implementation
3. **Use as reference** - For performance comparison

**Key Algorithms to Re-implement:**
1. SAT solver blocker optimization
2. Binary clause inlining
3. Simplex with theory propagation
4. BV interval abstraction
5. 64-bit BV fast paths

---

## 10. Key Papers

1. **Simplex for SMT:**
   - Dutertre & de Moura, CAV 2006 - "A Fast Linear-Arithmetic Solver for DPLL(T)"

2. **MC-SAT:**
   - de Moura & Jovanovic, VMCAI 2013 - "A Model-Constructing Satisfiability Calculus"

3. **Bit-Blasting:**
   - Multiple techniques from SAT competition winners
