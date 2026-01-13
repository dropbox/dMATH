# Z4 Competitive Solver Analysis Report

**Date**: 2026-01-02
**Purpose**: Technical analysis of CaDiCaL, CryptoMiniSat, and CVC5 for porting to Z4
**Status**: Complete

---

## Executive Summary

This report analyzes three world-class SAT/SMT solvers to inform Z4's development roadmap:

| Solver | Type | LOC | License | Key Strengths |
|--------|------|-----|---------|---------------|
| **CaDiCaL** | SAT | 55K | MIT | Clean architecture, competition-winning techniques |
| **CryptoMiniSat** | SAT | 59K | MIT | XOR handling, ML clause prediction, SLS integration |
| **CVC5** | SMT | 345K | BSD | Strings, quantifiers, proof production, theory combination |

**Z4 Current State**: 17K LOC SAT solver with CDCL, inprocessing, proofs. Beats Z3 on QF_LIA, QF_UF, QF_BV, QF_LRA.

**Goal**: Incorporate best techniques from all three to beat both Z3 and CVC5.

---

## Part 1: CaDiCaL Analysis

### 1.1 Architecture Summary

CaDiCaL uses a monolithic `Internal` class containing all solver state:

```
Internal {
  vals[]       - Assignment array (indexed by literal)
  trail[]      - Assignment stack
  control[]    - Decision level markers
  clauses[]    - All clause pointers
  wtab[]       - Watch lists by literal
  scores       - VSIDS heap
  queue/links  - VMTF doubly-linked list
}
```

**Key Design Choices**:
- Flexible array member for clause literals (no pointer indirection)
- Blocking literals in watch entries (skip clause access when satisfied)
- Dual value arrays (both +lit and -lit have entries)
- Arena allocator with GC that sorts by first watched literal

### 1.2 Techniques to Port

| Technique | Z4 Status | Priority | Effort |
|-----------|-----------|----------|--------|
| 2-Watched Literals | Implemented | - | - |
| VSIDS/EVSIDS | Implemented | - | - |
| VMTF | **Missing** | High | Medium |
| 1-UIP Learning | Implemented | - | - |
| Chronological Backtracking | Implemented | - | - |
| Glucose Restarts | Implemented | - | - |
| Luby Restarts | Implemented | - | - |
| Tier-based Clause DB | Implemented | - | - |
| Vivification | Implemented | - | - |
| BVE | Implemented | - | - |
| BCE | Implemented | - | - |
| Failed Literal Probing | Implemented | - | - |
| On-the-fly Strengthening | **Missing** | High | Low |
| Reason-side Bumping | **Missing** | High | Low |
| Target Phase Saving | **Missing** | Medium | Low |
| SAT Sweeping | Implemented | - | - |
| Internal LRAT Generation | **Missing** | Medium | Medium |
| Mode Switching (stable/focused) | **Missing** | High | Medium |
| Position Saving (Gent) | **Missing** | Medium | Low |

### 1.3 CaDiCaL Performance Tricks

1. **Branch-less swap**: `other = lits[0] ^ lits[1] ^ lit`
2. **Position saving**: Remember last replacement position per clause
3. **Binary clause optimization**: Never access clause body during propagation
4. **Watch list compaction**: Lazy two-pointer technique
5. **Cache prefetching**: `__builtin_prefetch` after assignment
6. **Tick-based limits**: Memory-proportional cost model

### 1.4 Key Files Reference

| File | Purpose |
|------|---------|
| `internal.hpp` | Main state container |
| `propagate.cpp` | Unit propagation (585 LOC) |
| `analyze.cpp` | Conflict analysis (1,360 LOC) |
| `decide.cpp` | VMTF/VSIDS (347 LOC) |
| `reduce.cpp` | Clause DB reduction (279 LOC) |
| `vivify.cpp` | Vivification (1,893 LOC) |
| `restart.cpp` | Restart strategies |
| `reluctant.hpp` | Luby sequence generator |

---

## Part 2: CryptoMiniSat Analysis

### 2.1 Architecture Summary

CryptoMiniSat uses layered inheritance:

```
CNF (clause storage, variables)
  └── PropEngine (2-watched propagation)
        └── HyperEngine (hyper-binary resolution)
              └── Searcher (CDCL search)
                    └── Solver (top-level interface)
```

**Key Design Choices**:
- Custom arena allocator (`ClauseAllocator`)
- Watch entries encode type (clause/binary/BNN/idx)
- Binary clauses stored inline in watch lists
- Template-based inprocessing to avoid branching

### 2.2 Techniques to Port

| Technique | Z4 Status | Priority | Effort |
|-----------|-----------|----------|--------|
| **XOR Recovery** | **Missing** | High | High |
| **Gaussian Elimination** | **Missing** | High | High |
| **ML Clause Prediction** | **Missing** | Low | High |
| **CCNR (SLS) Integration** | **Missing** | Medium | High |
| **Cardinality Detection** | **Missing** | Medium | Medium |
| **Backbone Detection** | **Missing** | Medium | Medium |
| Variable Replacement (equiv) | Partial | Medium | Medium |
| Distillation | Implemented (vivify) | - | - |
| FRAT Proofs | **Missing** | Low | Medium |

### 2.3 XOR Handling Deep Dive

CryptoMiniSat's killer feature is native XOR clause handling:

**Phase 1: XOR Recovery** (`xorfinder.cpp`)
```
Input: CNF encoding of XOR (2^n clauses for n-variable XOR)
Output: Native XOR constraint
Method: Pattern matching using clause abstraction
```

**Phase 2: Gaussian Elimination** (`gaussian.cpp`)
```
Data: Packed bit matrix representation
Operation: Gauss-Jordan elimination during search
Innovation: Incremental row elimination on variable assignment
```

**Watch Integration**:
- XOR clauses watched via `GaussWatched` entries
- When watched variable assigned, perform row operations
- Can propagate units or detect conflicts

### 2.4 ML Clause Prediction

CryptoMiniSat integrates XGBoost/LightGBM for clause quality:

**22-Feature Vector per Clause**:
- Size, glue, activity
- UIP1 usage statistics
- Propagation counts
- Age, solver time
- Relative ranking

**Prediction Types**:
- `short_pred`: Short-term usefulness
- `long_pred`: Long-term usefulness
- `forever_pred`: Permanent retention

### 2.5 Key Files Reference

| File | Purpose |
|------|---------|
| `solver.cpp` | Top-level API (4K LOC) |
| `searcher.cpp` | CDCL search (3.8K LOC) |
| `gaussian.cpp` | XOR handling (1.6K LOC) |
| `xorfinder.cpp` | XOR recovery |
| `cl_predictors_*.cpp` | ML integration |
| `ccnr*.cpp` | SLS integration |
| `reducedb.cpp` | Clause DB with ML |

---

## Part 3: CVC5 Analysis

### 3.1 Architecture Summary

CVC5 follows DPLL(T) with clean separation:

```
SolverEngine (top-level API)
  └── SmtSolver (solving orchestration)
        ├── PropEngine (SAT + CNF)
        │     ├── CDCLTSatSolver (CaDiCaL/MiniSat)
        │     ├── TheoryProxy (SAT-theory bridge)
        │     └── CnfStream (Tseitin encoding)
        └── TheoryEngine (theory coordination)
              ├── Theory instances (arith, strings, bv, ...)
              ├── CombinationEngine (care graph)
              └── EqualityEngine (congruence closure)
```

### 3.2 Techniques to Port

| Technique | Z4 Status | Priority | Effort |
|-----------|-----------|----------|--------|
| **CDCL(T) Propagator** | Partial | Critical | High |
| **Care Graph Combination** | **Missing** | High | High |
| **Simplex (LRA)** | Implemented | - | - |
| **Diophantine Solver (LIA)** | Implemented | - | - |
| **String Word Equations** | **Missing** | High | Very High |
| **E-Matching** | **Missing** | High | High |
| **MBQI** | **Missing** | Medium | High |
| **Bit-blasting** | Implemented | - | - |
| **Array Axiom Instantiation** | **Missing** | Medium | Medium |
| **Alethe Proofs** | **Missing** | Medium | High |
| **72 Preprocessing Passes** | Partial | Medium | High |

### 3.3 CDCL(T) Integration Pattern

CVC5's `CadicalPropagator` implements the key callbacks:

```cpp
class CadicalPropagator : public ExternalPropagator {
  // SAT assigns variable - notify theories
  void notify_assignment(const vector<int>& lits);

  // After BCP - check theories, get propagations
  int cb_propagate();

  // Complete model found - final theory check
  bool cb_check_found_model(const vector<int>& model);

  // Theory-guided decision
  int cb_decide();

  // Explain theory propagation as clause
  int cb_add_reason_clause_lit(int propagated_lit);
};
```

### 3.4 Theory Combination (Care Graph)

For theories sharing terms:
1. Each theory reports pairs of terms it cares about
2. If theories disagree on equality, add splitting lemma
3. Repeat until fixed point

### 3.5 String Solver Components

CVC5's string solver (98K LOC) uses multiple sub-solvers:

| Component | Purpose |
|-----------|---------|
| `BaseSolver` | Basic constraints |
| `CoreSolver` | Word equation solving |
| `ExtfSolver` | Extended functions (contains, replace) |
| `RegexpSolver` | Regular expression membership |
| `ArithEntail` | Length reasoning |

### 3.6 Key Files Reference

| File | Purpose |
|------|---------|
| `theory_engine.cpp` | Theory coordination (76K LOC) |
| `cdclt_propagator.h` | DPLL(T) interface (386 LOC) |
| `theory_arith_private.cpp` | Arithmetic (170K LOC) |
| `core_solver.cpp` | String solving (98K LOC) |
| `combination_care_graph.cpp` | Theory combination |
| `dual_simplex.cpp` | LRA simplex |
| `dio_solver.cpp` | LIA Diophantine |

---

## Part 4: Comparative Analysis

### 4.1 SAT Solver Feature Matrix

| Feature | CaDiCaL | CryptoMiniSat | Z4 |
|---------|---------|---------------|-----|
| 2-Watched Literals | Yes | Yes | Yes |
| VSIDS | Yes | Yes | Yes |
| VMTF | Yes | Yes | **No** |
| Chronological BT | Yes | Yes | Yes |
| Vivification | Yes | Yes | Yes |
| BVE/BCE | Yes | Yes | Yes |
| Failed Literal Probing | Yes | Yes | Yes |
| Gaussian Elimination | No | **Yes** | **No** |
| ML Clause Prediction | No | **Yes** | **No** |
| SLS Integration | No | **Yes** | Partial |
| XOR Detection | No | **Yes** | **No** |
| SAT Sweeping | Yes | No | Yes |
| Gate Extraction | Yes | Yes | Yes |
| Internal LRAT | Yes | No | **No** |
| Mode Switching | Yes | Yes | **No** |

### 4.2 SMT Feature Matrix

| Feature | CVC5 | Z3 | Z4 |
|---------|------|-----|-----|
| QF_LIA | Yes | Yes | Yes (1.5x faster) |
| QF_LRA | Yes | Yes | Yes (1.37x faster) |
| QF_BV | Yes | Yes | Yes (1.19x faster) |
| QF_UF | Yes | Yes | Yes (1.79x faster) |
| QF_SLIA (Strings) | **Best** | Yes | **Minimal** |
| Quantifiers | **Best** | Yes | **No** |
| CHC | Yes | Yes | Yes (improving) |
| Proof Production | **Best** | Yes | DRAT only |
| Theory Combination | Care Graph | Nelson-Oppen | **Basic** |

### 4.3 Performance Characteristics

| Solver | Startup | Small Problems | Large Problems |
|--------|---------|----------------|----------------|
| CaDiCaL | Fast | Excellent | Excellent |
| CryptoMiniSat | Medium | Good | Excellent (XOR) |
| CVC5 | Slow | Good | Variable |
| Z3 | Medium | Good | Good |
| **Z4** | **Very Fast** | **Excellent** | Good |

---

## Part 5: Test Suite Inventory

### 5.1 Available Test Resources

| Source | Type | Count | Location |
|--------|------|-------|----------|
| CaDiCaL | CNF | 88 | `reference/cadical/test/cnf/` |
| CaDiCaL | Trace | Many | `reference/cadical/test/trace/` |
| CryptoMiniSat | CNF | 12 | `reference/cryptominisat/tests/cnf-files/` |
| CryptoMiniSat | Unit | 20+ | `reference/cryptominisat/tests/*.cpp` |
| CVC5 | SMT2 | 3,994 | `reference/cvc5/test/regress/` |
| Z4 Current | Mixed | ~500 | `benchmarks/` |

### 5.2 Test Categories to Add

| Category | Source | Priority |
|----------|--------|----------|
| SAT Competition | SAT-COMP archives | High |
| SMT-COMP QF_* | SMT-LIB | High |
| CHC-COMP | CHC-COMP archives | High |
| XOR-heavy | CryptoMiniSat tests | Medium |
| String benchmarks | CVC5 regress/strings | High |
| Quantifier benchmarks | CVC5 regress/quantifiers | Medium |
| Proof checking | drat-trim corpus | High |

---

## Part 6: Gap Analysis

### 6.1 Critical Gaps (Must Fix)

| Gap | Impact | Effort | Source Reference |
|-----|--------|--------|------------------|
| CDCL(T) proper integration | Theory performance | High | CVC5 `cdclt_propagator.h` |
| String theory | SMT-COMP competitiveness | Very High | CVC5 `theory/strings/` |
| Quantifier instantiation | Completeness | Very High | CVC5 `theory/quantifiers/` |
| Theory combination | Multi-theory correctness | High | CVC5 `combination_care_graph.cpp` |

### 6.2 High-Value Gaps

| Gap | Impact | Effort | Source Reference |
|-----|--------|--------|------------------|
| VMTF heuristic | SAT performance | Medium | CaDiCaL `queue.hpp` |
| Mode switching | Robustness | Medium | CaDiCaL `decide.cpp` |
| XOR handling | Crypto benchmarks | High | CryptoMiniSat `gaussian.cpp` |
| On-the-fly strengthening | Learning quality | Low | CaDiCaL `analyze.cpp` |
| Reason-side bumping | Heuristic quality | Low | CaDiCaL `analyze.cpp` |

### 6.3 Nice-to-Have Gaps

| Gap | Impact | Effort | Source Reference |
|-----|--------|--------|------------------|
| ML clause prediction | Marginal | High | CryptoMiniSat `cl_predictors_*.cpp` |
| CCNR integration | Hybrid solving | High | CryptoMiniSat `ccnr*.cpp` |
| Alethe proofs | Proof checking | High | CVC5 `proof/alethe/` |
| 72 preprocessing passes | Various | Very High | CVC5 `preprocessing/passes/` |

---

## Appendix A: Code Size Comparison

```
CaDiCaL:       55,000 LOC C++
CryptoMiniSat: 59,000 LOC C++
CVC5:         345,000 LOC C++
Z3:           800,000 LOC C++ (estimated)
Z4:            17,000 LOC Rust (SAT only)
               40,000 LOC Rust (total, estimated)
```

## Appendix B: License Summary

| Solver | License | Can Port Code? |
|--------|---------|----------------|
| CaDiCaL | MIT | Yes, with attribution |
| CryptoMiniSat | MIT | Yes, with attribution |
| CVC5 | BSD-3 | Yes, with attribution |
| Z3 | MIT | Yes, with attribution |
| Yices | GPL | **Papers only** |
| MathSAT | Proprietary | **Papers only** |

## Appendix C: Key Papers

| Topic | Paper | Implemented In |
|-------|-------|----------------|
| CDCL | MiniSat (Een & Sorensson 2003) | All |
| VSIDS | Chaff (Moskewicz 2001) | All |
| Glucose Restarts | Audemard & Simon 2009 | CaDiCaL, Z4 |
| Chronological BT | Nadel & Ryvchin 2018 | CaDiCaL, Z4 |
| XOR Gaussian | Han & Jiang CAV 2012 | CryptoMiniSat |
| DPLL(T) | Nieuwenhuis et al. 2006 | CVC5, Z3 |
| Care Graph | Tinelli & Zarba 2005 | CVC5 |
| String Solving | CVC4 strings (Liang et al. 2014) | CVC5 |
