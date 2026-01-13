# CVC5 SMT Solver - Research Analysis

**Repository:** https://github.com/cvc5/cvc5
**License:** BSD 3-Clause
**Language:** C++ (C++17)
**Developers:** Stanford / University of Iowa (Clark Barrett, Cesare Tinelli, et al.)
**Predecessor:** CVC4

---

## 1. Overview

CVC5 is the successor to CVC4, known for excellent string theory support, quantifier handling, and frequently winning SMT-COMP categories. It features clean architecture and is particularly strong on complex theories.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Public API                              │
│                (C, C++, Python, Java via cvc5.h)                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                      SolverEngine                               │
│               (Main orchestration layer)                        │
└───────┬─────────────────────────────────────────────────┬───────┘
        │                                                 │
┌───────▼───────────────────┐         ┌───────────────────▼───────┐
│       PropEngine          │         │      TheoryEngine         │
│  (SAT + CNF conversion)   │◄───────►│  (Theory combination)     │
└───────────────────────────┘         └───────────────────────────┘
        │                                       │
┌───────▼───────────────────┐   ┌───────────────▼───────────────────┐
│     SAT Solvers           │   │        Theory Solvers             │
│ CaDiCaL│MiniSat│CMS│Kissat│   │ UF│Arith│BV│Arrays│Strings│Quant│
└───────────────────────────┘   └───────────────────────────────────┘
                                        │
                                ┌───────▼───────┐
                                │ EqualityEngine │
                                │  (Shared EUF)  │
                                └───────────────┘
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/smt/` | SolverEngine and SMT orchestration |
| `src/prop/` | SAT solver abstraction |
| `src/theory/` | All theory solvers |
| `src/theory/strings/` | String theory (CVC5's strength) |
| `src/theory/quantifiers/` | Quantifier handling |
| `src/preprocessing/` | Preprocessing passes |
| `src/proof/` | Proof infrastructure |

---

## 3. Strengths

### 3.1 String Theory (Best in Class)

CVC5's string solver is exceptional:

**Architecture:**
```
theory_strings.h
├── CoreSolver       - Word equation solving (Liang et al.)
├── BaseSolver       - Constant handling
├── RegExpSolver     - Regex membership
├── ExtfSolver       - Extended functions (contains, replace)
├── ArraySolver      - Sequence-as-array reasoning
└── CodePointSolver  - Unicode handling
```

**Key Algorithms:**
1. **Normal Form Algorithm** (Liang et al. CAV 2014)
   - Computes canonical forms for string terms
   - Handles concatenation equations efficiently

2. **Length-Based Reasoning**
   - Tight integration with arithmetic
   - Length constraints propagate to string solver

3. **Derivative-Based Regex**
   - Brzozowski derivatives for membership
   - Efficient regex elimination

4. **Sequences Rewriter** (132KB of rules)
   - Aggressive simplification
   - Based on Reynolds et al. papers

### 3.2 Quantifier Handling (Best in Class)

CVC5 excels at quantified formulas:

**Modules:**
- **E-matching** - Efficient pattern-based instantiation
- **CEGQI** - Counterexample-Guided Quantifier Instantiation
  - Complete for linear real arithmetic
  - BV-specific instantiation
- **Conflict-Based Instantiation** - Learns from conflicts
- **MBQI** - Model-Based Quantifier Instantiation
- **Enumerative Instantiation** - For finite domains

### 3.3 SyGuS (Syntax-Guided Synthesis)

CVC5 is the leading SyGuS solver:
- CEGIS algorithm implementation
- Grammar-based enumeration
- I/O-based synthesis
- Integrated with quantifier handling

### 3.4 Proof Production

Comprehensive proof infrastructure:
- Multiple proof formats (LFSC, Alethe, DOT)
- Theory-specific proof rules
- Proof reconstruction support

### 3.5 Theory Coverage

Supports 14+ theories:
- Boolean, UF, Arithmetic (Linear/Nonlinear)
- Bitvectors, Arrays, Datatypes
- **Strings/Sequences** (excellent)
- Floating-point, Sets, Bags
- Separation Logic
- Finite Fields

---

## 4. Weaknesses

### 4.1 Raw Performance
- Often slower than Yices/Bitwuzla on pure BV
- Arithmetic not as optimized as Yices
- Overhead from generality

### 4.2 Memory Usage
- Can be memory-intensive
- Many intermediate data structures

### 4.3 Build Complexity
- Many optional dependencies
- CaDiCaL now required

### 4.4 Codebase Size
- Large codebase (~400K lines)
- Complex theory interaction

---

## 5. Key Algorithms

### 5.1 String Solving (Study These!)

**File:** `src/theory/strings/core_solver.h`

```cpp
// Key methods:
void checkFlatForm();   // Flat form inference
void checkNormalForm(); // Normal form computation
void checkMembership(); // Regex membership
```

**Normal Form Algorithm:**
1. Build equivalence classes of string terms
2. Compute "flat form" - list of components
3. Compare flat forms for equality/disequality
4. Generate splitting lemmas when needed

### 5.2 CEGQI for Arithmetic

**File:** `src/theory/quantifiers/cegqi/ceg_arith_instantiator.cpp`

- Projects quantified variables to bounds
- Complete for linear real arithmetic
- Handles strict/non-strict inequalities

### 5.3 E-matching with Code Trees

**File:** `src/theory/quantifiers/ematching/`

- Compiles patterns to matching automata
- Efficient multi-pattern matching
- Relevancy-based filtering

---

## 6. What Z4 Should Adopt from CVC5

### Must Have
1. **String Solver Architecture** - Modular design with sub-solvers
2. **Normal Form Algorithm** - For word equation solving
3. **Length-Arithmetic Integration** - Critical for strings
4. **CEGQI** - For complete quantifier handling in arithmetic

### Should Have
1. **Regex Derivatives** - Elegant approach to membership
2. **Conflict-Based Instantiation** - Improves quantifier performance
3. **SyGuS Integration** - For synthesis applications
4. **Proof Infrastructure** - Multiple output formats

### Could Adopt
1. **Sequences as Arrays** - Alternative representation
2. **Separation Logic** - Specialized memory reasoning
3. **Finite Fields** - Cryptography applications

---

## 7. Key Files for Study

| Component | File | Why Study |
|-----------|------|-----------|
| String Core | `src/theory/strings/core_solver.cpp` | Word equation algorithm |
| String Rewriter | `src/theory/strings/sequences_rewriter.cpp` | Rewrite rules |
| Regex | `src/theory/strings/regexp_solver.cpp` | Membership checking |
| CEGQI | `src/theory/quantifiers/cegqi/ceg_instantiator.cpp` | Quantifier instantiation |
| E-matching | `src/theory/quantifiers/ematching/inst_match_generator.cpp` | Pattern matching |
| Theory Base | `src/theory/theory.h` | Theory interface |

---

## 8. Performance Characteristics

Based on SMT-COMP results:

| Category | CVC5 Performance | Notes |
|----------|------------------|-------|
| QF_S (Strings) | **Excellent** | Often wins |
| QF_SLIA | **Excellent** | Strings + arithmetic |
| UFLIA/UFLRA | **Excellent** | Strong quantifiers |
| BV_* | Good | But not fastest |
| QF_LIA/LRA | Good | Competitive |
| SyGuS | **Excellent** | Leading solver |

---

## 9. Integration Notes

CVC5's BSD license allows integration. Key integration points:

1. **String Solving:**
   - Study the normal form algorithm in detail
   - The modular sub-solver design is excellent
   - Length-arithmetic bridge is critical

2. **Quantifiers:**
   - CEGQI is well-documented in papers
   - E-matching code tree compilation is sophisticated
   - Consider MBQI for finite domains

3. **API Design:**
   - CVC5's API is clean and modern
   - Uses RAII and modern C++ patterns

---

## 10. Key Papers

Essential reading for implementing CVC5's techniques:

1. **Strings:**
   - Liang et al., CAV 2014 - "A DPLL(T) Theory Solver for a Theory of Strings and Regular Expressions"
   - Reynolds et al., CAV 2017 - "Scaling Up DPLL(T) String Solvers Using Context-Dependent Simplification"

2. **Quantifiers:**
   - Reynolds et al., IJCAR 2014 - "Counterexample-Guided Quantifier Instantiation"
   - Ge & de Moura, CAV 2009 - "Complete Instantiation for Quantified Formulas"

3. **SyGuS:**
   - Reynolds et al., TACAS 2019 - "Syntax-Guided Synthesis"
