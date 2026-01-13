# Z4 Academic References

This document indexes the academic literature underlying Z4's design and implementation. Papers are organized by topic with arXiv links where available.

---

## CRITICAL: Implementation-From-Papers Strategy

Due to licensing restrictions, Z4 cannot port code from **Yices 2** (GPL) or **MathSAT 5** (Proprietary). However, these solvers have the best performance in key areas. Our strategy:

| Solver | License | Best At | Strategy |
|--------|---------|---------|----------|
| **Yices 2** | GPL | SAT, Arithmetic, BV speed | Study source → Write report → AI generates clean-room implementation |
| **MathSAT 5** | Proprietary | Interpolation, AllSAT, OMT | Implement from published papers only |

### Yices 2 Key Papers (Must Implement)

These papers document the algorithms we need to re-implement:

1. **Yices 2.2** - Dutertre, CAV 2014
   - System overview, architecture, supported theories
   - [PDF](http://yices.csl.sri.com/papers/yices-06.pdf)

2. **A Fast Linear-Arithmetic Solver for DPLL(T)** - Dutertre & de Moura, CAV 2006
   - Core simplex algorithm with theory propagation
   - **CRITICAL** for z4-lra

3. **Solving Bitvectors with MCSAT** - Graham-Lengrand, Jovanovic, Dutertre
   - arXiv: [2004.07940](https://arxiv.org/abs/2004.07940)
   - Decision procedure for fixed-sized bitvectors

4. **MCSat-based Finite Field Reasoning in Yices2** - arXiv: [2402.17927](https://arxiv.org/abs/2402.17927)
   - MCSAT for non-linear polynomial systems

5. **A Model-Constructing Satisfiability Calculus** - de Moura & Jovanovic, VMCAI 2013
   - MCSAT framework foundations

### MathSAT 5 Key Papers (Must Implement)

1. **The MathSAT5 SMT Solver** - Cimatti, Griggio, Schaafsma, Sebastiani, TACAS 2013
   - Architecture and design decisions

2. **Efficient Generation of Craig Interpolants in SMT** - Cimatti, Griggio, Sebastiani
   - arXiv: [0906.4492](https://arxiv.org/abs/0906.4492)
   - **CRITICAL** for interpolation support

3. **Efficient Interpolant Generation in SMT(LA(Z))** - Griggio, Le, Sebastiani
   - arXiv: [1010.4422](https://arxiv.org/abs/1010.4422)
   - Integer arithmetic interpolation

4. **OptiMathSAT** Papers:
   - Optimization Modulo Theories foundations
   - arXiv: [1905.02838](https://arxiv.org/abs/1905.02838) - BV/FP optimization
   - arXiv: [1702.02385](https://arxiv.org/abs/1702.02385) - MaxSMT and sorting networks
   - arXiv: [2502.19963](https://arxiv.org/abs/2502.19963) - Partial-assignment enumeration

5. **Disjoint Projected Enumeration for SAT and SMT** - Spallitta, Sebastiani, Biere
   - arXiv: [2410.18707](https://arxiv.org/abs/2410.18707)
   - AllSAT/AllSMT without blocking clauses

### Source Code Study Process (Yices 2)

Since Yices is GPL, we use a clean-room implementation process:

```
Phase 1: AI Worker A studies Yices source code
         ↓
Phase 2: AI Worker A writes detailed algorithm report
         (NO CODE, only algorithm descriptions)
         ↓
Phase 3: AI Worker B reads report (never sees Yices source)
         ↓
Phase 4: AI Worker B implements in Rust from algorithm description
```

This ensures no GPL-contaminated code enters Z4.

---

## Table of Contents

0. [Implementation-From-Papers Strategy](#critical-implementation-from-papers-strategy)
1. [Foundational SAT Solving](#1-foundational-sat-solving)
2. [CDCL and Conflict Analysis](#2-cdcl-and-conflict-analysis)
3. [Modern SAT Solvers](#3-modern-sat-solvers)
4. [SMT Architecture (DPLL(T))](#4-smt-architecture-dpllt)
5. [Theory Combination](#5-theory-combination)
6. [Linear Arithmetic](#6-linear-arithmetic)
7. [Bitvector Theory](#7-bitvector-theory)
8. [Array Theory](#8-array-theory)
9. [String Theory](#9-string-theory)
10. [Quantifier Handling](#10-quantifier-handling)
11. [Craig Interpolation](#11-craig-interpolation)
12. [Proof Production](#12-proof-production)
13. [SMT Solver Systems](#13-smt-solver-systems)
14. [Machine Learning for SAT/SMT](#14-machine-learning-for-satsmt)
15. [Verified SAT/SMT Solvers](#15-verified-satsmt-solvers)
16. [Formal Verification Tools for Rust](#16-formal-verification-tools-for-rust)

---

## 1. Foundational SAT Solving

### MiniSat: A SAT Solver with Conflict-Clause Minimization
**Authors:** Niklas Een, Niklas Sorensson
**Venue:** SAT 2003
**Abstract:** Introduces the MiniSat solver with efficient conflict clause minimization and the two-watched literal scheme that became standard in modern SAT solvers. Foundational paper for CDCL implementation.
**Relevance:** Core reference for z4-sat implementation.

### Chaff: Engineering an Efficient SAT Solver
**Authors:** Matthew W. Moskewicz, Conor F. Madigan, Ying Zhao, Lintao Zhang, Sharad Malik
**Venue:** DAC 2001
**Abstract:** Introduces VSIDS (Variable State Independent Decaying Sum) heuristic and efficient BCP (Boolean Constraint Propagation) techniques.
**Relevance:** VSIDS implementation in z4-sat.

### GRASP: A Search Algorithm for Propositional Satisfiability
**Authors:** Joao Marques-Silva, Karem Sakallah
**Venue:** IEEE Transactions on Computers, 1999
**Abstract:** Introduces conflict-driven clause learning (CDCL) with non-chronological backtracking.
**Relevance:** Foundational CDCL algorithm.

---

## 2. CDCL and Conflict Analysis

### Extended Resolution Clause Learning via Dual Implication Points
**arXiv:** [2406.14190](https://arxiv.org/abs/2406.14190)
**Authors:** Sam Buss, et al.
**Abstract:** Presents a new extended resolution clause learning (ERCL) algorithm using Dual Implication Points in a CDCL SAT solver, enabling learning of extended resolution clauses during search.
**Relevance:** Advanced conflict analysis techniques.

### Clause Vivification by Unit Propagation in CDCL SAT Solvers
**arXiv:** [1807.11061](https://arxiv.org/abs/1807.11061)
**Authors:** Chu-Min Li, et al.
**Abstract:** Proposes a clause vivification approach that eliminates redundant literals by applying unit propagation, improving learned clause quality.
**Relevance:** Clause minimization techniques.

### Too Much Information: Why CDCL Solvers Need to Forget Learned Clauses
**arXiv:** [2202.01030](https://arxiv.org/abs/2202.01030)
**Authors:** Tom Kruger, et al.
**Abstract:** Examines how clause learning without deletion can deteriorate solver performance, providing theoretical foundations for clause deletion policies.
**Relevance:** Clause deletion strategy design.

### On Improving the Backjump Level in PB Solvers
**arXiv:** [2107.13085](https://arxiv.org/abs/2107.13085)
**Author:** Romain Wallon
**Abstract:** Studies optimality of first unique implication point (1-UIP) in pseudo-Boolean solvers, with implications for SAT conflict analysis.
**Relevance:** 1-UIP conflict analysis optimization.

### Towards a Complexity-theoretic Understanding of Restarts in SAT Solvers
**arXiv:** [2003.02323](https://arxiv.org/abs/2003.02323)
**Authors:** Chunxiao Li, Noah Fleming, Marc Vinyals, Toniann Pitassi, Vijay Ganesh
**Abstract:** Analyzes computational power of CDCL SAT solvers with VSIDS branching and restarts, providing theoretical foundations for restart strategies.
**Relevance:** Restart policy design.

---

## 3. Modern SAT Solvers

### CaDiCaL, Kissat, Paracooba, Plingeling and Treengeling
**Authors:** Armin Biere
**Venue:** SAT Competition 2020
**Abstract:** System descriptions of the CaDiCaL family of SAT solvers, featuring modern techniques like chronological backtracking, in-processing, and LBD-based clause management.
**Relevance:** State-of-the-art SAT techniques for z4-sat.

### Revisiting DRUP-based Interpolants with CaDiCaL 2.0
**arXiv:** [2501.02608](https://arxiv.org/abs/2501.02608)
**Authors:** Basel Khouri, Yakir Vizel
**Abstract:** Implementation of DRUP-based interpolants in CaDiCaL 2.0, evaluated in bit-level model checking context.
**Relevance:** Proof production and interpolation.

### Life Span of SAT Techniques
**arXiv:** [2402.01202](https://arxiv.org/abs/2402.01202)
**Authors:** Mathias Fleury, Daniela Kaufmann
**Abstract:** Studies effectiveness of four CaDiCaL features across SAT Competition benchmarks, testing hypotheses about technique longevity.
**Relevance:** Feature selection for z4-sat.

### Lazy Reimplication in Chronological Backtracking
**arXiv:** [2501.07457](https://arxiv.org/abs/2501.07457)
**Authors:** Robin Coutelier, Mathias Fleury, Laura Kovacs
**Abstract:** Introduces lazy reimplication procedure for chronological backtracking in SAT solvers.
**Relevance:** Advanced backtracking techniques.

### Deeply Optimizing the SAT Solver for the IC3 Algorithm
**arXiv:** [2501.18612](https://arxiv.org/abs/2501.18612)
**Authors:** Yuheng Su, et al.
**Abstract:** Optimizations for SAT solver in IC3 model checking, achieving 3.61x speedup over MiniSat-based implementation.
**Relevance:** SAT solver optimization techniques.

---

## 4. SMT Architecture (DPLL(T))

### Z3: An Efficient SMT Solver
**Authors:** Leonardo de Moura, Nikolaj Bjorner
**Venue:** TACAS 2008
**Abstract:** Introduces Z3's architecture including DPLL(T) integration, theory plugins, and tactic framework.
**Relevance:** Core SMT architecture reference.

### A Survey of Satisfiability Modulo Theory
**arXiv:** [1606.04786](https://arxiv.org/abs/1606.04786)
**Author:** David Monniaux
**Abstract:** Comprehensive survey explaining the combination of propositional satisfiability and decision procedures known as DPLL(T).
**Relevance:** SMT fundamentals.

### Generalized Optimization Modulo Theories
**arXiv:** [2404.16122](https://arxiv.org/abs/2404.16122)
**Authors:** Nestan Tsiskaridze, Clark Barrett, Cesare Tinelli
**Abstract:** Extension of SMT with theory-agnostic optimization calculus inspired by DPLL(T) approach.
**Relevance:** Optimization modulo theories.

### A DPLL(T) Framework for Verifying Deep Neural Networks
**arXiv:** [2307.10266](https://arxiv.org/abs/2307.10266)
**Authors:** Hai Duong, ThanhVu Nguyen, Matthew Dwyer
**Abstract:** NeuralSAT adapts DPLL(T) algorithm for neural network verification.
**Relevance:** Novel DPLL(T) applications.

---

## 5. Theory Combination

### On the Convexity of a Fragment of Pure Set Theory with Applications within a Nelson-Oppen Framework
**arXiv:** [2109.08309](https://arxiv.org/abs/2109.08309)
**Authors:** Domenico Cantone, Andrea De Domenico, Pietro Maugeri
**Abstract:** Studies SMT solvers based on variants of the Nelson-Oppen combination method.
**Relevance:** Theory combination implementation.

### Being Polite is Not Enough (and Other Limits of Theory Combination)
**arXiv:** [2505.04870](https://arxiv.org/abs/2505.04870)
**Authors:** Guilherme V. Toledo, Benjamin Przybocki, Yoni Zohar
**Abstract:** Explores limitations of different theory combination methods including Nelson-Oppen.
**Relevance:** Understanding theory combination boundaries.

### Combining Combination Properties: Minimal Models
**arXiv:** [2405.01478](https://arxiv.org/abs/2405.01478)
**Authors:** Guilherme Vicentin de Toledo, Yoni Zohar
**Abstract:** Analyzes properties related to theory combination in SMT.
**Relevance:** Theory combination properties.

### Predicate Abstraction via Symbolic Decision Procedures
**arXiv:** [cs/0612003](https://arxiv.org/abs/cs/0612003)
**Authors:** Shuvendu K. Lahiri, Thomas Ball, Byron Cook
**Abstract:** Constructs symbolic decision procedures using extension of Nelson-Oppen combination method.
**Relevance:** Theory combination for predicate abstraction.

---

## 6. Linear Arithmetic

### A Fast Linear-Arithmetic Solver for DPLL(T)
**Authors:** Bruno Dutertre, Leonardo de Moura
**Venue:** CAV 2006
**Abstract:** Efficient simplex-based linear arithmetic solver designed for SMT integration, with theory propagation from tableau bounds.
**Relevance:** Core reference for z4-lra implementation.

### FMplex: A Novel Method for Solving Linear Real Arithmetic Problems
**arXiv:** [2310.00995](https://arxiv.org/abs/2310.00995)
**Authors:** Jasper Nalbach, Valentin Promies, Erika Abraham, Paul Kobialka
**Abstract:** Novel quantifier elimination method for conjunctions of linear real arithmetic constraints.
**Relevance:** Alternative arithmetic solving approaches.

### FMplex: Exploring a Bridge between Fourier-Motzkin and Simplex
**arXiv:** [2309.03138](https://arxiv.org/abs/2309.03138)
**Authors:** Valentin Promies, Jasper Nalbach, Erika Abraham, Paul Kobialka
**Abstract:** Quantifier elimination method combining Fourier-Motzkin and simplex techniques.
**Relevance:** Hybrid arithmetic approaches.

### On Using Floating-Point Computations to Help an Exact Linear Arithmetic Decision Procedure
**arXiv:** [0904.3525](https://arxiv.org/abs/0904.3525)
**Author:** David Monniaux
**Abstract:** Using floating-point computations to accelerate exact linear arithmetic decision procedures in SMT.
**Relevance:** Arithmetic solver optimization.

---

## 7. Bitvector Theory

### Bitwuzla at the SMT-COMP 2020
**arXiv:** [2006.01621](https://arxiv.org/abs/2006.01621)
**Authors:** Aina Niemetz, Mathias Preiner
**Abstract:** System description of Bitwuzla SMT solver for bitvectors, floating-points, arrays and uninterpreted functions.
**Relevance:** Primary reference for z4-bv design.

### Solving Bitvectors with MCSAT: Explanations from Bits and Pieces
**arXiv:** [2004.07940](https://arxiv.org/abs/2004.07940)
**Authors:** Stephane Graham-Lengrand, Dejan Jovanovic, Bruno Dutertre
**Abstract:** Decision procedure for fixed-sized bitvectors in SMT using MCSAT framework.
**Relevance:** Alternative BV solving approach.

### Algebraic Reasoning Meets Automata in Solving Linear Integer Arithmetic
**arXiv:** [2403.18995](https://arxiv.org/abs/2403.18995)
**Authors:** Peter Habermehl, Vojtěch Havlena, et al.
**Abstract:** Combines automata-based approach (numbers as bitvectors) with algebraic approaches.
**Relevance:** Hybrid BV/arithmetic techniques.

### Source-Level Bitwise Branching for Temporal Verification
**arXiv:** [2111.02938](https://arxiv.org/abs/2111.02938)
**Authors:** Yuandong Cyrus Liu, Ton-Chanh Le, Eric Koskinen
**Abstract:** Verification tools for programs with bitvector operations using SMT techniques.
**Relevance:** BV applications in verification.

---

## 8. Array Theory

### Generalized, Efficient Array Decision Procedures
**Authors:** Leonardo de Moura, Nikolaj Bjorner
**Venue:** TACAS 2009
**Abstract:** Array decision procedure with lazy axiom instantiation based on weak equivalence.
**Relevance:** Core reference for z4-arrays.

### Weakly Equivalent Arrays
**arXiv:** [1405.6939](https://arxiv.org/abs/1405.6939)
**Authors:** Jurgen Christ, Jochen Hoenicke
**Abstract:** Algorithm that lazily instantiates lemmas based on weak equivalence classes for array theory.
**Relevance:** Lazy array lemma generation.

### Encoding and Reasoning About Arrays in Set Theory
**arXiv:** [2508.11447](https://arxiv.org/abs/2508.11447)
**Authors:** Maximiliano Cristia, Gianfranco Rossi
**Abstract:** Encodes arrays as functions represented by sets, defining a fragment of set theory for array reasoning.
**Relevance:** Alternative array encodings.

### Array Folds Logic
**arXiv:** [1603.06850](https://arxiv.org/abs/1603.06850)
**Authors:** Przemyslaw Daca, Thomas A. Henzinger, Andrey Kupriyanov
**Abstract:** Extension to quantifier-free theory of integer arrays allowing counting properties, with PSPACE-complete satisfiability.
**Relevance:** Extended array theories.

---

## 9. String Theory

### A DPLL(T) Theory Solver for a Theory of Strings and Regular Expressions
**Authors:** Tianyi Liang, Andrew Reynolds, Cesare Tinelli, Clark Barrett, Morgan Deters
**Venue:** CAV 2014
**Abstract:** Normal form algorithm for string equations with tight arithmetic integration.
**Relevance:** Core reference for z4-strings.

### Z3str3: A String Solver with Theory-aware Branching
**arXiv:** [1704.07935](https://arxiv.org/abs/1704.07935)
**Authors:** Murphy Berzish, Yunhui Zheng, Vijay Ganesh
**Abstract:** Novel string SMT solver using theory-aware branching in DPLL(T) architecture.
**Relevance:** String solving optimizations.

### An SMT Solver for Regular Expressions and Linear Arithmetic over String Length
**arXiv:** [2010.07253](https://arxiv.org/abs/2010.07253)
**Authors:** Murphy Berzish, et al.
**Abstract:** Length-aware solving algorithm for quantifier-free theory over regex membership and linear arithmetic over string length.
**Relevance:** String-arithmetic integration.

### Decision Procedure for A Theory of String Sequences
**arXiv:** [2509.00948](https://arxiv.org/abs/2509.00948)
**Authors:** Denghang Hu, Taolue Chen, Philipp Rummer, Fu Song, Zhilin Wu
**Abstract:** Theory of string sequences with decidability results for straight-line fragment.
**Relevance:** Extended string theories.

### Undecidability of a Theory of Strings, Linear Arithmetic over Length, and String-Number Conversion
**arXiv:** [1605.09442](https://arxiv.org/abs/1605.09442)
**Authors:** Vijay Ganesh, Murphy Berzish
**Abstract:** Studies undecidability boundaries for string theories with length and conversion operations.
**Relevance:** Understanding string theory limits.

---

## 10. Quantifier Handling

### Counterexample-Guided Quantifier Instantiation
**Authors:** Andrew Reynolds, et al.
**Venue:** IJCAR 2014
**Abstract:** CEGQI approach for quantifier instantiation, complete for linear real arithmetic.
**Relevance:** Core quantifier technique for z4.

### An Instantiation-Based Approach for Solving Quantified Linear Arithmetic
**arXiv:** [1510.02642](https://arxiv.org/abs/1510.02642)
**Authors:** Andrew Reynolds, Tim King, Viktor Kuncak
**Abstract:** Framework to derive instantiation-based decision procedures for quantified linear arithmetic.
**Relevance:** Quantified arithmetic solving.

### Machine Learning for Quantifier Selection in cvc5
**arXiv:** [2408.14338](https://arxiv.org/abs/2408.14338)
**Authors:** Jan Jakubuv, Mikolas Janota, Jelle Piepenbrock, Josef Urban
**Abstract:** ML-guided quantifier selection improving SMT solving on first-order quantified problems.
**Relevance:** ML-enhanced quantifier handling.

### Quantifier Instantiations: To Mimic or To Revolt?
**arXiv:** [2508.13811](https://arxiv.org/abs/2508.13811)
**Authors:** Jan Jakubuv, Mikolas Janota
**Abstract:** Novel instantiation approach using probabilistic context-free grammars to generate term instantiations.
**Relevance:** Advanced instantiation techniques.

### Fair and Adventurous Enumeration of Quantifier Instantiations
**arXiv:** [2105.13700](https://arxiv.org/abs/2105.13700)
**Authors:** Mikolas Janota, Haniel Barbosa, et al.
**Abstract:** Studies strategies for ordering quantifier instantiation tuples.
**Relevance:** Instantiation ordering heuristics.

---

## 11. Craig Interpolation

### Efficient Generation of Craig Interpolants in Satisfiability Modulo Theories
**arXiv:** [0906.4492](https://arxiv.org/abs/0906.4492)
**Authors:** Alessandro Cimatti, Alberto Griggio, Roberto Sebastiani
**Abstract:** Computing Craig interpolants for different logic theories in SMT.
**Relevance:** Core interpolation reference.

### Efficient Interpolant Generation in Satisfiability Modulo Linear Integer Arithmetic
**arXiv:** [1010.4422](https://arxiv.org/abs/1010.4422)
**Authors:** Alberto Griggio, Thi Thieu Hoa Le, Roberto Sebastiani
**Abstract:** Craig interpolant computation specifically for linear integer arithmetic.
**Relevance:** LIA interpolation.

### Proof Tree Preserving Interpolation
**arXiv:** [1705.05309](https://arxiv.org/abs/1705.05309)
**Authors:** Jurgen Christ, Jochen Hoenicke, Alexander Nutz
**Abstract:** New scheme for computing interpolants addressing challenges from mixed literals.
**Relevance:** Advanced interpolation techniques.

### Augmenting Interpolation-Based Model Checking with Auxiliary Invariants
**arXiv:** [2403.07821](https://arxiv.org/abs/2403.07821)
**Authors:** Dirk Beyer, Po-Chun Chien, Nian-Ze Lee
**Abstract:** Using Craig interpolation for generating program invariants in model checking.
**Relevance:** Interpolation applications.

---

## 12. Proof Production

### DRAT Proofs of Unsatisfiability for SAT Modulo Monotonic Theories
**arXiv:** [2401.10703](https://arxiv.org/abs/2401.10703)
**Authors:** Nick Feng, Alan J. Hu, Sam Bayless, et al.
**Abstract:** Generating DRAT proofs for SAT Modulo Monotonic Theories (SMMT).
**Relevance:** Proof production techniques.

### DRAT-based Bit-Vector Proofs in CVC4
**arXiv:** [1907.00087](https://arxiv.org/abs/1907.00087)
**Authors:** Alex Ozdemir, Aina Niemetz, Mathias Preiner, et al.
**Abstract:** Integrating DRAT proofs from SAT solver into CVC4's proof infrastructure for bitvectors.
**Relevance:** BV proof integration.

---

## 13. SMT Solver Systems

### cvc5: A Versatile and Industrial-Strength SMT Solver
**Authors:** Barbosa, Barrett, Brain, et al.
**Venue:** TACAS 2022
**Abstract:** System description of CVC5 with comprehensive theory support and SyGuS capabilities.
**Relevance:** Reference for string/quantifier design.

### Yices 2.2
**Authors:** Bruno Dutertre
**Venue:** CAV 2014
**Abstract:** System description of Yices 2 with optimized arithmetic and bitvector solving.
**Relevance:** Performance optimization reference.

### The MathSAT5 SMT Solver
**Authors:** Alessandro Cimatti, Alberto Griggio, Bastiaan Schaafsma, Roberto Sebastiani
**Venue:** TACAS 2013
**Abstract:** System description of MathSAT 5 with interpolation and AllSAT support.
**Relevance:** Interpolation reference.

---

## 14. Machine Learning for SAT/SMT

### LangSAT: A Novel Framework Combining NLP and Reinforcement Learning for SAT Solving
**arXiv:** [2512.04374](https://arxiv.org/abs/2512.04374)
**Authors:** Muyu Pan, Matthew Walter, Dheeraj Kodakandla, Mahfuza Farooque
**Abstract:** Uses RL to optimize heuristic selection in CDCL, converting English descriptions to CNF.
**Relevance:** Future ML integration possibilities.

### Boolean Satisfiability via Imitation Learning
**arXiv:** [2509.25411](https://arxiv.org/abs/2509.25411)
**Authors:** Zewei Zhang, Huan Liu, Yuanhao Yu, Jun Chen, Xiangyu Xu
**Abstract:** ImitSAT learns branching policy based on expert trace to reduce propagations.
**Relevance:** ML-guided branching.

### Circuit-Aware SAT Solving: Guiding CDCL via Conditional Probabilities
**arXiv:** [2508.04235](https://arxiv.org/abs/2508.04235)
**Authors:** Jiaying Zhu, Ziyang Zheng, Zhengyuan Shi, Yalun Cai, Qiang Xu
**Abstract:** Graph neural networks to guide CDCL heuristics for circuit satisfiability.
**Relevance:** GNN-guided solving.

### Guiding High-Performance SAT Solvers with Unsat-Core Predictions
**arXiv:** [1903.04671](https://arxiv.org/abs/1903.04671)
**Authors:** Daniel Selsam, Nikolaj Bjorner
**Abstract:** Modified MiniSat to use NeuroSAT predictions for performance improvement.
**Relevance:** Neural network guidance.

### AutoSAT: Automatically Optimize SAT Solvers via Large Language Models
**arXiv:** [2402.10705](https://arxiv.org/abs/2402.10705)
**Authors:** Yiwen Sun, et al.
**Abstract:** Using LLMs to optimize CDCL solvers, outperforming MiniSat on most datasets.
**Relevance:** LLM-based solver optimization.

### Enhancing SAT Solvers with Glue Variable Predictions
**arXiv:** [2007.02559](https://arxiv.org/abs/2007.02559)
**Author:** Jesse Michael Han
**Abstract:** Neural network predictions for glue variables to improve CaDiCaL performance.
**Relevance:** Neural heuristic enhancement.

### Proof-Driven Clause Learning in Neural Network Verification
**arXiv:** [2503.12083](https://arxiv.org/abs/2503.12083)
**Authors:** Omri Isac, Idan Refaeli, Haoze Wu, Clark Barrett, Guy Katz
**Abstract:** Applies CDCL to improve scalability of deep neural network verification.
**Relevance:** DNN verification applications.

---

## Citation Index

### By First Author (Alphabetical)

| Author | Papers |
|--------|--------|
| Berzish, M. | String solving (2017, 2020) |
| Biere, A. | CaDiCaL family (2020) |
| Cimatti, A. | MathSAT, Interpolation (2009, 2013) |
| Christ, J. | Arrays, Interpolation (2014, 2017) |
| de Moura, L. | Z3, DPLL(T), Arrays (2006, 2008, 2009) |
| Dutertre, B. | Yices, Simplex, MCSAT (2006, 2014, 2020) |
| Een, N. | MiniSat (2003) |
| Griggio, A. | Interpolation (2009, 2010) |
| Liang, T. | String theory (2014) |
| Moskewicz, M. | Chaff, VSIDS (2001) |
| Niemetz, A. | Bitwuzla (2020) |
| Reynolds, A. | Quantifiers, CEGQI (2014, 2015) |

---

---

## 15. Verified SAT/SMT Solvers

**CRITICAL REFERENCE**: These solvers prove that formal verification of SAT/SMT solvers is achievable. Z4 should study these for verification techniques, not performance.

### IsaSAT: Formally Verified SAT Solver
**Authors:** Mathias Fleury, Armin Biere, Peter Lammich
**Venue:** CAV 2020
**Repository:** https://bitbucket.org/isafol/isafol/src/master/IsaSAT/
**Verification:** Isabelle/HOL with code extraction to Standard ML/LLVM

**Key Facts:**
- Full CDCL with 2-watched literals, VSIDS-like heuristics, restarts, clause learning
- Generates DRAT proofs
- SAT Competition 2023: 141 problems (vs winner's 352) — **40% of best**
- SAT Competition 2022: 107 problems (vs winner's 238) — **45% of best**
- Verification approach: Stepwise refinement from abstract specification to efficient implementation

**What IsaSAT Proves:**
1. Termination of main solving loop
2. Soundness: SAT answer implies valid model
3. Soundness: UNSAT answer implies valid DRAT proof
4. Memory safety (via Isabelle's type system + code extraction)

**Why IsaSAT is Slower:**
- Missing advanced inprocessing (vivification, subsumption elimination)
- Missing advanced restart policies (glucose-style EMA)
- Missing chronological backtracking optimization
- Verification constraints limit some micro-optimizations

### CreuSAT: Formally Verified SAT Solver in Rust
**Author:** Sarek Skotåm
**Thesis:** University of Oslo, 2021
**Repository:** https://github.com/sarsko/CreuSAT
**Verification:** Creusot → Why3 → Z3/CVC5

**Key Facts:**
- Full CDCL with 2-watched literals, VMTF heuristics, phase saving, EMA restarts
- Written in Rust, verified using Creusot (Rust → Why3 translation)
- Claims to be "world's fastest deductively verified SAT solver"
- Statically proven panic-free

**What CreuSAT Proves:**
1. Soundness: SAT → formula satisfiable
2. Soundness: UNSAT → formula unsatisfiable
3. Panic freedom: No runtime crashes
4. Memory safety (via Rust + Creusot)

**CreuSAT Verification Approach:**
```rust
// Example from CreuSAT - verified invariants
#[requires(self.invariant())]
#[ensures(result == SatResult::Sat ==> self.formula.satisfied_by(self.model))]
#[ensures(result == SatResult::Unsat ==> !exists_model(self.formula))]
fn solve(&mut self) -> SatResult
```

### Comparison: Verified vs Unverified

| Solver | Verified | SAT Comp 2023 | Gap to Winner |
|--------|----------|---------------|---------------|
| VBS (portfolio) | No | 352 problems | — |
| Kissat | No | ~340 problems | ~3% |
| CaDiCaL | No | ~330 problems | ~6% |
| **IsaSAT** | Yes (Isabelle) | 141 problems | **60%** |
| **CreuSAT** | Yes (Creusot) | Not entered | Unknown |

**Key Insight:** The verification penalty is real (40-60% fewer problems solved) but not catastrophic. This suggests Z4 CAN be both verified and competitive.

### CakeML-based Proof Checkers
**cake_lpr:** Verified LRAT/DRAT proof checker in CakeML
**Repository:** https://github.com/tanyongkiam/cake_lpr
**Verification:** HOL4 → CakeML extraction

This is a verified proof CHECKER, not solver. Use for checking Z4's DRAT proofs.

### Related: Verified SMT Components

**SMTCoq:** Coq plugin for checking SMT proofs
- Verifies proofs from CVC4, veriT
- Could be extended to verify Z4's Alethe proofs

**VeriT:** SMT solver with extensive proof production
- One of the best proof-producing SMT solvers
- Alethe format originates here

### Academic Papers on Verified SAT Solving

1. **A Verified SAT Solver with Watched Literals Using Imperative HOL**
   - Lammich & Fleury, CPP 2019
   - First verified CDCL with watched literals in Isabelle/HOL

2. **Formalizing the LLVM Intermediate Representation in Isabelle/HOL**
   - IsaSAT uses LLVM code extraction for performance

3. **Simulation, Refinement, and Certification with Isabelle/HOL**
   - Stepwise refinement methodology used in IsaSAT

4. **Why3 - Where Programs Meet Provers**
   - Filliatre & Paskevich, ESOP 2013
   - Verification backend used by CreuSAT

5. **Creusot: A Foundationally Verified Deductive Verifier for Rust**
   - Denis, Jourdan, Marché
   - Tool used to verify CreuSAT

---

## 16. Formal Verification Tools for Rust

### Kani
**Repository:** https://github.com/model-checking/kani
**Backend:** CBMC (bounded model checking)
**License:** Apache 2.0 / MIT

**Capabilities:**
- Bounded model checking of Rust code
- Symbolic execution for unsafe code
- Proof harnesses similar to tests
- Automatic panic detection

**Limitations:**
- Bounded (can miss bugs beyond depth N)
- No concurrency support yet
- Some Rust features unsupported

### Verus
**Repository:** https://github.com/verus-lang/verus
**Backend:** Z3 (SMT solving)
**License:** MIT

**Capabilities:**
- Full deductive verification of Rust
- Preconditions, postconditions, invariants
- Ghost code for proofs
- Supports unsafe pointer manipulation

**Bootstrapping Note:** Uses Z3 as backend. Once Z4 is mature, could potentially replace Z3.

### Creusot
**Repository:** https://github.com/creusot-rs/creusot
**Backend:** Why3 → multiple provers (Z3, CVC5, Alt-Ergo)
**License:** LGPL-2.1

**Capabilities:**
- Translates Rust to WhyML
- Uses Why3's multi-prover architecture
- Supports ownership, borrowing, interior mutability

### Prusti
**Repository:** https://github.com/viperproject/prusti-dev
**Backend:** Viper → Z3
**License:** MPL 2.0

**Capabilities:**
- Automatic verification of safe Rust
- Overflow/panic checking by default
- Separation logic for heap reasoning

### Miri
**Repository:** https://github.com/rust-lang/miri
**Purpose:** Undefined behavior detection

Not formal verification, but essential for finding UB in unsafe code.

---

## Notes on Access

- **arXiv papers**: Freely available at https://arxiv.org/abs/[ID]
- **Conference papers**: Many available via author websites or institutional access
- **Foundational papers** (pre-2010): Often available via CiteSeerX or author pages
- **Verification tool papers**: Usually open access or on author websites

## Updates

This document should be updated as new relevant papers are published, particularly in:
- SAT Competition / SMT-COMP system descriptions
- CAV, TACAS, FMCAD proceedings
- Journal of Automated Reasoning
- ITP, CPP (for verified solver papers)
