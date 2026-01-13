# Z4 Paper Library

**32 papers** downloaded for Z4 implementation reference.

---

## Foundational Papers (Conference/Journal)

| File | Title | Authors | Year | Topic |
|------|-------|---------|------|-------|
| `minisat-sat04.pdf` | An Extensible SAT-solver | Een, Sorensson | 2004 | **CDCL Foundations** |
| `chaff-dac01.pdf` | Chaff: Engineering an Efficient SAT Solver | Moskewicz et al. | 2001 | **VSIDS Heuristic** |
| `dpll-t-jacm2006.pdf` | Solving SAT and SAT Modulo Theories | Nieuwenhuis, Oliveras, Tinelli | 2006 | **DPLL(T) Foundations** |
| `congruence-closure-rta2005.pdf` | Proof-Producing Congruence Closure | Nieuwenhuis, Oliveras | 2005 | **Congruence Closure** |
| `z3-tacas2008.pdf` | Z3: An Efficient SMT Solver | de Moura, Bjorner | 2008 | **SMT Architecture** |
| `simplex-dpll-cav2006.pdf` | A Fast Linear-Arithmetic Solver for DPLL(T) | Dutertre, de Moura | 2006 | **Simplex (Yices)** |
| `cadical-sat2018.pdf` | CaDiCaL at SAT Competition 2018 | Biere | 2018 | **Modern SAT** |
| `cvc5-tacas2022.pdf` | cvc5: A Versatile and Industrial-Strength SMT Solver | Barbosa et al. | 2022 | **CVC5 System** |
| `string-cav2014.pdf` | A DPLL(T) Theory Solver for Strings | Liang, Reynolds, Tinelli, Barrett | 2014 | **String Normal Form** |
| `cegqi-cav2017.pdf` | Solving Quantified Bitvector Formulas | Reynolds et al. | 2017 | **CEGQI** |

---

## ArXiv Papers - Core SMT

| File | Title | Topic |
|------|-------|-------|
| `1606.04786.pdf` | A Survey of Satisfiability Modulo Theory | **SMT Overview** |
| `2404.16122.pdf` | Generalized Optimization Modulo Theories | **OMT/DPLL(T)** |

---

## ArXiv Papers - SAT/CDCL

| File | Title | Topic |
|------|-------|-------|
| `2003.02323.pdf` | Complexity-theoretic Understanding of Restarts | **VSIDS/Restarts** |

---

## ArXiv Papers - Arithmetic

| File | Title | Topic |
|------|-------|-------|
| `1510.02642.pdf` | Instantiation-Based Approach for Quantified Linear Arithmetic | **Quantified Arith** |

---

## ArXiv Papers - Bitvectors

| File | Title | Topic |
|------|-------|-------|
| `2004.07940.pdf` | Solving Bitvectors with MCSAT | **BV (Yices/MCSAT)** |
| `2006.01621.pdf` | Bitwuzla at SMT-COMP 2020 | **BV (Bitwuzla)** |
| `2402.17927.pdf` | MCSat-based Finite Field Reasoning in Yices2 | **MCSAT** |

---

## ArXiv Papers - Arrays

| File | Title | Topic |
|------|-------|-------|
| `1405.6939.pdf` | Weakly Equivalent Arrays | **Array Theory** |

---

## ArXiv Papers - Strings

| File | Title | Topic |
|------|-------|-------|
| `1704.07935.pdf` | Z3str3: String Solver with Theory-aware Branching | **Strings** |
| `2010.07253.pdf` | SMT Solver for Regular Expressions and Linear Arithmetic | **Strings/Regex** |
| `1306.6054.pdf` | (Un)Decidability Results for Word Equations | **String Theory** |
| `2308.00175.pdf` | Decision Procedures for Sequence Theories | **Sequences** |
| `2509.00948.pdf` | Decision Procedure for A Theory of String Sequences | **String Sequences** |
| `2112.06039.pdf` | CertiStr: A Certified String Solver | **Certified Strings** |

---

## ArXiv Papers - Quantifiers

| File | Title | Topic |
|------|-------|-------|
| `1502.04464.pdf` | Counterexample-Guided Techniques for Program Synthesis | **CEGIS** |

---

## ArXiv Papers - E-graphs & Equality Saturation

| File | Title | Topic |
|------|-------|-------|
| `2504.10246.pdf` | Simplified Union-Find for Congruence Closure | **Union-Find/CC** |
| `2304.04332.pdf` | Better Together: Unifying Datalog and Equality Saturation | **egglog** |
| `2501.02413.pdf` | Semantic Foundations of Equality Saturation | **E-graph Theory** |

---

## ArXiv Papers - Interpolation & AllSAT

| File | Title | Topic |
|------|-------|-------|
| `0906.4492.pdf` | Efficient Generation of Craig Interpolants in SMT | **Interpolation (MathSAT)** |
| `1010.4422.pdf` | Efficient Interpolant Generation in SMT(LA(Z)) | **LIA Interpolation** |
| `1705.05309.pdf` | Proof Tree Preserving Interpolation | **Interpolation** |
| `2410.18707.pdf` | Disjoint Projected Enumeration for SAT and SMT | **AllSAT (MathSAT)** |

---

## Priority by Phase

### Phase 1: SAT Core
1. `minisat-sat04.pdf` - CDCL architecture
2. `chaff-dac01.pdf` - VSIDS decision heuristic
3. `cadical-sat2018.pdf` - Modern optimizations
4. `2003.02323.pdf` - Restart theory

### Phase 2: SMT Infrastructure
5. `dpll-t-jacm2006.pdf` - **CRITICAL**: DPLL(T) foundations (43 pages!)
6. `z3-tacas2008.pdf` - SMT architecture
7. `1606.04786.pdf` - SMT survey/overview

### Phase 3: Core Theories (EUF, LRA, LIA)
8. `congruence-closure-rta2005.pdf` - **CRITICAL**: Congruence closure
9. `simplex-dpll-cav2006.pdf` - **CRITICAL**: Arithmetic (Yices)
10. `2504.10246.pdf` - Modern union-find for CC

### Phase 4: BV & Arrays
11. `2006.01621.pdf` - Bitwuzla system
12. `2004.07940.pdf` - MCSAT for bitvectors
13. `1405.6939.pdf` - Weak equivalence arrays

### Phase 5: Strings & Quantifiers
14. `string-cav2014.pdf` - **CRITICAL**: String normal form (CVC5 approach)
15. `cvc5-tacas2022.pdf` - CVC5 system
16. `cegqi-cav2017.pdf` - CEGQI for quantifiers
17. `1502.04464.pdf` - CEGIS techniques

### Phase 6: E-graphs (Optimization)
18. `2304.04332.pdf` - egglog (Datalog + equality saturation)
19. `2501.02413.pdf` - E-graph foundations

### Phase 7: Interpolation & AllSAT
20. `0906.4492.pdf` - **CRITICAL**: Craig interpolation (MathSAT)
21. `1010.4422.pdf` - LIA interpolation
22. `2410.18707.pdf` - AllSAT enumeration

---

## License Notes

| Paper | Describes | Can Port Code? |
|-------|-----------|----------------|
| MiniSat | MiniSat | Yes (MIT) |
| Chaff | zChaff | Study only |
| DPLL(T) JACM | Foundations | Algorithm only |
| Z3 | Z3 | Yes (MIT) |
| Simplex | **Yices** | **NO (GPL)** - Papers only |
| CaDiCaL | CaDiCaL | Yes (MIT) |
| CVC5 | CVC5 | Yes (BSD) |
| String CAV14 | CVC4/5 | Yes (BSD) |
| Bitwuzla | Bitwuzla | Yes (MIT) |
| MathSAT papers | **MathSAT** | **NO (Proprietary)** - Papers only |
| Congruence Closure | Algorithm | Algorithm only |

---

## Complete File List (32 papers)

```
0906.4492.pdf          - Craig Interpolation
1010.4422.pdf          - LIA Interpolation
1306.6054.pdf          - Word Equations Decidability
1405.6939.pdf          - Weak Equivalent Arrays
1502.04464.pdf         - CEGIS
1510.02642.pdf         - Quantified Linear Arithmetic
1606.04786.pdf         - SMT Survey
1704.07935.pdf         - Z3str3
1705.05309.pdf         - Proof Tree Interpolation
2003.02323.pdf         - VSIDS/Restarts
2004.07940.pdf         - MCSAT Bitvectors
2006.01621.pdf         - Bitwuzla
2010.07253.pdf         - Strings + Regex
2112.06039.pdf         - CertiStr
2304.04332.pdf         - egglog
2308.00175.pdf         - Sequence Theories
2402.17927.pdf         - MCSAT Finite Fields
2404.16122.pdf         - Generalized OMT
2410.18707.pdf         - AllSAT
2501.02413.pdf         - Equality Saturation Semantics
2504.10246.pdf         - Union-Find for CC
2509.00948.pdf         - String Sequences
cadical-sat2018.pdf    - CaDiCaL System
cegqi-cav2017.pdf      - CEGQI
chaff-dac01.pdf        - Chaff/VSIDS
congruence-closure-rta2005.pdf - CC Algorithm
cvc5-tacas2022.pdf     - CVC5 System
dpll-t-jacm2006.pdf    - DPLL(T) Foundations
minisat-sat04.pdf      - MiniSat
simplex-dpll-cav2006.pdf - Simplex (Yices)
string-cav2014.pdf     - String Normal Form
z3-tacas2008.pdf       - Z3 Architecture
```

---

## Summary

All critical papers for Z4 implementation are now available:
- CDCL/SAT foundations
- DPLL(T) architecture
- Congruence closure algorithm
- Simplex for arithmetic (Yices approach)
- String normal form (CVC5 approach)
- CEGQI for quantifiers
- Craig interpolation (MathSAT approach)
- E-graph/equality saturation foundations
