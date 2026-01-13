# MathSAT 5 SMT Solver - Research Analysis

**Website:** https://mathsat.fbk.eu/
**License:** Proprietary (free for academic/research, commercial license required)
**Language:** C++
**Developer:** FBK (Fondazione Bruno Kessler), Trento, Italy
**Key Developers:** Alessandro Cimatti, Alberto Griggio, Roberto Sebastiani

---

## 1. Overview

MathSAT 5 is known for excellent interpolation support, AllSAT enumeration, and incremental solving. While proprietary (limiting direct code study), its techniques are well-documented in academic papers.

**Note:** Source code is not available. This analysis is based on:
- Academic papers by the MathSAT team
- Public documentation
- SMT-COMP performance data
- API documentation

## 2. Architecture (from papers)

```
┌─────────────────────────────────────────────────────────────────┐
│                      MathSAT 5 API                              │
│                  (C, C++, Python, OCaml)                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                   SMT Solver Core                               │
│              (DPLL(T) with optimizations)                       │
└───────┬─────────────────────────────────────────────────┬───────┘
        │                                                 │
┌───────▼───────────────────┐         ┌───────────────────▼───────┐
│      SAT Engine           │◄───────►│    Theory Engine          │
│   (MiniSat-based CDCL)    │         │  (Nelson-Oppen combo)     │
└───────────────────────────┘         └───────────────────────────┘
        │                                       │
        │                    ┌──────────────────┼──────────────────┐
        │                    │                  │                  │
        │             ┌──────▼──────┐   ┌───────▼───────┐  ┌───────▼───────┐
        │             │    LA(Q)    │   │     EUF      │  │      BV       │
        │             │   Simplex   │   │  (E-graph)   │  │ (Bit-blast)   │
        │             └─────────────┘   └──────────────┘  └───────────────┘
        │
┌───────▼───────────────────────────────────────────────────────────┐
│                    Special Modules                                 │
│  Interpolation │ AllSAT │ Optimization │ Proof Generation         │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Strengths

### 3.1 Interpolation (Best in Class)

MathSAT 5 is the leading solver for Craig interpolation:

**What is Interpolation?**
Given unsatisfiable A ∧ B, find I such that:
- A ⊨ I
- I ∧ B is unsatisfiable
- I only uses symbols common to A and B

**Applications:**
- Model checking (IC3/PDR algorithms)
- Program verification
- Abstraction refinement

**MathSAT's Approach:**
1. **Proof-based interpolation** - Extract from resolution proof
2. **Theory-specific interpolation** - LRA, EUF, BV
3. **Interpolation sequences** - For multiple partitions

### 3.2 AllSAT / Model Enumeration

Enumerate all satisfying assignments:

**Techniques:**
- Blocking clauses (standard)
- Projection to relevant variables
- Efficient model rotation

**Use Cases:**
- Fault tree analysis
- Test case generation
- Synthesis

### 3.3 Optimization Modulo Theories (OMT)

MathSAT excels at optimization:

**Approaches:**
- Linear search
- Binary search
- Bisection with lemmas
- OptiMathSAT extension

### 3.4 Incremental Solving

Highly optimized for incremental queries:
- Efficient push/pop
- Assumption-based solving
- Minimal clause set retention

### 3.5 Theory Support

Supports:
- Linear Rational Arithmetic (LRA)
- Linear Integer Arithmetic (LIA)
- Difference Logic (IDL, RDL)
- Uninterpreted Functions (EUF)
- Bitvectors
- Arrays
- Floating-point (via reduction)

---

## 4. Weaknesses

### 4.1 Proprietary License

- Cannot study source code
- Cannot modify for Z4
- Commercial license required for products

### 4.2 Limited String Support

- No native string theory
- Must encode strings as other theories

### 4.3 Quantifier Support

- Limited compared to CVC5
- Focus on quantifier-free theories

### 4.4 Less Active Development

- Smaller team than Z3/CVC5
- Slower to add new features

---

## 5. Key Algorithms (from papers)

### 5.1 Proof-Based Interpolation

**Paper:** "Efficient Generation of Craig Interpolants" (Griggio, 2009)

Algorithm:
1. Solve A ∧ B, get proof of unsatisfiability
2. Label proof steps as A-local, B-local, or shared
3. Transform proof into partial interpolant tree
4. Combine partial interpolants

For LRA specifically:
- Use Farkas lemmas for conflict clauses
- Interpolate Farkas coefficients

### 5.2 AllSAT Enumeration

**Paper:** "AllSMT: A Theory-Aware Solver for AllSAT" (2007)

Algorithm:
1. Solve formula
2. If SAT, record model
3. Add blocking clause (¬model)
4. Repeat until UNSAT

Optimizations:
- Project to user variables only
- Dual propagation for minimal models
- Clause subsumption

### 5.3 Optimization

**Paper:** "OptiMathSAT: A Tool for Optimization Modulo Theories" (2015)

Linear search with lemmas:
```
cost = infinity
while SAT(formula ∧ cost < current_cost):
    cost = model_cost
    add: cost < current_cost
return cost
```

Improvements:
- Binary search on cost bounds
- Unsat core extraction for bounds
- Incremental solving

### 5.4 Lazy SMT

**Paper:** "Lazy Satisfiability Modulo Theories" (2006)

Key ideas:
- Theory atoms abstracted as Booleans
- SAT solver finds propositional model
- Theory solver checks consistency
- Conflicts generate blocking clauses

---

## 6. What Z4 Should Adopt from MathSAT

### Must Have
1. **Interpolation Infrastructure**
   - Proof-based interpolation
   - Theory-specific interpolation for LRA
   - Interpolation sequences

   *Why:* Essential for verification applications (IC3/PDR)

2. **AllSAT Support**
   - Model enumeration API
   - Projection to user variables
   - Efficient blocking

3. **Incremental Solving Optimizations**
   - Efficient push/pop
   - Assumption-based solving
   - Clause retention strategies

### Should Have
1. **OMT (Optimization Modulo Theories)**
   - Linear/binary search
   - MaxSMT support
   - Pareto optimization

2. **Proof Production for Interpolation**
   - Resolution proof extraction
   - Theory lemma proofs
   - Farkas lemma handling

### Could Adopt
1. **Lazy SMT Refinements**
   - Clause minimization
   - Conflict generalization

---

## 7. Key Papers to Study

These papers document MathSAT's techniques:

### Interpolation
1. **McMillan, CAV 2005** - "Applications of Craig Interpolation to Model Checking"
2. **Griggio, IJCAR 2009** - "Efficient Generation of Craig Interpolants"
3. **Cimatti et al., TACAS 2010** - "A Simple and Efficient Approach to Computing Interpolants"

### AllSAT
4. **Sebastiani et al., 2007** - "AllSMT: A Theory-Aware Solver for AllSAT"

### Optimization
5. **Sebastiani & Trentin, TACAS 2015** - "OptiMathSAT: A Tool for Optimization Modulo Theories"

### Architecture
6. **Cimatti et al., TACAS 2013** - "The MathSAT5 SMT Solver"

---

## 8. Performance Characteristics

Based on SMT-COMP results:

| Category | MathSAT 5 Performance | Notes |
|----------|----------------------|-------|
| QF_LRA | **Excellent** | Very competitive |
| QF_LIA | Good | Strong |
| QF_IDL/RDL | **Excellent** | Difference logic |
| QF_BV | Good | Competitive |
| QF_AUFLIA | Good | Strong |
| Interpolation | **Best** | Leading solver |
| AllSAT | **Best** | Leading solver |
| OMT | **Excellent** | OptiMathSAT |

---

## 9. Integration Strategy

Since MathSAT is proprietary, Z4 cannot use its code. However:

### 1. Study Published Algorithms
- Interpolation algorithms are well-documented
- AllSAT techniques are published
- OMT approaches are in papers

### 2. Re-implement from Papers
- Proof-based interpolation (Griggio 2009)
- AllSAT enumeration
- Optimization search strategies

### 3. Use as Benchmark
- Compare Z4 performance against MathSAT
- Particularly for interpolation benchmarks
- OMT benchmarks

### 4. Consider Feature Parity
- Interpolation is critical for verification
- AllSAT is valuable for enumeration
- OMT extends applicability

---

## 10. API Features to Emulate

From MathSAT 5 documentation:

### Configuration
```c
msat_config cfg = msat_create_config();
msat_set_option(cfg, "interpolation", "true");
```

### Interpolation
```c
msat_env env = msat_create_env(cfg);
int group_a = msat_create_itp_group(env);
int group_b = msat_create_itp_group(env);

msat_set_itp_group(env, group_a);
msat_assert_formula(env, formula_a);

msat_set_itp_group(env, group_b);
msat_assert_formula(env, formula_b);

msat_result res = msat_solve(env);
if (res == MSAT_UNSAT) {
    int groups[] = {group_a};
    msat_term itp = msat_get_interpolant(env, groups, 1);
}
```

### AllSAT
```c
msat_all_sat(env, important_vars, num_vars, callback, user_data);
```

Z4 should provide similar APIs for these advanced features.

---

## 11. Summary

MathSAT 5 contributes:

| Feature | Importance for Z4 | Implementation |
|---------|-------------------|----------------|
| Interpolation | **Critical** | Re-implement from papers |
| AllSAT | High | Standard technique |
| OMT | Medium | Search-based approach |
| Incremental | High | Standard but optimize |
| Proof | High | For interpolation support |

The key differentiator is interpolation. Z4 must support this for verification applications.
