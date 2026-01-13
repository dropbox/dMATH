# CHC-COMP Winning Techniques: Golem & Eldarica Analysis

**Date:** 2026-01-05
**Author:** MANAGER AI
**Purpose:** Reference for porting CHC-COMP winning techniques to Z4

---

## Executive Summary

This document analyzes techniques from the top CHC-COMP solvers:
- **Golem** (2023-2025 LIA track winner): Portfolio of 3 engines + Model-Based Projection
- **Eldarica** (2023 LRA track winner): CEGAR with interpolation-based predicate synthesis

Z4's current CHC solver (35/55 @ 30s) already beats Z3 (17/55) but trails behind Golem and Eldarica on complex benchmarks.

---

## Golem Architecture (MIT License - Can Port Directly)

Source: `reference/golem/src/`

### Portfolio Strategy

Golem runs **3 engines in parallel**:
1. **Spacer** - General-purpose PDR/IC3 (default)
2. **LAWI** - Lazy Abstraction with Interpolants (IMPACT algorithm)
3. **Split-TPA** - Transition Power Abstraction

First to solve wins. This strategy exploits algorithmic complementarity.

### Engine 1: Spacer (`engine/Spacer.cc`)

**Algorithm:** Property-Directed Reachability (PDR/IC3)

**Key Data Structures:**
```cpp
class SpacerContext {
    UnderApproxMap under;  // Must summaries (reachable states)
    OverApproxMap over;    // May summaries (unreachable states)
    PriorityQueue pqueue;  // Proof obligations (lower bound first)
    DerivationDatabase database;  // For proof reconstruction
};
```

**Key Techniques:**

1. **Mixed Summaries** (`getEdgeMixedSummary`):
   - For hyperedges with multiple sources
   - Use may-summary for some sources, must-summary for others
   - Enables incremental refinement

2. **Interpolation-Based Lemma Learning** (`interpolatingSat`):
   - When all edges are blocked, compute interpolant
   - Interpolant becomes new lemma blocking proof obligation
   - Uses OpenSMT's interpolation with simplification level 4

3. **Component Pushing** (`tryPushComponents`):
   - After bounded safety check, try pushing lemmas to higher levels
   - Uses `impliedBy()` to check multiple candidates in parallel
   - Detects fixed point (induction) when all components push

4. **Model-Based Projection** (`projectFormula`):
   - Projects formula onto subset of variables using model
   - Critical for computing predecessors efficiently
   - Uses `ModelBasedProjection` class (separate file)

**Z4 Gap:** Z4 lacks interpolation-based lemma learning. Currently uses blocking lemmas from counterexample analysis.

### Engine 2: LAWI (`engine/Lawi.cc`)

**Algorithm:** Lazy Abstraction with Interpolants (IMPACT)

**Paper:** McMillan, "Lazy Abstraction with Interpolants", CAV 2006

**Key Data Structures:**
```cpp
class LawiContext {
    AbstractReachabilityTree art;  // Unfolding tree
    LabelingFunction labels;       // Node -> formula
    CoveringRelation coveringRelation;  // Subsumption
    ImplicationChecker implicationChecker;  // Cached checks
};
```

**Key Techniques:**

1. **Abstract Reachability Tree (ART)**:
   - Unfolds CHC system into tree
   - Each node has label (formula characterizing reachable states)
   - Edges connect parent to children via transitions

2. **Covering Relation**:
   - `(v,w)` in relation means `Lab(v) implies Lab(w)`
   - Enables pruning: covered nodes don't need further exploration
   - `vertexStrengthened()` updates relation when labels change

3. **Interpolation-Based Refinement**:
   - When spurious counterexample found
   - Get sequence of interpolants along path
   - Strengthen node labels to eliminate spurious path

4. **Forced Covering** (optional):
   - Aggressively try to cover nodes
   - Limit how many ancestors to check

**Z4 Gap:** Z4 uses Houdini-style induction, not ART-based CEGAR.

### Engine 3: Split-TPA (`engine/TPA.cc`)

**Algorithm:** Transition Power Abstraction

**Key Innovation:** Separates exact and less-than reachability relations

```cpp
class TPASplit : public TPABase {
    vec<PTRef> exactPowers;      // R^{2^k} exactly
    vec<PTRef> lessThanPowers;   // R^{<2^k}
    vec<SolverWrapper*> reachabilitySolvers;  // Incremental
};
```

**Key Techniques:**

1. **Power-Based Iteration**:
   - Compute `R^{2^0}`, `R^{2^1}`, `R^{2^2}`, ...
   - Check reachability at each power
   - Fixed point when power doesn't change

2. **Incremental Solvers** (`SolverWrapperIncremental`):
   - Reuses solver state across queries
   - Periodic restarts (every 100 pushes) to prevent slowdown
   - Keeps interpolation partition mask for lemma extraction

3. **Transition Invariant Extraction**:
   - When safe, extracts k-inductive invariant
   - `inductiveInvariantFromEqualsTransitionInvariant()`

**Z4 Gap:** Z4 doesn't have TPA-style accelerated reachability.

### Model-Based Projection (`ModelBasedProjection.h`)

**Purpose:** Eliminate variables from formula while preserving satisfiability under model

**Key Method:**
```cpp
PTRef project(PTRef fla, vec<PTRef> const & varsToEliminate, Model & model);
```

**Techniques:**
1. **LIA Variable Elimination**:
   - Extract bounds on variable
   - Resolve pairs (lower bound + upper bound)
   - Handle divisibility constraints

2. **LRA Variable Elimination**:
   - Standard Fourier-Motzkin style
   - Model-guided to pick right combinations

**Z4 Gap:** Z4 has basic MBP but may lack the divisibility constraint handling.

---

## Eldarica Architecture (BSD License - Can Study)

Source: GitHub `uuverifiers/eldarica` (Scala)

### Core Algorithm: CEGAR with Interpolation

**Flow:**
1. Start with trivial abstraction
2. Check if abstraction proves property
3. If counterexample found, check if spurious
4. If spurious, refine abstraction using interpolants
5. Repeat until proven safe or real counterexample found

### Key Techniques

1. **Predicate Abstraction**:
   - Abstract transition relations using predicates
   - Predicates discovered via interpolation

2. **Interpolation Strategies** (command-line options):
   - `-disj`: Disjunctive interpolants
   - `-abstract`: Abstract interpolation
   - `-stac`: Requires Yices v1

3. **Multiple Input Format Support**:
   - SMT-LIB 2
   - Prolog-style Horn clauses
   - Source code (Scala, C)

### Eldarica vs Golem

| Aspect | Eldarica | Golem |
|--------|----------|-------|
| Core Algorithm | CEGAR | Portfolio (PDR + LAWI + TPA) |
| Language | Scala | C++ |
| Interpolation | Central | Used in all engines |
| Strength | LRA benchmarks | LIA benchmarks |

---

## Techniques to Port to Z4

### Priority 1: Interpolation-Based Lemma Learning

**Impact:** High (enables principled lemma synthesis)

**Current Z4 Approach:** Extract blocking lemmas from counterexample analysis

**Golem Approach:**
```cpp
auto res = interpolatingSat(logic.mkOr(edgeRepresentations), pob.constraint);
PTRef newLemma = VersionManager(logic).targetFormulaToBase(res.interpolant);
addMaySummary(pob.vertex, pob.bound, newLemma);
```

**Implementation Plan:**
1. Add interpolation capability to z4-dpll theory solver
2. Modify PDR to use interpolants for lemma generation
3. Compare quality of interpolant-based vs. heuristic lemmas

### Priority 2: Model-Based Projection with LIA Support

**Impact:** Medium-High (better predecessor computation)

**Current Z4 Approach:** Variable elimination via substitution

**Golem Approach:**
- `projectSingleVar()`: Projects one variable at a time
- `projectIntegerVars()`: Special handling for LIA
- `processDivConstraints()`: Handles divisibility

**Implementation Plan:**
1. Review z4's current MBP implementation
2. Add divisibility constraint handling
3. Benchmark on LIA-heavy CHC problems

### Priority 3: Portfolio Engine Strategy

**Impact:** High (algorithmic complementarity)

**Current Z4 Approach:** Single PDR engine

**Implementation Plan:**
1. Factor out common interface (`Engine` trait)
2. Implement LAWI as alternative engine
3. Add parallel execution with first-to-finish

### Priority 4: Abstract Reachability Tree (LAWI)

**Impact:** Medium (different algorithmic approach)

**Data Structures Needed:**
```rust
struct ART {
    nodes: Vec<ARTNode>,
    edges: Vec<ARTEdge>,
    labels: HashMap<NodeId, Formula>,
    covering: Vec<(NodeId, NodeId)>,  // coveree, coverer
}
```

**Implementation Plan:**
1. Implement ART data structure
2. Add covering relation with update
3. Implement interpolation-based refinement

### Priority 5: TPA (Transition Power Abstraction)

**Impact:** Medium (good for specific problem classes)

**Key Insight:** Powers of 2 iteration enables fast fixed-point detection

**Implementation Plan:**
1. Implement power-based iteration
2. Add incremental solver support
3. Extract k-inductive invariants

---

## Quick Wins (Port Without Major Architecture Changes)

### 1. Interpolation Simplification Level

Golem uses:
```cpp
solver.getConfig().setSimplifyInterpolant(4);
```

Add similar simplification to z4 lemma generation.

### 2. Component Pushing in Parallel

Golem checks multiple candidates at once:
```cpp
auto pushed = impliedBy(std::move(targetCandidates), body, logic);
```

Z4 could batch induction checks similarly.

### 3. Incremental Solver with Restarts

Golem restarts incremental solver every 100 pushes:
```cpp
const unsigned limit = 100;
if (levels > limit) { rebuildSolver(); }
```

Prevents solver slowdown on long-running problems.

### 4. Priority Queue by Bound

Golem processes lower-bound POBs first:
```cpp
std::priority_queue<ProofObligation, ..., std::greater<>> pqueue;
```

Z4 already has level priority but verify it matches.

---

## References

1. **Golem Source**: `reference/golem/` (MIT License)
2. **Spacer Paper**: Komuravelli et al., "SMT-Based Model Checking for Recursive Programs", CAV 2014
3. **LAWI/IMPACT Paper**: McMillan, "Lazy Abstraction with Interpolants", CAV 2006
4. **TPA Paper**: (Need to find - referenced in Golem)
5. **Eldarica**: Hojjat & Rümmer, "The ELDARICA Horn Solver", FMCAD 2018

---

## Recommended Implementation Order

1. **Interpolation infrastructure** (enables multiple techniques)
2. **Enhanced MBP** (immediate benefit to PDR)
3. **Portfolio strategy** (low risk, high reward)
4. **LAWI engine** (alternative approach)
5. **TPA engine** (specialized problems)

Each technique should be benchmarked on CHC-COMP suite after implementation.
