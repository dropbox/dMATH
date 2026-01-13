# Z4 Benchmarks and Techniques Reference

## Goal: Union of All Winning Techniques

Z4 should implement the **union** of techniques from all competition-winning solvers, while maintaining formal verification. If IsaSAT can verify CDCL with watched literals, we can verify inprocessing too.

---

## Phase Mapping

| Phase | Component | Competition | Benchmarks | Techniques From |
|-------|-----------|-------------|------------|-----------------|
| **1** | z4-sat | SAT Competition | DIMACS CNF | CaDiCaL, Kissat, IsaSAT |
| **2** | z4-core, z4-frontend, z4-dpll | — | — | Z3, CVC5 |
| **3** | z4-euf, z4-lra, z4-lia | SMT-COMP (QF_UF, QF_LRA, QF_LIA) | SMT-LIB | Z3, CVC5, Yices |
| **4** | z4-bv, z4-arrays | SMT-COMP (QF_BV, QF_ABV) | SMT-LIB | Bitwuzla, Boolector |
| **5** | z4-strings, z4-fp, z4-dt | SMT-COMP (QF_S, QF_FP) | SMT-LIB | CVC5, Z3 |
| **6** | Quantifiers | SMT-COMP (AUFLIA, etc.) | SMT-LIB | Z3, CVC5 |

---

# PHASE 1: SAT SOLVER (z4-sat)

This section covers everything needed for Phase 1.

---

## Competitions

### SAT Competition (2002-present)

**What**: Annual competition for Boolean satisfiability solvers
**Website**: https://satcompetition.github.io/
**Years**: 2002-2025 (24 competitions)

**Tracks**:
- Main Track (~400 benchmarks/year)
- Parallel Track
- Cloud Track
- CaDiCaL Hack Track (2024+)
- No-Limits Track

**Benchmark Access**:
```bash
# Global Benchmark Database (recommended)
# Get 2023 main track benchmark URLs
curl "https://benchmark-database.de/getinstances?query=track%3Dmain_2023&context=cnf" > benchmarks_2023.uri

# Download all benchmarks
wget --content-disposition -i benchmarks_2023.uri

# Available tracks (2006-2025):
# track=main_YYYY, track=application_YYYY, track=random_YYYY, track=crafted_YYYY
```

**Zenodo Archive**: https://doi.org/10.5281/zenodo.13379892 (2024 benchmarks)

---

## CaDiCaL Techniques (Complete List for Phase 1)

From `src/options.hpp` — every option represents a technique:

### Preprocessing (Inprocessing)

| Technique | Description | IsaSAT? |
|-----------|-------------|---------|
| **Bounded Variable Elimination (BVE)** | Eliminate variables by resolving all clauses | NO |
| **Blocked Clause Elimination** | Remove clauses blocked by some literal | NO |
| **Covered Clause Elimination** | Remove asymmetric covered clauses | NO |
| **Subsumption** | Remove subsumed clauses | Partial |
| **Self-Subsumption (Strengthening)** | Remove redundant literals | Partial |
| **Vivification** | Strengthen clauses via unit propagation | NO |
| **Failed Literal Probing** | Detect forced assignments | NO |
| **Hyper Ternary Resolution** | Add ternary resolvents | NO |
| **Transitive Reduction** | Simplify binary implication graph | NO |
| **Gate Extraction** | Recognize AND/XOR/ITE gates | NO |
| **Congruence Closure** | Equality reasoning in preprocessing | NO |
| **SAT Sweeping** | Merge equivalent variables | NO |
| **Variable Instantiation** | Substitute variables | NO |

### Search Heuristics

| Technique | Description | IsaSAT? |
|-----------|-------------|---------|
| **EVSIDS** | Exponential VSIDS with decaying | YES |
| **VMTF** | Variable Move-To-Front | NO (CreuSAT has this) |
| **CHB** | Conflict History Based branching | NO |
| **Phase Saving** | Remember last assignment polarity | YES |
| **Lucky Phases** | Try satisfying assignment early | NO |
| **Target Phases** | Save best phases found | NO |
| **Random Decisions** | Occasional random choices | Partial |

### Backtracking

| Technique | Description | IsaSAT? |
|-----------|-------------|---------|
| **Non-Chronological Backtracking** | Jump to asserting level | YES |
| **Chronological Backtracking** | Sometimes backtrack one level | NO |
| **Lazy Reimplication** | Defer reimplication after chrono BT | NO |

### Restarts

| Technique | Description | IsaSAT? |
|-----------|-------------|---------|
| **Luby Restarts** | Luby sequence based | Partial |
| **Glucose-style EMA Restarts** | Based on LBD moving average | Partial |
| **Reluctant Doubling** | Delayed restart strategy | NO |
| **Focused Restarts** | Stabilize promising branches | NO |
| **Warm Restarts** | Preserve some state | NO |

### Clause Management

| Technique | Description | IsaSAT? |
|-----------|-------------|---------|
| **2-Watched Literals** | Efficient unit propagation | YES |
| **Blocking Literals** | Extra literal in watch | YES |
| **LBD (Glue) Tracking** | Measure clause quality | Partial |
| **Tier-based Clause Management** | Core/mid/local tiers | NO |
| **Clause Minimization** | Remove redundant literals | YES |
| **Recursive Minimization** | Deep minimization | Partial |
| **On-the-fly Self-Subsumption** | Strengthen during analysis | NO |

### Proof Generation

| Technique | Description | IsaSAT? |
|-----------|-------------|---------|
| **DRAT Proofs** | Standard format | YES |
| **LRAT Proofs** | Efficient checking | NO |
| **FRAT Proofs** | Fast format | NO |
| **Binary Proofs** | Compressed format | Partial |
| **Incremental Proofs (IDRUP)** | For incremental solving | NO |

---

## What IsaSAT is Missing (The Gap)

### Techniques causing the 60% gap:

1. **Inprocessing** (biggest impact)
   - Vivification
   - BVE
   - Blocked clause elimination
   - Failed literal probing

2. **Chronological Backtracking** (significant on some instances)
   - Lazy reimplication

3. **Advanced Restart Policies**
   - Reluctant doubling
   - Focused restarts

4. **Preprocessing**
   - Most preprocessing disabled for verification simplicity

5. **Micro-optimizations**
   - Cache-friendly data structures
   - SIMD operations

### Why these weren't verified:

- **Complexity**: Inprocessing changes clause database non-trivially
- **Invariants**: Harder to state and prove correctness
- **Time**: Verification effort prioritized core CDCL

### Key insight: These CAN be verified

CreuSAT proves more invariants than IsaSAT in some areas. With sufficient effort, every technique can be verified. The question is investment, not possibility.

---

## Phase 1 Implementation Checklist (z4-sat)

### 1A: Core CDCL (Match IsaSAT) - COMPLETE

**Verified from Day 1:**
- [x] Literal/Variable encoding (u32-based)
- [x] Clause database with arena allocation
- [x] 2-Watched literals with blocking literals
- [x] EVSIDS scoring with decay
- [x] Phase saving
- [x] 1-UIP conflict analysis
- [x] Clause minimization (recursive)
- [x] Non-chronological backtracking
- [x] Luby restarts
- [x] DRAT proof generation

**Verification tools:**
- Kani proofs for all unsafe blocks
- TLA+ spec for CDCL state machine
- proptest for clause database invariants
- Differential testing against MiniSat

### 1B: Performance (Match CaDiCaL techniques) - COMPLETE

**Add incrementally, verify each:**
- [x] Chronological backtracking + lazy reimplication
- [x] Glucose-style EMA restarts
- [x] LBD-based clause management
- [x] Tier-based clause database (core/mid/local)
- [x] Reluctant doubling restarts (Luby sequence - already implemented)
- [x] Target phases (save best phases from longest conflict-free trail)

### 1C: Inprocessing (Close the Gap) - COMPLETE

**The techniques that make the 60% difference:**
- [x] Vivification (strengthen clauses)
- [x] Bounded variable elimination (BVE)
- [x] Blocked clause elimination (BCE)
- [x] Subsumption & self-subsumption
- [x] Failed literal probing
- [x] Hyper-ternary resolution

### 1D: Advanced SAT (Competition-winning) - COMPLETE

**Nice to have for SAT Competition placement:**
- [x] Gate extraction (AND/XOR/ITE)
- [x] Congruence closure in preprocessing
- [x] SAT sweeping
- [x] LRAT proof generation
- [x] Binary proof format (DRAT and LRAT)

---

## Benchmark Testing Strategy

### SAT Testing

```bash
# Download competition benchmarks
mkdir -p benchmarks/sat
cd benchmarks/sat

# Get 2023 main track
curl "https://benchmark-database.de/getinstances?query=track%3Dmain_2023&context=cnf" > main_2023.uri
wget --content-disposition -i main_2023.uri

# Run Z4 against all
for f in *.cnf; do
    timeout 5000 z4 $f > results/$f.out 2>&1
done

# Compare against reference (MiniSat, CaDiCaL)
./scripts/differential_test_sat.sh
```

### SMT Testing

```bash
# Download SMT-LIB benchmarks
mkdir -p benchmarks/smt
cd benchmarks/smt

# Get core logics
for logic in QF_LIA QF_LRA QF_UF QF_BV; do
    wget "https://zenodo.org/api/records/16740866/files/${logic}.tar.zst/content" -O ${logic}.tar.zst
    tar -xf ${logic}.tar.zst
done

# Run Z4 against all
./scripts/run_smt_benchmarks.sh

# Compare against Z3, CVC5
./scripts/differential_test_smt.sh
```

---

## Success Metrics

### SAT Competition Parity

| Metric | IsaSAT (2023) | Target for Z4 |
|--------|---------------|---------------|
| Problems solved | 141/400 (35%) | 300/400 (75%) |
| Rank | Last | Top 5 |
| UNSAT with proof | 100% | 100% |

### SMT-COMP Parity

| Division | Z3 (typical) | Target for Z4 |
|----------|--------------|---------------|
| QF_LIA | ~95% | ~90% |
| QF_LRA | ~98% | ~95% |
| QF_BV | ~90% | ~85% |
| QF_UF | ~99% | ~98% |

---

---

# PHASE 2+: SMT SOLVER (Future)

This section covers Phases 2-6 (SMT infrastructure and theory solvers).

## SMT-COMP (Phase 3+)

**What**: Annual competition for SMT solvers
**Website**: https://smt-comp.github.io/
**Years**: 2005-2024 (20 competitions)

**Tracks**:
- Single Query (standard benchmarks)
- Incremental (push/pop sequences)
- Unsat Core
- Model Validation
- Parallel / Cloud

**Benchmark Access**:
```bash
# SMT-LIB 2025 (latest release)
# Zenodo: https://zenodo.org/communities/smt-lib

# Non-incremental benchmarks by logic:
wget https://zenodo.org/api/records/16740866/files/QF_LIA.tar.zst/content -O QF_LIA.tar.zst
wget https://zenodo.org/api/records/16740866/files/QF_BV.tar.zst/content -O QF_BV.tar.zst
wget https://zenodo.org/api/records/16740866/files/QF_LRA.tar.zst/content -O QF_LRA.tar.zst
```

**SMT-LIB Logics by Phase**:

| Phase | Logics | Description |
|-------|--------|-------------|
| **3** | QF_UF, QF_LIA, QF_LRA, QF_IDL, QF_RDL | Core theories |
| **4** | QF_BV, QF_ABV, QF_AUFBV, QF_AUFLIA | Bitvectors + Arrays |
| **5** | QF_S, QF_SLIA, QF_FP, QF_DT | Strings, FP, Datatypes |
| **6** | AUFLIA, AUFLIRA, UFNIA, LIA, LRA | Quantified logics |

---

# LEAN 4 → LEAN 5 INTEGRATION

## Current State: Lean 4 + Z3

Lean 4 uses external SMT solvers for:
1. **`decide` tactic**: Propositional logic (uses native kernel)
2. **`omega` tactic**: Linear arithmetic (native implementation)
3. **`bv_decide` tactic**: Bitvectors (bit-blasting to SAT)
4. **External `smt` tactic**: Calls Z3/CVC5 for complex problems

Z3 is the de facto backend for serious SMT work in Lean.

## Goal: Lean 5 + Z4

Replace Z3 with Z4 as Lean's SMT backend:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LEAN 5                                       │
│                                                                      │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│   │   decide    │  │    omega    │  │  bv_decide  │                │
│   │  (native)   │  │  (native)   │  │  (native)   │                │
│   └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                      smt tactic                              │  │
│   │                                                              │  │
│   │   ┌─────────────────────────────────────────────────────┐   │  │
│   │   │                      Z4                              │   │  │
│   │   │                                                      │   │  │
│   │   │  • Verified (proofs checked by Lean)                │   │  │
│   │   │  • Fast (CaDiCaL-level SAT, Z3-level SMT)           │   │  │
│   │   │  • Native Rust → Lean FFI                           │   │  │
│   │   │  • Produces Lean-checkable proof certificates       │   │  │
│   │   └─────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Bootstrap Strategy

### Stage 1: Z4 Verified by Z3

```
Z3 (unverified, trusted)
  ↓
Verus/Dafny proofs about Z4 algorithms
  ↓
Z4 produces DRAT/Alethe proofs
  ↓
Verified checkers (cake_lpr, carcara) validate
  ↓
Z4 becomes trusted
```

### Stage 2: Z4 Integrated with Lean 4

```
Lean 4
  ↓
Z4 as external solver (like Z3 today)
  ↓
Z4 returns proof certificates
  ↓
Lean kernel checks certificates
  ↓
Soundness: Even if Z4 has bugs, Lean catches them
```

### Stage 3: Z4 Self-Hosting (Lean 5)

```
Z4 replaces Z3 as Verus backend
  ↓
Z4 verifies itself (or next-gen Z5)
  ↓
Lean 5 uses Z4 natively
  ↓
Full bootstrap: Lean 5 ←→ Z4 mutually verified
```

## Proof Certificate Flow

```
User writes: theorem foo : P := by smt

                    ↓

Lean 5 smt tactic:
1. Encode goal as SMT-LIB
2. Call Z4
3. Z4 returns: (unsat, proof_certificate)
4. Lean kernel checks certificate
5. If valid → theorem proven
   If invalid → tactic fails (soundness preserved)
```

## Z4 → Lean Proof Format

Options for proof certificates:

| Format | Pros | Cons |
|--------|------|------|
| **Alethe** | Standard, existing checkers | Needs Lean integration |
| **LFSC** | Used by CVC4/CVC5 | Complex |
| **Native Lean terms** | Direct kernel checking | Must generate valid Lean |
| **Custom Z4 format** | Optimized for Lean | Non-standard |

**Recommended**: Alethe proofs + Lean Alethe checker (to be written).

## Why This Matters

1. **Soundness**: Lean's kernel is the trust anchor, not Z4
2. **Performance**: Z4 can be fast without compromising soundness
3. **Dogfooding**: Lean verifies Z4, Z4 powers Lean
4. **Independence**: No reliance on Microsoft (Z3) or Stanford (CVC5)

---

## References

- SAT Competition: https://satcompetition.github.io/
- SMT-COMP: https://smt-comp.github.io/
- Global Benchmark Database: https://benchmark-database.de/
- SMT-LIB: https://smt-lib.org/
- CaDiCaL: https://github.com/arminbiere/cadical
- Kissat: https://github.com/arminbiere/kissat
- IsaSAT: https://bitbucket.org/isafol/isafol/
- CreuSAT: https://github.com/sarsko/CreuSAT
- Lean 4: https://lean-lang.org/
- Lean Mathlib: https://github.com/leanprover-community/mathlib4
