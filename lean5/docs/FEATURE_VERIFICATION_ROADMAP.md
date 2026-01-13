# Lean5 Feature Verification Roadmap

**Date:** 2026-01-06
**Status:** APPROVED
**Goal:** Rigorous verification of Lean5 feature completeness and performance parity with Lean 4

**Related:** `VERIFICATION_ROADMAP.md` covers kernel correctness (V1-V5, complete).
This document covers **feature completeness** - does Lean5 implement all Lean 4 features?

---

## Problem Statement

Current completion percentages in ROADMAP_LEAN4_REPLACEMENT.md are **self-reported worker estimates with no methodology**:

| Component | Claimed | Evidence | Realistic |
|-----------|---------|----------|-----------|
| Lake | 95% | Missing: lint, check-*, pack/unpack/upload | ~65-70% |
| LSP | 98% | Has 12 methods, missing: formatting, folding, Lean-specific infoview | ~60-70% |
| .olean | 90% | Import works, export not done | ~70% (import only) |
| Tactics | 90% | 120+ functions, but untested against real proofs | Unknown |

**What's verifiable:** 4,860 tests pass, ~316k LOC compiles, basic features work.

**What's NOT verifiable:** "Lean 4 parity", "Mathlib compatible", percentage claims.

---

## Ultimate Success Criterion

**Lean5 replaces Lean 4 when:**

1. Every Lean 4 feature has a passing test in Lean5
2. Every Mathlib module imports and type-checks correctly
3. Every Mathlib proof elaborates without sorry
4. Performance is **equal or better** than Lean 4 (faster, more memory-efficient)

**This requires testing every single Mathlib item with benchmarks against Lean 4.**

---

## Verification Strategy

### Execution Model

- **Single worker, sequential BFS execution**
- Phase 0 is BLOCKING - must complete before Phase 1
- Phases 1A/1B/1C can interleave within single worker
- Phase 2 depends on Phase 1 completion

### External Dependencies

| Dependency | Handling |
|------------|----------|
| Lean 4 toolchain | Pin to v4.13.0. Warn and skip tests if not installed. |
| .olean fixtures | Check in ~100KB of fixtures for reproducible CI |
| Mathlib4 | NOT checked in. Full install required for Phase 2 validation. |

---

## Phase 0: Foundation Verification (BLOCKING)

**Goal:** Verify the core infrastructure that everything else depends on.

**Estimated effort:** 15-20 commits

### 0.1 Parser Verification

Create exhaustive test suite for Lean 4 syntax coverage.

**Source:** Lean 4 grammar specification, lean4 repo test files

**Deliverables:**
- `tests/lean4_features/parser/` - 50+ test files
- Each syntax construct has positive test (parses) and negative test (rejects invalid)
- Coverage matrix in `docs/PARSER_COVERAGE.md`

**Features to verify:**
```
[ ] Universe levels: Type u, Sort u, forall (α : Type u)
[ ] Binders: (x : T), {x : T}, [x : T], {{x : T}}
[ ] Lambda: fun x => e, fun x y => e, fun | pat => e
[ ] Application: f x, f (x), f x y, @f x
[ ] Let: let x := e; body, let rec f := e; body
[ ] Match: match e with | pat => body
[ ] Do notation: do { stmts }
[ ] Notation: infix, prefix, postfix, notation
[ ] Structure: structure S where field : T
[ ] Class: class C where method : T
[ ] Instance: instance : C T where
[ ] Inductive: inductive T | ctor : T
[ ] Mutual: mutual def f, def g end
[ ] Where: def f where helper := e
[ ] Calc: calc a = b := h1; _ = c := h2
[ ] Have/Let/Show in terms
[ ] Anonymous constructor: ⟨a, b, c⟩
[ ] Field notation: x.field, x.1
[ ] If-then-else: if c then t else e
[ ] Decidable: if h : p then t else e
[ ] Syntax quotations: `(term), `($x)
[ ] Antiquotations: $x, $x:term, $[xs]*
```

### 0.2 Kernel Verification

Verify type checking correctness against Lean 4.

**Note:** Much of this is already covered by VERIFICATION_ROADMAP.md (V1-V5).
This phase extends that work with additional coverage.

**Deliverables:**
- Expand differential test suite to 2000+ expressions
- Property tests for all kernel invariants
- `docs/KERNEL_COVERAGE.md` with pass/fail matrix

**Invariants to verify:**
```
[ ] whnf is idempotent (V2 - DONE)
[ ] is_def_eq is reflexive, symmetric, transitive
[ ] infer_type is deterministic
[ ] Type : Type 1, Prop : Type 0
[ ] Pi types: (x : A) → B x
[ ] Lambda typing: fun (x : A) => b : (x : A) → B
[ ] Application typing: f a : B[a/x] when f : (x : A) → B
[ ] Let typing: let x : A := v; b : B[v/x]
[ ] Universe constraints solved correctly
[ ] Inductive types: constructors, recursors, reduction rules
```

### 0.3 .olean Verification

Verify .olean import works correctly.

**Deliverables:**
- Check in fixtures: `tests/fixtures/olean/v4.13.0/` (~100KB)
  - `Init.Core.olean`
  - `Init.Prelude.olean`
  - `Init.Data.Nat.Basic.olean`
  - `README.md` with regeneration instructions
- Test suite that works offline (fixtures only)
- Test suite that works with full elan install
- `docs/OLEAN_COVERAGE.md` with module import matrix

**Modules to verify (with elan):**
```
[ ] Init.Prelude - loads, all constants type-check
[ ] Init.Core - loads, all constants type-check
[ ] Init.Data.Nat.Basic - loads, Nat operations work
[ ] Init.Data.List.Basic - loads, List operations work
[ ] Std (if installed) - loads without errors
```

### 0.4 Acceptance Criteria for Phase 0

- [ ] Parser: 50+ syntax tests, documented coverage
- [ ] Kernel: 2000+ differential tests, 0 mismatches
- [ ] .olean: Fixtures checked in, offline tests pass
- [ ] All three coverage docs published

**Phase 0 is complete when all boxes are checked.**

---

## Phase 1: Component Verification

**Goal:** Verify each major component against Lean 4 feature set.

**Estimated effort:** 30-40 commits

**Execution:** Sequential BFS - work on 1A, 1B, 1C in rotation, not strict order.

### 1A: Tactic Verification

**Goal:** Every claimed tactic works correctly on representative goals.

**Deliverables:**
- `tests/lean4_features/tactics/` - one file per tactic
- Each tactic tested on 5-10 representative goals
- `docs/TACTIC_COVERAGE.md` with pass/fail matrix

**Tactics to verify (120 claimed):**

Core:
```
[ ] rfl - reflexivity on definitionally equal terms
[ ] rw/rewrite - rewriting with equalities
[ ] simp - simplification with simp lemmas
[ ] ring - polynomial ring solver
[ ] linarith - linear arithmetic
[ ] omega - Presburger arithmetic
[ ] decide - decidable propositions
[ ] norm_num - numeric normalization
[ ] positivity - positivity of expressions
[ ] nlinarith - nonlinear arithmetic
```

Structural:
```
[ ] cases/rcases - case analysis
[ ] induction - structural induction
[ ] constructor - apply constructor
[ ] exists - provide witness
[ ] left/right - disjunction
[ ] split - conjunction
[ ] contradiction - derive False
[ ] exfalso - switch to proving False
[ ] by_contra - proof by contradiction
[ ] by_cases - case split on decidable
```

Automation:
```
[ ] trivial - try simple tactics
[ ] solve_by_elim - backward reasoning
[ ] library_search - lemma search
[ ] aesop - automated proof search
[ ] tauto - propositional tautology
```

(Full list in tactic.rs - all 120 must be tested)

### 1B: Macro Verification

**Goal:** Macro system handles all Lean 4 macro patterns.

**Deliverables:**
- `tests/lean4_features/macros/` - test files
- `docs/MACRO_COVERAGE.md` with pass/fail matrix

**Features to verify:**
```
[ ] Syntax quotations: `(term)
[ ] Antiquotations: $x
[ ] Typed antiquotations: $x:term
[ ] Splice antiquotations: $[xs]*
[ ] Hygiene: fresh names don't capture
[ ] macro keyword: macro "name" ... => ...
[ ] macro_rules: macro_rules | `(...) => ...
[ ] syntax: syntax "keyword" ... : term
[ ] notation: notation "sym" => term
[ ] declare_syntax_cat
[ ] Built-in macros: do, if-let, match, have, let, calc
[ ] Deriving: BEq, Repr, Hashable, Inhabited, DecidableEq
```

### 1C: Elaborator Verification

**Goal:** Elaboration handles all Lean 4 definition forms.

**Deliverables:**
- `tests/lean4_features/elaborator/` - test files
- `docs/ELABORATOR_COVERAGE.md` with pass/fail matrix

**Features to verify:**
```
[ ] def - simple definitions
[ ] theorem/lemma - proof terms
[ ] abbrev - abbreviations (unfold eagerly)
[ ] structure - record types
[ ] class - type classes
[ ] instance - type class instances
[ ] inductive - inductive types
[ ] mutual - mutual recursion
[ ] where - local definitions
[ ] variable - section variables
[ ] universe - universe declarations
[ ] attribute - attribute assignment
[ ] open - namespace opening
[ ] namespace - namespace declaration
[ ] section - section scoping
[ ] import - module imports
```

### 1.4 Acceptance Criteria for Phase 1

- [ ] Tactics: All 120 tested, coverage doc published
- [ ] Macros: All patterns tested, coverage doc published
- [ ] Elaborator: All forms tested, coverage doc published
- [ ] No regressions in Phase 0 tests

---

## Phase 2: Integration Verification

**Goal:** End-to-end testing against real-world usage.

**Estimated effort:** 40-60 commits

### 2A: Standalone Proof Corpus

**Goal:** Exhaustive, careful audit of every Lean4/5 feature through real proofs.

This is the **primary verification artifact** - a curated corpus of 200+ proofs that:
1. Exercises every Lean 4 feature systematically
2. Does NOT require Mathlib installed (uses Lean5 built-in math)
3. Serves as permanent regression test suite

**Deliverables:**
- `tests/proof_corpus/` - 200+ proof files
- `docs/PROOF_CORPUS.md` - catalog with feature coverage map
- Automated test: `cargo test proof_corpus`

**Corpus structure:**
```
tests/proof_corpus/
├── 01_foundations/          # 20 proofs
│   ├── prop_logic.lean      # And, Or, Not, Implies, Iff
│   ├── predicate_logic.lean # Forall, Exists, negation
│   ├── equality.lean        # refl, symm, trans, subst, congr
│   └── ...
├── 02_data_types/           # 30 proofs
│   ├── nat_basic.lean       # Nat arithmetic, induction
│   ├── int_basic.lean       # Int operations
│   ├── list_basic.lean      # List operations, induction
│   ├── option_sum.lean      # Option, Sum types
│   └── ...
├── 03_algebra/              # 40 proofs
│   ├── group_basic.lean     # Group axioms, simple theorems
│   ├── ring_basic.lean      # Ring axioms, polynomial identities
│   ├── field_basic.lean     # Field axioms, division
│   └── ...
├── 04_analysis/             # 30 proofs
│   ├── real_basic.lean      # Real number properties
│   ├── limits.lean          # Epsilon-delta proofs
│   ├── continuity.lean      # Continuous functions
│   └── ...
├── 05_topology/             # 20 proofs
│   ├── open_sets.lean       # Open set definitions
│   ├── continuity_top.lean  # Topological continuity
│   └── ...
├── 06_tactics/              # 40 proofs
│   ├── simp_stress.lean     # Complex simp chains
│   ├── ring_stress.lean     # Complex polynomial goals
│   ├── omega_stress.lean    # Integer arithmetic stress
│   ├── aesop_stress.lean    # Automated search stress
│   └── ...
├── 07_macros/               # 20 proofs
│   ├── custom_notation.lean # User-defined notation
│   ├── do_notation.lean     # Monadic proofs
│   ├── calc_proofs.lean     # Calculational proofs
│   └── ...
└── README.md                # Corpus design, coverage map
```

**Each proof file includes:**
```lean
/-
Feature coverage: [list of features exercised]
Derived from: [Mathlib source or original]
Tactics used: [list]
Expected behavior: [what Lean5 should do]
-/

-- Proof code here
```

### 2B: Lake Verification

**Goal:** Lake commands work on real projects.

**Deliverables:**
- `tests/lake_projects/` - test project fixtures
- `docs/LAKE_COVERAGE.md` with command pass/fail matrix

**Test projects:**
```
tests/lake_projects/
├── hello_world/        # Minimal project
├── with_deps/          # Project with git dependencies
├── multi_lib/          # Multiple lean_lib targets
├── with_tests/         # Project with lean_test
└── with_scripts/       # Project with scripts
```

**Commands to verify:**
```
[ ] lake build - builds all targets
[ ] lake build <target> - builds specific target
[ ] lake clean - removes build artifacts
[ ] lake new - creates new project
[ ] lake init - initializes in current dir
[ ] lake fetch - fetches dependencies
[ ] lake update - updates dependencies
[ ] lake exe - runs executable
[ ] lake test - runs tests
[ ] lake env - shows environment
```

### 2C: LSP Verification

**Goal:** IDE features work correctly in VSCode.

**Deliverables:**
- Manual test protocol in `docs/LSP_TEST_PROTOCOL.md`
- `docs/LSP_COVERAGE.md` with feature pass/fail matrix

**Features to verify:**
```
[ ] Hover - shows type information
[ ] Go to definition - jumps to correct location
[ ] Find references - finds all usages
[ ] Document symbols - lists all definitions
[ ] Workspace symbols - searches across files
[ ] Completion - suggests valid completions
[ ] Diagnostics - shows errors at correct locations
[ ] Code actions - quick fixes work
[ ] Rename - renames across files
[ ] Semantic tokens - syntax highlighting correct
```

### 2D: Mathlib Parity Testing

**Goal:** Prove Lean5 handles every Mathlib item with equal or better performance.

**This is the ultimate validation.** Requires Mathlib4 installed.

**Deliverables:**
- `scripts/mathlib_parity_test.sh` - automated comparison harness
- `docs/MATHLIB_PARITY.md` - results matrix
- Performance benchmarks vs Lean 4

**Testing methodology:**

```
For each Mathlib module M:
  1. Load M.olean in Lean5, measure time T5_load
  2. Load M.olean in Lean4, measure time T4_load
  3. Type-check all constants in Lean5, measure time T5_check
  4. Type-check all constants in Lean4, measure time T4_check
  5. Record: module, constants, T5_load, T4_load, T5_check, T4_check
  6. Flag any constants that fail in Lean5 but pass in Lean4
```

**Success criteria:**
- Every Mathlib module loads in Lean5
- Every constant type-checks in Lean5
- Lean5 time <= Lean4 time (equal or better)
- Zero regressions

**Phases:**
1. Mathlib.Init (~10 modules) - baseline
2. Mathlib.Data (~100 modules) - data structures
3. Mathlib.Algebra (~200 modules) - algebra
4. Mathlib.Analysis (~150 modules) - analysis
5. Full Mathlib (~2000 modules) - complete validation

### 2.4 Acceptance Criteria for Phase 2

- [ ] Proof corpus: 200+ proofs, all pass, coverage doc published
- [ ] Lake: All commands tested on fixture projects
- [ ] LSP: Manual test protocol executed, results documented
- [ ] Mathlib: Parity test harness built, initial modules tested
- [ ] Performance: Lean5 equal or better than Lean4 on benchmarks

---

## Verification Artifacts Summary

| Artifact | Location | Purpose |
|----------|----------|---------|
| Parser coverage | `docs/PARSER_COVERAGE.md` | Syntax feature matrix |
| Kernel coverage | `docs/KERNEL_COVERAGE.md` | Type checking matrix |
| .olean coverage | `docs/OLEAN_COVERAGE.md` | Module import matrix |
| Tactic coverage | `docs/TACTIC_COVERAGE.md` | Tactic pass/fail matrix |
| Macro coverage | `docs/MACRO_COVERAGE.md` | Macro feature matrix |
| Elaborator coverage | `docs/ELABORATOR_COVERAGE.md` | Definition form matrix |
| Proof corpus | `tests/proof_corpus/` | 200+ standalone proofs |
| Lake coverage | `docs/LAKE_COVERAGE.md` | Command pass/fail matrix |
| LSP coverage | `docs/LSP_COVERAGE.md` | IDE feature matrix |
| Mathlib parity | `docs/MATHLIB_PARITY.md` | Full Mathlib comparison |
| .olean fixtures | `tests/fixtures/olean/v4.13.0/` | Reproducible CI fixtures |

---

## Timeline

| Phase | Commits | Cumulative | Milestone |
|-------|---------|------------|-----------|
| Phase 0 | 15-20 | 15-20 | Foundation verified |
| Phase 1 | 30-40 | 45-60 | Components verified |
| Phase 2 | 40-60 | 85-120 | Integration verified |
| Mathlib Full | 50+ | 135-170+ | Complete Mathlib parity |

**Total estimated effort:** 135-170+ commits for complete verification.

---

## Success Definition

**Lean5 verification is COMPLETE when:**

1. All coverage docs show 100% pass rate
2. Proof corpus has 200+ passing proofs
3. Every Mathlib module loads and type-checks
4. Performance benchmarks show Lean5 >= Lean4
5. Zero known regressions from Lean 4 behavior

**Lean5 REPLACES Lean 4 when:**

1. Verification is complete (above)
2. A real Lean 4 user can switch to Lean5 with zero code changes
3. Their workflow (Lake, LSP, tactics, proofs) works identically or better
4. **Every single Mathlib item tested and proven equal or better performance**

---

## Appendix A: External Dependency Handling

### Lean 4 Toolchain

**Required version:** v4.13.0 (pinned)

**Detection:** Check `~/.elan/toolchains/leanprover--lean4---v4.13.0/`

**If not installed:**
```
WARNING: Lean 4 toolchain not found.
Some tests will be skipped.
Install with: elan toolchain install leanprover/lean4:v4.13.0
```

### .olean Fixtures

**Location:** `tests/fixtures/olean/v4.13.0/`

**Contents:** ~100KB of core .olean files for offline testing

**Regeneration:**
```bash
elan override set leanprover/lean4:v4.13.0
cp ~/.elan/toolchains/leanprover--lean4---v4.13.0/lib/lean/Init/Core.olean \
   tests/fixtures/olean/v4.13.0/
# Update README.md with date and commit hash
```

### Mathlib4

**NOT checked in** (4GB+)

**Required for:** Phase 2D Mathlib Parity Testing

**If not installed:**
```
WARNING: Mathlib not found.
Mathlib parity tests will be skipped.
Install with:
  git clone https://github.com/leanprover-community/mathlib4
  cd mathlib4 && lake build
  export MATHLIB_PATH=$(pwd)/.lake/build/lib
```

---

## Appendix B: Honest Assessment Update

After verification is complete, this section will be updated with **actual measured percentages** replacing the self-reported estimates.

| Component | Claimed (Before) | Measured (After) | Notes |
|-----------|------------------|------------------|-------|
| Parser | 97% | TBD | |
| Kernel | 100% | TBD | |
| .olean import | 90% | TBD | |
| Tactics | 90% | TBD | |
| Macros | 95% | TBD | |
| Elaborator | 60% | TBD | |
| Lake | 95% | TBD | |
| LSP | 98% | TBD | |
| Mathlib | 30% | TBD | |

**Measured = (passing tests) / (total tests) based on verification artifacts.**

---

## Appendix C: What is Mathlib?

**Mathlib4** is the community-maintained mathematical library for Lean 4.

```
Lean 4 (language/compiler)
    ├── Init/     ← Built-in: Basic types (Nat, Bool, List), core logic
    ├── Std/      ← Built-in: Standard library (HashMap, IO, etc.)
    └── [external]
            ↓
        Mathlib4  ← 1M+ lines of formalized mathematics
```

**Analogy:** Mathlib is to Lean what NumPy/SciPy is to Python.

**Contents:**
- Algebra (Groups, Rings, Fields, Modules, Galois theory)
- Analysis (Real analysis, measure theory, functional analysis)
- Category Theory (Categories, functors, natural transformations)
- Topology (Topological spaces, continuity, compactness)
- Number Theory (Primes, modular arithmetic)
- ~500,000 theorems total

**Why it matters:** Most serious Lean 4 users depend on Mathlib. If Lean5 can't handle Mathlib, it can't replace Lean 4 for real users.

**Why it's hard:** Mathlib exercises every Lean 4 feature - type classes, tactics, macros, universe polymorphism, coercions, attributes. Any incompleteness in Lean5 will be exposed.
