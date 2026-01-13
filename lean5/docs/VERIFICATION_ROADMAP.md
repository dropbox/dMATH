# Verification Roadmap - SUPER EXCELLENT PROOFS

## Goal
Zero doubt. If Lean5 accepts a proof, it is correct.

---

## Phase V1: Micro-Checker Cross-Validation (COMPLETE - #76)

**Every type inference verified by independent checker**

### Implementation (Worker #76)
- [x] Added `MicroCert::from_proof_cert()` to convert kernel certificates to micro-checker format
- [x] Added `cross_validate_with_micro()` function for independent verification
- [x] Integrated cross-validation into `infer_type()` in debug builds
- [x] Added proper level normalization to MicroLevel (max, imax simplification)
- [x] Any disagreement = panic with full diagnostic

### Files modified
- `crates/lean5-kernel/src/tc.rs` - infer_type now calls cross-validation in debug mode
- `crates/lean5-kernel/src/micro.rs` - added ProofCert → MicroCert conversion, level normalization

### Coverage
- 100% of `infer_type` calls validated for supported expressions
- Gracefully skips validation for: FVars, Consts, Lits, Projs (unsupported by micro-checker)
- Zero disagreements on full test suite (241 kernel tests)

---

## Phase V2: WHNF Invariant Checking (COMPLETE - #75)

**Runtime verification of reduction correctness**

### Implementation (Worker #75)
- [x] Added idempotence check: `whnf(whnf(e)) == whnf(e)` in debug builds
- [x] Split whnf into `whnf()` (with assertion) and `whnf_core()` (implementation)
- [x] Debug assertion panics with full diagnostic on failure

### Files modified
- `crates/lean5-kernel/src/tc.rs` - whnf function with debug assertions

### Success Criteria
- All idempotence checks pass on every whnf call in debug mode
- Zero violations on full test suite

---

## Phase V3: Differential Testing vs Lean 4 (COMPLETE)

**Byte-for-byte comparison with reference implementation**

### Tasks
- [x] Create `tests/differential/` directory
- [x] Write harness that runs both `lean` and `lean5` on input
- [x] Compare type inference results exactly
- [x] Start with 100 simple expressions
- [x] Expand to 266 expressions (Worker #78)
- [x] Expand to 790 expressions (Worker #79)
- [x] Expand to 1044 expressions (Worker #80)
- [x] Scale to 1000+ expressions - TARGET ACHIEVED

Current harness:
- Dataset: `tests/differential/expressions.txt` (1044 expressions including dependent types)
- Runner: `crates/lean5-kernel/tests/differential.rs` (calls Lean 4 CLI + Lean5 parser/elab/tc)
- Normalization: arrows, forall, binder names canonicalized for comparison
- Requirement: Lean 4 binary on PATH (`lean --version` must succeed)

### Categories (Worker #78)
- Prop/Type/Sort universes
- Lambda expressions with polymorphism
- Forall/Pi expressions (dependent and non-dependent)
- Arrow types with nesting
- Higher-order types (functors, type families)
- Church encodings (booleans, naturals)
- Identity and composition patterns
- K/S combinator patterns

### Infrastructure needed
```bash
# Test script pseudocode
for file in test_cases/*.lean; do
    lean4_result=$(lean --print-type "$file")
    lean5_result=$(lean5 --print-type "$file")
    if [ "$lean4_result" != "$lean5_result" ]; then
        echo "MISMATCH: $file"
        exit 1
    fi
done
```

### Success Criteria
- 1000+ differential test cases
- Zero mismatches

---

## Phase V4: Mutation Testing (COMPLETE - #96)

**Prove our tests catch real bugs**

### Tasks
- [x] Install cargo-mutants: `cargo install cargo-mutants`
- [x] Run on kernel: `cargo mutants -p lean5-kernel`
- [x] Document surviving mutants (Worker #81)
- [x] Add tests to kill survivors - Round 1 (Worker #81)
- [x] Add tests to kill survivors - Round 2 (Worker #82)
- [x] Kill targeted survivors from MANAGER_DIRECTIVE_90 (Worker #91)
- [x] Full-crate file-by-file sweep (Worker #96)
- [x] Target: 0% survival rate - ACHIEVED

### Progress
- Initial run: 165 survivors (54.8% kill rate)
- After Round 1 (#81): 131 survivors (73.5% kill rate)
- After Round 2 (#82): 74 survivors (85.1% kill rate)
- After Round 3 (#84): 23 survivors (95.3% kill rate)
- After Round 4 (#85): ~18-20 survivors estimated (96%+ kill rate)
- Targeted run (Worker #91): 10 mutants, 6 caught, 4 unviable, **0 missed**
- **Complete file-by-file sweep (Workers #95-96)**: All kernel files have **0 missed** mutants

### Final File-by-File Results (Worker #96)
| File | Total | Caught | Unviable | Missed |
|------|-------|--------|----------|--------|
| tc.rs | 138 | 118 | 20 | **0** |
| micro.rs | 98 | 83 | 15 | **0** |
| expr.rs | 79 | 56 | 23 | **0** |
| cert.rs | 78 | 76 | 2 | **0** |
| env.rs | 55 | 40 | 15 | **0** |
| level.rs | 50 | 39 | 11 | **0** |
| inductive.rs | 41 | 36 | 5 | **0** |
| quot.rs | 35 | 24 | 11 | **0** |
| name.rs | 8 | 3 | 5 | **0** |
| lean4_compat.rs | 0 | 0 | 0 | **0** |
| **Total** | **582** | **475** | **107** | **0** |

**Kill rate**: 475/475 = **100%** (excluding unviable mutants)

### Refactoring (Worker #95)
- Removed duplicated `infer_type_unchecked` function (102 lines of code duplication)
- Both debug and release builds now use the same `infer_type_with_cert` implementation
- This ensures consistent behavior and eliminates drift between debug/release paths

### Success Criteria
- [x] `cargo mutants` reports 0 surviving mutants in kernel - **ACHIEVED**

---

## Phase V5: lean4lean Theorem Coverage (COMPLETE - 100%)

**Track formal specification coverage**

### Tasks
- [x] Create `docs/LEAN4LEAN_COVERAGE.md` (Worker #92)
- [x] List all lean4lean theorems about kernel (74 theorems inventoried)
- [x] For each theorem, link to Lean5 test that validates it (74 mapped)
- [x] Track coverage percentage (100%)
- [x] Target: 100% of lean4lean kernel theorems tested (Worker #93)

### Coverage by Category (Worker #93)
| Category | Theorems | Tested | Coverage |
|----------|----------|--------|----------|
| Level Operations | 25 | 25 | 100% |
| Expression Operations | 20 | 20 | 100% |
| Typing Rules | 12 | 12 | 100% |
| Definitional Equality | 10 | 10 | 100% |
| Quotient Types | 5 | 5 | 100% |
| Inductive Types | 2 | 2 | 100%* |

*lean4lean's inductive formalization is incomplete (`sorry`)

### Key lean4lean files to cover
- `Theory/Typing/Basic.lean` - core typing rules ✓
- `Theory/VLevel.lean` - universe levels ✓ (100% as of Worker #93)
- `Theory/Quot.lean` - quotient types ✓
- `Theory/Inductive.lean` - inductive types ✓

### Tests Added (Worker #93) - Final Coverage
- `test_equiv_congr_left` - Verifies lean4lean equiv_congr_left
- `test_equiv_congr_right` - Verifies lean4lean equiv_congr_right
- `test_inst_id` - Verifies lean4lean inst_id (identity substitution)
- `test_inst_map_id` - Verifies lean4lean inst_map_id

### Tests Added (Worker #92)
- `test_lift_inst_commutation_lo` - Verifies lean4lean lift_instN_lo
- `test_lift_inst_commutation_hi` - Verifies lean4lean lift_inst_hi
- `test_inst_lift_identity` - Verifies lean4lean inst_liftN

### Success Criteria
- [x] 100% of lean4lean kernel theorems have corresponding tests

---

## Metrics Dashboard

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Micro-checker coverage | 100%* | 100% | COMPLETE |
| WHNF invariant checks | 100% | 100% | COMPLETE |
| Differential tests | 1044 | 1000+ | COMPLETE |
| Mutation survival | 0 missed (all 10 files) | 0% | COMPLETE |
| lean4lean coverage | 100% | 100% | COMPLETE |

*Mutation testing: 582 total mutants, 475 caught, 107 unviable, **0 missed** (Worker #96)
*lean4lean coverage: 74/74 theorems (Worker #93)

*Micro-checker coverage is 100% for expressions supported (no FVars, Consts, Lits, Projs).
For unsupported expressions, cross-validation is gracefully skipped.

---

## Priority Order

1. **V1 + V2**: Immediate - add to next 2 commits
2. **V3**: This week - create differential harness
3. **V4**: This week - run cargo mutants
4. **V5**: Ongoing - track coverage

Worker: Start with V1 (micro-checker) in your next commit.
