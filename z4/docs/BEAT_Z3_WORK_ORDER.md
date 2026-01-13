# BEAT Z3 WORK ORDER

**Status**: ACTIVE
**Goal**: Z4 must be FASTER than Z3 on raw speed
**Constraint**: No unnecessary new features - only what's needed for performance and integrations
**Integrations**: Kani Fast, Lean 5, TLA+, tRust

---

## CURRENT STATUS (Iteration 226)

### Local Benchmarks (verified)
| Theory | Z4 vs Z3 | Agreement | Files |
|--------|----------|-----------|-------|
| QF_LIA | **1.85x faster** | 100% | 50 |
| QF_UF | **1.14x faster** | 100% | 20 |
| QF_BV | **1.11x faster** | 100% | 50 |
| QF_LRA | **1.07x faster** | 100% | 10 |

### SMT-COMP Benchmarks
- QF_UF (30-file sample): **1.09x faster**, 100% agreement
- QF_LRA (SMT-COMP): Many return "unknown" due to unsupported `define-fun` with `ite`

**Note:** LIA outliers fixed by Diophantine solver (iterations 220-223).

### LIA Outlier Analysis (Iteration 191)

Current outliers (equality-dense problems):
- **system_05.smt2**: 104ms vs 11ms (0.11x) - 3 equalities, 4 vars
- **system_06.smt2**: 91ms vs 10ms (0.11x) - 3 equalities, 4 vars
- **system_11.smt2**: 47ms vs 12ms (0.26x) - 2 equalities, 4 vars

**Iteration 191 finding: HNF cuts are fundamentally incremental**

Investigated: Increased HNF iterations per check() call (5→20 for equality-dense).
Result: No significant improvement. The issue is that HNF cuts tighten bounds one step at a time.

Analysis:
- For system_05.smt2: Z4 does 341 LRA checks, 15 solver resets, 14 branch-and-bound calls
- Each HNF iteration generates 1 cut (row 3 has non-integer RHS)
- Cuts have same coefficient pattern, different bounds (incremental tightening)
- Eventually converges but requires many iterations

**Root cause: Iterative cutting planes vs direct integer lattice solving**

Z3's Diophantine solver (dioph_eq.cpp) uses Griggio's algorithm:
1. Directly solves systems of linear equations over integers
2. Uses extended GCD to parameterize solutions
3. Eliminates variables through substitution
4. Reduces a 4-variable problem to 1 free parameter instantly

Z4's HNF approach:
1. Generates cuts from tight constraint matrix
2. Each cut tightens bounds by a small amount
3. Requires many iterations to converge
4. Branch-and-bound explores many combinations

**Required fix: Implement Diophantine equation solving**

The correct approach is to implement variable elimination for unit-coefficient equalities:
1. When asserting an equality like `-4*v0 + 2*v1 - v2 + v3 = -16`, detect unit coefficient on v3
2. Solve: `v3 = -16 + 4*v0 - 2*v1 + v2`
3. Substitute into all other constraints
4. Reduces problem dimension from 4 to 3
5. Repeat until all equalities are processed

This is what Z3 does in `src/math/lp/dioph_eq.cpp` (Griggio's algorithm).

### Previous Iterations
- **190**: Documented variable elimination approach
- **189**: HNF cuts for equality-dense problems
- **188**: Fixed non-determinism in branch variable selection
- **187**: Soft reset for HNF cut persistence
- **186**: Infrastructure for incremental HNF cut persistence

### Gomory Cut Limitation (Iteration 182)

Z4's simplex uses slack variables for constraints:
- `a*x + b*y <= c` becomes `slack = a*x + b*y` with `slack <= c`
- Slack variables are internal (no term mapping)
- Gomory cuts require non-basic variables to be at bounds
- When cuts involve slack variables, they can't be expressed in original terms

**Result:** Gomory cuts disabled in favor of HNF cuts. HNF works but is incremental.

### SMT-COMP QF_LRA Benchmark Results (100 random files)
**Soundness: 100% (0 disagreements with Z3)**

| Category | Count | Notes |
|----------|-------|-------|
| Z4 agrees with Z3 | 16 | Correct sat/unsat |
| Z4 returns unknown | 50 | Correctly detects unsupported features (ITE, etc.) |
| Z4 timeout | 34 | Performance issue on hard benchmarks |
| **Z4 disagrees with Z3** | **0** | **No soundness bugs** |

**Fixed in iterations 172-177:**
- LRA solver now marks `saw_unsupported=true` for Boolean combinations
- Disequality checking skipped when solver has incomplete info
- Simplex iteration limit reduced to prevent runaway execution
- Deep arithmetic ITE lifting transforms ITEs inside predicates
- Result: All soundness bugs eliminated, ITEs now handled properly

### Root Cause Analysis (Iteration 178)

**Why QF_LRA benchmarks timeout:**

The main bottleneck is the **non-incremental theory solving architecture**:

1. Every SAT model triggers `sync_theory()` which calls `reset()` on the theory solver
2. After reset, ALL theory atoms must be re-parsed and re-asserted (O(n) per atom)
3. For vpm2-30.smt2 with 1507 atoms, this is 1507 parse+assert operations per theory check
4. Z4 calls theory check hundreds of times → parsing overhead dominates

**Z3's approach (9318 bound propagations in 0.92s):**
- Incremental theory solving: doesn't reset on each model
- Bound propagation: theory actively propagates bounds to SAT solver
- Result: 114 arith conflicts (vs Z4's hundreds of 2-literal conflicts)

**Required architectural changes:**
1. Incremental theory solving (don't call reset() on every SAT model)
2. Theory propagation (LRA should propagate implied literals, not just detect conflicts)
3. Better conflict analysis (return more informative conflict clauses)

**Remaining issue:**
- 34 benchmarks timeout due to non-incremental architecture
- These benchmarks have complex Boolean+LRA structure
- Future work: Implement incremental DPLL(T)

---

## PRIORITY 1: Fix LIA (Integer Arithmetic) - URGENT

### Root Cause

Z4's LIA solver (819 lines) uses naive branch-and-bound:
1. Solve LRA relaxation
2. If non-integer value found → generate conflict
3. Hope DPLL(T) finds right branches

Z3's LIA solver uses sophisticated techniques:
1. **GCD test** - Quick infeasibility detection ✓ (implemented in Z4)
2. **Gomory cuts** - Add valid inequalities that cut off fractional solutions
3. **HNF cuts** - Hermite Normal Form cutting planes
4. **Diophantine equation solving** - Direct integer solutions
5. **Smart branching** - Variable selection heuristics

### Gomory Cuts Investigation (Iteration 179)

**Finding: Gomory cuts cannot be generated with the current simplex representation.**

The Z4 simplex solver uses internal "slack" variables to represent constraints:
- Inequality `a*x + b*y <= c` becomes `a*x + b*y + s = c` with `s >= 0`
- These slack variables are internal (no term mapping)

When generating Gomory cuts from a tableau row:
- The row coefficients involve both user variables AND slack variables
- Gomory cuts are only valid when expressed in terms of the original problem variables
- Including slack variables in cuts can cause **incorrect UNSAT** (cuts eliminate valid solutions)

**Why Z3 doesn't have this problem:**
Z3 uses a different simplex representation where cuts can be expressed in terms of original
variables through coefficient substitution. Z4's current implementation skips cuts that
involve internal variables, which means cuts are rarely generated (slack vars are ubiquitous).

**Possible fixes (ranked by effort):**
1. **HNF cuts instead** - May work better with our representation (MEDIUM)
2. **Substitute out slack variables** - Express cuts in original vars only (HARD)
3. **Refactor simplex** - Use representation that supports cuts natively (VERY HARD)

**Current QF_LIA status (Iteration 179):**
- Overall ratio: 0.94x (Z3 slightly faster)
- Agreement: 100%
- Main outlier: system_06.smt2 (16x slower due to branch-and-bound)

### Z3 Source Reference (MIT Licensed - COPY IT)

Key files in `reference/z3/src/math/lp/`:

| File | Lines | Purpose |
|------|-------|---------|
| `int_solver.cpp` | 918 | Main integer solver - **copy the algorithm flow** |
| `int_solver.h` | 103 | Interface definition |
| `gomory.cpp` | 563 | **Gomory cuts - PORT THIS** |
| `gomory.h` | 37 | Gomory interface |
| `hnf_cutter.cpp` | 286 | HNF cuts |
| `int_gcd_test.h` | ~200 | GCD-based infeasibility test |
| `int_branch.h` | ~100 | Branching heuristics |
| `dioph_eq.cpp` | ~400 | Diophantine solver |

### Z3's Algorithm Flow (from int_solver.cpp:246-255)

```cpp
// TRY THESE IN ORDER:
if (r == lia_move::undef && should_hnf_cut()) r = hnf_cut();
if (r == lia_move::undef && should_gomory_cut()) r = gomory(lia).get_gomory_cuts(2);
if (r == lia_move::undef && should_solve_dioph_eq()) r = solve_dioph_eq();
if (r == lia_move::undef) r = int_branch(lia)();  // branching is LAST RESORT
```

### Worker Tasks for LIA

**Task 1.1**: Add GCD test to Z4 LIA (quick win)
- Location: `crates/z4-theories/lia/src/lib.rs`
- Reference: `reference/z3/src/math/lp/int_gcd_test.h`
- Test: `benchmarks/smt/QF_LIA/linear_05.smt2` should go from 0.043s to <0.010s

**Task 1.2**: Port Gomory cuts
- Reference: `reference/z3/src/math/lp/gomory.cpp` (563 lines)
- This is the key technique for cutting off fractional solutions
- Test: `benchmarks/smt/QF_LIA/system_03.smt2` should solve (currently TIMEOUT)

**Task 1.3**: Add smart branching heuristics
- Reference: `reference/z3/src/math/lp/int_branch.h`
- Current Z4 has no variable selection heuristics

**Task 1.4**: Port HNF cutter (if still needed after Gomory)
- Reference: `reference/z3/src/math/lp/hnf_cutter.cpp`

### Acceptance Criteria for LIA

```bash
# This MUST pass with Z4 faster than Z3:
python3 scripts/benchmark_smt.py benchmarks/smt/QF_LIA --timeout 30
# Expected: Overall ratio > 1.0x (Z4 faster)
# Current: Overall ratio = 0.01x (Z3 100x faster)
```

---

## PRIORITY 2: SMT-COMP Benchmarks

Current benchmarks are tiny (50 files). Need real SMT-COMP benchmarks.

### Worker Tasks

**Task 2.1**: Download SMT-COMP 2024 benchmarks
```bash
mkdir -p benchmarks/smtcomp
# QF_LIA benchmarks
git clone --depth 1 https://github.com/SMT-COMP/benchmarks-QF_LIA benchmarks/smtcomp/QF_LIA
# QF_BV benchmarks
git clone --depth 1 https://github.com/SMT-COMP/benchmarks-QF_BV benchmarks/smtcomp/QF_BV
# QF_LRA benchmarks
git clone --depth 1 https://github.com/SMT-COMP/benchmarks-QF_LRA benchmarks/smtcomp/QF_LRA
```

**Task 2.2**: Run Z4 vs Z3 on SMT-COMP and report results
```bash
python3 scripts/benchmark_smt.py benchmarks/smtcomp/QF_LIA --timeout 60 --jobs 4
```

**Task 2.3**: Profile and identify bottlenecks
- Use `perf` or `samply` to find hot spots
- Document where Z4 loses to Z3

### Acceptance Criteria

- Z4 solves >= Z3 count on QF_BV, QF_LRA
- Z4 solves >= 90% of Z3 count on QF_LIA (after LIA fixes)
- Total time competitive (within 20% of Z3)

---

## PRIORITY 3: Integration APIs

### Existing Bridges

| Crate | Lines | Purpose |
|-------|-------|---------|
| `z4-lean-bridge` | 36110 | Lean 5 integration |
| `z4-tla-bridge` | 26851 | TLA+ integration |

### Worker Tasks

**Task 3.1**: Review `z4-lean-bridge` for Lean 5 compatibility
- Check: Does it match current Lean 5 API?
- Check: Is the interface what Lean needs for `omega`/`decide` tactics?

**Task 3.2**: Review `z4-tla-bridge` for TLA+ compatibility
- Check: Does it work with TLC?
- Check: Can it be used from TLAPS?

**Task 3.3**: Document Kani Fast integration requirements
- What API does Kani Fast need from Z4?
- Is incremental solving working? (push/pop)
- Are proofs being generated correctly?

**Task 3.4**: Document tRust integration requirements
- What does tRust need from Z4?
- Any missing features?

---

## Execution Order

1. **IMMEDIATE**: Fix LIA with Gomory cuts (Task 1.2) - this is the 100x gap
2. **NEXT**: Add GCD test (Task 1.1) - quick win
3. **THEN**: Download SMT-COMP benchmarks (Task 2.1)
4. **THEN**: Run comprehensive benchmarks (Task 2.2)
5. **PARALLEL**: Review integrations (Tasks 3.x)

---

## Success Metrics

### Phase 1: LIA Parity (1-2 weeks)
- [ ] QF_LIA benchmark ratio > 0.8x (Z4 within 20% of Z3)
- [ ] No timeouts on current QF_LIA benchmarks
- [ ] All results agree with Z3

### Phase 2: Beat Z3 (2-4 weeks)
- [ ] QF_LIA benchmark ratio > 1.0x (Z4 faster)
- [ ] QF_BV benchmark ratio > 1.0x (maintain lead)
- [ ] QF_LRA benchmark ratio > 1.0x (maintain lead)
- [ ] Overall SMT-COMP competitive

### Phase 3: Integration Complete
- [ ] Lean 5 integration tested and working
- [ ] TLA+ integration tested and working
- [ ] Kani Fast integration tested and working
- [ ] tRust integration tested and working

---

## Notes for Workers

1. **Z3 source is MIT licensed** - you can and should copy algorithms
2. **Study the Z3 code** in `reference/z3/src/math/lp/` before implementing
3. **Run benchmarks** after every change to verify improvement
4. **Don't add unnecessary features** - focus on performance
5. **Commit frequently** with clear messages about what improved

---

## References

- Z3 int_solver: `reference/z3/src/math/lp/int_solver.cpp`
- Z3 gomory: `reference/z3/src/math/lp/gomory.cpp`
- Z4 LIA: `crates/z4-theories/lia/src/lib.rs`
- Benchmark script: `scripts/benchmark_smt.py`

---

## Investigation: count_by_2_000.smt2 Timeout (Iteration 323)

### Root Cause
The count_by_2 benchmark times out because:

1. **Non-inductive blocking lemma**: At level 1, PDR generates blocking lemma `(= a0 a1)` to block the bad state `a0 > a1`. This lemma is correct but NOT inductive (can't be pushed to higher levels).

2. **Fixed point detection blocked**: The frames_equivalent check compares frame lemma counts. Frame 1 has 14 lemmas (13 invariants + 1 blocking), Frame 2 has 13 lemmas (only the pushed invariants). Since counts differ, no fixed point is detected between frames 1 and 2.

3. **Slow SMT queries**: The benchmark uses modular arithmetic (`mod 2`, `mod 3`). Each lemma push attempt does SMT queries involving these mod constraints, which are expensive.

### Key Insight
The relational invariant `a0 <= a1` is discovered and propagated correctly. It DOES block the bad state `a0 > a1`. But the blocking lemma generated at level 1 (`a0 = a1`) is stronger than necessary and not inductive.

### Attempted Fixes (None Worked)
1. **Use cumulative constraints in block_obligation**: Caused regression on other benchmarks
2. **Skip already-pushed lemmas in push_lemmas**: Minor optimization, didn't help with root cause
3. **Semantic frame equivalence**: Too expensive for mod constraints

### Potential Future Fixes
1. **Smarter blocking lemma generation**: When a relational invariant already blocks the bad state, don't generate a new blocking lemma at all
2. **Monotonic frame comparison**: Compare frames by checking if lower frame constraints are subsumed by higher frame constraints
3. **Mod constraint optimization**: Cache or precompute mod-related queries

### Files
- Benchmark: `benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/count_by_2_000.smt2`
- Key invariant: `a0 <= a1` (counter is always <= bound)

---

## Investigation: Integer ITE Splitting Bug (Iteration 325)

### Background
The CHC solver has ITE splitting (`try_split_ites_in_clauses`) that transforms:
- `body ∧ φ(ite(c,t,e)) -> head` into:
  - `body ∧ c ∧ φ(t) -> head`
  - `body ∧ ¬c ∧ φ(e) -> head`

Originally restricted to **Boolean-typed ITEs only** (where `t` and `e` have sort `Bool`).

### Attempted Fix
Iteration 325 tried to extend ITE splitting to **Integer-typed ITEs** (where `t` and `e` have sort `Int`). This would enable solving benchmarks like `three_dots_moving_2_000.smt2` that Z3 solves.

### Bug Found
The fix introduced a **soundness bug** on `three_dots_moving_2_000.smt2`:
- Z4 returned `unsat` (unsafe)
- Expected result: `sat` (safe)
- Z3 correctly returns `sat`

### Root Cause
The ITE splitting adds the condition `c` or `¬c` at the **top level** of the clause body. This is incorrect when the ITE is inside a disjunction.

Example: `body ∧ (P ∨ (Q ∧ G=ite(c,t,e))) => head`

Current (buggy) split:
- Clause 1: `body ∧ c ∧ (P ∨ (Q ∧ G=t)) => head`
- Clause 2: `body ∧ ¬c ∧ (P ∨ (Q ∧ G=e)) => head`

**Problem**: When disjunct `P` is true, the original clause doesn't require `c` or `¬c`. But the split clauses do require `c` (clause 1) or `¬c` (clause 2), which over-constrains the clause.

Correct split would push condition inside disjunction:
- Clause 1: `body ∧ (P ∨ (Q ∧ c ∧ G=t)) => head`
- Clause 2: `body ∧ (P ∨ (Q ∧ ¬c ∧ G=e)) => head`

### Why Bool ITEs Work
Boolean ITEs in CHC constraints typically appear at the top level of Boolean expressions (e.g., `ite(c, P, Q)` where P, Q are predicates). In this context, adding the condition at the top level is equivalent to adding it inside the ITE's context.

### Resolution
Reverted the fix. The Bool-only restriction in `find_ite_path` is intentional:
```rust
ChcExpr::Op(ChcOp::Ite, args)
    if args.len() == 3
        && args[1].sort() == ChcSort::Bool  // <-- Intentional restriction
        && args[2].sort() == ChcSort::Bool =>
```

### Proper Fix (Future Work)
To support Int ITE splitting correctly:
1. When splitting an ITE inside a disjunction, add the condition at the **same nesting level** as the ITE
2. This requires tracking the expression context during path-finding
3. Complexity: Medium-High - requires significant refactoring of `split_clause_once_on_ite`

### Files
- ITE splitting code: `crates/z4-chc/src/problem.rs:299` (`try_split_ites_in_clauses`)
- Buggy benchmark: `benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/three_dots_moving_2_000.smt2`
