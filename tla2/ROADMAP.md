# TLA2 Roadmap

**Date:** 2026-01-13
**Goal:** Bug-free TLA+ model checker, faster than TLC

---

## Current State

| Metric | Status |
|--------|--------|
| Specs passing | ~115 (+15 from TLAPS modules) |
| Known correctness bugs | **1** (#120 bind/unbind LET handler) |
| Recently fixed | #86 PaxosCommit symmetry (commit de2a944, verified 1ae8200), #97 MCYoYoPruning |
| Active investigation | #80 MCBakery 20x perf gap - root cause: architectural (bind/unbind vs symbolic) |
| P1 Bugs | #120 Missing Expr::Let handler in bind/unbind (test_let_binding_with_primed_variable fails) |
| Performance regressions | #80 (20x slower than TLC on MCBakery N=3) |
| Performance vs TLC | **20x slower** on MCBakery N=3 (was 6.3x before #79 fix), ~1.1x on bosco |
| Skipped specs | ~30 (6 legitimate, ~24 hiding bugs/gaps) |

**All P1 correctness bugs FIXED.** #86 fixed (de2a944), #87 fixed (34b73b1). Performance gap remains (#80).

**#79 MCBoulanger Fix (W#149):**
Fixed false positive invariant violation caused by IF conditions with primed variables.
Before: 1829 states, false Inv violation. After: correct exploration (7.8M+ states like TLC).

**Hidden problems in "skipped" specs:**

| Problem | Specs | Type |
|---------|-------|------|
| Algorithm bugs | MCInnerSerial | **P1** |
| Performance | AST enum (#72 3.6x, was 5x) | **P1** - in progress |
| Init enumeration | YoYoAllGraphs, LockHS | Bug/Feature gap |
| Over-exploration | bcastByz, bcastByzNoBcast (7.76x states) | Bug (#121) |
| Primed variables | MCTwoPhase | Bug |
| Constant evaluation mode | 10 specs | Feature gap |
| Simulation mode | 2 specs | Feature gap |
| RandomElement | 1 spec | Feature gap |

---

## P0: Architecture Cleanup (#51)

**Status:** COMPLETED (Fixes #51)

**Decision: Delete `enumerate_preprocessed` entirely. Use AST path only.**

All tests pass with AST-only path.

**Performance impact:** Initially 5x slower (0.061s → 0.328s on MCBakery). See follow-up #72.

**#72 Progress:**
- Initial: 0.328s (5x slower than preprocessed)
- Pattern checks: 0.22s (37% improvement)
- Continuation-passing: 0.12s (46% speedup from 0.22s)
- Target: ~0.06s (match preprocessed path) - currently 2x off
- Continuation correctness bug (#78) fixed - parity tests pass on 16 specs
- Continuation now enabled by default (disable with TLA2_NO_CONTINUATION=1)

**#72 Progress (M#146/M#147 - PARTIAL):**
- substitute_let_bindings eliminated (~330 lines deleted)
- CapturedConstraint approach: O(bindings) capture instead of O(AST) substitution
- Performance on N=2 toy config improved 60%
- **BUT:** N=3 realistic config is **52x slower than TLC** (14.6K vs 758K states/s)

**#72 Progress (W#160 - mark/restore):**
- Implemented index-based mark/restore pattern (no more cloning)
- Stack snapshot: O(n) Vec clone → O(1) four integers
- Correctness verified: MCBakery N=2 = 2303 states (matches TLC)
- **KEY FINDING:** Enumeration is only ~10% of total time!
  - N=3 profile: Enumeration 50s, Total 500s → 90% is OUTSIDE enumeration
  - Bottleneck is state management (fingerprinting, queue, invariants), not enumeration
  - mark/restore optimization correct but wrong target

**#88 RESOLVED (R session - fingerprint-only storage):**
- Fingerprint-only mode now default (was storing 280 bytes/state, now 32 bytes)
- Memory reduced 4.3x (6 GB → 1.4 GB for 500K states)
- **Performance improved 6.7x** - Gap reduced from 42x to **6.3x**
- MCBakery N=3: 6,016,610 states in 412s (TLC: 65s)
- State counts match TLC exactly - correctness verified
- Remaining 6.3x gap is algorithmic (successor enumeration, invariant checking)
- **Next:** Focus on #72 for successor enumeration optimization

**Acceptance:**
- [x] enumerate_preprocessed deleted
- [x] Zero fallback counters
- [x] Codebase reduced significantly

---

## P1: Fix Algorithm Bugs

Three specs timeout due to wrong algorithms:

### MCCheckpointCoordination (#13)
- 4x slower than TLC, times out
- Study TLC source, fix algorithm

### YoYoAllGraphs (#74 - Init enumeration)
- Correctness bug #70 fixed (commit b574b33)
- LET memoization fix committed (commit b9c457e)
- **Root cause (M#130):** Not memory/LET - Init uses `\E Nbrs \in SUBSET Edges`
- TLA2 doesn't support existential quantifiers over SUBSET in Init
- **MOVED TO FEATURE GAPS** - This is #74, not P1 bug

### MCInnerSerial
- Eager SUBSET enumeration times out
- Implement lazy enumeration

**Acceptance for all:**
- [ ] Completes within 2x of TLC time
- [ ] State count matches TLC exactly
- [ ] Removed from SKIP_SPECS / LARGE_SPECS

---

## P2: Feature Gaps (13 specs)

**Constant evaluation mode (10 specs):**
CarTalkPuzzle_M1/M2/M3, SimpleMath, Stones, TransitiveClosure, MC_sums_even, SmokeEWD998, SmokeEWD998_SC, PrintValues

**Simulation mode (2 specs):**
SimKnuthYao, SimTokenRing

**RandomElement (1 spec):**
SpanTreeRandom

---

## P3: Refactoring Debt (#75)

Filed M#134. Not blocking - clean up after #72:
- ~~4 dead functions in enumerate.rs~~ (DONE - commit 8426028)
- 3 duplicate `and_contains_X` functions (could be generic)
- 3 clippy warnings (needless_borrow, needless_lifetimes, too_many_arguments)
- Flag proliferation: `TLA2_USE_HANDLE_STATE` - benchmark and choose default
- Dead test code: `is_constant_expr` functions unused in tests
- 2 TODOs not tracked: tla-smt/translate.rs:75, enumerate.rs:1336

---

## TLAPS Library Modules (DONE)

**Status:** COMPLETED - TLAPS modules now available in `test_specs/tla_library/`

Copied from https://github.com/tlaplus/tlapm - both TLC and TLA2 can now use:
- TLAPS.tla (proof backend pragmas)
- FiniteSetTheorems.tla, NaturalsInduction.tla, WellFoundedInduction.tla
- FunctionsFork.tla, FunctionForkTheorems.tla
- SequenceTheorems.tla, SequencesExtFork.tla, BagsTheorems.tla

**Verified working specs (TLA2 = TLC):**
- MCFindHighest: 742 states
- Simple: 723 states
- MCVoting: 6,752 states
- MCConsensus: 4 states
- MCPaxos: 25 states
- SimpleRegular: 277,726 states

**TLA2 bugs blocking (not TLAPS-related):**
- LockHS: Init enumeration doesn't support Stuttering module expressions
- **bcastByz:** Over-exploration - 111,848 states vs TLC's 14,424 (7.76x). Init works correctly (64 states). See #121 - hypothesis: lazy value fingerprinting inconsistency.
- MCTwoPhase: Primed variable evaluation in non-next-state context

---

## Legitimate Skips (6 specs)

These cannot be verified against TLC baseline:
- 2 specs have no config (Consensus, VoteProof - extended by other specs)
- 1 spec (Einstein) needs Apalache
- 3 animation/export specs are not model checking

---

## Completed

### Correctness (Fixed - Verified)
- FastPaxos - 25,617 states = TLC
- All 4 liveness bugs (#62)

### ACTIVE BUGS
- Stack overflow (#57) - needs investigation
- Liveness stuttering (#59) - needs investigation

### FIXED (Verified)
- **#86** PaxosCommit - 119,992 states = TLC (symmetry canonicalization fix, commit de2a944, verified 1ae8200)
- **#87** AllocatorImplementation - 17,701 states = TLC (bind_local + guard check fix, commit 34b73b1, verified R#113)
- #97 MCYoYoPruning - 102 states = TLC (LIFO fix, commit dbcd169, verified 05f0f63)
- MCCheckpointCoordination (#13) - Fixed tla_library shadowing (FiniteSets.tla, Bags.tla)

### Performance
- bosco - 6:49 vs TLC 6:00 (~1.1x)
- #88 - Fingerprint-only storage: 6.7x speedup on MCBakery N=3 (42x → 6.3x gap)

---

## Quick Reference

```bash
cargo test                                                    # All tests
pytest tests/tlc_comparison/test_tlaplus_examples.py -x       # TLC comparison
samply record ./target/profiling/tla check <spec>             # Profile
```
