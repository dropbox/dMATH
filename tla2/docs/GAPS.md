# TLA2 Gap Analysis

**Last Updated**: 2026-01-13 (by MANAGER)
**Status**: Active development - see priorities at bottom

---

This document tracks known gaps between TLA2 and TLC/TLAPM baseline implementations.

---

## Parser (tla-core)

### Status: ~95% Complete

**Working**:
- [x] Module structure (MODULE/EXTENDS/VARIABLES/CONSTANTS)
- [x] Operator definitions (with parameters)
- [x] RECURSIVE operator declarations
- [x] LOCAL definitions
- [x] Basic expressions (logic, arithmetic, sets, functions)
- [x] Quantifiers (FORALL, EXISTS, CHOOSE)
- [x] Temporal operators syntax (always, eventually, leads-to)
- [x] Prime notation (x')
- [x] UNCHANGED
- [x] INSTANCE (parsing only, not resolution)
- [x] Tuple and record syntax
- [x] Function syntax ([x \in S |-> e], f[x], DOMAIN)
- [x] Set comprehension ({x \in S : P}, {e : x \in S})
- [x] CASE expressions
- [x] LET..IN
- [x] IF..THEN..ELSE

**Parse Test Results**: 304/304 tracked `~/tlaplus-examples` `.tla` files parse successfully (see `reports/main/phase0-parse-audit-2025-12-26-23-41.md`)

**NOT Working**:
- [ ] PlusCal algorithm blocks (--algorithm ... end algorithm)
- [ ] Full proof syntax (THEOREM, PROOF, BY, QED, etc.)
- [ ] Some Unicode operators (partial support)
- [ ] Identifiers starting with underscore (seen in ignored TLC-generated `*_TTrace_*.tla` files)
- [ ] Subexpression naming (::)

**Priority**: LOW - Parser is mostly complete for model checking

---

## Config Parser (tla-check/config.rs)

### Status: ~85% Complete

**Working**:
- [x] INIT <name>
- [x] NEXT <name>
- [x] INVARIANT <name>
- [x] INVARIANTS <names> (space or comma separated)
- [x] INVARIANTS (multi-line block format)
- [x] SPECIFICATION <name> (extracts Init/Next from temporal formula)
- [x] SPECIFICATION (multi-line block format)
- [x] CONSTANTS (multi-line block format)
- [x] CONSTANT Name = Value (integers, model values, sets)
- [x] CONSTANT Name <- { model values }
- [x] CONSTANT Name <- [ model value ]
- [x] Comments (\* and (* ... *))
- [x] CONSTRAINT / CONSTRAINTS
- [x] ACTION_CONSTRAINT / ACTION_CONSTRAINTS
- [x] PROPERTY / PROPERTIES (parsing only)
- [x] PROPERTIES (multi-line block format)
- [x] SYMMETRY
- [x] CHECK_DEADLOCK TRUE/FALSE

**NOT Working**:
- [ ] Nested tuple/record syntax in CONSTANTS (e.g., `Clauses = { {<<v1, "neg">>} }`)
- [ ] CONSTANT Name <- OtherOp (operator replacement - runtime only)
- [ ] Complex constant expressions (function construction, etc.)
- [ ] ALIAS directive
- [ ] TERMINAL states declaration (distinguishing intentional termination from deadlock)

**Priority**: MEDIUM - Nested config syntax requested by Z4 project (see `docs/Z4_USER_FEEDBACK.md`)

---

## Evaluator (tla-check/eval.rs)

### Status: ~80% Complete

**Working**:
- [x] Boolean operations (and, or, not, implies, equiv)
- [x] Arithmetic (+, -, *, /, %, ^, ..)
- [x] Comparison (<, >, <=, >=, =, /=)
- [x] Set operations (union, intersection, difference, membership)
- [x] Set comprehension
- [x] Sequences (Seq, Len, Head, Tail, Append, \o)
- [x] SubSeq, SelectSeq, SortSeq
- [x] Quantifiers (forall, exists, choose)
- [x] Functions ([x \in S |-> e], DOMAIN, f[x])
- [x] Records ([field |-> value], r.field)
- [x] Tuples (<<a, b, c>>)
- [x] IF..THEN..ELSE
- [x] CASE expressions
- [x] LET..IN
- [x] User-defined operator application (with argument binding)
- [x] Stdlib: Cardinality, SUBSET, UNION
- [x] Stdlib: Abs, Sign, Range, Min, Max
- [x] Primed variable lookup (with next_state context)
- [x] ENABLED operator
- [x] Bags module (all operators)
- [x] TLCExt module (TLCModelValue)
- [x] Function override (@@) operator

**NOT Working**:
- [ ] Temporal operators (Always, Eventually, LeadsTo) - evaluated in liveness checker, not eval.rs
- [ ] EXCEPT nested paths (r.field.subfield) - single-level works
- [ ] Real numbers
- [ ] TLC module (Print, Assert, etc.) - partial

**Priority**: MEDIUM - Core evaluator is mostly complete

---

## Model Checker (tla-check/check.rs)

### Status: ~70% Complete

**Working**:
- [x] BFS state exploration
- [x] Parallel exploration (adaptive worker selection)
- [x] Invariant checking (safety properties)
- [x] Deadlock detection
- [x] State fingerprinting
- [x] Counterexample trace generation
- [x] Max states/depth limits
- [x] Progress callbacks
- [x] Liveness checking (temporal properties via Büchi automata)
- [x] Weak fairness (WF_vars(A))
- [x] Strong fairness (SF_vars(A))
- [x] State constraints (CONSTRAINT directive)
- [x] Action constraints (ACTION_CONSTRAINT directive)
- [x] JSON output format for AI agents (--output json)

**NOT Working**:
- [ ] Symmetry reduction
- [ ] Stuttering equivalence

**Priority**: MEDIUM - Core model checking is functional

---

## Module System

### Status: ~10% Complete

**Working**:
- [x] EXTENDS parsing
- [x] Built-in modules: Naturals, Integers, Sequences, FiniteSets

**NOT Working**:
- [ ] EXTENDS resolution (loading external .tla files)
- [ ] INSTANCE resolution
- [ ] INSTANCE WITH substitution
- [ ] Parameterized instantiation
- [ ] Circular dependency detection
- [ ] Module path searching

**Priority**: HIGH - Many specs use INSTANCE

---

## Proof System (tla-prove, tla-zenon, tla-cert)

### Status: ~30% Complete

**Working**:
- [x] Basic SMT translation (tla-smt)
- [x] Z3 integration (when installed)
- [x] Zenon tableau prover (tla-zenon) - 51 tests
  - [x] Tableau construction with alpha/beta/gamma/delta rules
  - [x] Equality reasoning
  - [x] Proof tree generation
- [x] Proof certificate checker (tla-cert) - 39 tests
  - [x] Certificate verification (modus ponens, quantifier rules, etc.)
  - [x] Axiom verification (equality, arithmetic, set theory)
  - [x] JSON serialization/deserialization
  - [x] Certificate file I/O
  - [x] Alpha-equivalence for quantifier matching
- [x] Integration: tla-zenon generates certificates verifiable by tla-cert

**NOT Working**:
- [ ] Full TLAPM proof syntax (BY, SUFFICES, HAVE, TAKE, WITNESS, PICK, etc.)
- [ ] SMT-based proof completion
- [ ] Full TLAPM compatibility

**Priority**: MEDIUM - Core tableau prover and certificate checker operational

---

## Validation Status

### Verified Test Suite (scripts/verify_correctness.sh)

**Result: 41 PASS, 0 FAIL, 0 SKIP** (as of 2025-12-30)

| Spec | States | Features Tested |
|------|--------|----------------|
| DieHard | 14 | Basic model checking |
| Counter | 6 | Deadlock detection |
| DiningPhilosophers | 67 | Multiple actions |
| MissionariesAndCannibals | 61 | State constraints |
| TCommit | 34 | Config parsing |
| MCChangRoberts | 137 | Module constants |
| EWD840+Liveness | 302 | **Liveness checking, temporal properties** |
| EnabledFairness+Liveness | 4 | **ENABLED, WF_vars, []<> properties** |
| BidirectionalTransitions1 | 3 | WF with disjunctive actions |
| BidirectionalTransitions2 | 4 | Modular arithmetic |
| BagsTest | 1 | **Bags module operators** |
| TLCExtTest | 1 | **TLCModelValue** |
| CigaretteSmokers | 6 | Multiple action patterns |
| TokenRing | 46,656 | Large state space |
| SimTokenRing | 823,494 | Large-scale verification |
| GameOfLife | 65,536 | Cellular automata |
| HourClock | 12 | Simple state machines |
| MCEcho | 75 | Echo algorithm |
| test1-test30 | 1-2 | TLC baseline compatibility |

### Unit Test Counts (as of #312)

| Crate | Tests |
|-------|-------|
| tla-core | 569 |
| tla-check | 418 |
| tla-zenon | 51 |
| tla-cert | 39 |
| **Total** | **~1077** |

### Performance Benchmark (Updated #314)

| Spec | States | TLA2 (1 worker) | TLC (1 worker) | Gap |
|------|--------|-----------------|----------------|-----|
| DieHard | 14 | 0.27s | <0.1s | 3x |
| DiningPhilosophers | 67 | 0.05s | <0.1s | ~1x |
| bcastFolklore | 501,552 | ~19s | ~16s | **~19%** |

| Spec | TLA2 (8 workers) | TLC (8 workers) | Gap |
|------|------------------|-----------------|-----|
| bcastFolklore | ~6.9s | ~3.5s | ~97% |

**Sequential Performance**: At near-parity with TLC (~19% gap).
**Parallel Scaling**: 2.75x speedup at 8 workers (TLC achieves ~5x).

**Performance Target**: Further optimize parallel scaling to match TLC.

**Recent Improvements** (#314-#316):
- Skip State construction for duplicate states (contains_key pre-check)
- 27% improvement in parallel wall time (9.4s → 6.9s)
- 27% reduction in parallel user time (66s → 48s)
- (#316) Increased ShardedFingerprintSet shards from 64 to 256 (for no-trace mode)
- (#316) No-trace mode achieves 3.3x scaling at 8 workers (vs 2.8x for trace mode)

### Remaining Blockers

1. **Parallel Performance Gap** (MEDIUM PRIORITY)
   - TLA2 is ~19% slower than TLC sequentially (~19s vs ~16s) - **near parity**
   - TLA2 is ~97% slower than TLC in parallel (~6.9s vs ~3.5s at 8 workers)
   - Root cause: Lock contention, cache invalidation, work stealing overhead
   - Recent: #314 added contains_key pre-check (27% improvement in parallel)
   - See `reports/main/expression_profiling_305_2025-12-30.md` for analysis

2. **Module resolution** (MEDIUM)
   - Pattern: EXTENDS UserModule
   - Affects: Many complex specs requiring custom modules
   - Fix: Implement file resolution

3. **Large set overflow** (LOW)
   - Pattern: Large power sets (SUBSET), large function spaces
   - Affects: SlidingPuzzles and similar specs
   - Fix: Lazy evaluation or graceful limits

### Resolved Blockers

- ~~Liveness checking~~ (RESOLVED in #86-#100)
- ~~ENABLED operator~~ (RESOLVED in #85)
- ~~Fairness (WF_, SF_)~~ (RESOLVED in #90)
- ~~Parallel checker context~~ (RESOLVED in #77)
- ~~Function constructor in Init~~ (RESOLVED in #76)
- ~~Function space membership~~ (RESOLVED in #78)
- ~~Sequential performance~~ (NEAR PARITY in #300-#305)
- ~~Zenon prover~~ (RESOLVED in #306-#309)
- ~~Certificate checker~~ (RESOLVED in #306-#311)
- ~~Certificate serialization~~ (RESOLVED in #311)
- ~~Alpha-equivalence for quantifiers~~ (RESOLVED in #313)

---

## Current Priorities

### Priority 1: Parallel Performance Optimization
**Status**: Sequential at parity, parallel improved by 27%
**Target**: Improve parallel scaling from 2.75x to ~5x at 8 workers

Implemented (achieving sequential parity):
- [x] SmallInt optimization (avoid BigInt allocations)
- [x] String interning (O(1) pointer equality)
- [x] ArrayState for internal BFS (O(1) variable lookup)
- [x] IntFunc for array-backed integer-indexed functions
- [x] Liveness consistency caching (15x faster liveness)
- [x] Depth embedding in work queue
- [x] ShardedFingerprintSet (TLC-style MultiFPSet)
- [x] VarRegistry Arc wrapping (O(1) clone)
- [x] FuncApp profiling and IntFunc fingerprint caching
- [x] Contains_key pre-check to skip State construction for duplicates (#314)
- [x] Increased ShardedFingerprintSet shards to 256 for no-trace mode (#316)

Potential parallel optimizations:
- [ ] Work-stealing queue improvements
- [ ] Per-worker memory arenas
- [ ] Lock-free data structures
- [ ] SIMD fingerprinting

### Priority 2: Module Resolution
**Status**: Deferred
**Target**: Support EXTENDS/INSTANCE for user modules

### Priority 3: Proof System Completion
**Status**: In progress (30% complete)
**Recent**: Zenon prover, certificate checker, certificate serialization
**Target**: Full TLAPM compatibility

Future phases:
- [ ] Full proof syntax parsing
- [ ] SMT-based proof completion
- [x] Alpha-equivalence for quantifiers (#313)

---

## Testing Commands

```bash
# Run correctness verification (41 specs)
./scripts/verify_correctness.sh

# Run performance benchmark
./scripts/benchmark_performance.sh

# Single spec model check
./target/release/tla check <spec.tla> --config <config.cfg>

# JSON output for AI agents
./target/release/tla check <spec.tla> --output json

# Run unit tests (excluding z3-dependent crates)
cargo test -p tla-core -p tla-check -p tla-cert -p tla-zenon
```

---

## Notes

- Gap analysis updated based on iteration #312
- **Sequential model checking at near-parity with TLC** (~21% gap)
- **Liveness checking is complete and verified**
- **Proof system actively developed**: Zenon prover + certificate checker operational
- Parser is solid (304/304 specs parse successfully)
- Module system is deferred
- Focus: parallel performance optimization and proof system completion
