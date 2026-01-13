# CaDiCaL Feature Checklist for Z4

**GOAL: 100% CaDiCaL Feature Parity**

We have the MIT-licensed CaDiCaL source. Copy every optimization. No exceptions.

**CaDiCaL has 93 source files and 253 configurable options. Z4 must implement ALL of them.**

## Feature Summary

| Category | Implemented | Missing | Total |
|----------|-------------|---------|-------|
| Core SAT | 8 | 6 | 14 |
| Decision Heuristics | 7 | 3 | 10 |
| Restart Strategy | 4 | 2 | 6 |
| Inprocessing | 12 | 24 | 36 |
| Clause Management | 6 | 1 | 7 |
| **Local Search** | 1 | 3 | 4 | ⚠️ **KEY GAP** (walk O(n²)) |
| Lookahead | 0 | 3 | 3 |
| Proof Generation | 2 | 4 | 6 |
| Memory/Performance | 1 | 9 | 10 |
| Incremental Solving | 1 | 8 | 9 |
| External Integration | 0 | 4 | 4 |
| **TOTAL** | **34** | **83** | **117** |

**Progress: 29% implemented → Goal: 100%**

**CaDiCaL is single-threaded with NO GPU. Z4 Phase 6 adds GPU = our edge to beat CaDiCaL.**

**Current Performance:** uf200 ~1.2x, uf250 ~2.5x vs CaDiCaL (lower is better)

---

## Core SAT Solving

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| Propagate | propagate.cpp | ✓ Implemented | - | Unit propagation with watched literals |
| Analyze (1UIP) | analyze.cpp | ✓ Implemented | - | First UIP conflict analysis |
| Decide | decide.cpp | ✓ Implemented | - | Variable selection for branching |
| Backtrack | backtrack.cpp | ✓ Implemented | - | Non-chronological backtracking |
| **Score heap** | score.cpp, heap.hpp | ✓ Implemented | - | O(log n) binary heap for VSIDS |
| Queue (VMTF) | queue.cpp | ✓ Implemented | - | Variable-Move-To-Front heuristic (alternative to VSIDS) |
| Restart | restart.cpp | ✓ Implemented | - | Glucose-style dynamic restarts |
| Reduce | reduce.cpp | ✓ Implemented | - | Learned clause database reduction |
| **Reduce opt** | reduce.cpp | ❌ Missing | P3 | `opts.reduceopt` - interval calc (0=prct, 1=sqrt, 2=max) |
| **Dynamic tier** | tier.cpp | ❌ Missing | P2 | `opts.recomputetier` - recompute tier limits dynamically |
| **Chronological backtrack** | propagate.cpp | ❌ Missing | P2 | `opts.chrono` - out-of-order backtracking |
| **Trail reuse** | decide.cpp | ❌ Missing | P2 | `opts.restartreusetrail` - keep assignments on restart |
| **OTFS** | analyze.cpp | ❌ Missing | P2 | On-The-Fly Self-Subsumption during conflict analysis |

## Decision Heuristics

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| VSIDS heap | score.cpp | ✓ Implemented | - | Activity-based variable ordering with decay |
| VMTF queue | queue.cpp | ✓ Implemented | - | Doubly-linked list ordering by recency of bumping |
| Phase saving | phases.cpp | ✓ Implemented | - | Save last assigned polarity per variable |
| Rephasing | rephase.cpp | ✓ Implemented | - | Periodically reset phases to target/best/random |
| Target phases | phases.cpp | ✓ Implemented | - | Track best phases seen during search |
| **Lucky phases** | lucky.cpp | ✓ Implemented | - | 6 initial phase strategies (forward/backward, horn) |
| **Reason bumping** | analyze.cpp | ✓ Implemented | - | Bump variables in reason clauses |
| **Random decisions** | decide.cpp | ❌ Missing | P3 | `opts.randec` - random decision sequences |
| **Stubborn I/O phases** | decide.cpp | ❌ Missing | P3 | `opts.stubbornIOfocused` - periodic I/O phase forcing |
| **Forced phase** | decide.cpp | ❌ Missing | P3 | `opts.forcephase` - always use initial phase |

### Lucky Phases Detail (lucky.cpp)
Quick pre-solving checks that can solve easy formulas instantly:
1. `trivially_false_satisfiable()` - All variables false
2. `trivially_true_satisfiable()` - All variables true
3. `forward_false_satisfiable()` - Assign in variable order, prefer false
4. `forward_true_satisfiable()` - Assign in variable order, prefer true
5. `backward_false_satisfiable()` - Reverse variable order, prefer false
6. `backward_true_satisfiable()` - Reverse variable order, prefer true
7. `positive_horn_satisfiable()` - First positive literal per clause
8. `negative_horn_satisfiable()` - First negative literal per clause

## Restart Strategy

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| Glucose restarts | restart.cpp | ✓ Implemented | - | LBD-based restart decisions |
| Stable/Unstable | stable.cpp, unstable.cpp | ✓ Implemented | - | Alternating restart modes |
| Reluctant doubling | restart.cpp | ✓ Implemented | - | Luby-like restart sequence |
| **Stabilize only** | restart.cpp | ❌ Missing | P3 | `opts.stabilizeonly` - only stabilizing phases |
| **Restart margins** | restart.cpp | ❌ Missing | P3 | Different margins for stable vs focused mode |
| EMA averages | averages.cpp, ema.cpp | ✓ Implemented | - | Exponential moving averages for LBD/trail |

## Inprocessing

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| Vivification | vivify.cpp | ✓ Implemented | - | Strengthen clauses by unit propagation |
| **Vivify once** | vivify.cpp | ❌ Missing | P3 | `opts.vivifyonce` - only vivify each clause once |
| **Vivify instantiate** | vivify.cpp | ❌ Missing | P3 | `opts.vivifyinst` - instantiate last literal |
| **Vivify retry** | vivify.cpp | ❌ Missing | P3 | `opts.vivifyretry` - re-vivify if successful |
| **Vivify demote** | vivify.cpp | ❌ Missing | P3 | `opts.vivifydemote` - demote irredundant clauses |
| **Vivify flush** | vivify.cpp | ❌ Missing | P3 | `opts.vivifyflush` - flush subsumed before rounds |
| **Vivify tier calc** | vivify.cpp | ❌ Missing | P3 | `opts.vivifycalctier` - recalculate tier limits |
| Subsumption | subsume.cpp | ✓ Implemented | - | Remove subsumed clauses |
| **Subsume strengthen** | subsume.cpp | ❌ Missing | P3 | `opts.subsumestr` - strengthen during subsumption |
| **Subsume limited** | subsume.cpp | ❌ Missing | P3 | `opts.subsumelimited` - limit subsumption checks |
| BVE | elim.cpp | ✓ Implemented | - | Bounded variable elimination |
| **BVE scoring** | elim.cpp | ❌ Missing | P3 | `opts.elimsum/elimprod` - scoring weights |
| **BVE def mining** | elim.cpp | ❌ Missing | P3 | `opts.elimdef` - use Kitten for definitions |
| **BVE limited** | elim.cpp | ❌ Missing | P3 | `opts.elimlimited` - limit effort |
| **BVE bound** | elim.cpp | ❌ Missing | P3 | `opts.elimboundmax` - dynamic bound increase |
| **Fast BVE** | elimfast.cpp | ❌ Missing | P2 | Lower-limit BVE for preprocessing |
| **Pure literal detection** | flags.cpp | ❌ Missing | P3 | `mark_pure()` - eliminate pure literals |
| **Eager subsumption** | subsume.cpp | ❌ Missing | P2 | `opts.eagersubsume` - subsume recently learned |
| **Inprobing** | probe.cpp | ❌ Missing | P3 | `opts.inprobing` - inprocessing probing (separate from preproc) |
| BCE | block.cpp | ✓ Implemented | - | Blocked clause elimination |
| **BCE limits** | block.cpp | ❌ Missing | P3 | `opts.blockminclslim/blockmaxclslim/blockocclim` |
| Probing | probe.cpp | ✓ Implemented | - | Failed literal detection |
| **Probe HBR** | probe.cpp | ❌ Missing | P2 | `opts.probehbr` - learn hyper binary clauses |
| Gates | gates.cpp | ✓ Implemented | - | Gate extraction (AND/XOR/ITE) |
| **Gate AND** | gates.cpp | ❌ Missing | P3 | `opts.elimands` - find AND gates |
| **Gate ITE** | gates.cpp | ❌ Missing | P3 | `opts.elimites` - find ITE gates |
| **Gate XOR** | gates.cpp | ❌ Missing | P3 | `opts.elimxors` - find XOR gates with limit |
| **Gate Equiv** | gates.cpp | ❌ Missing | P3 | `opts.elimequivs` - find equivalence gates |
| Sweeping | sweep.cpp | ✓ Implemented | - | SAT sweeping for equivalences |
| **Sweep complete** | sweep.cpp | ❌ Missing | P3 | `opts.sweepcomplete` - run to completion |
| **Sweep random** | sweep.cpp | ❌ Missing | P3 | `opts.sweeprand` - randomize sweeping |
| **Sweep flip** | sweep.cpp | ❌ Missing | P3 | `opts.sweepfliprounds` - flip rounds |
| HTR | ternary.cpp | ✓ Implemented | - | Hyper ternary resolution |
| **Ternary rounds** | ternary.cpp | ❌ Missing | P3 | `opts.ternaryrounds` - number of rounds |
| **Ternary limits** | ternary.cpp | ❌ Missing | P3 | `opts.ternaryocclim/ternarymaxadd` - limits |
| **Backbone** | backbone.cpp | ❌ Missing | P2 | Detect forced literals via binary clause probing |
| **Decompose (SCC)** | decompose.cpp | ❌ Missing | P2 | Tarjan's SCC algorithm for equivalent literal detection |
| **Transitive reduction** | transred.cpp | ❌ Missing | P3 | Remove transitive binary clauses |
| **Factoring** | factor.cpp | ❌ Missing | P3 | Introduce fresh variables to reduce clause count |
| **Deduplicate** | deduplicate.cpp | ❌ Missing | P3 | Remove duplicate clauses + hyper unary resolution |
| **Compact** | compact.cpp | ❌ Missing | P3 | Variable renumbering to remove holes |
| **Congruence closure** | congruence.cpp | ❌ Missing | P3 | Gate-based equivalence propagation |
| **Congruence AND** | congruence.cpp | ❌ Missing | P3 | `opts.congruenceand` - AND gate extraction |
| **Congruence XOR** | congruence.cpp | ❌ Missing | P3 | `opts.congruencexor` - XOR gate extraction |
| **Congruence ITE** | congruence.cpp | ❌ Missing | P3 | `opts.congruenceite` - ITE gate extraction |
| **Congruence binaries** | congruence.cpp | ❌ Missing | P3 | `opts.congruencebinaries` - binary extraction |
| **Instantiate** | instantiate.cpp | ❌ Missing | P3 | Variable instantiation to remove literals |
| **Backward subsumption** | backward.cpp | ❌ Missing | P3 | Eager backward subsumption during BVE |
| **Conditioning** | condition.cpp | ❌ Missing | P3 | Globally blocked clause elimination |
| **Covered clause elim** | cover.cpp | ❌ Missing | P3 | CCE/ACCE with asymmetric literal addition |
| **Definition extraction** | definition.cpp | ❌ Missing | P3 | Extract definitions using Kitten sub-solver |

### Decompose Detail (decompose.cpp)
Uses Tarjan's SCC algorithm on binary implication graph to:
- Find equivalent literals (same SCC)
- Substitute equivalent literals
- Detect contradictions (literal and its negation in same SCC → UNSAT)

### Backbone Detail (backbone.cpp)
Detects "backbone" literals - variables that must have same value in all solutions:
- Probe binary clauses to find forced assignments
- Multiple rounds with tick-based limits
- Derives unit clauses

### Transitive Reduction (transred.cpp)
Removes redundant binary clauses:
- If `a → b` and `a → c → b`, then `a → b` is transitive
- BFS-based path finding to detect transitivity
- Also finds failed literals in binary implication graph

### Backward Subsumption (backward.cpp)
Eager subsumption check during BVE:
- Check if newly resolved clauses subsume/strengthen existing clauses
- Hyper unary resolution: `(a ∨ b) ∧ (a ∨ ¬b) → a`
- `elim_backward_clause()` - check one clause against occurrence list

### Conditioning (condition.cpp)
Globally blocked clause elimination (Kiesl PhD thesis 2019):
- Simulates "headlines" from FAN algorithm
- Uses conditional autarkies
- `condition_round()` - main elimination pass
- Triggers based on decision level vs backjump average

### Covered Clause Elimination (cover.cpp)
CCE/ACCE from LPAR-10/JAIR'15:
- Asymmetric literal addition (ALA) - propagation-based extension
- Covered literal addition (CLA) - resolution-based extension
- `cover_clause()` - try CCE on single clause
- Extension stack management for witness reconstruction

### Variable Instantiation (instantiate.cpp)
Remove literals with few occurrences:
- Try assigning literal true, propagate clause to false
- If conflict found, literal can be removed from clause
- `instantiate_candidate()` - single instantiation attempt
- Triggered at end of BVE rounds

### Definition Extraction (definition.cpp)
Uses Kitten sub-solver to find definitions:
- Export clauses to Kitten (lightweight SAT solver)
- If UNSAT, definition exists
- Extract core clauses as gate definitions
- Can derive unit clauses from one-sided cores

## Clause Management

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| Watched literals | watch.cpp | ✓ Implemented | - | Two-literal watching scheme |
| Binary clauses | bins.cpp | ⚠️ Partial | P1 | Special handling for binary clauses |
| Ternary handling | ternary.cpp | ⚠️ Partial | P2 | Special handling for ternary clauses |
| Tier system | tier.cpp | ✓ Implemented | - | Clause quality tiers for reduction |
| LBD scoring | - | ✓ Implemented | - | Literal Block Distance metric |
| **Clause shrinking** | shrink.cpp | ❌ Missing | P2 | Block-level UIP to shrink learned clauses |
| Clause minimization | minimize.cpp | ✓ Implemented | - | Remove redundant literals via resolution |
| **Minimize depth** | minimize.cpp | ❌ Missing | P3 | `opts.minimizedepth` - recursion depth limit |
| **Minimize ticks** | minimize.cpp | ❌ Missing | P3 | `opts.minimizeticks` - count ticks during minimize |

### Clause Shrinking Detail (shrink.cpp)
More aggressive than minimization:
- Groups literals by decision level into "blocks"
- Finds UIP within each block
- Can replace multiple literals with single UIP
- Uses `reap` heap for efficiency
- Multiple modes: `opts.shrink > 1` enables large clause resolution

## Local Search

**⚠️ KEY PERFORMANCE GAP: Local search/walk is likely the primary reason for Z4's heuristic variance vs CaDiCaL.**

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| **Random walk** | walk.cpp | ⚠️ O(n²) | P2 | ProbSAT walk exists but O(n²) per flip; disabled for < 500 vars |
| **Warmup** | warmup.cpp | ✓ Implemented | - | O(1) propagation-based phase initialization |
| **Flip** | flip.cpp | ❌ Missing | P2 | Flip individual variables |
| **Walk full occs** | walk_full_occs.cpp | ❌ Missing | **P1** | O(1) walk with occurrence lists - KEY for performance |

### Local Search Detail (walk.cpp)
ProbSAT implementation:
- CB (constant base) values from Balint's thesis
- Break-count scoring: probability inversely proportional to `base^break_count`
- Adaptive CB selection based on average clause size
- Tracks minimum unsatisfied clauses
- Updates saved phases with best assignment found
- Binary clause handling via `TaggedBinary`

### Warmup Detail (warmup.cpp)
Prepares for local search by:
- Propagating with decisions (ignoring conflicts)
- Updates target phases for walk initialization
- `warmup_propagate_beyond_conflict()` continues past conflicts

## Lookahead

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| **Lookahead probing** | lookahead.cpp | ❌ Missing | P2 | Enhanced probing for cube generation |
| **Cover** | cover.cpp | ❌ Missing | P3 | Covering number heuristic |
| **Cube generation** | lookahead.cpp | ❌ Missing | P3 | Generate cubes for parallel solving |

### Lookahead Detail (lookahead.cpp)
Enhanced probing for cube-and-conquer:
- `most_occurring_literal()` - finds best split variable
- `lookahead_probing()` - probing without propagation limit
- `generate_cubes()` - recursive cube generation to specified depth
- Integrates with decompose and ternary preprocessing

## Proof Generation

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| DRAT | drattracer.cpp | ✓ Implemented | - | Deletion-based Resolution Asymmetric Tautology |
| LRAT | lrattracer.cpp | ✓ Implemented | - | Labeled RAT with clause IDs |
| FRAT | frattracer.cpp | ❌ Missing | P3 | Flexible RAT format |
| IDRUP | idruptracer.cpp | ❌ Missing | P3 | Incremental DRUP |
| LIDRUP | lidruptracer.cpp | ❌ Missing | P3 | Labeled incremental DRUP |
| VeriPB | veripbtracer.cpp | ❌ Missing | P3 | Pseudo-boolean proof format |
| Checker | checker.cpp | ⚠️ External | - | Internal proof checker |

## Memory/Performance

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| **Arena allocator** | arena.cpp | ❌ Missing | P2 | Two-space copying collector for clauses |
| **Arena type** | arena.cpp | ❌ Missing | P3 | `opts.arenatype` - clause locality (1=clause, 2=var, 3=queue) |
| **Arena compact** | arena.cpp | ❌ Missing | P3 | `opts.arenacompact` - keep clauses compact |
| **Arena sort** | collect.cpp | ❌ Missing | P3 | `opts.arenasort` - sort clauses after GC |
| Occurrence lists | occs.cpp | ⚠️ Basic | P2 | Full occurrence lists for preprocessing |
| **Radix sort** | radix.hpp | ❌ Missing | P3 | O(n) radix sort for clause sorting |
| **Reap heap** | reap.cpp/hpp | ❌ Missing | P3 | Special heap for shrinking |
| **Variable shuffle** | score.cpp | ❌ Missing | P3 | `opts.shuffle` - shuffle variable order |
| **Score shuffle** | score.cpp | ❌ Missing | P3 | `opts.shufflescores` - randomize initial scores |
| **Flush redundant** | reduce.cpp | ❌ Missing | P3 | `opts.flush` - periodically flush redundant clauses |

### Arena Allocator Detail (arena.cpp)
Memory management for clause compaction:
- Two-space copying collector (`from` and `to` spaces)
- `prepare()` allocates new space
- `swap()` switches spaces after copying live clauses
- Reduces fragmentation, improves cache locality

## Incremental Solving

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| Assumptions | assume.cpp | ⚠️ Partial | P2 | Incremental solving with assumptions |
| **Restore clauses** | restore.cpp | ❌ Missing | P2 | Restore eliminated clauses for incremental |
| **Extend solution** | extend.cpp | ❌ Missing | P2 | Reconstruct witness from extension stack |
| **Flip literals** | flip.cpp | ❌ Missing | P3 | Flip assigned literals (for local search) |
| Constraints | constrain.cpp | ❌ Missing | P3 | Constraint clauses |

### Restore Detail (restore.cpp)
For incremental solving after preprocessing:
- Clauses eliminated by BVE/BCE need restoration when witnesses involved
- Tracks "tainted" literals (added after simplification)
- `restore_clauses()` - restore all needed clauses
- Recomputes witness bits after restoration

### Extend Detail (extend.cpp)
Witness reconstruction from extension stack:
- For eliminated clauses, need to flip "blocking" literals
- Extension stack stores: witness literals + clause literals + ID
- `extend()` - reconstruct satisfying assignment
- Handles conditional autarkies from conditioning

## External Integration

| Feature | CaDiCaL File | Z4 Status | Priority | Description |
|---------|--------------|-----------|----------|-------------|
| IPASIR | ipasir.cpp | ❌ Missing | P3 | Standard incremental SAT interface |
| External propagate | external_propagate.cpp | ❌ Missing | P3 | User-provided propagation |
| **ILB (In-processing Lazy Backtrack)** | external_propagate.cpp | ❌ Missing | P3 | Lazy backtracking with external propagator |
| **Observed variables** | external_propagate.cpp | ❌ Missing | P3 | Track external variable assignments |

---

## Priority Legend

- **P0**: Critical - blocking performance parity
- **P1**: High - significant performance impact
- **P2**: Medium - noticeable improvement
- **P3**: Low - marginal or situational benefit

---

## Implementation Progress

### Completed (✓)
1. ~~**P0: Implement heap-based VSIDS** (score.cpp)~~ ✓ DONE (#127)
   - Implemented O(log n) binary heap for variable selection

2. ~~**P1: Better clause minimization** (minimize.cpp)~~ ✓ DONE (#128)
   - Implemented CaDiCaL-style poison/removable marking
   - Added depth limiting for recursion

3. ~~**P1: Reason bumping** (analyze.cpp)~~ ✓ DONE (#129)
   - Bump variables in reason clauses of learned clause literals
   - Depth limit: 1 in focused mode, 2 in stable mode

4. ~~**P1: Lucky phases** (lucky.cpp)~~ ✓ DONE (#130)
   - Quick pre-solving checks (6 strategies)
   - Skipped for larger formulas (>220 vars or >1000 clauses)

5. ~~**P1: VMTF decision queue** (queue.cpp)~~ ✓ DONE (#131)
   - CaDiCaL-style doubly-linked VMTF queue with unassigned cursor
   - Used in focused mode; VSIDS heap used in stable mode

### Current Performance Status (commit #132)

**uf200 (100 files):** 1.23x ratio, Z4 wins 51/100
**uf250 (100 files):** 2.67x ratio, Z4 wins 38/100, 0 disagreements

**Root Cause Analysis:**
The gap is NOT due to implementation speed - it's due to heuristic variance:
- When Z4 wins: finds solutions with 10-50x fewer conflicts than CaDiCaL
- When CaDiCaL wins: Z4 explores 5-30x more conflicts

Example comparisons on uf250:
- uf250-055.cnf: Z4 wins (21ms, 1K conflicts) vs CaDiCaL (1.57s, 28K conflicts)
- uf250-07.cnf: CaDiCaL wins - Z4 (7.0s, 428K conflicts) vs CaDiCaL (0.55s, 28K conflicts)

**Key Missing Feature:** Walk-based phase initialization (walk.cpp, warmup.cpp)
- CaDiCaL uses ProbSAT local search to find good initial phases
- Z4 only uses the assignment from longest conflict-free trail
- This likely explains the high variance in heuristic quality

---

## Implementation Roadmap: 33 Missing Features

### Phase A: Local Search (P1 - Critical for heuristic quality)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 1 | **Walk (ProbSAT)** | walk.cpp | High - ~800 lines |
| 2 | **Warmup** | warmup.cpp | Medium - ~150 lines |
| 3 | Walk full occs | walk_full_occs.cpp | Medium - variant of walk |
| 4 | Flip | flip.cpp | Low - ~150 lines |

### Phase B: Clause Optimization (P2 - Reduce learned clause overhead)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 5 | **Clause shrinking** | shrink.cpp | High - ~400 lines, reap heap |
| 6 | Reap heap | reap.cpp/hpp | Medium - bucket-based priority queue |

### Phase C: Inprocessing (P2 - Simplification techniques)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 7 | **Decompose (SCC)** | decompose.cpp | High - Tarjan's algorithm |
| 8 | **Backbone detection** | backbone.cpp | Medium - binary probing |
| 9 | Transitive reduction | transred.cpp | Medium - BFS path finding |
| 10 | Deduplicate | deduplicate.cpp | Low - hash-based |
| 11 | Compact | compact.cpp | Medium - variable renumbering |
| 12 | Backward subsumption | backward.cpp | Medium - during BVE |
| 13 | Factoring | factor.cpp | High - fresh variable introduction |
| 14 | Instantiate | instantiate.cpp | Medium - literal removal |
| 15 | Conditioning | condition.cpp | High - globally blocked clauses |
| 16 | Covered clause elim | cover.cpp | High - ALA/CLA |
| 17 | Definition extraction | definition.cpp | High - needs Kitten sub-solver |
| 18 | Congruence closure | congruence.cpp | Very High - ~7000 lines |

### Phase D: Lookahead (P2 - Cube generation)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 19 | Lookahead probing | lookahead.cpp | High |
| 20 | Cover | cover.cpp | Medium |
| 21 | Cube generation | lookahead.cpp | Part of lookahead |

### Phase E: Memory & Performance (P2)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 22 | **Arena allocator** | arena.cpp | High - copying GC |
| 23 | Radix sort | radix.hpp | Low - template |

### Phase F: Incremental Solving (P2)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 24 | Restore clauses | restore.cpp | Medium |
| 25 | Extend solution | extend.cpp | Medium |

### Phase G: Proof Formats (P3)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 26 | FRAT | frattracer.cpp | Medium |
| 27 | IDRUP | idruptracer.cpp | Medium |
| 28 | LIDRUP | lidruptracer.cpp | Medium |
| 29 | VeriPB | veripbtracer.cpp | Medium |

### Phase H: External Integration (P3)
| # | Feature | File | Est. Complexity |
|---|---------|------|-----------------|
| 30 | IPASIR interface | ipasir.cpp | Medium |
| 31 | External propagate | external_propagate.cpp | High |
| 32 | ILB | external_propagate.cpp | Part of ext propagate |
| 33 | Observed variables | external_propagate.cpp | Part of ext propagate |

---

## Immediate Priority Order

**Week 1-2: Local Search (close the heuristic gap)**
1. walk.cpp - ProbSAT random walk
2. warmup.cpp - Phase initialization

**Week 3-4: Core Inprocessing**
3. decompose.cpp - SCC/equivalent literals
4. backbone.cpp - Forced literal detection
5. shrink.cpp + reap.cpp - Clause shrinking

**Week 5-6: Advanced Inprocessing**
6. transred.cpp - Transitive reduction
7. deduplicate.cpp - Duplicate removal
8. compact.cpp - Variable compaction
9. backward.cpp - Backward subsumption

**Week 7-8: Memory & Incremental**
10. arena.cpp - Copying garbage collector
11. restore.cpp + extend.cpp - Incremental support

**Week 9+: Remaining features**
- All P3 features
- External integration
- Advanced proof formats

---

## CaDiCaL Configuration Options (253 total)

Key options affecting performance (from options.hpp):

### Search
- `stable` / `unstable` - mode switching
- `restartint`, `restartmargin` - restart intervals
- `reducetier1`, `reducetier2` - clause reduction thresholds

### Inprocessing
- `vivify`, `vivifyonce`, `vivifytier1`, `vivifytier2`
- `subsume`, `subsumeclslim`, `subsumeocclim`
- `elim`, `elimbound`, `elimclslim`, `elimocclim`
- `probe`, `probeint`, `proberounds`
- `transred`, `transredeffort`
- `decompose`, `decomposerounds`
- `factor`, `factorsize`, `factorcandrounds`
- `backbone`, `backbonerounds`, `backbonemaxrounds`
- `deduplicate`
- `compact`, `compactlim`, `compactmin`

### Local Search
- `walk`, `walkeffort`, `walkmineff`, `walkmaxeff`
- `warmup`
- `walkredundant`

### Decision
- `phase`, `target`, `rephase`
- `lucky`, `luckyassumptions`

### Minimization
- `minimize`, `minimizedepth`
- `shrink`, `shrinkreap`

---

## GPU Acceleration (Phase 6)

GPU is planned for Phase 6 per DESIGN.md:

- **Parallel SAT solving**: Multiple CDCL instances on GPU cores
- **BV operations**: Parallel bit-blasting and evaluation
- **Local search**: GPU-accelerated stochastic local search
- **Target**: 10x speedup on parallelizable workloads

GPU is NOT a priority until we match CaDiCaL on single-threaded performance.
Focus: Get all CaDiCaL optimizations first, then parallelize.

---

## References

- CaDiCaL source: MIT licensed, available at reference/cadical/
- Total source files: 93
- Lines of code: ~40,000

### Key Papers
- Biere, "CaDiCaL, Kissat, Paracooba, Plingeling and Treengeling" (SAT 2020)
- Biere et al., "Preprocessing in Incremental SAT" (SAT 2011)
- Balint & Schöning, "Choosing probability distributions for stochastic local search" (SAT 2012)
- Kiesl PhD thesis (2019) - Globally blocked clauses / conditioning
- Heule et al., "Covered Clause Elimination" (LPAR 2010, JAIR 2015)
- Fazekas, Biere, Scholl, "Incremental Inprocessing in SAT Solving" (SAT 2019)
- Van Gelder, "Improved Conflict-Clause Minimization" (SAT 2009) - Poison marking
