# PLAN: `tla2` as a Full Replacement + Upgrade for TLC

**Premise:** `tla2` is not a harness around TLC; it is a from-scratch (or ground-up) **next-generation explicit-state TLA+ model checker** that replaces TLC and targets modern workloads: highly concurrent protocols, large state spaces, and “systems-of-systems” (e.g., DPLL(T) orchestration with caches/GPU workers). Z4 handles SMT; `tla2` focuses on being the best possible **explicit-state** checker and workflow engine.

**Primary use in this org:** Catch protocol-level correctness bugs in verifier stacks (γ-CROWN v2), LLM safety orchestration, caching/scoping, concurrency, and replayability—quickly enough to be used daily.

---

## 1. What `tla2` Must Do Better Than TLC (non-negotiables)

### 1.1 Performance and Scale

TLC’s main limits: single-machine scaling ceilings, state explosion for concurrent systems, and weak reduction/analysis tooling. `tla2` should be designed around:

1. **Multi-core first** (near-linear speedups to at least 32–64 cores).
2. **Distributed/cluster execution** (horizontal scaling when needed).
3. **Aggressive reduction** (POR + symmetry + slicing) so “more cores” isn’t the only lever.
4. **Memory efficiency** (state compression + disk spill + snapshots).
5. **Incremental/restartable runs** (resume after crash; reuse state when spec changes slightly).

### 1.2 Usability (debug time dominates compute time)

TLC produces counterexamples; it does not optimize for *debug latency*. `tla2` must:

1. **Minimize counterexample traces** (smallest failing prefix; optional delta-debugging of nondeterministic choices).
2. **Emit structured traces** (JSON + human) with stable event schemas.
3. **Map traces to source** (TLA+ module/line, PlusCal label, and user-defined event tags).
4. **Offer “replay”** (a deterministic reproduction script / seed + explicit schedule).

---

## 2. Core Architectural Requirements (engine-level)

### 2.1 Fast Explicit-State Exploration Engine

Core engine should support multiple exploration strategies:
- BFS (shortest counterexample), DFS (deep), iterative deepening, randomized, heuristic-guided.
- Pluggable scheduling policies for actions (useful for concurrency bug hunting).

Key primitives:
- **Canonicalization** of states (hash-consing of immutable values; stable ordering).
- **High-performance hashing** (64/128-bit; optional cryptographic mode for reproducibility).
- **State storage backends**:
  - in-memory hash table (baseline),
  - compressed in-memory,
  - disk-backed store (LSM-like) for very large runs.

### 2.2 Parallel + Distributed Model Checking

At minimum:
- **Work stealing** across threads (lock-minimized frontier queues).
- **Partitioned state space** for distributed runs (hash partitioning; consistent hashing for elasticity).
- **Deterministic mode** that yields stable results for CI (repeatable scheduling / fixed seeds).

Distributed reliability features:
- checkpoints/snapshots,
- crash recovery,
- run metadata for audit/repro.

### 2.3 Reduction Techniques (make the state space smaller)

These are the “upgrade” features that most directly differentiate a TLC replacement:

1. **Partial Order Reduction (POR)**:
   - Persistent sets / ample sets style POR.
   - Focused on interleavings of independent actions (critical for queue/worker models).
   - Must preserve safety properties; liveness requires careful treatment.
2. **Symmetry reduction**:
   - User-declared symmetry sets (common in distributed protocols: identical workers).
   - Automatic symmetry detection is hard; start with explicit symmetry groups and canonicalization.
3. **State slicing / cone of influence**:
   - Track which variables affect which properties; avoid exploring irrelevant parts.
4. **Stubborn sets / sleep sets** (if practical alongside POR).

### 2.4 Liveness + Fairness (first-class, not bolted on)

For real protocol verification, liveness/fairness matters. `tla2` should implement:
- standard TLA+ temporal operators supported by TLC today,
- fairness constraints (weak/strong fairness),
- efficient SCC-based liveness checking in parallel settings,
- diagnostic output for liveness counterexamples (cycle extraction + explanation).

---

## 3. Language/Frontend Requirements (TLA+ and PlusCal)

### 3.1 Parser + Evaluator

`tla2` needs:
- a robust TLA+ parser (full module system; operator precedence; LET/IN; EXTENDS; overrides),
- fast expression evaluation with memoization where safe,
- a value representation optimized for model checking (not “general interpreter performance”).

Performance-relevant decisions:
- specialized representations for common types (ints in bounded ranges, finite sets, tuples, records),
- bitset/bitvector encodings when domains are finite (huge for speed/memory),
- canonical ordering for sets/records to enable hashing and equality quickly.

### 3.2 PlusCal integration

PlusCal is how many engineers write protocols. `tla2` should:
- provide a PlusCal compiler with stable label mapping,
- preserve source-level debug info in traces,
- optionally support a “single-step semantics” replay mode (step-by-step label execution).

---

## 4. “Fast Enough for Daily Use” Features (the ones that matter most)

If we only build a faster TLC, adoption will still be slow unless we optimize developer feedback loops.

### 4.1 CI-friendly profiles

Support named profiles:
- `--profile smoke`: tiny bounds, seconds.
- `--profile pr`: moderate bounds, minutes.
- `--profile nightly`: big sweeps, hours.

Each profile specifies:
- bounds / constants,
- exploration strategy,
- reductions enabled,
- determinism mode,
- resource limits.

### 4.2 Parameter sweeps + regression management

Built-in sweep runner:
- run N configs (constants, bounds, reductions),
- parallelize across cores/machines,
- aggregate results (which configs fail, minimal failing config),
- store artifacts (trace, minimized trace, spec hash).

### 4.3 Counterexample minimization + explanation

Beyond “shortest trace”:
- **delta-debug** schedules and nondet choices to minimize failing conditions,
- “why this step mattered” explanations (dependency analysis on variables touched).

### 4.4 Deterministic replay

For any found counterexample:
- emit a replay file (seed + action choices + schedule decisions),
- provide `tla2 replay <file>` that reproduces it bit-for-bit in deterministic mode.

This is critical for debugging concurrency-sensitive specs and for integration into code-level simulators.

---

## 5. Why This Matters Specifically for NN/LLM/Verifier Work

The NN math is not what TLA is for; it’s the **protocol correctness** of complex verification systems:
- DPLL(T) decision levels vs SMT push/pop alignment,
- guarded lemma scoping,
- caching keyed by dependency signatures (assignment/config/numeric mode),
- async GPU batching (stale results, race conditions),
- “certified vs heuristic” policy gates (what can affect a “verified” claim).

These bugs are common, expensive, and not reliably caught by unit tests.

### 5.1 First Specs `tla2` should target (high ROI)

1. **Lemma scoping correctness**
   - No guarded lemma active without its guard.
   - No lemma leaks across backtracking levels.
2. **Context alignment**
   - SMT context depth == DPLL decision level.
3. **Cache validity**
   - Cached bounds/α solutions used only when dependency signatures match.
4. **Certified-mode gates**
   - Any inference affecting certified UNSAT must have a justification token.
5. **Parallel worker discipline**
   - No stale propagation results applied to a new decision level.

These are exactly the class of failures that can lead to “false UNSAT” in a verifier.

---

## 6. Non-Goals (to keep scope crisp)

1. `tla2` is not an SMT solver (Z4 handles that).
2. `tla2` does not try to prove real-analysis theorems (Lean handles math objects).
3. `tla2` should not require a GUI; CLI-first is the default. (A UI can come later.)

---

## 7. Implementation Strategy (practical roadmap)

### Phase A: TLC compatibility baseline
- Parse a meaningful subset of TLA+ used in-house.
- Support constants, invariants, and safety checking.
- Basic explicit-state BFS/DFS with in-memory state store.
- Produce traces with source mapping.

### Phase B: Performance + parallelism
- Multi-threaded exploration with work stealing.
- Highly optimized state representation and hashing.
- Snapshot/resume and disk spill.

### Phase C: Reduction & liveness
- POR for safety (first).
- Symmetry reduction with user-declared symmetry groups.
- Liveness/fairness checking with SCC algorithms.

### Phase D: Developer experience
- Trace minimization (delta-debugging).
- Deterministic replay.
- Parameter sweeps and CI profiles.

### Phase E: Distributed execution
- Partitioned state space, checkpoints, recovery.
- Artifact store integration for large runs.

---

## 8. Interfaces to Other Systems (tCore, gamma-crown, Lean)

Even if `tla2` replaces TLC, it must integrate cleanly:

- Emit counterexamples in a **stable machine-readable schema** that `tCore` can store and index.
- Support user-defined **event tags** in specs so traces match code-level events (e.g., `PushLevel`, `AssertLemma`, `CacheHit`).
- Provide a replay artifact that a Rust simulator/test can consume (for “spec ↔ implementation” alignment tests).

---

## 9. Success Metrics

1. **Speed:** 5–20× faster than TLC on our concurrency-heavy specs at the same bounds.
2. **Scale:** sustain 50M–500M states with disk spill and resume.
3. **Usability:** counterexample traces minimized to near-minimal failing schedules; replay works.
4. **Reduction:** POR + symmetry reduce explored states by ≥10× on worker-queue protocols.
5. **Integration:** CI profiles run in seconds/minutes and gate regressions.

