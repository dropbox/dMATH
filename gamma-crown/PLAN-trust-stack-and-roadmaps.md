# PLAN: Trust Stack + Roadmaps (tCore, z4, Lean5, TLA2, tRust/Trusted-C, tensorforge)

**Purpose:** Provide a roadmap-ready plan for building *proven building blocks* that enable γ-CROWN v2 (unified DPLL(T)+bounds) to be fast **and** trustworthy, and to support longer-horizon ambitions (verified training, repair, and eventually “cannot be jailbroken” style claims) without collapsing under semantic ambiguity.

**Audience:** Maintainers setting near-term roadmaps and long-term research goals.  
**Status:** Draft for roadmap planning (not an implementation PR).

---

## 0. Guiding Principles

1. **Make claims match artifacts.** If we say “verified,” there must exist a checkable artifact whose validation is cheaper than generating it.
2. **Separate math soundness from implementation soundness.** Proving a relaxation is sound over reals is different from proving an `f32` GPU implementation is conservative.
3. **Proven building blocks > monolithic proof.** Prefer small verified components with clear interfaces and composition rules.
4. **Sound-by-default modes.** The system must have a mode where it will not report a false proof, even if that costs performance.
5. **Scope honesty.** “Complete” must be stated only for a clearly defined operator fragment and semantics.

---

## 1. What We’re Ultimately Building

### 1.1 The “Trust Spine”

`tCore` is the trust spine that makes all systems agree on:
- **Semantics:** what function/model is being verified (real vs float; operator set; preprocessing; property language).
- **Canonical IR:** a stable representation of networks + properties that all backends (SMT, bounds, GPU, proofs) consume.
- **Evidence:** proof/certificate schemas for SAT/UNSAT, learned lemmas, numeric error envelopes, and replay traces.
- **Checkers:** cheap validators for evidence (Rust) plus optional deep checkers (Lean) in CI.

Everything else plugs into `tCore`:
- `gamma-crown` produces bounds, lemmas, and (eventually) certificates using the IR.
- `z4` is the authoritative theory checker and (ideally) certificate generator.
- `Lean5` proves math objects and/or checks certificates and composition theorems.
- `TLA2` specifies solver/protocol invariants (scoping, caching, concurrency correctness).
- `tRust` / Trusted-C enforce implementation safety and kernel correctness assumptions.
- `tensorforge` supplies numerically conservative tensor primitives (esp. on GPU).

### 1.2 Evidence-Driven Verification (what “verified” means)

We explicitly support multiple result tiers, each with a different evidence requirement:

1. **Heuristic:** fast result, minimal evidence (debugging, exploration). Not acceptable for “verified safety” claims.
2. **Verified (Replay):** SAT has a concrete witness; UNSAT has a deterministic replay trace that replays in the same solver context.
3. **Verified (Certified):** UNSAT has a certificate checkable by an independent checker (Lean or small Rust checker). SAT witness replay remains required.

Roadmap priority is to make **(2)** ubiquitous quickly and make **(3)** available for the core fragment (e.g., LRA-based proofs).

---

## 2. Semantics Contract (must be explicit early)

### 2.1 Verified Semantics Targets (choose and label)

We should support *two* explicit semantics targets, each with different engineering/proof requirements:

1. **Real semantics (primary for “math proofs”):** Network interpreted over ℝ with exact arithmetic, for a restricted operator set (PL + selected relaxations).
2. **Float semantics (implementation semantics):** Network and bounds computed in IEEE-754 (`f32`/`f16`) possibly on GPU.

If we claim Real-semantics verification but compute in float, we must either:
- compute **conservative enclosures** (outward rounding / error envelopes), or
- restrict claims to Replay/Heuristic tiers.

### 2.2 Operator Fragment Policy

Define a matrix: {supported natively, rewritten away, over-approximated with proof obligations, excluded}.

Recommended early “complete” fragment:
- affine + conv (as affine), ReLU/max, pooling-as-max, linear constraints.

Operators like softmax, layernorm, GELU:
- are not “complete” under standard SMT without heavy approximations;
- should be explicitly treated as **(a)** rewritten away (verify logits), **(b)** bounded with sound relaxations, or **(c)** out-of-scope for completeness claims.

### 2.3 LLM generation semantics (long-horizon)

For roadmap planning, treat “cannot be jailbroken” as a **stack of increasingly complete system models**, not one claim:
- logits-only single-step properties (tractable),
- constrained decoding (masking/filters) with formally modeled policy,
- autoregressive loop with bounded length + deterministic decoding,
- stochastic decoding with explicit probability semantics (hard; may require different machinery).

This belongs in `tCore` as a versioned semantics spec, not in ad-hoc docs.

---

## 3. Root-of-Trust Options (propose, compare, choose later)

### Option A: z4 as root of trust (proof logs / certificates)
**Idea:** z4 produces proofs/certificates; we store them; independent checker validates them.

Pros:
- Strongest “SMT proved it” story if proofs are checkable.
- Natural match to DPLL(T).

Cons:
- Depends on z4 proof/cert feature maturity and stability.
- Proof formats can be complex; checking can be nontrivial.

### Option B: Lean5 as root of trust (certificate checking + selective proofs)
**Idea:** Lean checks UNSAT certificates for the fragment and proves key lemmas about bound transformers; the solver is trusted only insofar as it produces checkable evidence.

Pros:
- Independent checker with strong foundations.
- Enables compositional theorems for training/design.
- Doesn’t require proving the whole solver to get strong guarantees.

Cons:
- Requires designing a certificate format Lean can check efficiently.
- Still leaves implementation/numeric issues unless included in the semantics.

### Option C: Replay as root of trust (deterministic replay + cross-checking)
**Idea:** treat verification as “replayable computation” in a controlled deterministic mode; cross-check with multiple solvers or redundant runs.

Pros:
- Fast to implement; good engineering discipline.
- Useful for debugging and regression.

Cons:
- Not independent in the strong sense; can replay the same bug.
- Weak against numeric unsoundness unless the replay engine is conservative.

### Recommended near-term default
Adopt **Option B + C** immediately (Lean checks certificates where possible; otherwise replay), and keep Option A as an engineering goal for z4.

---

## 4. z4 Roadmap (Detailed)

### 4.1 What γ-CROWN v2 needs from z4

1. **Incremental contexts** with cheap push/pop (decision levels).
2. **Assumption literals / activation** so lemmas/cuts can be guarded.
3. **Unsat cores** under assumptions (minimum: a core; ideal: minimal-ish).
4. **Certificate hooks** for LRA (best: Farkas multipliers for contradiction).
5. **Models** for SAT (counterexample extraction).

### 4.2 How we keep learning sound

Bound propagation may suggest:
- implied phase literals,
- conflicts (empty bounds),
- cuts/lemmas.

But for soundness:
- **Any pruning that affects “Verified” must be justified by z4**, either by:
  - checking implication: `Base ∧ A ∧ ¬lit` is UNSAT (then core/cert becomes justification), or
  - producing a guarded lemma and validating it in z4 before making it global.

Policy:
- “Empty bounds” may **trigger** a z4 check, but must not itself be the learned reason unless it yields a certificate.

### 4.3 Deliverables for a z4 epic

1. Stable IDs for asserted constraints/lemmas (for core/cert referencing).
2. `unsat_core()` API that returns those IDs under assumptions.
3. LRA certificate support (Farkas) OR enough data to reconstruct a certificate.
4. `model()` extraction API with value assignment for relevant variables.
5. Replay tool: given `tCore` IR + lemma set + assumptions, re-check result deterministically.

### 4.4 Stretch goal: proof-producing z4 for our fragment

If z4 can emit proof logs for the relevant fragment, `tCore` can store them and Lean can check them. This becomes the cleanest end-to-end story:

`gamma-crown` search → `z4` UNSAT proof → `Lean5` checks proof → “Verified (Certified)”.

---

## 5. Lean5 Roadmap (Detailed)

### 5.1 “Lean should not try to prove the whole solver” — why, and when it becomes feasible

**Why not by default (pragmatic reasons):**
1. **The solver is a large, stateful, performance-driven system** with caches, mutable state, GPU calls, and FFI; a literal end-to-end proof would require modeling all of that or restricting the implementation severely.
2. **Numeric semantics explode complexity.** If we include IEEE-754, GPU nondeterminism, and performance tricks, full formalization becomes a research program on its own.
3. **Maintenance cost.** A fully verified solver ties every refactor to proof refactors; this can halt iteration unless the system is designed for verification from day one.

**Why it might be possible (ambitious path):**
If we architect the solver around:
- a small verified kernel (pure functional core),
- proof-producing procedures (certificates for every critical inference),
- and isolate performance optimizations behind proven refinement steps,
then Lean can scale to proving “the whole solver” *at the level of the kernel + certificates*, while the implementation becomes a refinement of that kernel.

### 5.2 Recommended Lean focus (high leverage building blocks)

1. **Formal semantics for the core fragment** (PL networks over ℝ).
2. **Soundness of bound transformers** (IBP/CROWN relaxations) stated against that semantics.
3. **Soundness of explanation rules** (if the solver uses a specific explanation calculus).
4. **Composition theorems** for modular verification (key to scaling and training/design).
5. **Certificate checking** for UNSAT in LRA (Farkas) and for simple propagation claims.

### 5.3 Certificate strategy that scales

Design certificates so they are:
- **small** (or compressible),
- **checkable fast** (linear-time checking if possible),
- **stable** across solver refactors (avoid proof objects tied to internal heuristics).

Concrete proposal:
- For LRA conflicts: Farkas multipliers over a named set of inequalities.
- For learned clauses: a record of the z4 query + its core/cert, plus guard literals.

### 5.4 The most ambitious Lean5 endgame (credible “maximum ambition”)

**Goal:** A proof-producing, certificate-checked verifier where UNSAT claims are independently checkable, and solver learning is justified.

Ambition ladder (increasing difficulty):

1. **Lean checks LRA certificates** (UNSAT) + validates key math lemmas for bounds (IBP/CROWN).
2. **Lean validates the encoding correctness** for the supported fragment: the IR-to-constraints translation is semantics-preserving.
3. **Lean proves the DPLL(T) kernel** (search + clause learning protocol) correct *assuming* theory oracle correctness.
4. **Lean checks theory certificates produced by z4** (or a smaller verified LRA engine used only for certificates).
5. **Full refinement proof:** The production implementation refines the verified kernel (with a translation validation layer and a certificate pipeline).

This yields the strongest story without requiring Lean to model GPU execution directly: GPU is used to propose bounds/lemmas, but correctness relies on checked certificates for every decisive inference.

---

## 6. TLA2 Roadmap (Detailed)

### 6.1 Why TLA2 belongs in this stack

Even if math is correct, system-level bugs can invalidate results:
- wrong lemma scoping across decision levels,
- stale caches reused under different assumptions,
- race conditions in GPU batching,
- incorrect push/pop discipline with the SMT context.

TLA2 is well-suited to specifying and model-checking these *protocol invariants*.

### 6.2 What to model (minimum viable spec)

State components:
- DPLL assignment stack (decision levels, trail).
- SMT context depth and active assumptions.
- Lemma DB with scopes (global vs guarded vs level-scoped).
- BoundEngine cache dependency signatures.
- Optional: parallel work queue for GPU/CPU propagation.

Key invariants:
1. **Scope correctness:** a lemma/cut is active iff its guard is satisfied (or it is global).
2. **Justification completeness:** every learned lemma has an attached justification artifact (or is marked heuristic and barred from “Verified”).
3. **Context alignment:** SMT push/pop mirrors DPLL decision levels.
4. **Cache validity:** cached results are used only when dependency signatures match.
5. **Termination sanity:** backtracking makes progress; no cycles caused by stale implications.

### 6.3 Deliverables for a TLA2 epic

1. PlusCal spec of the DPLL(T)+propagation loop with lemma scoping.
2. TLC model checking for bounded toy instances (to catch protocol bugs).
3. A “spec-to-code checklist” that turns invariants into runtime assertions and tests.

---

## 7. tensorforge + Trusted Languages (what we need and why)

### 7.1 tensorforge (numerically conservative tensor runtime)

If bounds/propagation influence learned clauses, we need numeric conservatism:
- outward rounding or proven error envelopes,
- deterministic or bounded-nondeterministic reductions,
- explicit NaN/Inf policy and preconditions,
- stable kernel contracts that can be independently tested/verified.

The roadmap goal is: **GPU speed without forfeiting soundness**.

### 7.2 tRust / Trusted-C (implementation safety layer)

Use tRust to enforce:
- no UB, safe FFI boundaries,
- tight control of arithmetic overflow/NaN propagation policies,
- (where feasible) structural invariants (shapes, indexing).

Use Trusted-C only behind a very small, auditable interface if necessary for kernels.

---

## 8. Roadmap Proposal (phases / epics)

### Epic 0: Establish tCore v0 (trust spine)
- Define canonical IR for network + property.
- Define semantics spec (Real vs Float; operator support; preprocessing).
- Define evidence schema (SAT witness, UNSAT replay, UNSAT certificate, lemma justification).
- Implement replay checker (Rust) and a minimal artifact store.

### Epic 1: z4 integration for evidence
- Assumptions/activation literals + incremental API integration.
- Unsat core extraction with stable IDs.
- Model extraction for SAT witnesses.
- Replay-by-construction: export the exact asserted constraint set for re-check.

### Epic 2: Lean5 v0 (proof-carrying building blocks)
- Formalize semantics for PL fragment over ℝ.
- Prove IBP/ReLU/CROWN lemmas for that fragment (math soundness).
- Implement Lean checker for LRA certificates (or for a chosen certificate subset).
- Add CI target: validate certificates from a regression suite.

### Epic 3: TLA2 protocol spec v0
- Specify DPLL(T) + scoping + caching protocol.
- Model-check invariants on toy instances.
- Translate invariants into runtime assertions in γ-CROWN v2.

### Epic 4: “Certified UNSAT” pipeline for the PL fragment
- z4 produces cores/certs; γ-CROWN stores them; Lean checks them.
- Policy: only “Verified (Certified)” emits UNSAT if certificate passes.

### Epic 5 (moonshot): Verified DPLL(T) kernel + refinement
- Prove a small DPLL(T) kernel in Lean.
- Restrict production system to emit certificates that justify every learned/pruning inference.
- Prove/validate translation correctness from IR to constraints.

---

## 9. Key Decision Points (to schedule explicitly)

1. **Semantics primary:** Real-only (with conservative float envelopes) vs Float-as-truth.
2. **Proof/evidence tier default:** Replay vs Certified for shipped “verified” mode.
3. **Certificate format:** start with LRA/Farkas; expand later.
4. **Scope of completeness claims:** PL fragment only vs expanded operator set.
5. **tensorforge priority:** required for sound GPU bounds vs “GPU is heuristic-only.”

---

## 10. What “Most Ambitious” Looks Like (explicitly)

The most ambitious credible endpoint is:

> A proof-producing verifier where every UNSAT result is accompanied by a certificate checkable by an independent checker (Lean), and every learned/pruning inference that could affect correctness is justified by checkable evidence—while GPU/heuristics are allowed only as proposal mechanisms that must be validated before they influence certified results.

This is stronger (and more maintainable) than trying to directly prove a large, optimized, stateful implementation end-to-end, because:
- the trusted core stays small,
- performance work happens in untrusted “oracle/proposal” layers,
- correctness is enforced by certificate validation and refinement boundaries.

