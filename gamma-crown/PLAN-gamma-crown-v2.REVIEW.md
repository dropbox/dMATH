# Technical Review: `PLAN-gamma-crown-v2.md` (γ-CROWN v2 Unified Verification Architecture)

**Document type:** Academic/technical review (rigorous, skeptical, ambitious)  
**Reviewed plan:** `gamma-crown/PLAN-gamma-crown-v2.md`  
**Review date:** 2026-01-05  
**Reviewer stance:** Assume the goals are sincere and worth achieving; optimize for correctness, credibility, and the probability of winning.

---

## Abstract

The plan proposes a unified verifier that embeds bound-propagation techniques (IBP/CROWN/α-CROWN) as a DPLL(T) theory propagator within an SMT engine (`z4`), with conflict learning and cutting planes, plus a “correctness stack” of compile-time shape safety and Kani proofs. The direction is conceptually aligned with NeuralSAT-style frameworks and with the broader trend of combining fast incomplete relaxations with complete search. However, the current document makes multiple strong claims (notably “complete: will find answer or counterexample” and “z4 proves results are correct”) without stating the semantic, numeric, and proof obligations required for those claims to be true. Several phases (type-level shapes, GPU acceleration, ML guidance) are underspecified relative to the real blockers: sound explanations/learning, well-defined operator semantics beyond ReLU networks, and rigorous numeric soundness in a float-heavy GPU implementation.

This review identifies those gaps, articulates the missing formal contracts, and proposes a revised set of acceptance criteria and research milestones that preserve ambition while increasing technical plausibility and scientific credibility.

---

## Scope and Assumptions

1. The review treats the plan as a **research + engineering roadmap**, not an implementation PR.
2. The review evaluates:
   - **Logical soundness** of the intended solver (no false UNSAT, no false proof).
   - **Completeness** claims under stated theories and supported operators.
   - **Numerical correctness** (float vs real semantics; GPU effects).
   - **Scalability** to large modern networks (including transformers).
   - **Project management realism** (deliverables, dependencies, timeline).
3. The plan is evaluated against the repository’s stated goals (γ-CROWN as a trusted developer tool for large models) and also against the plan’s VNN-COMP competition framing.

---

## High-Level Evaluation

### What is strong and worth keeping

1. **Unification via DPLL(T)** is directionally correct. Treating phase decisions as Boolean search with an LRA theory engine matches known successful verifiers and creates a principled interface for propagation/learning.
2. **Multi-level strategy (IBP → CROWN → α-CROWN → complete)** is the right performance envelope: fast “incomplete” wins easy cases; completeness is reserved for hard cases.
3. **Learning + cuts** is the main plausible path to materially outperforming BaB-only approaches on hard instances, if explanations are sound and useful.

### What is currently not defensible / risks invalidating results

The plan currently mixes three different notions of “correctness” without pinning down which one is claimed:

1. **Mathematical correctness of the relaxation algorithm** (IBP/CROWN are sound over reals given exact arithmetic and correct case handling).
2. **Correctness of the SMT encoding** (the constraints represent the intended network/property).
3. **Correctness of the implementation** (floating-point evaluation, GPU kernels, overflow/NaNs, and non-associativity do not invalidate soundness).

As written, several “proof” statements imply (2) and (3), but the plan sketches only (1).

---

## Core Technical Gaps and Risks

### 1) Completeness and theory coverage are underspecified (and likely false as stated)

The plan claims, in the strategy diagram, that Level 3 is “Complete: will find answer or counterexample” (`gamma-crown/PLAN-gamma-crown-v2.md:158-161`) and later that “SMT-based → will always find answer or counterexample” (`gamma-crown/PLAN-gamma-crown-v2.md:654-657`).

These claims are only valid if all of the following are simultaneously true:

1. **Supported operator set is decidable under the chosen theory.**
   - The plan’s layer table includes **Softmax/exp** and suggests QF_NRA or bounded LRA. Full correctness/completeness with exp/transcendentals is not available in standard SMT; “bounded LRA” introduces approximation and must be framed as incomplete or as a verified over-approximation with explicit error envelopes.
2. **The property being verified matches the encoded semantics.**
   - Typical certified robustness verifies **logits** and avoids softmax in the formal model; if that is the intended approach, it must be stated explicitly.
3. **Numeric semantics are defined.**
   - If the network executes in `f32` but the SMT model uses exact reals, the system is verifying a different function unless floating-point error is explicitly bounded or floats are modeled directly.

**Ambitious but credible fix:** add a “Supported Semantics and Operator Contract” (see below) and restrict completeness claims to the piecewise-linear fragment (ReLU/MaxPool/linear/conv unrolled) under real arithmetic, with explicit handling of any non-linear layers (either eliminated by rewriting, over-approximated with formal error bounds, or excluded).

### 2) “Bounds as theory propagation” requires proof-carrying explanations, not just heuristics

The plan’s key innovation is to implement DPLL(T)’s `propagate` and `explain` with bound propagation (`gamma-crown/PLAN-gamma-crown-v2.md:166-209`), learning a clause from conflicts.

This is exactly where most integrated verifiers fail: if a conflict explanation is derived from an **approximate relaxation** or from a **numerically unsound bound**, the learned clause may prune a feasible region, producing a false UNSAT (a catastrophic “verified” result).

In DPLL(T), the `explain()` must return a clause that is a **logical consequence** of the base constraints (and any globally valid lemmas) under the theory. The plan currently says “find minimal subset … that cause conflict” and calls `analyze_conflict()` (`gamma-crown/PLAN-gamma-crown-v2.md:204-209`), but does not specify:

1. What constitutes a conflict (empty bounds? infeasible LRA? infeasible relaxation?).
2. How to compute a sound explanation (unsat core? Farkas certificate? guarded lemma?).
3. How to ensure learned clauses remain valid under backtracking and across different relaxations.

**Required for soundness:** explanations must be validated against the underlying theory solver (e.g., via unsat cores or certificates) or constructed from proof-carrying propagation rules whose antecedents are tracked precisely.

### 3) Global vs conditional validity of cutting planes is not specified

The plan suggests “cuts persist across backtracking (global learning)” in the unified architecture diagram, i.e., adding learned cuts as lemmas. Many strong cuts in NN verification are only valid **under specific phase assignments** or depend on a particular relaxation state. Persisting such cuts globally is unsound unless they are guarded.

**Required:** define cut classes and scoping rules:
- **Globally valid cuts**: unconditional consequences of the base constraints; safe to keep across backtracking.
- **Conditionally valid cuts**: valid only under a conjunction of phase literals; must be asserted as guarded lemmas or at the appropriate decision level.

### 4) The plan is ReLU-centric; transformer-class networks break assumptions

The DPLL(T) framing is centered on “ReLU phase decisions (b_i ∈ {0,1})” (diagram around `gamma-crown/PLAN-gamma-crown-v2.md:104-117`). This fits piecewise-linear networks. But modern targets in this repo include transformers, where critical operators include:
- **Softmax** (transcendentals + normalization),
- **LayerNorm** (division by variance; non-linear),
- **GELU** (smooth non-linear),
- attention masking and large matmuls.

Unless the v2 scope explicitly restricts to PL networks (or provides verified relaxations for these operators with stated semantics), the “complete” solver narrative does not apply. A complete solver for a PL fragment is still valuable, but the scope must be honest.

### 5) LLM generation/jailbreaking applications need a precise threat model and semantics

The plan’s “LLM Security Implication” section introduces dual-use framing (“Find counterexamples → Jailbreak any LLM”) and a very strong target property:

```
Property: ∀ inputs x, P(harmful_output | x) = 0
Result: Model that CANNOT produce harmful outputs (proven)
```

(`gamma-crown/PLAN-gamma-crown-v2.md:763-781`)

This is an inspiring north star, but it currently lacks the definitions required to be a verifiable statement:

1. **What is the function being verified?**
   - For an LLM, the primitive object is typically `x ↦ logits(x)` (next-token distribution parameters). “Generation” is an *iterated closed-loop system* involving autoregressive state updates plus a decoding algorithm (greedy/beam/sampling) and often additional system components (system prompt, tools, RAG, filters).
   - Verification needs a fixed, formal semantics for that whole loop; otherwise the target is ill-posed.
2. **What does `P(harmful_output | x)` mean operationally?**
   - If decoding is stochastic, the probability is with respect to the sampling procedure (temperature, top-k/top-p, RNG). If decoding is deterministic, the probability is degenerate and the statement reduces to a universal safety predicate on the deterministic trace.
   - Either way, “probability zero for all x” is far stronger than most safety goals and will be brittle to minor system changes.
3. **What is the input domain `x`?**
   - “All strings” is infinite. “All token sequences up to length L over vocabulary V” is finite but astronomically large.
   - Any credible verification target must specify domain restrictions (templates, grammars, bounded length, bounded perturbations, etc.).
4. **What is `harmful_output` as a predicate?**
   - If it is defined via a classifier, that classifier is part of the system and must be included in the verification story (or treated as an oracle, which weakens the guarantee).
   - If it is defined by human judgment, it is not mechanically checkable; the guarantee becomes non-operational.

**Recommendation:** explicitly scope the LLM-safety direction to a sequence of formalizable intermediate targets, e.g.:
- single-step properties over logits (e.g., “danger-token set has logit margin below safe set” under bounded prompt perturbations),
- constrained decoding where unsafe tokens are hard-masked (guarantee comes from the constraint mechanism, not from hoping the model never wants them),
- compositional guarantees: prove properties of components (router, filter, refusal head) and then prove system-level invariants for a specified orchestration/decoding policy.

This still supports an ambitious “cannot be jailbroken” narrative, but only once the threat model, domains, and system semantics are made explicit.

### 6) Verification-guided training and design: high upside, but missing feasibility gates

The plan’s long-term vision expands from “verify existing networks” to “use verification to create better networks” (V-NAS, verification-aware loss, verified adversarial training, repair, verified-by-construction) and also advances a “sample efficiency” conjecture (`gamma-crown/PLAN-gamma-crown-v2.md:700-797`).

This direction is plausible and potentially transformative, but the plan currently treats it as a straight-line extrapolation from “a faster verifier” without acknowledging several hard constraints:

1. **Differentiability and stability are not optional.**
   - Training-time use (verification-aware loss, verified adversarial training) requires bounds that are not only sound but **smooth enough to optimize** and stable under SGD noise.
   - Many complete-verification mechanisms (BaB/DPLL(T) with learning) are discontinuous; they can be used as *oracles* or for curriculum mining, but not directly as a differentiable loss without careful design.
2. **What exactly is being optimized?**
   - “Tighter bounds” can correlate with verifiability but is not identical to the target property; optimizing surrogate objectives can cause Goodhart effects (tight bounds but wrong region; improved certificate but degraded task performance).
   - The plan should specify candidate objectives and how they relate to guarantees.
3. **Compute economics at LLM scale are unforgiving.**
   - Even with a 100× speedup, verifying/training with certificates on large transformers may remain too expensive unless the verification target is made local/compositional (per-layer, per-block, per-head), or unless properties are drastically simplified.
4. **Design/search requires a tractable verification oracle.**
   - Architecture search over verifiability is only realistic if the verifier produces **cheap, monotone, and comparable scores** across candidates; complete verification is too expensive as an inner loop unless heavily amortized.

**Recommended “feasibility gates” (so this stays ambitious but credible):**

1. **Verified training pilot (small but real):**
   - Choose a modest architecture where differentiable bounds are already known to work (e.g., small CNN/MLP or a tiny attention model with a restricted operator set).
   - Demonstrate that adding a verification-aware regularizer improves certified metrics without collapsing accuracy.
2. **Compositional guarantees first:**
   - For transformers, pursue properties that decompose (e.g., bounds on attention outputs, stability of a block under bounded perturbations) and prove composition rules; avoid the full sequence-generation loop initially.
3. **Architecture scoring oracle:**
   - Define a cheap verifiability score (bound tightness, fraction of stable activations, certificate margin) that predicts success, then validate correlation with actual verification outcomes.
4. **Repair as a bridge to “by construction”:**
   - “Repair” (minimal weight edits to satisfy a property) is often a more tractable intermediate step than full synthesis; it can be driven by counterexamples and certificate gradients.

Framed this way, the plan’s “design/training” ambitions become a sequence of falsifiable experiments rather than a speculative epilogue.

### 7) Numeric soundness is not addressed (GPU makes it harder, not easier)

The plan introduces GPU acceleration as Phase 4 (`gamma-crown/PLAN-gamma-crown-v2.md:448-455`). GPU acceleration is compatible with soundness, but only if the system enforces:
- outward rounding / conservative error envelopes,
- explicit NaN/Inf handling and preconditions,
- deterministic reductions (or bounding nondeterminism).

Otherwise, the most likely failure mode is: float under-approximation of bounds → invalid conflict → learned clause → false UNSAT.

Given the repo already emphasizes “Sound Verification: Never report incorrect bounds,” the plan must treat numeric soundness as a first-class acceptance criterion, not a post-hoc hope.

### 8) Compile-time shape safety is valuable, but the proposed approach risks non-scalable complexity

The plan proposes const-generic shape types for compile-time checking (`gamma-crown/PLAN-gamma-crown-v2.md:217-299`). This can eliminate a class of bugs, but real models often include partially dynamic dimensions (batch, sequence length, variable image sizes). Full static shape typing for end-to-end model ingestion can become brittle.

**Likely best compromise:** a “typed core” (internal IR with fixed shapes) with explicit boundaries:
- dynamic ingestion → validated canonical IR → typed execution/propagation core where feasible.

### 9) Kani proofs are promising but currently conflated with end-to-end correctness

The plan suggests “Kani proves our verifier is correct” (diagram around `gamma-crown/PLAN-gamma-crown-v2.md:621-627`) and outlines proofs over `f32` (`gamma-crown/PLAN-gamma-crown-v2.md:302-399`).

Concerns:
- Unconstrained `f32` includes NaNs/Infs; proofs may be vacuous or false without strong assumptions.
- Proving the math of IBP/CROWN over reals is different from proving the `f32` implementation is sound.
- “Proof of SMT encoding equisatisfiability” (`gamma-crown/PLAN-gamma-crown-v2.md:497-500`) is a major research deliverable and will likely require a proof-producing encoding pipeline or an independently checked translation, not just bounded model checking.

**Recommendation:** split “algorithmic soundness” proofs (reals, small kernels) from “implementation soundness” (floats, bounded ranges, error envelopes). Do not market Kani as a blanket correctness certificate unless that is literally what is proven.

### 10) Timeline and effort estimates are not credible

The plan estimates 75 commits in ~1.5 weeks (`gamma-crown/PLAN-gamma-crown-v2.md:684`). For the scope claimed (new solver architecture, explanations, learning, GPU acceleration, proofs, plus competition dominance), this is not realistic. A more credible plan needs:
- explicit acceptance tests,
- stage gates that prevent “fast but wrong” progress,
- recognition that explanation soundness and numeric soundness are the critical path.

---

## Missing “Formal Contract” Sections (recommended additions to the plan)

To make the roadmap scientifically credible, the plan should explicitly define:

### A) Semantics Contract

1. **Verified semantics target:** Real arithmetic vs IEEE-754 (and which precision).
2. **Operator subset:** which layers are supported natively, which are rewritten, which are over-approximated, which are out-of-scope.
3. **Preprocessing + property semantics:** normalization constants, input domains, and property language (e.g., VNNLIB) must match the encoding.
4. **LLM generation semantics (if in scope):** tokenization, context bounds, decoding policy (greedy/beam/sampling), and what randomness/probability means in the verified claim.
5. **Soundness requirement:** what “verified” means in the presence of floats/GPU (e.g., “sound wrt real semantics within δ”).

### B) Proof Obligations for DPLL(T) Integration

1. **Propagation soundness:** each implied phase literal must be justified by constraints.
2. **Conflict soundness:** each reported conflict must imply inconsistency in the theory.
3. **Learning soundness:** each learned clause/lemma must be a consequence of the base theory (global) or guarded by the correct decision literals (conditional).
4. **Cut soundness:** same as learning; include scoping rules.

### C) Artifact Contract (for trust and debuggability)

1. SAT results include a **counterexample witness** and a deterministic replay check.
2. UNSAT results include:
   - at minimum, an independently-checkable replay trace that reproduces the unsat in `z4` with the same lemmas, or
   - ideally, theory certificates (e.g., Farkas proofs) where available.

---

## Recommendations on Phasing (keeping ambition, improving probability of success)

### Phase 2 should be restructured around “sound-first” milestones

Current Phase 2 items (“conflict clause extraction,” “cuts,” “beat α,β-CROWN”) are correct goals but missing gates.

Suggested staged milestones:

1. **MVP complete solver (slow but correct):**
   - Encode PL networks with exact LRA + ReLU disjunction (or indicator constraints).
   - Use `z4` incrementally with push/pop.
   - No bound-engine learning yet; just correctness and baseline completeness for small networks.
2. **Add bound-engine as *advisory propagation* with validation:**
   - Use bounds to propose implied literals, but validate them via theory constraints or via a proof-carrying derivation.
3. **Add learning only when explanations are validated:**
   - Clauses derived from unsat cores/certificates, or guarded and checked.
4. **Add cuts with explicit validity classes:**
   - Global vs conditional; enforce scoping rules.

This sequencing prevents the most dangerous failure: “fast incorrect verifier that returns UNSAT.”

### Type-level shapes: avoid a flag-day migration

Instead of “migrate everything,” define:
- a typed internal tensor/IR for propagation kernels where shapes are fixed after model canonicalization,
- explicit dynamic boundary checks at import time,
- and keep compile-time typing scoped to modules where it provides leverage without fighting the entire ecosystem.

### ML-guidance: defer until measurement proves it is the bottleneck

The plan’s Phase 6 is plausible but should be gated on:
- a stable verification corpus format,
- an ablation showing that strategy selection is the limiting factor (not kernel speed, not encoding, not explanation quality),
- and reproducible evaluation (train/test separation across benchmark families).

Otherwise ML becomes a high-effort distraction from solver fundamentals.

---

## Evaluation of Success Metrics (need sharpening)

Current metrics include “No soundness bugs” and competition scores (`gamma-crown/PLAN-gamma-crown-v2.md:467-483`). These are directionally fine but not measurable enough.

Recommended additions:

1. **Soundness regression suite:**
   - adversarially generated counterexamples,
   - numeric edge-case tests (NaNs/Infs/overflows),
   - differential testing vs a reference solver on overlapping operator sets.
2. **Proof/trace reproducibility:**
   - given a seed, results must be bitwise reproducible (or explain nondeterminism).
3. **Ablation benchmarks:**
   - bounds-only vs bounds+learning vs bounds+cuts vs full DPLL(T).
   - report not just time, but node counts, learned clauses, and bound tightness improvements.

---

## Research Questions to Add (high leverage)

1. **What explanations are both sound and small?** (core extraction vs certificate-based vs proof-carrying propagation)
2. **How to control propagation cost?** (lazy propagation schedules; incremental warm-started α optimization; caching across branches)
3. **What is the numeric soundness strategy on GPU?** (outward rounding, mixed precision, error envelopes)
4. **How to support transformer operators rigorously?** (verified relaxations for softmax/layernorm/GELU; scope limits if not)

---

## Concluding Assessment

The plan’s central architectural thesis—integrating bound propagation as theory propagation in a DPLL(T) solver—is strong and worth pursuing. The biggest blockers are not “wiring z4” or “adding GPU,” but the rigorous definition of semantics and the production of **sound explanations** for conflicts, learning, and cuts in a numerically conservative implementation. Without those, the system risks becoming fast but scientifically untrustworthy. With those explicitly designed and staged, the plan can plausibly deliver a verifier that is both competitive and reliable, and can serve as a foundation for the longer-term “verification-guided training/design” ambitions.
