# PLAN: γ-CROWN v3 - Sound Verification Architecture

**Vision:** A neural network verifier that is provably correct, numerically sound, and competitive.

**Core Principle:** Soundness first. Performance second. No false UNSATs, ever.

---

## Response to Technical Review

This plan addresses the concerns raised in `PLAN-gamma-crown-v2.REVIEW.md`.

### Key Corrections from Review

| Review Concern | v2 Claim | v3 Response |
|----------------|----------|-------------|
| "Complete: will find answer" | Implied for all networks | Restricted to piecewise-linear fragment |
| Softmax/exp handling | QF_NRA mentioned | Explicitly out of scope for completeness |
| Conflict explanations | "find minimal subset" | Proof-carrying derivations required |
| Cut persistence | "persist across backtracking" | Scoping rules: global vs conditional |
| Numeric soundness | Not addressed | First-class requirement with error envelopes |
| Kani proofs | "proves verifier correct" | Split: algorithmic vs implementation |
| Timeline | 75 commits in 1.5 weeks | Phases with stage gates |

### Clarifications on Our Tool Stack

**The reviewer assumed external dependencies. We control the stack:**

| Tool | Status | Implication |
|------|--------|-------------|
| **z4** | We own it | Add features as needed (e.g., proof certificates) |
| **tRust** | We own it | "Proof IN language" - compilation IS verification |
| **kani_fast** | We own it | Stepping stone; features migrate to tRust |

**On tRust (reviewer didn't have context):**

tRust is not "another external verifier." It implements **proof-in-language**: the type system IS the proof system. Compilation doesn't call a separate verifier - the act of type-checking IS the proof. This eliminates the soundness gap between compiler and verifier.

---

## Formal Contracts (Per Review Section)

### A) Semantics Contract

**1. Verified Semantics Target:**
- **Primary:** Real arithmetic over piecewise-linear networks
- **Implementation:** IEEE-754 f32 with conservative outward rounding
- **Guarantee:** If bounds computed in f32 satisfy property, real-valued network also satisfies it

**2. Operator Subset:**

| Operator | Status | Handling |
|----------|--------|----------|
| Linear (Gemm, MatMul) | Complete | Exact under LRA |
| ReLU | Complete | Disjunctive encoding |
| Conv2d | Complete | Unrolled to linear |
| MaxPool | Complete | Disjunctive encoding |
| Add, Sub, Mul (const) | Complete | Linear transformation |
| BatchNorm | Complete | Folded to linear (frozen) |
| Flatten, Reshape, Transpose | Complete | Index mapping |
| **Softmax** | **Out of scope** | Verifies logits, not probabilities |
| **LayerNorm** | **Incomplete** | Over-approximation with explicit error |
| **GELU/SiLU** | **Incomplete** | Piecewise-linear approximation |
| **Exp, Log, Sin, etc.** | **Out of scope** | Transcendentals not supported |

**3. Completeness Scope:**
```
COMPLETE for: Piecewise-linear networks (ReLU, MaxPool, linear, conv)
INCOMPLETE for: Networks with softmax, GELU, LayerNorm in verification path
```

**4. Soundness Requirement:**
- "Verified SAFE" means: ∀x in input region, real-valued f(x) satisfies property
- f32 implementation uses outward rounding to preserve soundness
- GPU kernels must produce bounds ⊇ CPU bounds (conservative)

**5. LLM/Transformer Semantics (Future Scope):**

The v2 plan mentioned LLM jailbreaking applications. Per review, this requires precise definitions:

| Aspect | Required Definition | Status |
|--------|---------------------|--------|
| **Function under verification** | `x ↦ logits(x)` (single forward pass) vs autoregressive generation | Single-pass only for now |
| **Input domain** | Bounded token sequences, perturbation sets | Must be finite/bounded |
| **Output property** | Logit margins, hard token constraints | Not probability statements |
| **Decoding policy** | Greedy, beam, sampling | Out of scope (changes semantics) |
| **System components** | System prompt, tools, RAG, filters | Out of scope |

**What we CAN verify (near-term):**
- Single-step logit properties: "danger-token logits below safe-token logits under bounded perturbation"
- Component properties: classifier heads, refusal mechanisms
- Piecewise-linear approximations of transformer blocks

**What we CANNOT verify (without further research):**
- Full autoregressive generation (unbounded loop)
- Stochastic decoding (probability statements require different semantics)
- "Model CANNOT produce harmful outputs" (requires operational definition of harmful)

**Honest framing:** LLM safety verification is a research direction enabled by a fast verifier, not a near-term deliverable. The VNN-COMP work builds the engine; LLM applications require additional formalization.

### B) Proof Obligations for DPLL(T)

**1. Propagation Soundness:**
Each implied phase literal must be derivable from:
- Direct bound computation (lb > 0 → active; ub < 0 → inactive)
- Theory constraints in z4 context
- Tracked antecedents for explanation

```rust
struct ImpliedLiteral {
    literal: Literal,           // The implied phase
    antecedents: Vec<Literal>,  // Phase decisions that imply it
    bound_derivation: BoundProof, // Certificate from bound engine
}
```

**2. Conflict Soundness:**
A conflict is valid iff the theory solver (z4 LRA) returns UNSAT for the current constraints.

```rust
enum ConflictSource {
    /// z4 returned unsat core
    TheoryUnsat { core: Vec<ConstraintId> },
    /// Bound propagation detected empty interval
    EmptyBounds {
        variable: NodeId,
        derivation: BoundProof,  // How we derived lb > ub
    },
}
```

**3. Learning Soundness:**
Learned clauses must be consequences of base theory.

**Implementation:** We use z4's unsat core to generate clauses:
```rust
fn explain_conflict(&self, conflict: ConflictSource) -> Clause {
    match conflict {
        ConflictSource::TheoryUnsat { core } => {
            // z4 provides unsat core - convert to clause
            self.core_to_clause(core)
        }
        ConflictSource::EmptyBounds { derivation, .. } => {
            // Extract relevant phase decisions from derivation
            // Then VALIDATE via z4 that negation is unsat
            let clause = derivation.extract_antecedents();
            self.validate_learned_clause(&clause)?;
            clause
        }
    }
}
```

**4. Cut Validity Classes:**

| Cut Type | Scope | Example |
|----------|-------|---------|
| **Global** | Always valid | Network structure constraints |
| **Conditional** | Valid under phase assignment | ReLU-specific tightening |
| **Level-scoped** | Valid at decision level | Branch-specific bounds |

```rust
enum Cut {
    Global(LinearConstraint),
    Conditional {
        guard: Vec<Literal>,  // Must be true for cut to apply
        constraint: LinearConstraint,
    },
}
```

### C) Artifact Contract

**1. SAT (Counterexample Found):**
- Output: concrete input x* violating property
- Validation: evaluate f(x*) and check property violation
- Reproducibility: deterministic replay from seed

**2. UNSAT (Verified SAFE):**
- Output: verification trace with learned clauses
- Validation option A: replay trace in z4 with same lemmas
- Validation option B: theory certificates (Farkas proofs) where available
- Reproducibility: bitwise reproducible given seed

**3. UNKNOWN (Timeout/Resource):**
- Output: current bounds, search state
- No soundness claim - just "didn't finish"

---

## Numeric Soundness Strategy

### The Problem (Per Review)

Float under-approximation → invalid conflict → false UNSAT

### The Solution: Conservative Arithmetic

**1. Outward Rounding:**
```rust
/// Compute lower bound with downward rounding
fn compute_lower_bound_safe(a: f32, b: f32) -> f32 {
    // Use nextafter to ensure conservative bound
    let result = a * b;  // May round either way
    if result.is_nan() || result.is_infinite() {
        f32::NEG_INFINITY  // Conservative: any value is >= -inf
    } else {
        f32::from_bits(result.to_bits().wrapping_sub(1))  // Round down
    }
}
```

**2. NaN/Inf Handling:**
- NaN in bounds → widen to [-inf, +inf] (conservative)
- Overflow → widen to infinity (conservative)
- Underflow → round toward conservative direction

**3. GPU Kernel Requirements:**
```rust
/// GPU bound propagation must satisfy:
/// cpu_lower <= gpu_lower <= gpu_upper <= cpu_upper
///
/// If GPU is less conservative than CPU, results are invalid.
trait ConservativeKernel {
    fn propagate(&self, input: &Bounds) -> Bounds;

    /// Validation: GPU bounds must be at least as conservative as CPU
    fn validate_against_cpu(&self, gpu: &Bounds, cpu: &Bounds) -> bool {
        gpu.lower >= cpu.lower && gpu.upper <= cpu.upper
    }
}
```

**4. Determinism:**
- GPU reductions must be deterministic or bounds must account for variation
- Prefer tree-reduction over atomic operations
- Document any nondeterminism with explicit bounds on variation

---

## Revised Architecture

### Layer 1: z4 SMT Solver (We Own This)

```
┌─────────────────────────────────────────────────────────────────┐
│                         z4 SMT Solver                           │
│                                                                 │
│  DPLL(T) Core                                                   │
│  ├── Boolean search with VSIDS, restarts                       │
│  ├── Incremental: push/pop for backtracking                    │
│  ├── Clause learning with minimization                         │
│  └── Proof production (unsat cores, certificates)  ← REQUIRED  │
│                                                                 │
│  LRA Theory                                                     │
│  ├── Simplex for feasibility checking                          │
│  ├── Farkas certificates for unsat  ← REQUIRED                 │
│  └── Conflict explanation                                       │
│                                                                 │
│  Neural Network Theory (NEW)                                    │
│  ├── Bound propagation as theory propagation                   │
│  ├── Proof-carrying explanations                               │
│  └── Conservative float arithmetic                              │
└─────────────────────────────────────────────────────────────────┘
```

**Key additions to z4 for gamma-crown:**
1. Proof production for unsat cores
2. Farkas certificates for LRA conflicts
3. Custom theory propagator interface
4. Incremental bound updates

### Layer 2: gamma-crown Verification Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                    γ-CROWN v3 Verifier                          │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 0: IBP (fastest)                                         │
│  ├── Interval propagation                                       │
│  ├── Conservative f32 arithmetic                                │
│  └── If verified → DONE                                         │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 1: CROWN (tighter)                                       │
│  ├── Linear relaxation backward pass                           │
│  ├── Bounds intersected with IBP                               │
│  └── If verified → DONE                                         │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 2: α-CROWN (optimized)                                   │
│  ├── Gradient-based α optimization                             │
│  ├── Early stopping if verified                                │
│  └── If verified → DONE                                         │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 3: DPLL(T) with z4 (complete for PL fragment)           │
│  ├── ReLU phases as Boolean variables                          │
│  ├── Bound engine as theory propagator                         │
│  ├── Proof-carrying conflict explanations                      │
│  ├── Sound clause learning                                     │
│  └── Scoped cutting planes                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 3: tRust Compiler (Proof-in-Language)

**This is NOT an external verifier. The type system IS the proof system.**

```rust
// In tRust, proofs are embedded in types
// This compiles IFF the proof is valid

/// Shape-safe linear layer
/// The type signature IS the shape correctness proof
impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    // Compiles only if shapes match - no runtime check needed
    fn forward(&self, x: Tensor<[IN]>) -> Tensor<[OUT]>;
}

/// Soundness-carrying bound propagation
/// The return type includes a proof certificate
impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    fn propagate_ibp(
        &self,
        input: BoundedTensor<[IN]>,
    ) -> (BoundedTensor<[OUT]>, SoundnessProof)
    // The SoundnessProof type cannot be constructed unless
    // the implementation actually satisfies IBP soundness
}
```

**Migration path:**
1. TODAY: Kani proofs external to compilation
2. SOON: Proof harnesses structured for tRust migration
3. FUTURE: `cargo +trust build` proves gamma-crown correct

---

## Revised Implementation Phases

### Phase 0: Fix Current Bugs
**No change.** Shape mismatch must be fixed.

### Phase 1: Typed Core + Numeric Soundness

**Changed from v2:** Focus on typed internal IR, not full type-level migration.

**Deliverables:**
1. Canonical internal IR with fixed shapes (post-import)
2. Shape validation at import boundary
3. Conservative float arithmetic library
4. NaN/Inf handling throughout bound propagation
5. Outward rounding for all bound computations

**Acceptance tests:**
- [ ] No shape errors in typed core (compile-time)
- [ ] Numeric edge case suite (NaN, Inf, overflow)
- [ ] GPU bounds ⊇ CPU bounds validation

### Phase 2: Sound DPLL(T) Integration

**Changed from v2:** Staged milestones per review recommendation.

**Stage 2a: MVP Complete Solver (Slow but Correct)**
- Encode PL networks with exact LRA + ReLU disjunction
- Use z4 incrementally with push/pop
- No bound-engine learning yet
- Pass: correctness on small networks

**Stage 2b: Advisory Propagation with Validation**
- Bound engine proposes implied literals
- Validate via theory constraints
- Track antecedents for explanations
- Pass: implied literals match z4 deductions

**Stage 2c: Sound Learning**
- Conflict explanations from unsat cores
- Learned clauses validated by z4
- Pass: no false UNSATs on adversarial tests

**Stage 2d: Scoped Cuts**
- Global cuts: always valid
- Conditional cuts: guarded by literals
- Pass: cut soundness regression suite

### Phase 3: Algorithmic Soundness Proofs

**Changed from v2:** Split algorithmic vs implementation proofs.

**Algorithmic proofs (Kani, then tRust):**
- IBP over reals is sound (symbolic execution)
- CROWN relaxation over reals is sound
- ReLU linear relaxation is tight

**Implementation proofs (future tRust):**
- f32 with outward rounding preserves soundness
- No NaN/Inf escapes bounds
- GPU kernel conservativeness

**Deliverables:**
- Kani proof harnesses for core algorithms
- Documentation of proof structure for tRust migration
- Numeric soundness test suite (not proofs yet)

### Phase 4: GPU Acceleration

**Changed from v2:** Numeric soundness is prerequisite.

**Requirements:**
- Conservative arithmetic on GPU
- Deterministic reductions
- Validation: gpu_bounds ⊇ cpu_bounds

**Deliverables:**
- GPU IBP with outward rounding
- GPU CROWN with conservative matmul
- Correctness validation suite

### Phase 5: VNN-COMP 2026

**Target:** Win on piecewise-linear benchmarks. Honest incomplete on others.

**Deliverables:**
1. Full 2021-2025 benchmark assessment
2. Performance tuning per benchmark family
3. Ablation: bounds-only vs bounds+learning vs full DPLL(T)
4. Comparison vs α,β-CROWN (apples-to-apples)

**Acceptance:**
- VNN-COMP 2021 PL benchmarks: >95%
- VNN-COMP 2025 PL benchmarks: >60%
- Zero false UNSATs (soundness regression suite)

### Phase 6: ML-Assisted (Deferred)

**Per review:** Gate on evidence that strategy selection is the bottleneck.

**Prerequisites:**
- Stable verification corpus
- Ablation showing strategy selection limits performance
- Train/test separation across benchmark families

---

## Research Questions (Revised Per Review)

### 1. What explanations are both sound and small?

**Options:**
- Unsat core extraction (sound, may be large)
- Farkas certificates (sound, geometric)
- Proof-carrying propagation (tracked antecedents)

**Plan:** Start with unsat cores, evaluate size/cost tradeoff.

### 2. How to control propagation cost?

**Options:**
- Lazy propagation (only on demand)
- Incremental warm-starting
- Caching across branches

**Plan:** Profile first, optimize hot paths.

### 3. Numeric soundness on GPU?

**Strategy:** Outward rounding + validation against CPU.

**Research:** Can we do better than 2x width expansion?

### 4. Transformer operators?

**Current plan:** Out of scope for completeness.

**Future research:** Verified relaxations for softmax/LayerNorm with explicit error bounds.

---

## Phase Overview

| Phase | Goal | Stage Gate |
|-------|------|------------|
| 0 | Fix bugs | ViT loads without error |
| 1 | Typed core + numeric soundness | Numeric soundness tests pass |
| 2a | MVP complete solver | Correct on small PL networks |
| 2b | Advisory propagation | Implied literals validated by z4 |
| 2c | Sound learning | Zero false UNSATs on adversarial suite |
| 2d | Scoped cuts | Cut soundness tests pass |
| 3 | Algorithmic proofs | Kani proofs for IBP/CROWN |
| 4 | GPU acceleration | GPU bounds ⊇ CPU bounds |
| 5 | VNN-COMP 2026 | >95% on 2021 PL, zero false UNSAT |
| 6 | ML-assisted (conditional) | Only if bottleneck proven |

**Critical path:** Phase 2 (sound DPLL(T)) is the risk. Numeric soundness and explanation quality determine success.

---

## Success Criteria (Sharpened Per Review)

### Soundness (Non-Negotiable)

- [ ] **Zero false UNSATs** on soundness regression suite
- [ ] **Numeric edge cases pass**: NaN, Inf, overflow, underflow
- [ ] **Adversarial tests pass**: deliberately constructed hard cases
- [ ] **GPU bounds ⊇ CPU bounds**: validation on all benchmarks
- [ ] **Proof replay**: UNSAT results reproducible with trace

### Soundness Regression Suite (Per Review)

The review specifically recommends measurable soundness criteria:

**1. Adversarially Generated Tests:**
- Networks designed to expose numeric edge cases
- Properties at the boundary of SAT/UNSAT
- Relaxations that are tight (easy to get wrong)

**2. Numeric Edge Case Tests:**
- Weights/biases containing NaN, Inf, subnormals
- Inputs at f32 precision boundaries
- Operations that overflow/underflow intermediate results

**3. Differential Testing:**
- Compare γ-CROWN results against reference solver on shared operator set
- Any disagreement on SAT/UNSAT is a bug (investigate which is wrong)

**4. Reproducibility:**
- Given seed, results must be bitwise reproducible
- Document any sources of nondeterminism
- UNSAT traces must replay to same conclusion

### Correctness

- [ ] **Kani proofs**: IBP, CROWN, ReLU relaxation (algorithmic)
- [ ] **Type-safe core**: no shape errors in IR

### Performance + Ablations (Per Review)

- [ ] **Easy instances**: within 2x of α,β-CROWN
- [ ] **Hard instances**: better via conflict learning
- [ ] **Ablation benchmarks** (not just time):

| Configuration | Metrics to Report |
|---------------|-------------------|
| IBP only | Time, verified count |
| CROWN only | Time, verified count, bound tightness |
| α-CROWN | Time, verified count, optimization iterations |
| Bounds + learning | Time, verified, learned clauses, node count |
| Bounds + cuts | Time, verified, cuts generated, cut effectiveness |
| Full DPLL(T) | Time, verified, conflicts, backtracks |

### Competition

- [ ] **VNN-COMP 2021 PL**: >95%
- [ ] **VNN-COMP 2025 PL**: >60%
- [ ] **VNN-COMP 2026**: competitive (honest scope)

---

## Appendix: Why We Will Win

### Our Advantages

1. **Sound-first design**: Others sacrifice soundness for speed. We don't.
2. **Unified architecture**: SMT + bounds in one framework with proof production.
3. **We own the tools**: z4, tRust, kani_fast - we add what we need.
4. **Honest scope**: We claim completeness where it's true, incomplete elsewhere.

### The Winning Formula

```
Easy instances:  IBP/CROWN (milliseconds)
Medium instances: α-CROWN (seconds)
Hard instances:   Sound DPLL(T) with learning (complete for PL)

Non-PL instances: Honest "incomplete" with best-effort bounds
```

**Differentiator:** When we say "verified," we can produce a proof trace. When others say "verified," they hope their floats didn't lie.

---

## Summary: v3 vs v2 Changes

| Aspect | v2 | v3 |
|--------|----|----|
| Completeness claim | "All networks" | Piecewise-linear only |
| Softmax/GELU | Vaguely supported | Explicitly out of scope |
| Conflict explanations | "Find minimal subset" | Proof-carrying, validated |
| Cut persistence | Global | Scoped (global vs conditional) |
| Numeric soundness | Not addressed | First-class with outward rounding |
| Type safety | Full migration | Typed core + dynamic boundary |
| Kani proofs | "Proves verifier correct" | Split: algorithmic vs implementation |
| Timeline | Commit estimates | Phases with stage gates |
| Phase 2 | "Wire up z4" | 4-stage sound-first milestones |

**Core message:** The v2 architecture was directionally correct. v3 adds the rigor required for scientific credibility and actual soundness.
