# PLAN: γ-CROWN v2 - Unified Verification Architecture

**Vision:** Neural network verification as constraint solving, with compile-time correctness guarantees.

**Core Insight:** A neural network IS a system of equations. Verification IS satisfiability checking. We have z4 (a world-class SMT solver). Let's use it properly.

---

## The Mathematical Foundation

### Neural Networks as Equations

A neural network `f(x)` is a composition of functions:
```
f(x) = f_n(f_{n-1}(...f_1(x)))
```

Each layer is expressible in SMT theories:

| Layer | Mathematical Form | SMT Theory |
|-------|-------------------|------------|
| Linear | y = Wx + b | QF_LRA (linear real arithmetic) |
| ReLU | y = max(0, x) | QF_LRA + case split |
| BatchNorm | y = scale * normalize(x) + bias | QF_LRA |
| Conv2d | y = W ⊛ x + b | QF_LRA (unrolled) |
| Softmax | y_i = exp(x_i) / Σexp(x_j) | QF_NRA or bounded LRA |

### Verification as Satisfiability

A verification query:
```
∀x ∈ [l, u]: f(x) satisfies property P
```

Is equivalent to checking UNSAT of:
```
∃x ∈ [l, u]: f(x) violates P
```

**If UNSAT:** Property holds (verified)
**If SAT:** Counterexample found (violated)

### Why Pure SMT is Slow

For a network with n ReLU units:
- 2^n possible phase combinations
- Pure enumeration is exponential
- Most combinations are infeasible (bounds prove it)

### Why Bound Propagation Helps

Bound propagation provides:
1. **Pruning:** If ReLU input ∈ [3, 7], it's always active → no case split
2. **Tightening:** Narrower bounds → smaller search space
3. **Early conflicts:** Empty bounds = infeasible immediately
4. **Hints to solver:** Linear relaxations guide the search

---

## Current Architecture Problems

### Problem 1: Separation of Concerns (Wrong Way)

```
Current gamma-crown:
├── gamma-propagate (bound propagation) ← Does its own BaB
├── gamma-smt (SMT verification)        ← Separate, underutilized
└── These don't talk to each other properly
```

α,β-CROWN does bound propagation + BaB, but no SMT.
gamma-smt has SMT, but it's bolted on, not integrated.

### Problem 2: Runtime Shape Checking

```rust
// Current: Shapes are runtime values
fn propagate(&self, input: &Tensor) -> Result<Tensor> {
    if input.shape() != expected {
        return Err(ShapeMismatch);  // Runtime error!
    }
}
```

Shape errors like "expected [1632], got [96]" are symptoms of this.

### Problem 3: No Formal Verification of Algorithms

How do we know IBP is sound? We have tests, but tests can't cover all cases.
We should PROVE that our bound propagation is correct.

---

## The v2 Architecture

### Unified DPLL(T) Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    γ-CROWN v2 UNIFIED VERIFIER                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    z4 DPLL(T) Core                        │  │
│  │                                                           │  │
│  │  - Boolean variables: ReLU phase decisions (b_i ∈ {0,1}) │  │
│  │  - Theory: Linear Real Arithmetic (LRA)                   │  │
│  │  - Incremental: push/pop for backtracking                 │  │
│  │  - Learning: conflict clauses from failed branches        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Theory Propagator: Bound Engine              │  │
│  │                                                           │  │
│  │  PROPAGATE(assignment) → new_bounds, implied_phases       │  │
│  │    - Given partial ReLU phase assignment                  │  │
│  │    - Run IBP/CROWN to compute bounds                      │  │
│  │    - Deduce implied phases (if lb > 0, phase = active)    │  │
│  │    - Return to DPLL core                                  │  │
│  │                                                           │  │
│  │  EXPLAIN(conflict) → clause                               │  │
│  │    - Given: bounds became infeasible                      │  │
│  │    - Return: minimal set of phase decisions causing it    │  │
│  │    - This clause is learned, prevents same failure        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Cutting Planes Engine                   │  │
│  │                                                           │  │
│  │  - Generate GCP-CROWN cuts from bound analysis            │  │
│  │  - Add cuts as learned lemmas to SMT context              │  │
│  │  - Cuts persist across backtracking (global learning)     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  GPU Acceleration Layer                   │  │
│  │                                                           │  │
│  │  - Batch bound propagation on GPU (wgpu/Metal)            │  │
│  │  - Parallel α-CROWN optimization                          │  │
│  │  - Memory-efficient for large networks                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      Verification Strategy                      │
│                                                                 │
│  LEVEL 0: IBP                                                   │
│    - Cheapest bounds, often sufficient for easy instances       │
│    - If verified → DONE                                         │
│                                                                 │
│  LEVEL 1: CROWN / CROWN-IBP                                     │
│    - Tighter bounds via linear relaxation                       │
│    - If verified → DONE                                         │
│                                                                 │
│  LEVEL 2: α-CROWN                                               │
│    - Optimize relaxation parameters                             │
│    - If verified → DONE                                         │
│                                                                 │
│  LEVEL 3: DPLL(T) with z4                                       │
│    - Full SMT solving with learned bounds and cuts              │
│    - Complete: will find answer or counterexample               │
│    - Uses all bounds from levels 0-2 as hints                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovation: Bounds as Theory Propagation

In DPLL(T), the theory solver does two things:
1. **Propagate:** Given partial assignment, deduce consequences
2. **Explain:** Given conflict, explain why (for learning)

We implement this with bound propagation:

```rust
impl TheoryPropagator for BoundEngine {
    /// Given ReLU phase decisions, compute bounds and deduce new phases
    fn propagate(&mut self, assignment: &PartialAssignment) -> PropagationResult {
        // Set ReLU phases according to assignment
        for (relu_id, phase) in assignment.iter() {
            self.network.set_relu_phase(relu_id, phase);
        }

        // Run CROWN with fixed phases
        let bounds = self.network.propagate_crown(&self.input)?;

        // Check for conflicts (empty bounds)
        if bounds.is_empty() {
            return PropagationResult::Conflict;
        }

        // Deduce implied phases
        let mut implied = Vec::new();
        for (relu_id, pre_bounds) in bounds.pre_activation_iter() {
            if pre_bounds.lower > 0.0 {
                implied.push((relu_id, Phase::Active));
            } else if pre_bounds.upper < 0.0 {
                implied.push((relu_id, Phase::Inactive));
            }
        }

        PropagationResult::Ok { implied, bounds }
    }

    /// Explain why current assignment is infeasible
    fn explain(&self, conflict: Conflict) -> Clause {
        // Find minimal subset of phase decisions that cause conflict
        // This becomes a learned clause: ¬(phase_1 ∧ phase_2 ∧ ... ∧ phase_k)
        self.analyze_conflict(conflict)
    }
}
```

---

## Software Engineering Excellence

### Compile-Time Shape Safety

**Goal:** Shape mismatches should be compile errors, not runtime errors.

```rust
// === Type-Level Tensor Shapes ===

/// Marker trait for tensor shapes
trait Shape: 'static {
    const DIMS: &'static [usize];
    fn ndim() -> usize { Self::DIMS.len() }
    fn numel() -> usize { Self::DIMS.iter().product() }
}

/// Concrete shape types
struct S<const D0: usize>;
struct S2<const D0: usize, const D1: usize>;
struct S3<const D0: usize, const D1: usize, const D2: usize>;
struct S4<const D0: usize, const D1: usize, const D2: usize, const D3: usize>;

impl<const D0: usize> Shape for S<D0> {
    const DIMS: &'static [usize] = &[D0];
}
impl<const D0: usize, const D1: usize> Shape for S2<D0, D1> {
    const DIMS: &'static [usize] = &[D0, D1];
}
// ... etc

/// Typed tensor - shape is part of the type
struct Tensor<S: Shape> {
    data: Vec<f32>,
    _shape: PhantomData<S>,
}

impl<S: Shape> Tensor<S> {
    fn shape(&self) -> &'static [usize] { S::DIMS }
}

// === Type-Safe Layer Operations ===

/// Linear layer: In → Out
struct Linear<const IN: usize, const OUT: usize> {
    weight: [[f32; IN]; OUT],
    bias: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Type signature GUARANTEES correct shapes
    fn forward(&self, input: Tensor<S<IN>>) -> Tensor<S<OUT>> {
        // Can't pass wrong shape - won't compile!
        todo!()
    }

    fn propagate_ibp(
        &self,
        input: BoundedTensor<S<IN>>
    ) -> BoundedTensor<S<OUT>> {
        // Shape correctness is compile-time guaranteed
        todo!()
    }
}

// === Network as Type-Level Composition ===

/// Type-safe network builder
struct Network<In: Shape, Out: Shape> {
    // Internal layers are type-erased but In/Out are known
    layers: Vec<Box<dyn Layer>>,
    _phantom: PhantomData<(In, Out)>,
}

impl<In: Shape, Out: Shape> Network<In, Out> {
    fn verify(&self, input: BoundedTensor<In>, spec: Spec<Out>) -> VerificationResult {
        // Type system ensures input and output shapes match
        todo!()
    }
}
```

**Benefits:**
- Shape mismatch = compile error
- IDE shows expected shapes
- Refactoring is safe
- Documentation is automatic

### Formal Verification with Kani

**Goal:** Prove that our bound propagation is mathematically sound.

```rust
// === Kani Proofs for Soundness ===

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Prove IBP is sound: true output always within computed bounds
    #[kani::proof]
    #[kani::unwind(10)]  // Unroll loops up to 10 iterations
    fn ibp_soundness_linear() {
        // Arbitrary linear layer (small for tractability)
        let weight: [[f32; 4]; 4] = kani::any();
        let bias: [f32; 4] = kani::any();
        let layer = Linear { weight, bias };

        // Arbitrary input within bounds
        let lower: [f32; 4] = kani::any();
        let upper: [f32; 4] = kani::any();
        kani::assume(lower.iter().zip(&upper).all(|(l, u)| l <= u));

        let input: [f32; 4] = kani::any();
        kani::assume(input.iter().zip(&lower).zip(&upper)
            .all(|((x, l), u)| *l <= *x && *x <= *u));

        // Compute true output and IBP bounds
        let output = layer.forward_concrete(&input);
        let (out_lower, out_upper) = layer.propagate_ibp(&lower, &upper);

        // PROVE: output is within IBP bounds
        for i in 0..4 {
            kani::assert(
                out_lower[i] <= output[i] && output[i] <= out_upper[i],
                "IBP bounds must contain true output"
            );
        }
    }

    /// Prove ReLU relaxation is sound
    #[kani::proof]
    fn relu_relaxation_soundness() {
        let lower: f32 = kani::any();
        let upper: f32 = kani::any();
        kani::assume(lower <= upper);

        let x: f32 = kani::any();
        kani::assume(lower <= x && x <= upper);

        let y = x.max(0.0);  // True ReLU output

        // Compute relaxation bounds
        let (y_lower, y_upper) = relu_ibp(lower, upper);

        kani::assert(y_lower <= y && y <= y_upper, "ReLU IBP sound");

        // For CROWN: linear relaxation
        let (slope_l, intercept_l, slope_u, intercept_u) = relu_crown(lower, upper);
        let crown_lower = slope_l * x + intercept_l;
        let crown_upper = slope_u * x + intercept_u;

        kani::assert(crown_lower <= y && y <= crown_upper, "ReLU CROWN sound");
    }

    /// Prove conflict clause extraction is correct
    #[kani::proof]
    fn conflict_clause_correctness() {
        // If we learn clause ¬(a ∧ b ∧ c), then (a ∧ b ∧ c) must be infeasible
        // This ensures our conflict analysis is sound
        todo!()
    }
}
```

**Benefits:**
- Mathematical guarantee of soundness
- Catches subtle bugs that tests miss
- Documentation of algorithm invariants
- Confidence in refactoring

### Contract-Based Programming

Use `contracts` crate for runtime-checked pre/post conditions:

```rust
use contracts::*;

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    #[requires(input.len() == IN, "Input dimension must match")]
    #[ensures(ret.len() == OUT, "Output dimension must match")]
    #[ensures(
        ret.iter().zip(&expected_lower).zip(&expected_upper)
            .all(|((y, l), u)| l <= y && y <= u),
        "Output must be within bounds (soundness)"
    )]
    fn propagate_ibp(&self, input: &BoundedTensor) -> BoundedTensor {
        // Implementation
    }
}
```

---

## Implementation Phases

### Phase 0: Fix Current Bugs (1-2 commits)
**Owner:** Current Worker

- [ ] Debug and fix ViT shape mismatch (1632 vs 96)
- [ ] Add defensive shape assertions throughout
- [ ] Verify VNN-COMP 2025 ACAS-Xu and CIFAR-100 still work

### Phase 1: Type-Safe Tensors (5-8 commits)
**Goal:** Eliminate runtime shape errors

1. [ ] Design type-level shape system (`Shape` trait, `Tensor<S>`)
2. [ ] Implement for core tensor operations
3. [ ] Migrate `Linear`, `Conv2d`, `ReLU` layers
4. [ ] Migrate `BoundedTensor` and `LinearBounds`
5. [ ] Update network builders
6. [ ] Ensure all shape errors become compile errors

### Phase 2: z4 DPLL(T) Integration (10-15 commits)
**Goal:** Unified SMT-based verification

1. [ ] Study NeuralSAT paper and z4 incremental API
2. [ ] Design `TheoryPropagator` trait
3. [ ] Implement `BoundEngine` as theory propagator
4. [ ] Implement conflict clause extraction
5. [ ] Wire up DPLL(T) main loop with z4
6. [ ] Implement cutting plane integration
7. [ ] Benchmark: easy instances (should match α,β-CROWN speed)
8. [ ] Benchmark: hard instances (should beat α,β-CROWN)

### Phase 3: Kani Verification (5-8 commits)
**Goal:** Prove algorithm soundness

1. [ ] Set up Kani in CI
2. [ ] Prove IBP soundness for Linear, ReLU, Conv2d
3. [ ] Prove CROWN relaxation soundness
4. [ ] Prove conflict clause correctness
5. [ ] Add proofs for new features going forward

### Phase 4: GPU Acceleration (8-12 commits)
**Goal:** Fast bound propagation on GPU

1. [ ] Profile current bottlenecks
2. [ ] Move batch matrix multiply to GPU
3. [ ] GPU-accelerated α-CROWN optimization
4. [ ] Memory-efficient streaming for large networks
5. [ ] Benchmark: should be 10-100x faster than CPU

### Phase 5: VNN-COMP 2025 Domination (10-15 commits)
**Goal:** Win the competition

1. [ ] Full assessment on all 26 benchmarks
2. [ ] Targeted optimizations for each benchmark category
3. [ ] Tune heuristics (branching, cut generation, timeouts)
4. [ ] Final performance comparison vs α,β-CROWN

---

## Success Metrics

### Correctness
- [ ] Zero shape mismatch errors (compile-time guarantee)
- [ ] Kani proofs for all core algorithms
- [ ] No soundness bugs (verified bounds always contain true output)

### Performance
- [ ] Easy instances: Within 2x of α,β-CROWN
- [ ] Hard instances: 2-10x better than α,β-CROWN (conflict learning)
- [ ] GPU acceleration: 10-100x over CPU

### Competition
- [ ] VNN-COMP 2021: >90% (currently 80%)
- [ ] VNN-COMP 2025: >50% (currently <15%)
- [ ] Beat α,β-CROWN on aggregate score

---

## Research Questions

1. **Can z4 replace NeuralSAT's custom SAT solver?**
   - NeuralSAT uses custom DPLL with domain-specific heuristics
   - z4 is more general but highly optimized
   - Need to benchmark

2. **How to extract good conflict clauses from bound propagation?**
   - NeuralSAT has specific techniques
   - May need adaptation for our CROWN-based bounds

3. **Can we use Kani to verify the SMT encoding is correct?**
   - Prove that network encoding is equisatisfiable
   - This would be a strong correctness guarantee

4. **What's the right balance between bound propagation and SMT?**
   - Too much bound propagation = slow on easy instances
   - Too little = slow on hard instances
   - Need adaptive strategy

---

---

## ML-Assisted Verification (New Direction)

### The Insight

We have thousands of public neural networks (HuggingFace, PyTorch Hub, ONNX Model Zoo).
We have years of VNN-COMP benchmark results.
We have kani_fast's AI synthesis pattern.

**Why not learn verification strategies from data?**

### What ML Could Learn

| Task | Training Data | Model Output |
|------|---------------|--------------|
| **Easy/Hard Classification** | (network, property) → verification time | Predict: IBP-solvable? Needs BaB? Timeout? |
| **Branching Heuristic** | (network, split history) → outcome | Which neuron split leads to fastest verification? |
| **Cut Selection** | (network, cuts) → bound improvement | Which cuts will tighten bounds most? |
| **α Initialization** | (network structure) → optimal α | Good starting point for α-CROWN optimization |
| **Counterexample Guidance** | (network, failed proofs) → CE region | Where should PGD focus? |

### Architecture: ICE-Style Learning for NN Verification

Borrowing from kani_fast's `kani-fast-ai`:

```
┌─────────────────────────────────────────────────────────────────┐
│                    gamma-crown-ai                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Verification Corpus                          │  │
│  │                                                           │  │
│  │  Store: (network_hash, property, strategy, result, time)  │  │
│  │  Lookup: Given new (network, property), find similar      │  │
│  │  Use: Apply strategy that worked on similar instances     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              GNN Predictor (Graph Neural Network)         │  │
│  │                                                           │  │
│  │  Input: Network as graph (layers=nodes, connections=edges)│  │
│  │  Features: Layer types, sizes, activation functions       │  │
│  │  Output: Difficulty score, recommended strategy           │  │
│  │  Training: VNN-COMP results + HuggingFace networks        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              LLM Strategy Advisor                         │  │
│  │                                                           │  │
│  │  Input: Network summary + property + failed attempts      │  │
│  │  Output: "Try splitting neuron X" or "Add cut on layer Y" │  │
│  │  Feedback: Success/failure → fine-tune suggestions        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              ICE Learner (Counterexample-Guided)          │  │
│  │                                                           │  │
│  │  Positive: Networks that verified quickly                 │  │
│  │  Negative: Networks that timed out or had counterexamples │  │
│  │  Learn: Decision tree/rules for strategy selection        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Data Sources

1. **VNN-COMP History (2020-2024)**
   - Thousands of (network, property, result, time) tuples
   - Ground truth for difficulty classification

2. **HuggingFace Model Hub**
   - Extract network structures (don't need weights for features)
   - Generate synthetic properties
   - Build corpus of "real-world" architectures

3. **PyTorch Hub / ONNX Model Zoo**
   - More architectures
   - Different domains (vision, NLP, audio)

4. **Our Own Runs**
   - Every verification we run → add to corpus
   - Continuous learning

### Implementation Plan (Phase 6)

1. [ ] Create `gamma-crown-ai` crate (mirror kani-fast-ai structure)
2. [ ] Implement Verification Corpus (SQLite backend)
3. [ ] Extract network features (graph structure, layer stats)
4. [ ] Train GNN on VNN-COMP data (predict difficulty)
5. [ ] Integrate LLM for strategy suggestions
6. [ ] ICE learner for branching heuristics
7. [ ] Benchmark: Does ML guidance improve verification speed?

---

## Integration with Tool Ecosystem

### The Full Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER CODE (Rust)                           │
│  Neural network implementation with specifications              │
├─────────────────────────────────────────────────────────────────┤
│                        tRust                                    │
│  Compile-time verification: shape safety, overflow, bounds      │
│  "If it compiles, shapes are correct"                           │
├─────────────────────────────────────────────────────────────────┤
│                      kani_fast                                  │
│  Prove algorithm soundness: IBP is correct, CROWN is sound      │
│  "Kani proves our verifier is correct"                          │
├─────────────────────────────────────────────────────────────────┤
│                     gamma-crown v2                              │
│  Runtime verification of neural networks                        │
│  Uses z4 DPLL(T), type-safe tensors, ML guidance                │
├─────────────────────────────────────────────────────────────────┤
│                         z4                                      │
│  SMT solving: DPLL, theories (LRA, LIA), conflict learning      │
│  The engine that powers everything                              │
└─────────────────────────────────────────────────────────────────┘
```

### Concrete Integrations

| Tool | How gamma-crown Uses It |
|------|------------------------|
| **tRust** | Compile gamma-crown with tRust → shape errors are compile errors |
| **kani_fast** | Prove IBP/CROWN soundness → mathematical guarantee |
| **z4** | DPLL(T) core → drives ReLU splitting, theory propagation |
| **gamma-crown-ai** | ML guidance → predict strategy, suggest cuts |

### Why This Wins

1. **Correctness Stack:**
   - tRust proves: no shape bugs, no overflow, no UB
   - kani_fast proves: algorithms are mathematically sound
   - z4 proves: verification results are correct

2. **Performance Stack:**
   - ML predicts easy instances → skip to simple IBP
   - Bound propagation handles 90% of cases fast
   - DPLL(T) + learning handles hard cases

3. **Completeness Stack:**
   - SMT-based → will always find answer or counterexample
   - No "unknown" except timeout
   - Conflict learning → gets faster over time

---

## References

1. **NeuralSAT:** "NeuralSAT: A DPLL(T) Framework for Neural Network Verification"
2. **α,β-CROWN:** "Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers"
3. **GCP-CROWN:** "GCP-CROWN: General Cutting Planes for Bound-Propagation-Based Neural Network Verification"
4. **z4:** Rust SMT solver (local: `~/z4/`)
5. **kani_fast:** Fast Rust verification with AI (local: `~/kani_fast/`)
6. **tRust:** Rustc fork with built-in verification (local: `~/trust/`)

---

## Timeline

| Phase | Goal | Commits | Cumulative |
|-------|------|---------|------------|
| Phase 0 | Fix shape bugs | 2 | 2 |
| Phase 1 | Type-safe tensors (tRust integration) | 8 | 10 |
| Phase 2 | z4 DPLL(T) integration | 15 | 25 |
| Phase 3 | kani_fast proofs (IBP/CROWN soundness) | 8 | 33 |
| Phase 4 | GPU acceleration | 12 | 45 |
| Phase 5 | VNN-COMP 2025 benchmarking | 15 | 60 |
| Phase 6 | ML-assisted verification (gamma-crown-ai) | 15 | 75 |

**Estimated total: 75 commits (~1.5 weeks of worker time)**

### Phase Dependencies

```
Phase 0 (bugs) ──┬──→ Phase 1 (types) ──→ Phase 3 (kani proofs)
                 │
                 └──→ Phase 2 (z4) ──→ Phase 4 (GPU) ──→ Phase 5 (VNN-COMP)
                                                              │
                                                              └──→ Phase 6 (ML)
```

Phase 6 (ML) can start after Phase 5 generates enough training data from VNN-COMP runs.

---

## Long-Term Vision: Verification-Guided Neural Network Design

**This section describes where we're going, not what we're doing now.**

Once the core engine is strong enough to win VNN-COMP, it becomes a tool for CREATING better networks, not just validating them.

### The Feedback Loop

```
DESIGN → TRAIN → VERIFY → LEARN FROM FAILURES → DESIGN BETTER
```

### Future Applications (Post VNN-COMP 2026)

| Application | Description | Prerequisite |
|-------------|-------------|--------------|
| **V-NAS** | Architecture search optimizing for verifiability | Fast verification |
| **Verified Adversarial Training** | Train against verified worst-case, not PGD heuristics | Differentiable bounds |
| **Verification-Aware Loss** | Penalize loose bounds during training | Efficient bound propagation |
| **Verification-Guided Pruning** | Remove neurons that hurt verifiability | Per-neuron analysis |
| **Architecture Repair** | Auto-fix architectures that fail verification | Failure diagnosis |
| **Verified-by-Construction** | Synthesize networks guaranteed to satisfy properties | Full stack working |

### For LLMs (Long-Term)

- Design verifiable attention mechanisms
- Verification-guided fine-tuning
- Verified safety constraints in training
- Component-level guarantees composed into system guarantees

**These are NOT current priorities. They require a working, competitive core engine first.**

### Priority Stack

```
NOW:     Fix bugs → Build engine → Win VNN-COMP 2026
THEN:    ML guidance → Verification-guided training
FUTURE:  V-NAS → Verified-by-construction synthesis
```

The advanced applications are WHY we build the engine. VNN-COMP 2026 is HOW we prove the engine works.

---

## Research Landscape: Certified Training

**Prior work exists but is limited.** Key papers:

| Paper | Method | Result |
|-------|--------|--------|
| **CROWN-IBP** (Zhang et al.) | Train with IBP + CROWN bounds | 66.94% verified on CIFAR-10 |
| **Gaussian Loss Smoothing** | Smooth loss for certified training | Tighter convex relaxations |
| **CACTUS** | Compression + certified training | Maintain accuracy while compressing |
| **Certified Robustness Limits** (Zhang & Sun) | Bayes error analysis | Fundamental limits identified |

**Gap in the literature:**
- Current certified training is SLOW (bounds computed every batch)
- Limited to small networks (MNIST, CIFAR-10)
- No integration with architecture search
- No LLM-scale applications

**Our opportunity:** A 100x faster verifier enables certified training at scale.

### The LLM Security Implication

A sufficiently powerful verifier has dual use:

| Capability | Offensive Use | Defensive Use |
|------------|---------------|---------------|
| **Find counterexamples** | Jailbreak any LLM (find inputs that bypass safety) | Red-team your own models |
| **Prove properties** | N/A | Train LLMs with mathematical safety guarantees |
| **Verified training** | N/A | Build models that CANNOT be jailbroken |

**The vision:** Instead of playing whack-a-mole with jailbreaks, train models where harmful outputs are mathematically impossible.

```
Property: ∀ inputs x, P(harmful_output | x) = 0
Training: Optimize task loss subject to verification constraint
Result: Model that CANNOT produce harmful outputs (proven, not hoped)
```

This is why verification matters for AI safety. Not post-hoc testing, but **verified-by-construction safety**.

### Sample Efficiency Hypothesis

**Conjecture:** Verification can substitute for data in robustness.

```
Traditional: Train on massive adversarial dataset → empirical robustness
Verified:    Train on small clean dataset + verification loss → proven robustness
```

If true, this enables:
- Training cutting-edge models on laptops
- Dramatically reduced data requirements
- Mathematical guarantees instead of empirical hopes

**This is an open research question worth pursuing.**

---

## Appendix: The "If It Compiles, It's Correct" Philosophy

The shape mismatch bug is a symptom of a deeper truth: **dynamic typing is a liability in safety-critical code.**

We are building a VERIFIER. Our job is to provide mathematical guarantees about neural networks. If our own code can have runtime type errors, how can we trust its outputs?

The solution is to push as much as possible into the type system:
- Tensor shapes → type parameters
- Layer compatibility → trait bounds
- Network structure → type-level composition
- Soundness properties → Kani proofs

This is not just about catching bugs earlier. It's about building a system where correctness is STRUCTURAL, not incidental.

When you can't express a shape mismatch in the type system, you can't have a shape mismatch bug.

When Kani proves your algorithm is sound, you know it's sound for ALL inputs, not just the ones you tested.

This is the level of rigor that a verification tool should have.
