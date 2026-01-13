# PLAN: Achieving Practically Useful Bounds

**Status:** RESEARCH
**Created:** 2026-01-02
**Problem:** Bounds of ±444,000 prove nothing. Need ±10 or better.

---

## The Core Problem

Neural network verification is **NP-hard**. CROWN provides polynomial-time bounds by linear relaxation, but:

1. Each layer's relaxation introduces "slack" (gap between true bounds and computed bounds)
2. Slack compounds **multiplicatively** through depth
3. For depth D with 10% slack per layer: final slack = 1.1^D

| Depth | Slack Factor |
|-------|--------------|
| 10 | 2.6x |
| 28 | 14x |
| 56 | 208x |
| 112 | 43,000x |

Qwen3-0.6B has 28 layers × 2 (attention + FFN) = 56 effective layers minimum.

---

## Approaches That Could Work

### Approach 1: Branch-and-Bound at Massive Scale (MOST PROMISING)

**How it works:**
- Split input space into N subregions
- Each subregion has tighter intermediate bounds
- Verify each subregion independently
- If all subregions verify → full verification

**Why it helps:**
- Smaller input region → tighter activation bounds
- Tighter activation bounds → less relaxation slack
- Exponential improvement with branching depth

**The math:**
```
Input region radius ε
Split into 2^k subregions, each radius ε/2^(k/d)
If slack ~ radius, then slack reduces 2^(k/d) per split
```

**Challenge:**
- Need 10^6 to 10^12 subproblems for large networks
- Each subproblem still takes O(network_size) time
- Total: O(10^12 × network_size) = infeasible naively

**Our advantage:**
- γ-CROWN is 10-100x faster than Python
- GPU parallelism for subproblems
- Distributed computing across cluster
- Smart branching (split dimensions that cause most slack)

**Implementation:**
```rust
// Already have BranchAndBound infrastructure
pub struct BranchAndBound {
    pub strategy: BranchingStrategy,
    pub max_domains: usize,  // Increase to 10^6+
    pub parallel: bool,       // GPU parallel verification
}
```

**Target:** 10^6 parallel subproblems on GPU, 10^3 sequential batches = 10^9 total

---

### Approach 2: Certified Training (DIFFERENT PARADIGM)

**How it works:**
- Train network with IBP/CROWN in the loop
- Network learns to have tight verification bounds
- Trade ~2-5% accuracy for 10-100x tighter bounds

**Papers:**
- IBP Training (Gowal et al., 2018)
- CROWN-IBP (Zhang et al., 2020)
- Auto-LiRPA training (Xu et al., 2020)

**Implementation:**
```python
# Training loop (Python, not Rust)
for batch in data:
    # Standard forward pass
    logits = model(x)
    clean_loss = cross_entropy(logits, y)

    # IBP bound computation
    lower, upper = ibp_bounds(model, x, epsilon)
    worst_case = lower[y]  # Worst case for correct class
    robust_loss = cross_entropy(-worst_case, y)

    # Combined loss
    loss = alpha * clean_loss + (1-alpha) * robust_loss
    loss.backward()
```

**Result:** Networks that are designed to be verifiable.

**Limitation:** Requires retraining. Can't verify existing models.

---

### Approach 3: Compositional Verification (ARCHITECTURAL)

**How it works:**
- Don't verify end-to-end
- Verify each component independently
- Compose guarantees mathematically

**For transformers:**
```
1. Verify: Embedding layer maps input to bounded representation
2. Verify: Each attention block is L-Lipschitz
3. Verify: Each FFN block is M-Lipschitz
4. Compose: Total Lipschitz = L^n × M^n for n layers
```

**Why it might work:**
- Per-layer verification is much tighter
- Lipschitz composition is multiplicative but controlled
- Can compute tighter Lipschitz bounds than end-to-end

**Challenge:**
- Need tight per-layer Lipschitz constants
- Composition may still explode for deep networks
- Attention Lipschitz is tricky (data-dependent)

---

### Approach 4: Probabilistic Bounds (WEAKER GUARANTEE)

**How it works:**
- Give up 100% soundness
- Prove: "With probability ≥ 1-δ, output ∈ [l, u]"
- Use concentration inequalities + sampling

**The math:**
```
Sample N points uniformly in ε-ball
Compute outputs for all samples
Use Hoeffding/Chernoff to bound probability of outliers
```

**For N = 10^6 samples, δ = 10^-9:**
- If all samples satisfy property → property holds w.p. ≥ 1-10^-9
- This is often "good enough" for practical deployment

**Challenge:**
- Not 100% sound
- Adversarial examples may be in the 10^-9 probability region
- Not acceptable for safety-critical applications

---

### Approach 5: Specification Weakening

**Instead of proving:**
> "For ALL x in ε-ball, output is in [l, u]"

**Prove weaker but useful properties:**

1. **Local Lipschitz:** "Network is K-Lipschitz around input x"
   - Easier to verify
   - Implies bounded output change for bounded input change

2. **Gradient bounds:** "‖∇f(x)‖ ≤ G for all x in region"
   - First-order approximation of Lipschitz
   - Verifiable via interval arithmetic on Jacobian

3. **Monotonicity:** "Output i is monotonic in input j"
   - Structural property, not numeric bound
   - Sometimes more useful than bounds

4. **Top-1 stability:** "Top predicted class doesn't change"
   - Don't need output bounds, just argmax stability
   - Often much easier to verify

---

## Recommended Path Forward

### Phase 1: Massive Branch-and-Bound (Next Priority)

1. **GPU-parallel subproblem verification**
   - Port core IBP/CROWN to WGPU
   - Verify 10^4 subproblems in parallel
   - Target: 10^6 subproblems/minute

2. **Smart branching heuristics**
   - Track which neurons cause most slack
   - Split on high-slack dimensions first
   - Implement BABSR scoring from α,β-CROWN

3. **Distributed verification**
   - Split across multiple GPUs/machines
   - Target: 10^9 subproblems in reasonable time

### Phase 2: Hybrid Approaches

1. **CROWN + sampling validation**
   - Use CROWN for fast approximation
   - Sample 10^6 points to validate bounds empirically
   - Report both: "Proven: [-100, 100], Empirical: [-5, 5]"

2. **Layer-wise analysis**
   - Report per-layer bound growth
   - Identify which layers cause explosion
   - Focus tightening efforts there

### Phase 3: Architecture-Specific Bounds

1. **Softmax-aware bounds**
   - Softmax outputs sum to 1
   - Exploit this constraint for tighter bounds

2. **Attention structure**
   - Attention is low-rank (head_dim << seq_len × hidden)
   - Exploit rank structure for tighter bounds

3. **RoPE/position encoding**
   - Positional encodings have known structure
   - Use for tighter bounds

---

## Success Metrics

| Model | Current Bounds | Target | Method |
|-------|---------------|--------|--------|
| whisper-tiny | ±4 | ±1 | Branch-and-bound |
| whisper-small | ±50 | ±5 | Branch-and-bound |
| whisper-large | ±177 | ±20 | GPU parallel B&B |
| Qwen3-0.6B | ±444,000 | ±1,000 | Distributed B&B |
| Qwen3-0.6B | ±444,000 | ±100 | Certified training |

**Honest assessment:** Getting Qwen3-0.6B to ±100 with existing architecture may require certified retraining. Branch-and-bound alone may only get us to ±10,000 (still 44x improvement).

---

## References

- α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- VNN-COMP results: https://sites.google.com/view/vnn2024
- Certified training: arxiv:2006.13311
- Branch-and-bound for NNV: arxiv:2103.06624

---

## Commands

```bash
# Current branch-and-bound test
cargo test -p gamma-propagate branch_and_bound

# Profile bound explosion per layer
./target/release/gamma verify model.safetensors --verbose --per-layer-bounds

# GPU parallel verification (once implemented)
./target/release/gamma verify model.safetensors --gpu --parallel-subproblems 10000
```
