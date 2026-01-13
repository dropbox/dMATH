# VNN-COMP Competing Systems and Methods

## γ-CROWN Design Philosophy

> **γ-CROWN must ALWAYS win because it has ALL available methods, chooses the best one dynamically, and has the best implementation.**

While α,β-CROWN wins overall, other tools win specific benchmarks:
- **nnenum** wins on some ACAS-Xu instances
- **ERAN** wins on abstract interpretation-friendly benchmarks
- **PyRAT** competitive on recent benchmarks
- **Marabou** strong on SMT-amenable problems

γ-CROWN implements ALL methods and automatically selects the optimal approach for each problem.

**Goal:** Implement ALL methods from ALL competing verification systems.

---

## VNN-COMP Winners and Top Performers

| Year | Winner | Runner-up |
|------|--------|-----------|
| 2020 | ERAN | VeriNet |
| 2021 | α,β-CROWN | ERAN |
| 2022 | α,β-CROWN | VeriNet |
| 2023 | α,β-CROWN | PyRAT |
| 2024 | α,β-CROWN | PyRAT |

**α,β-CROWN has won 5 consecutive years (2021-2024).** To beat them, we need ALL their methods.

---

## 1. α,β-CROWN (Reference Implementation)

**Repo:** https://github.com/Verified-Intelligence/alpha-beta-CROWN

### Methods to Implement

| Method | Paper | Status in γ-CROWN | Priority |
|--------|-------|-------------------|----------|
| CROWN | NeurIPS'18 | ✅ Done | - |
| α-CROWN | ICLR'21 | ✅ Done | - |
| β-CROWN | NeurIPS'21 | ✅ Done | - |
| **GCP-CROWN** | NeurIPS'22 | ❌ Missing | **CRITICAL** |
| **BICCOS** | ICLR'23 | ❌ Missing | **HIGH** |
| **MIP Integration** | - | ❌ Missing | **HIGH** |
| **INVPROP** | - | ❌ Missing | **MEDIUM** |
| GenBaB | ICLR'24 | ❌ Missing | MEDIUM |
| BaB-Attack | - | ❌ Missing | LOW |
| Clip-and-Verify | - | ❌ Missing | LOW |

### Key Papers

1. **GCP-CROWN** (arxiv:2208.05740)
   - General Cutting Planes for fast, scalable verification
   - Adds linear constraints (cuts) that tighten ReLU relaxation
   - **Critical for matching SOTA bound tightness**

2. **BICCOS** (arxiv:2302.08730)
   - BaB-Induced Cutting Planes with Constraint Optimization Search
   - Generates cutting planes during branch-and-bound
   - Further tightens bounds beyond GCP-CROWN

3. **INVPROP** (forward/backward constraint propagation)
   - Uses output constraints to tighten intermediate bounds
   - Propagates constraints backward through network

---

## 2. ERAN (ETH Zurich)

**Repo:** https://github.com/eth-sri/eran

### Methods to Implement

| Method | Paper | Status | Priority |
|--------|-------|--------|----------|
| DeepZ | NeurIPS'18 | ❌ Missing | MEDIUM |
| **DeepPoly** | POPL'19 | ❌ Missing | **HIGH** |
| GPUPoly | MLSys'21 | ❌ Missing | MEDIUM |
| RefineZono | ICLR'19 | ❌ Missing | LOW |
| RefinePoly | NeurIPS'19 | ❌ Missing | LOW |

### Key Techniques

- **Abstract Interpretation**: Uses abstract domains (Zonotopes, Polyhedra)
- **DeepPoly**: Combines intervals + polyhedra for tighter bounds
- **MILP Refinement**: LP/MILP solvers for bound tightening
- **GPU Acceleration**: GPUPoly for parallel abstract interpretation

---

## 3. Marabou (Stanford)

**Repo:** https://github.com/NeuralNetworkVerification/Marabou

### Methods

| Method | Status | Priority |
|--------|--------|----------|
| SMT-based verification | ❌ Different paradigm | LOW |
| DeepPoly bound tightening | ❌ Missing | MEDIUM |
| DeepSoI search | ❌ Missing | LOW |
| Split-and-Conquer (SNC) | ❌ Missing | LOW |

### Key Insight
Marabou uses SMT solvers - fundamentally different approach from bound propagation. Not directly applicable but worth understanding for completeness.

---

## 4. nnenum (Stanley Bak)

**Repo:** https://github.com/stanleybak/nnenum

### Methods

| Method | Status | Priority |
|--------|--------|----------|
| Multi-level zonotopes | ❌ Missing | MEDIUM |
| ImageStar trick | ❌ Missing | MEDIUM |
| Parallelized ReLU splitting | Partial (have B&B) | LOW |
| Star set overapproximation | ❌ Missing | LOW |

### Key Techniques
- **Three zonotope levels**: Different precision/cost tradeoffs
- **ImageStar**: Efficient layer propagation for CNNs
- **Completeness focus**: Emphasizes provable completeness

---

## 5. OVAL (Oxford)

**Repo:** https://github.com/oval-group/oval-bab

### Methods

| Method | Status | Priority |
|--------|--------|----------|
| IBP | ✅ Done | - |
| CROWN | ✅ Done | - |
| Fast-Lin | ❌ Missing | LOW |
| Lagrangian Decomposition | ❌ Missing | MEDIUM |
| **FSB Branching** | ❌ Missing | **HIGH** |
| SR branching | ❌ Missing | MEDIUM |
| Active Set solver | ❌ Missing | LOW |
| Saddle Point solver | ❌ Missing | LOW |
| MI-FGSM attack | ❌ Missing | LOW |

### Key Techniques
- **FSB (Filtered Smart Branching)**: State-of-the-art branching heuristic
- **Dual solvers**: Lagrangian decomposition for tighter bounds
- **Planet relaxation**: Efficient LP relaxation

---

## 6. NNV (Vanderbilt)

**Repo:** https://github.com/verivital/nnv

### Methods

| Method | Status | Priority |
|--------|--------|----------|
| Star set reachability | ❌ Missing | LOW |
| Zonotope analysis | ❌ Missing | MEDIUM |
| Polyhedra analysis | ❌ Missing | LOW |
| Exact verification | Different approach | LOW |

### Focus Areas
- Control systems verification
- Neural ODEs
- Recurrent networks
- Less relevant for our image/transformer focus

---

## 7. VeriNet

**Repo:** https://github.com/vas-group-imperial/VeriNet

### Methods
- Maxpool relaxations
- Residual network support
- Mixed precision verification

---

## 8. PyRAT (Recent Competitor)

Recent strong performer in VNN-COMP 2023-2024. Need to investigate methods.

---

## Implementation Priority Matrix

### CRITICAL (Must implement to match SOTA)

| Method | From | Effort | Impact |
|--------|------|--------|--------|
| **GCP-CROWN** | α,β-CROWN | ~30 commits | Tighter bounds |
| **FSB Branching** | OVAL | ~10 commits | Better branching |
| **GPU B&B** | α,β-CROWN | ~25 commits | Speed |

### HIGH (Important for competitiveness)

| Method | From | Effort | Impact |
|--------|------|--------|--------|
| BICCOS | α,β-CROWN | ~20 commits | Tighter bounds |
| MIP Integration | α,β-CROWN | ~15 commits | Completeness |
| DeepPoly | ERAN | ~15 commits | Alternative bounds |
| INVPROP | α,β-CROWN | ~10 commits | Bound tightening |

### MEDIUM (Nice to have)

| Method | From | Effort | Impact |
|--------|------|--------|--------|
| GenBaB | α,β-CROWN | ~20 commits | General activations |
| Zonotope analysis | nnenum/ERAN | ~15 commits | Alternative approach |
| Lagrangian Decomposition | OVAL | ~15 commits | Dual bounds |
| GPUPoly | ERAN | ~20 commits | GPU abstract interp |

### LOW (Future work)

- SMT integration (Marabou approach)
- Star set reachability (NNV)
- BaB-Attack
- Clip-and-Verify

---

## Papers to Read

### Must Read (CRITICAL)
1. **GCP-CROWN** - arxiv:2208.05740
2. **BICCOS** - arxiv:2302.08730
3. **FSB Branching** - arxiv:2104.06718
4. **β-CROWN** - NeurIPS'21

### Should Read (HIGH)
5. **DeepPoly** - POPL'19
6. **α-CROWN** - ICLR'21
7. **CROWN** - NeurIPS'18

### Good to Read (MEDIUM)
8. **DeepZ** - NeurIPS'18
9. **GenBaB** - ICLR'24
10. **nnenum** - CAV'20

---

## Repos to Clone

```bash
# Critical references
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
git clone https://github.com/oval-group/oval-bab.git

# Important alternatives
git clone https://github.com/eth-sri/eran.git
git clone https://github.com/stanleybak/nnenum.git
git clone https://github.com/NeuralNetworkVerification/Marabou.git

# Additional
git clone https://github.com/verivital/nnv.git
git clone https://github.com/vas-group-imperial/VeriNet.git
```

---

## Summary: What We Need

To beat α,β-CROWN, we MUST implement:

1. **GCP-CROWN** - Their key advantage for tight bounds
2. **BICCOS** - Further bound tightening during B&B
3. **FSB Branching** - Their best branching heuristic
4. **GPU B&B** - Their speed advantage
5. **MIP Integration** - For complete verification on hard instances

Total estimated effort: **~100 commits**

Our advantages:
- **Rust performance** - 10-100x faster than Python
- **1.5B scale** - Larger models than any competitor
- **ALL methods** - Dynamic selection of optimal approach

---

## Automatic Method Selection

γ-CROWN should automatically select the best method based on:

```
Problem Characteristics → Optimal Method
─────────────────────────────────────────
Small network (<1K neurons)     → MIP (complete)
Medium network (<100K neurons)  → β-CROWN + GCP-CROWN
Large network (>100K neurons)   → α-CROWN + GPU B&B
Transformers                    → Specialized attention bounds
Abstract interp-friendly        → DeepPoly/Zonotope
High branching factor           → FSB heuristic
Tight timeout                   → IBP → CROWN → α-CROWN (escalate)
```

### Method Portfolio

| Problem Type | Primary | Fallback | Complete |
|--------------|---------|----------|----------|
| ACAS-Xu | β-CROWN + GCP | α-CROWN | MIP |
| MNIST | α-CROWN | DeepPoly | β-CROWN |
| CIFAR | GPU α-CROWN | IBP | - |
| Transformers | Specialized | α-CROWN | - |
| Large (>1M) | IBP + GPU | - | - |

### Auto-Selection Algorithm

```rust
fn select_method(network: &Network, property: &Property, timeout: Duration) -> Method {
    let size = network.num_neurons();
    let depth = network.num_layers();
    let has_conv = network.has_conv_layers();
    let is_transformer = network.is_transformer();

    match (size, depth, timeout.as_secs()) {
        // Small + long timeout → complete verification
        (..1_000, _, 60..) => Method::MIP,

        // Medium → GCP-CROWN with B&B
        (1_000..100_000, _, 10..) => Method::GcpCrown { with_bab: true },

        // Large → GPU-accelerated α-CROWN
        (100_000.., _, _) => Method::AlphaCrown { gpu: true },

        // Transformer → specialized
        _ if is_transformer => Method::TransformerCrown,

        // Tight timeout → escalate
        (_, _, ..10) => Method::Escalate(vec![
            Method::IBP,
            Method::Crown,
            Method::AlphaCrown { gpu: false },
        ]),

        // Default
        _ => Method::AlphaCrown { gpu: true },
    }
}
```

---

## WORKER DIRECTIVE

**Phase 1: GCP-CROWN** (CRITICAL, ~30 commits)
- Read arxiv:2208.05740
- Implement cutting plane generation
- Integrate with existing CROWN
- Test on ACAS-Xu

**Phase 2: FSB Branching** (~10 commits)
- Read arxiv:2104.06718
- Implement filtered smart branching
- Replace current BoundImpact heuristic

**Phase 3: GPU B&B** (~25 commits)
- Batch domain processing on GPU
- Parallel bound propagation
- Target: 10^6 domains/second

**Phase 4: BICCOS** (~20 commits)
- Cut generation during B&B
- Integrate with GCP-CROWN cuts
