# PLAN: Path to State-of-the-Art

**Status:** PLANNING
**Created:** 2026-01-02
**Goal:** Match or exceed α,β-CROWN on standard benchmarks

---

## Current Gap Analysis

### We Have
- ✅ CROWN, β-CROWN, SDP-CROWN
- ✅ Branch-and-bound with BoundImpact heuristic
- ✅ α-CROWN structure (AlphaState, optimization config)
- ✅ GPU acceleration (WGPU)
- ✅ VNNLIB format support
- ✅ Full transformer support (Whisper, Qwen)
- ✅ 1.5B parameter scale
- ✅ Soundness infrastructure (property tests, directed rounding)

### We're Missing (CRITICAL for SOTA)
- ❌ **GCP-CROWN** (General Cutting Planes) - arxiv:2208.05740
- ❌ **BICCOS** (Cutting planes during B&B)
- ❌ **MIP integration** (Gurobi/CPLEX for exact verification)
- ❌ **INVPROP** (Bound tightening with output constraints)
- ❌ **VNN-COMP benchmark validation**

### Uncertain Status
- ⚠️ α-CROWN optimization loop - need to verify it actually improves bounds
- ⚠️ GPU B&B - have WGPU but not integrated with B&B
- ⚠️ Branching heuristics - have BoundImpact but not FSB (Filtered Smart Branching)

---

## Priority 1: Validate Current Implementation

Before adding new features, we must know where we stand.

### Task 1.1: VNN-COMP Benchmark Suite
```bash
# Download VNN-COMP 2024 benchmarks
git clone https://github.com/ChristopherBrix/vnncomp2024_benchmarks

# Run γ-CROWN on same benchmarks
./target/release/gamma verify benchmarks/acasxu/... --vnnlib ...

# Compare with α,β-CROWN published results
```

**Metrics:**
- Verified rate (% of properties verified)
- Unknown rate (% inconclusive)
- Timeout rate
- Average verification time

### Task 1.2: α-CROWN Effectiveness Test
```rust
// Test that α optimization actually tightens bounds
#[test]
fn test_alpha_crown_improves_over_crown() {
    let crown_bounds = network.propagate_crown(&input)?;
    let alpha_bounds = network.propagate_alpha_crown(&input, &config)?;

    // α-CROWN bounds should be tighter
    assert!(alpha_bounds.max_width() < crown_bounds.max_width());
}
```

### Task 1.3: Head-to-Head Comparison
```bash
# Same model, same input, same epsilon
# γ-CROWN
./target/release/gamma verify model.onnx --epsilon 0.01

# α,β-CROWN (Python)
python -m complete_verifier.main model.onnx --epsilon 0.01

# Compare bounds and time
```

---

## Priority 2: GCP-CROWN Implementation

**This is the single most impactful missing technique.**

### Paper: arxiv:2208.05740
"GCP-CROWN: General Cutting Planes for Fast and Scalable Neural Network Verification"

### Core Idea
Standard CROWN uses triangle relaxation for ReLU. GCP-CROWN adds cutting planes that tighten this relaxation by exploiting:
1. Inter-neuron dependencies
2. Multiple linear constraints per neuron
3. Lagrangian dual optimization

### Implementation Outline
```rust
// crates/gamma-propagate/src/gcp_crown.rs

/// Cutting plane constraint: a^T x <= b
pub struct CuttingPlane {
    pub coefficients: Array1<f32>,
    pub bias: f32,
    pub layer_idx: usize,
}

/// GCP-CROWN verifier with cutting plane generation
pub struct GcpCrown {
    pub base_config: BetaCrownConfig,
    pub max_cuts_per_layer: usize,
    pub cut_generation_iterations: usize,
}

impl GcpCrown {
    /// Generate cutting planes from current bounds
    fn generate_cuts(&self, network: &Network, bounds: &[BoundedTensor]) -> Vec<CuttingPlane> {
        // 1. Identify "violated" relaxations (where triangle is loose)
        // 2. Generate linear constraints that cut off the loose region
        // 3. Return strongest cuts
    }

    /// Propagate bounds with cutting plane constraints
    fn propagate_with_cuts(
        &self,
        network: &Network,
        input: &BoundedTensor,
        cuts: &[CuttingPlane],
    ) -> Result<BoundedTensor> {
        // Modified backward pass that incorporates cuts
    }
}
```

### Effort Estimate
- Core GCP-CROWN: 15-20 commits
- Integration with B&B: 5-10 commits
- Testing and validation: 5 commits
- **Total: ~30 commits**

---

## Priority 3: GPU Branch-and-Bound

Currently B&B runs on CPU. For SOTA, need GPU parallelism.

### Implementation
```rust
// Batch multiple domains on GPU
pub struct GpuBranchAndBound {
    pub device: WgpuDevice,
    pub batch_size: usize,  // Process 1000+ domains in parallel
}

impl GpuBranchAndBound {
    fn verify_batch(&self, domains: &[BabDomain]) -> Vec<DomainResult> {
        // 1. Pack all domain bounds into GPU buffers
        // 2. Run batched bound propagation on GPU
        // 3. Unpack results
        // 4. Prune verified/infeasible domains
    }
}
```

### Effort Estimate
- GPU domain batching: 10 commits
- Batched bound propagation: 10 commits
- Integration: 5 commits
- **Total: ~25 commits**

---

## Priority 4: FSB Branching Heuristic

Filtered Smart Branching (FSB) is the state-of-the-art branching strategy.

### Current: BoundImpact
```rust
// Select neuron that most affects output bounds
BranchingHeuristic::BoundImpact
```

### Missing: FSB
```rust
// FSB considers:
// 1. Bound impact (like BoundImpact)
// 2. Estimated search tree size reduction
// 3. Filtering based on neuron stability
// 4. History of successful splits

pub struct FsbBranching {
    pub k: usize,  // Number of candidates to evaluate
    pub history: HashMap<NeuronId, f32>,  // Historical impact scores
}
```

### Effort Estimate: ~10 commits

---

## Milestone Definitions

### Milestone 1: Benchmark Parity (COMPLETE ✅)
- [x] Run on VNN-COMP 2021 ACAS-Xu benchmark
- [x] Achieve >80% verified rate → **Achieved 96.8%**
- [x] Verification time within 2x of α,β-CROWN → **5.91s avg (faster)**
- [x] 100% resolution rate (all instances verified or falsified)

### Milestone 2: GCP-CROWN Complete (MEDIUM-TERM)
- [ ] Implement GCP-CROWN cutting plane generation
- [ ] Integrate with branch-and-bound
- [ ] Achieve >90% verified rate on ACAS-Xu
- [ ] Bounds at least as tight as α,β-CROWN

### Milestone 3: Scale + Tight (LONG-TERM)
- [ ] Verify useful properties on Whisper-tiny with bounds < 1.0
- [ ] GPU B&B at 10^6 domains/second
- [ ] New VNN-COMP benchmark category: Transformers

### Milestone 4: True SOTA
- [ ] Win or match VNN-COMP on existing benchmarks
- [ ] Demonstrate capability α,β-CROWN cannot match (1B+ scale)
- [ ] Published, reproducible results

---

## Success Metrics

| Metric | Current | Target for SOTA | Status |
|--------|---------|-----------------|--------|
| ACAS-Xu resolution rate | **100%** | >95% | ✅ EXCEEDED |
| ACAS-Xu verified rate | **96.8%** | >95% | ✅ EXCEEDED |
| whisper-tiny bound width | ~2-4 | <1.0 | Pending |
| Qwen-0.6B bound width | ~444,000 | <1,000 | Pending |
| VNN-COMP ranking | N/A | Top 3 | Pending |
| Verification speed (ACAS-Xu) | **5.91s** | <10s/property | ✅ EXCEEDED |

---

## TARGETS (Beat α,β-CROWN)

| Metric | α,β-CROWN | Our Target | Gap to Close |
|--------|-----------|------------|--------------|
| ACAS-Xu verified rate | ~95% | **>95%** | Must beat |
| Time per property | ~10s | **<10s** | Must beat |
| Bound tightness | Tight | Tight | Must match |

**Both metrics must be beaten. Not one or the other. BOTH.**

---

## Immediate Next Steps

### Step 0: Get ACAS-Xu Benchmarks ✅ DONE

Benchmarks are now set up in `benchmarks/` with pytest integration:

```bash
cd benchmarks
./download_benchmarks.sh     # Download VNN-COMP 2021-2024
pip install -r requirements.txt
```

### Step 1: WORKER - Run Baseline ACAS-Xu Measurement (DO THIS NOW)

**WORKER DIRECTIVE: Run ACAS-Xu benchmark and report results.**

```bash
# Build gamma first
cargo build --release

# Run ACAS-Xu 2021 benchmark with pytest
cd benchmarks
pytest test_vnncomp.py::TestAcasXu2021::test_full_benchmark -v --timeout=60 --method=crown \
    --save-results=../reports/main/acasxu_baseline_crown.json

# Also run with beta-CROWN
pytest test_vnncomp.py::TestAcasXu2021::test_full_benchmark -v --timeout=60 --method=beta \
    --save-results=../reports/main/acasxu_baseline_beta.json
```

**Worker must report:**
1. Verified count / Total
2. Verified rate (target: >95%)
3. Average time per property (target: <10s)
4. Error breakdown (Unknown, Timeout, Error counts)

### Step 2: Baseline Measurement (Manual if pytest fails)

```bash
# If pytest fails, run manually on single instance:
./target/release/gamma verify \
    benchmarks/vnncomp2021/benchmarks/acasxu/ACASXU_run2a_1_1_batch_2000.onnx \
    --vnnlib benchmarks/vnncomp2021/benchmarks/acasxu/prop_1.vnnlib \
    --method crown \
    --timeout 10000 \
    --json

# Loop over all instances (bash)
for net in benchmarks/vnncomp2021/benchmarks/acasxu/*.onnx; do
    for prop in benchmarks/vnncomp2021/benchmarks/acasxu/*.vnnlib; do
        echo "=== $net × $prop ==="
        timeout 60 ./target/release/gamma verify "$net" --vnnlib "$prop" --method crown --json
    done
done > acasxu_results.txt 2>&1
```

### Step 3: Identify Bottleneck

If verified rate < 95%:
- Are bounds too loose? → Need GCP-CROWN
- Are we timing out? → Need GPU B&B
- Are we getting Unknown? → Need better branching

If time > 10s:
- Is it bound propagation? → Optimize hot paths
- Is it branching? → GPU parallelism
- Is it memory? → Already optimized

### Step 4: Fix and Re-measure

Iterate until:
- ✅ >95% verified
- ✅ <10s average

---

## Priority Order (Updated)

1. **ACAS-Xu baseline** (know where we stand)
2. **GCP-CROWN** (if bounds are the bottleneck)
3. **GPU B&B** (if time is the bottleneck)
4. **Both** (likely need both)

---

---

## References

- α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- GCP-CROWN paper: arxiv:2208.05740
- VNN-COMP: https://sites.google.com/view/vnn2024
- FSB branching: arxiv:2104.06718
- BICCOS: arxiv:2302.08730
