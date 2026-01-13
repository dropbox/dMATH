# γ-CROWN BATTLE PLAN

## Mission: 1st Place in EVERY Category

> **γ-CROWN will be the undisputed champion of neural network verification.**

---

## The Goal

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   γ-CROWN: 1st PLACE in EVERY CATEGORY                         │
│                                                                 │
│   • FASTEST verification                                        │
│   • LOWEST memory usage                                         │
│   • HIGHEST verified rate                                       │
│   • LARGEST model scale                                         │
│                                                                 │
│   Not "competitive". Not "close". DOMINATE.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Three Pillars

| Metric | Target | Why We Win |
|--------|--------|------------|
| **Speed** | Fastest | Rust > Python (10-100x) |
| **Memory** | Lowest | Zero-copy, arena allocation |
| **Accuracy** | Highest | ALL methods + auto-select |

---

## Strategy: ALL Methods + Best Implementation

### Why We Will Win

1. **ALL Methods** - Every technique from every competitor
2. **Best Implementation** - Rust > Python (10-100x faster)
3. **Largest Scale** - 1.5B+ parameters (competitors max ~10M)
4. **Auto Selection** - Dynamically choose optimal method per problem

### The Formula

```
γ-CROWN = (α,β-CROWN methods)
        + (ERAN methods)
        + (OVAL methods)
        + (nnenum methods)
        + (Rust speed)
        + (GPU acceleration)
        + (Auto selection)
```

### The Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. We are ALWAYS faster (Rust vs Python)                  │
│                                                             │
│  2. If anyone beats us on correctness:                     │
│     → Study what method they used                          │
│     → Implement that method                                │
│     → Now we have their accuracy + our speed               │
│     → WIN                                                  │
│                                                             │
│  3. Repeat until #1 on everything                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Speed is guaranteed. Accuracy gaps are closed by adding methods.**

---

## Benchmark Domination Matrix

### VNN-COMP 2025 Benchmarks

| Benchmark | Beat On | Speed Target | Memory Target | Method | Status |
|-----------|---------|--------------|---------------|--------|--------|
| acasxu | α,β-CROWN | 2x faster | 2x less | β-CROWN + GCP | 🔴 |
| vit | α,β-CROWN | 3x faster | 2x less | Transformer CROWN | 🔴 |
| nn4sys | varies | Fastest | Lowest | Auto-select | 🔴 |
| cifar100 | α,β-CROWN | 2x faster | 2x less | GPU α-CROWN | 🔴 |
| tinyimagenet | α,β-CROWN | 2x faster | 2x less | GPU α-CROWN | 🔴 |
| cora | ? | Fastest | Lowest | Auto-select | 🔴 |
| safenlp | ? | Fastest | Lowest | Auto-select | 🔴 |
| soundnessbench | ? | Fastest | Lowest | All methods | 🔴 |
| malbeware | ? | Fastest | Lowest | Auto-select | 🔴 |
| vggnet16 | α,β-CROWN | 2x faster | 2x less | GPU + GCP | 🔴 |
| yolo | α,β-CROWN | 2x faster | 2x less | Specialized | 🔴 |
| traffic_signs | α,β-CROWN | 2x faster | 2x less | GPU α-CROWN | 🔴 |

**Target: 🟢 1st place on ALL rows in ALL metrics**

### Victory Conditions Per Benchmark

```
For EACH benchmark, we must be:
✓ #1 in verified rate
✓ #1 in speed (fastest)
✓ #1 in memory (lowest)
✓ #1 in scale (largest model handled)
```

---

## Implementation Phases

### Phase 1: Foundation (Current)
- [x] CROWN
- [x] α-CROWN
- [x] β-CROWN
- [x] GPU acceleration (wgpu)
- [x] Transformer support
- [ ] **GCP-CROWN** ← CURRENT PRIORITY

### Phase 2: Cutting Planes
- [ ] GCP-CROWN complete
- [ ] BICCOS
- [ ] MIP integration

### Phase 3: Speed
- [ ] GPU Branch-and-Bound
- [ ] FSB branching
- [ ] Parallel domain processing

### Phase 4: Alternative Methods
- [ ] DeepPoly
- [ ] Zonotope analysis
- [ ] Lagrangian decomposition

### Phase 5: Auto-Selection
- [ ] Problem classifier
- [ ] Method portfolio
- [ ] Dynamic escalation

### Phase 6: Domination
- [ ] Beat α,β-CROWN on ACAS-Xu
- [ ] Beat ERAN on abstract-friendly benchmarks
- [ ] Beat nnenum on zonotope-friendly problems
- [ ] Beat Marabou on SMT-friendly problems
- [ ] **WIN VNN-COMP 2025**

---

## Per-Benchmark Attack Plan

### ACAS-Xu (45 networks × 10 properties)

**Current champion:** α,β-CROWN (~95% verified, ~10s/property)

**Our attack:**
1. β-CROWN baseline
2. Add GCP-CROWN cuts → tighter bounds
3. Add BICCOS → even tighter
4. FSB branching → faster search
5. GPU B&B → massive parallelism

**Target:** >98% verified, <5s/property

### Vision Transformers (ViT)

**Current champion:** α,β-CROWN

**Our attack:**
1. Specialized attention bounds (already have)
2. GPU-accelerated propagation
3. Layer-wise verification for large models

**Target:** Verify properties α,β-CROWN cannot

### CIFAR/ImageNet

**Current champion:** α,β-CROWN

**Our attack:**
1. GPU α-CROWN for speed
2. GCP-CROWN for precision
3. DeepPoly as alternative

**Target:** Faster than Python competitors

### Large Models (>1M parameters)

**Current champion:** None can handle well

**Our attack:**
1. Already verified 1.5B parameters
2. GPU memory optimization
3. Compositional verification

**Target:** Verify what others cannot

---

## Metrics to Track

### Per Benchmark
- Verified count
- Verified rate (%)
- Average time (s)
- Timeout count
- Memory usage (MB)

### Comparison vs Competitors
- α,β-CROWN: Must beat on ALL
- ERAN: Must beat on abstract-friendly
- nnenum: Must beat on ACAS-Xu
- Marabou: Must beat on SMT-friendly

### Overall
- VNN-COMP score calculation
- Total verified across all benchmarks
- Total time across all benchmarks

---

## Speed Advantages (Why Rust Wins)

| Optimization | Rust γ-CROWN | Python α,β-CROWN | Speedup |
|--------------|--------------|------------------|---------|
| No GIL | Full parallelism | Limited | 4-8x |
| Zero-copy tensors | Yes | No (numpy copies) | 2-3x |
| SIMD vectorization | Native | Via numpy | 1.5-2x |
| Cache locality | Controlled | Unpredictable | 1.5-2x |
| No interpreter | Compiled | Interpreted | 5-10x |
| GPU dispatch | Direct wgpu | PyTorch overhead | 1.2-1.5x |

**Combined:** 10-100x faster on same algorithm

---

## Memory Advantages

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| Arena allocation | Per-verification arena | No fragmentation |
| In-place operations | Mutate bounds directly | 50% less allocation |
| Streaming propagation | Process layer-by-layer | Constant memory for depth |
| Sparse representations | Only store non-zero | 10x less for sparse |
| GPU unified memory | MLX backend | No CPU↔GPU copies |

**Target:** Use <50% memory of α,β-CROWN for same problem

---

## Success Criteria

### Minimum (Competitive)
- [ ] >95% on ACAS-Xu
- [ ] <10s average on ACAS-Xu
- [ ] Top 3 in VNN-COMP

### Target (Winning)
- [ ] >98% on ACAS-Xu
- [ ] <5s average on ACAS-Xu
- [ ] #1 overall in VNN-COMP

### Ultimate (Domination)
- [ ] **1st place in EVERY category**
- [ ] **Fastest** on every benchmark
- [ ] **Lowest memory** on every benchmark
- [ ] **Highest verified rate** on every benchmark
- [ ] Methods others don't have
- [ ] Scale others can't reach (1B+)
- [ ] **Undisputed champion**

---

## Gap Analysis Process

When any competitor outperforms us on any benchmark:

### Step 1: Identify the Gap
```bash
# Run benchmark, identify where we lose
pytest test_vnncomp.py::TestXXX -v --save-results=gap_analysis.json

# Find instances where competitor wins
# Look for: status=unknown when they got verified
```

### Step 2: Study Their Method
```
- Which tool won this instance?
- What method did they use?
- Is it in our implementation?
- If not, where is it documented? (paper, code)
```

### Step 3: Implement Their Method
```
- Clone their repo to references/
- Study their implementation
- Port to Rust (faster)
- Add to gamma-crown
```

### Step 4: Verify We Now Win
```bash
# Re-run benchmark
pytest test_vnncomp.py::TestXXX -v

# Confirm we now match or beat
# If still losing, goto Step 1
```

### Gap Tracking Table

| Benchmark | Instance | Winner | Their Method | Our Gap | Status |
|-----------|----------|--------|--------------|---------|--------|
| (fill as we find gaps) | | | | | |

---

## Timeline

### Now
- GCP-CROWN implementation

### +30 commits
- GCP-CROWN complete
- ACAS-Xu improvement measured

### +50 commits
- FSB branching
- GPU B&B started

### +80 commits
- BICCOS
- MIP integration

### +100 commits
- All core methods implemented
- Auto-selection working

### +120 commits
- DeepPoly alternative
- Full method portfolio

### VNN-COMP 2025
- **WIN**

---

## WORKER: Execute This Plan

Start with GCP-CROWN. Every commit should move us toward domination.

**Measure everything. Report results. Beat everyone.**

```
"The only acceptable outcome is total victory."
```
