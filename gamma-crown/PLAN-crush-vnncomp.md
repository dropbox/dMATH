# PLAN: Crush All VNN-COMP Competitions (2021-2025)

**Goal**: Beat α,β-CROWN's scores on ALL past VNN-COMP benchmarks
**Strategy**: Port every technique that makes α,β-CROWN win, implement in Rust

---

## Current Status Assessment

### What γ-CROWN Has (Working)

| Technique | Status | Notes |
|-----------|--------|-------|
| IBP | ✅ | Sound interval propagation |
| CROWN | ✅ | Linear relaxation backward pass |
| α-CROWN | ⚠️ Partial | Falls back to CROWN for some layers |
| β-CROWN (B&B) | ✅ | Branch-and-bound verification |
| Input splitting | ✅ | For low-dim inputs (ACAS-Xu) |
| ReLU splitting | ✅ | Neuron branching |
| FSB branching | ✅ | Filtered smart branching heuristic |
| PGD attack | ✅ | Counterexample finding |
| DAG support | ✅ | GraphNetwork for ResNets |
| VNN-LIB parsing | ✅ | Standard property format |

### What α,β-CROWN Has That We Need

| Technique | Priority | Effort | Impact |
|-----------|----------|--------|--------|
| **GCP-CROWN (cutting planes)** | P0 | 15-20 commits | Tighter bounds on hard instances |
| **BatchNorm backward CROWN** | ✅ DONE | - | Implemented in commit #361 |
| **Dynamic shape support** | ✅ WORKS | - | ViT models load OK (tested 2026-01-07) |
| **Conv backward CROWN** | P1 | 8-12 commits | Faster CNN verification |
| **Multi-GPU parallelism** | P1 | 10-15 commits | Scale to larger models |
| **BaB-Attack** | P2 | 5-8 commits | Better counterexample finding |
| **Auto branching selection** | P2 | 3-5 commits | Per-instance optimization |
| **CUDA backend** | P2 | 20-30 commits | NVIDIA GPU acceleration |

---

## VNN-COMP Benchmark Coverage

### VNN-COMP 2021 (Easiest - Start Here)

**Updated 2026-01-05 (Commit #443)**

| Benchmark | Instances | γ-CROWN Status | Gap |
|-----------|-----------|----------------|-----|
| **acasxu** | 186 | ✅ 100% | None |
| **cifar10_resnet** | 72 | ⚠️ **2.8%** (correct semantics) | Need multi-objective BaB |
| **mnistfc** | 90 | ✅ ~80% (with --proactive-cuts) | 256x6 network needs GCP-CROWN |
| **oval21** | 30 | ✅ 100% | All instances verify |
| **marabou-cifar10** | 72 | ✅ **100%** (all sizes) | Solved via maxdiff reduction |
| **eran** | ~72 | ✅ ~80% | Some deep instances timeout |
| **nn4sys** | 75 | ✅ ~80% (improved via early-exit) | Large models (42M params); some hard thresholds |
| **verivital** | 60 | ✅ 100% | Pool layers verified |

**Overall: ~75% verified (cifar10_resnet lowered from incorrect 45.8% to correct 2.8%)**

**Key Findings (Commit #443)**:
- **cifar10_resnet correct rate**: Ran full benchmark with correct disjunctive semantics
  - Rate: **2.8% (2/72)** - only resnet_2b props 39, 41 verify
  - Previous 45.8% was incorrect due to early exit on ANY constraint proved
  - Correct behavior: SAFE requires ALL 9 constraints proved violated
  - Timeout improvement: 3x more domains explored (650 vs 200) but no rate change
  - Root cause: bounds too loose to verify all 9 constraints per property
  - Fix needed: Multi-objective BaB to verify all constraints simultaneously
  - See: `reports/main/cifar10_resnet_disjunction_semantics_2026-01-05.md`

**Key Findings (Commit #441-442)**:
- **marabou-cifar10 100% solved**: Implemented maxdiff reduction for conjunctive targeted-attack constraints
  - Before (Commit #440): 33% verified (small only, medium/large 0%)
  - After (Commit #442): **100% verified** (72/72, all sizes)
  - Technique: Reduce conjunctive constraints `Y_t >= Y_j` for multiple `j` to single objective `maxdiff = max_j(Y_j - Y_t)`
  - All instances verify in 1 domain (no BaB needed), times: small ~4s, medium ~9s, large ~24s
- **cifar10_resnet semantics fix**: Fixed disjunctive property handling for GraphNetwork
  - **BUG FIXED**: Properties with `(or ...)` output constraints were incorrectly verified when ANY constraint was proved
  - **CORRECT**: Disjunctive properties (OR) require ALL constraints to be proved violated for SAFE
  - Previous 45.8% rate was INCORRECT; need to re-benchmark with correct semantics
  - The VnnLibSpec now tracks `is_disjunction` flag to distinguish property types

**Key Findings (Commit #440)**:
- **marabou-cifar10 breakthrough**: Enabled ReLU splitting for Conv networks (was wrongly forced to input splitting)
  - Before: 0% verified (input splitting ineffective for 3072-dim inputs)
  - After: 33% verified (24/72 - all small network instances)
  - Fix: Removed forced `has_conv` → input splitting; Conv CROWN backward was already implemented

**Key Findings (Commit #439)**:
- **Assessment script fix**: `assess_vnncomp2021.py` now uses `--proactive-cuts` for mnistfc when `--tuned` flag is passed
  - Without fix: MNISTFC reported as 40% (misleading)
  - With fix: MNISTFC correctly reports ~80%
- **Full assessment completed**: See `reports/main/vnncomp2021_full_assessment_2026-01-05.md`
- **Verified MNISTFC results**: All 90 instances tested with `--proactive-cuts`
  - 256x2: 30/30 (100%)
  - 256x4: 30/30 (100%)
  - 256x6: 12/30 (40%)
  - Total: 72/90 (80%)

**Key Findings (Commit #438)**:
- **Adaptive α-CROWN skipping**: Automatically skip α-CROWN for deep networks
  - Networks with >8 ReLU layers skip α-CROWN optimization (new default)
  - Pilot iteration check: verify α-CROWN helps before continuing optimization
  - ResNet-4b (10 ReLU nodes) now automatically skips α-CROWN
  - ResNet-2b (6 ReLU nodes) still runs α-CROWN (below threshold)
  - CLI options: `--no-adaptive-alpha-skip` (force α-CROWN), `--alpha-skip-depth N` (set threshold)
- **Why this helps**: For deep networks with loose bounds, α-CROWN optimization costs time
  but provides no improvement. Skipping allows more BaB exploration.

**Key Findings (Commit #437)**:
- **Sparse α-CROWN optimization**: Focus optimization on top 30% most influential alphas
  - New defaults: 3 iterations, 1 SPSA sample, sparse_ratio=0.3
  - ResNet-2b: 45.8% verified (unchanged), faster domain exploration
  - ResNet-4b: Still 0% (fundamentally bounds too loose)
- **α-CROWN tradeoff analysis for ResNet-4b**:
  - IBP only: 381 domains/60s, 0% verified - best for domain exploration
  - With α-CROWN: 234 domains/60s, 0% verified - slower but same result
  - For ResNet-4b, `--no-alpha` is recommended (more BaB exploration)
- **Recommendation**: Use `--no-alpha` for deep networks (>25 layers) where α-CROWN doesn't help (now automatic)

**Key Findings (Commit #436)**:
- `fix_interm_bounds=true` optimization: matches α,β-CROWN default behavior
  - ResNet-4b init: **82s → 0.1s** (IBP only) or **26.5s** (1 α-CROWN iteration)
  - ResNet-4b can now run BaB (173 domains avg) instead of timing out
  - ResNet-2b unchanged: 45.8% verified (22/48)
- ResNet-4b still 0% verified because:
  - IBP bounds are ~6x looser than α-CROWN (not tight enough)
  - α-CROWN optimization is expensive (27s for 1 iteration)
  - Need: GPU-accelerated CROWN or better initial bounds

**Key Findings (Commit #435)**:
- `cifar10_resnet` resnet_2b improved **0%→45.8%** (22/48 verified)
  - α-CROWN bounds now working for DAG models
  - 8 properties verify instantly (1 domain)
- `mnistfc` improved 40%→81% by enabling `--proactive-cuts` flag
  - 256x2: 100%, 256x4: 100%, 256x6: 40%
  - Proactive cuts solve 256x4 instantly (was timeout)
- `nn4sys` improved 10%→80% via early termination optimization (skip α-CROWN when CROWN-IBP verifies)
- `marabou-cifar10` has fundamental bound tightness issue (need proactive GCP-CROWN cuts)

### VNN-COMP 2023 (Medium)

**Updated 2026-01-07 (Commit #545)**

| Benchmark | Key Blocker | Status |
|-----------|-------------|--------|
| **acasxu** | None | ✅ Same as 2021 |
| **vit** | ~~Dynamic shapes~~ | ✅ **Models load OK** (performance issue) |
| **vggnet16** | Conv backward + BatchNorm | Test needed |
| **collins_rul_cnn** | Test needed | Test needed |
| **traffic_signs** | BatchNorm | Test needed |
| **yolo** | Complex architecture | Test needed |

**Key Finding (2026-01-07)**: ViT models (pgd_2_3_16.onnx) load and run without dynamic shapes error!
- Test: pgd_2_3_16_8835.vnnlib → explored 63 domains, timeout after 41s
- Issue is now **performance** not **architecture support**

### VNN-COMP 2024/2025 (Hardest)

**Updated 2026-01-07 (Commit #545)**

| Benchmark | Key Blocker | Status |
|-----------|-------------|--------|
| **cifar100** | ~~BatchNorm~~ Performance | Model loads but runs very slow (11+ min without timeout) |
| **tinyimagenet** | Large models | Test needed |
| **safenlp** | NLP architecture | Test needed |
| **cora** | Graph neural networks | Test needed |
| **soundnessbench** (2025) | Memory usage | Uses ~10GB RAM, needs optimization |

**Key Findings (2026-01-07)**:
- CIFAR100: Model loads (BatchNorm works!) but performance is very slow
- Soundnessbench: Very high memory usage (~10GB), possible memory leak or large model

---

## Execution Plan

### Phase 1: Win VNN-COMP 2021 (Estimated: 30-40 commits)

**Week 1-2: Assessment**
1. [ ] Run full 2021 benchmark suite, record per-benchmark scores
2. [ ] Compare to published α,β-CROWN 2021 scores
3. [ ] Identify exact gaps per benchmark

**Week 3-4: Quick Wins**
1. [ ] Fix MNIST FC (tighter bounds or more splits)
2. [ ] Test OVAL21, marabou-cifar10, eran, nn4sys
3. [ ] Fix any obvious bugs found

**Week 5-6: Hard Instances**
1. [ ] Implement GCP-CROWN cutting planes (main bound tightener)
2. [ ] Profile and optimize hot paths
3. [ ] Beat α,β-CROWN 2021 total score

### Phase 2: Win VNN-COMP 2023 (Estimated: 40-50 commits)

1. [ ] BatchNorm backward CROWN
2. [ ] Dynamic shape support (ViT)
3. [ ] Conv backward CROWN
4. [ ] Test and debug all 2023 benchmarks

### Phase 3: Win VNN-COMP 2024/2025 (Estimated: 50-60 commits)

1. [ ] Multi-GPU support
2. [ ] CUDA backend (optional but helps)
3. [ ] Any remaining architecture support
4. [ ] Optimize for competition rules

---

## α,β-CROWN Techniques to Port

### From `alpha-beta-CROWN/complete_verifier/`

| File | Technique | Priority |
|------|-----------|----------|
| `bab.py` | Main BaB loop | ✅ Have |
| `bab_attack.py` | BaB-Attack hybrid | P2 |
| `cut.py` | GCP-CROWN cuts | **P0** |
| `cut_utils.py` | Cut generation helpers | **P0** |
| `branching_domains.py` | Domain management | ✅ Have |
| `branching_heuristics.py` | Branching strategies | ✅ Have |
| `input_split/` | Input space splitting | ✅ Have |
| `batch_branch_and_bound.py` | Batched BaB | P1 |

### From `auto_LiRPA/`

| File | Technique | Priority |
|------|-----------|----------|
| `backward_bound.py` | CROWN backward | ✅ Have |
| `optimized_bounds.py` | α-CROWN optimization | ⚠️ Partial |
| `operators/linear.py` | Linear layer bounds | ✅ Have |
| `operators/relu.py` | ReLU bounds | ✅ Have |
| `operators/convolution.py` | Conv backward CROWN | **P1** |
| `operators/normalization.py` | BatchNorm backward | **P0** |
| `operators/softmax.py` | Softmax bounds | ✅ Have |

---

## Key Implementation Tasks

### 1. GCP-CROWN Cutting Planes (P0)

**What it does**: Generates linear constraints (cuts) that tighten bounds

**Reference**: `alpha-beta-CROWN/complete_verifier/cut.py`

**Algorithm**:
1. Identify "unstable" ReLUs (could be active or inactive)
2. Generate cuts of form: Σ(α_i * x_i) <= β
3. Add cuts to LP relaxation
4. Re-solve with tighter bounds

**Implementation steps**:
1. [ ] Port `CutGenerator` class
2. [ ] Implement cut scoring and selection
3. [ ] Integrate with BaB loop
4. [ ] Test on MNIST FC (should improve 43% → 90%+)

### 2. BatchNorm Backward CROWN - ✅ DONE

**Status**: IMPLEMENTED (as of commit #361)

BatchNorm backward CROWN is fully implemented in `layers/mod.rs`:
- `BatchNormLayer::propagate_ibp()` - IBP propagation
- `BatchNormLayer::propagate_linear()` - CROWN backward propagation
- Handles 1D, 2D, 3D, 4D inputs with channel dimension detection
- Tested on CIFAR-100 models (loads and runs, timeout due to model size)

**NOT a blocker** - CIFAR-100 models load and verify, they just timeout due to:
1. Large model size (3072 inputs, 100 outputs)
2. Many constraints (99 simultaneous)
3. Bound looseness requiring more BaB splits

### 3. Dynamic Shape Support (P0)

**What it does**: Handles models with dynamic batch dimensions

**Reference**: ViT benchmark models

**Problem**: ViT models use ops like:
- `Shape` - extract tensor dimensions
- `Gather` - index into shape tensor
- `ConstantOfShape` - create tensor with computed shape

**Solution options**:
1. Pre-process ONNX to fix batch=1 (quick hack)
2. Support shape inference in graph (proper fix)

**Implementation steps**:
1. [ ] Add `Shape` op support (extract dims as constants for batch=1)
2. [ ] Add `Gather` op for shape indexing
3. [ ] Add `ConstantOfShape` with static shape inference
4. [ ] Test ViT loading and verification

### 4. Conv Backward CROWN (P1)

**What it does**: Propagates linear bounds backward through Conv layers

**Reference**: `auto_LiRPA/operators/convolution.py`, `BoundConv`

**Algorithm**:
- Conv backward = transposed convolution
- ∂L/∂x = ConvTranspose(∂L/∂y, weights)

**Implementation steps**:
1. [ ] Implement `Conv2dLayer::propagate_linear_with_bounds()`
2. [ ] Handle padding, stride, dilation
3. [ ] Optimize with im2col or FFT
4. [ ] Test on OVAL21

---

## Success Metrics

### VNN-COMP 2021 Target
- Total score: Beat α,β-CROWN's 2021 score
- ACAS-Xu: 100% ✅ (already achieved)
- MNIST FC: >90% (currently 43%)
- CIFAR-10 ResNet: >95%
- Other benchmarks: >80%

### VNN-COMP 2023 Target
- ViT: >80% (currently 0%)
- VGGNet16: >80% (currently untested)
- All other benchmarks: competitive

### VNN-COMP 2024/2025 Target
- CIFAR-100: >80% (currently 0%)
- All benchmarks: competitive with α,β-CROWN

---

## Worker Directives

### Immediate (Next Worker)
1. Run full VNN-COMP 2021 assessment
2. Create `reports/main/vnncomp2021_full_assessment.md`
3. Record per-benchmark scores

### Short-term (Next 10 iterations)
1. Implement BatchNorm backward CROWN
2. Test CIFAR-100
3. Start GCP-CROWN implementation

### Medium-term (Next 30 iterations)
1. Complete GCP-CROWN
2. Dynamic shape support for ViT
3. Win VNN-COMP 2021

---

## Reference Commands

```bash
# Run single benchmark
./target/release/gamma beta-crown <model.onnx> --property <prop.vnnlib> --timeout 60

# Full ACAS-Xu suite
python scripts/run_acasxu_benchmark.py --timeout 30 --all

# Test specific benchmark
pytest benchmarks/test_vnncomp.py -v -k "acasxu" --timeout=30

# Compare to α,β-CROWN
# (need to run α,β-CROWN separately and compare)
```

---

## Timeline

| Phase | Target | Commits |
|-------|--------|---------|
| Phase 1: Win 2021 | 2 weeks | 30-40 |
| Phase 2: Win 2023 | 4 weeks | 40-50 |
| Phase 3: Win 2024/2025 | 4 weeks | 50-60 |
| **Total** | **10 weeks** | **120-150** |

With workers running continuously at ~12 commits/day, this is achievable in ~2 weeks real time.
