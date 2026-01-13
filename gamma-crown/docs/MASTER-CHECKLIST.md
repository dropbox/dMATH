# γ-CROWN Master Implementation Checklist

**Status:** ACTIVE TRACKING DOCUMENT
**Created:** 2026-01-02
**Purpose:** Track ALL methods, layers, and optimizations. Nothing falls through cracks.

---

## 1. VERIFICATION METHODS

### Core Methods (from DESIGN.md)

| Method | Status | Notes |
|--------|--------|-------|
| IBP (Interval Bound Propagation) | ✅ DONE | `propagate_ibp()` |
| CROWN (basic) | ✅ DONE | `propagate_crown()` |
| CROWN batched | ✅ DONE | `propagate_crown_batched()` |
| α-CROWN (optimizable slopes) | ⚠️ PARTIAL | Integrated into CROWN, needs explicit control |
| β-CROWN (split constraints) | ✅ DONE | `beta_crown.rs` module |
| SDP-CROWN | ✅ DONE | `sdp_crown.rs`, Linear/ReLU only |
| Branch-and-Bound | ✅ DONE | `branch_and_bound.rs` |
| Domain Clipping | ✅ DONE | `domain_clip.rs` |
| Zonotope | ⚠️ PARTIAL | Some layers support zonotope bounds |
| Streaming/Checkpointing | ✅ DONE | `streaming.rs` |

### Methods NOT YET IMPLEMENTED

| Method | Paper | Priority | Notes |
|--------|-------|----------|-------|
| GCP-CROWN | arxiv:2208.05740 | HIGH | General cutting planes |
| MIP-based verification | VNN-COMP | MEDIUM | Mixed integer programming |
| PRIMA | arxiv:2103.03638 | MEDIUM | Multi-neuron constraints |
| DeepPoly | arxiv:1804.10829 | LOW | Abstract interpretation |
| k-ReLU | arxiv:1811.01715 | LOW | k-activation relaxation |

---

## 2. LAYER TYPES (33 total)

### Linear Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| LinearLayer | ✅ | ✅ | ✅ | ✅ | Full support |
| MatMulLayer | ✅ | ⚠️ | ❌ | ⚠️ | CROWN partial |
| Conv1dLayer | ✅ | ✅ | ❌ | ❌ | Full CROWN support via transposed conv |
| Conv2dLayer | ✅ | ✅ | ❌ | ❌ | Full CROWN support via transposed conv |

### Activation Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| ReLULayer | ✅ | ✅ | ✅ | ✅ | Full support |
| GELULayer | ✅ | ✅ | ⚠️ | ✅ | Adaptive relaxation implemented |
| TanhLayer | ✅ | ✅ | ❌ | ❌ | |
| SigmoidLayer | ✅ | ✅ | ❌ | ❌ | |
| SoftplusLayer | ✅ | ⚠️ | ❌ | ❌ | |
| SinLayer | ✅ | ⚠️ | ❌ | ❌ | |
| CosLayer | ✅ | ⚠️ | ❌ | ❌ | |
| SqrtLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |
| PowConstantLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |

### Transformer Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| SoftmaxLayer | ✅ | ✅ | ❌ | ⚠️ | Sampling-based bounds |
| CausalSoftmaxLayer | ✅ | ✅ | ❌ | ⚠️ | Masked softmax |
| LayerNormLayer | ✅ | ✅ | ❌ | ⚠️ | Sampling-based bounds |
| BatchNormLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |

### Arithmetic Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| AddLayer | ✅ | ✅ | ❌ | ✅ | Binary add |
| SubLayer | ✅ | ✅ | ❌ | ✅ | Binary sub |
| MulBinaryLayer | ✅ | ⚠️ | ❌ | ⚠️ | Non-linear |
| DivLayer | ✅ | ⚠️ | ❌ | ❌ | Non-linear |
| AddConstantLayer | ✅ | ✅ | ✅ | ✅ | |
| SubConstantLayer | ✅ | ✅ | ✅ | ✅ | |
| MulConstantLayer | ✅ | ✅ | ✅ | ✅ | |
| DivConstantLayer | ✅ | ✅ | ✅ | ✅ | |

### Shape Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| TransposeLayer | ✅ | ✅ | ✅ | ✅ | |
| ReshapeLayer | ✅ | ✅ | ✅ | ✅ | |
| FlattenLayer | ✅ | ✅ | ✅ | ✅ | |
| TileLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |

### Pooling Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| AveragePoolLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |
| MaxPool2dLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |

### Reduction Layers

| Layer | IBP | CROWN | β-CROWN | Zonotope | Notes |
|-------|-----|-------|---------|----------|-------|
| ReduceMeanLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |
| ReduceSumLayer | ✅ | ⚠️ | ❌ | ❌ | Falls back to IBP |

---

## 3. OPTIMIZATIONS CHECKLIST

### Processing (12 items)

| ID | Optimization | Status | Commit |
|----|-------------|--------|--------|
| P1 | Inline hot paths | ✅ DONE | #264 |
| P2 | Parallel ndarray ops | ✅ DONE | #265-267 |
| P3 | Loop fusion | ✅ DONE | #279 (faer element-wise ops, 6-16% IBP speedup) |
| P4 | Cache-friendly access | ⚠️ N/A | #279 (attempted zero-copy views, 3-5x regression - reverted) |
| P5 | Batch size auto-tuning | ❌ PENDING | |
| P6 | More SIMD operations | ⚠️ PARTIAL | #256 (interval_mul, pos_neg_split) |
| P7 | Lazy computation | ❌ PENDING | |
| P8 | Prefetch hints | ❌ PENDING | |
| P9 | Branch prediction hints | ❌ PENDING | |
| P10 | Compile-time computation | ❌ PENDING | |
| P11 | Work stealing tuning | ❌ PENDING | |
| P12 | Async I/O for model loading | ❌ PENDING | |

### Memory (9 items)

| ID | Optimization | Status | Commit |
|----|-------------|--------|--------|
| M1 | Reduce clone calls | ✅ DONE | #268-272 |
| M2 | Pre-allocate vectors | ✅ DONE | #264 |
| M3 | Arena allocator | ❌ PENDING | |
| M4 | String interning | ❌ PENDING | |
| M5 | Lazy error messages | ❌ PENDING | |
| M6 | Reuse intermediate buffers | ⚠️ PARTIAL | Cow<LinearBounds> done |
| M7 | Memory-mapped model files | ❌ PENDING | |
| M8 | Compact tensor representation | ❌ PENDING | |
| M9 | Arc for layer parameters | ✅ DONE | #272 |

### Debug/Error/UX (11 items)

| ID | Optimization | Status | Commit |
|----|-------------|--------|--------|
| D1 | Replace unwrap/expect | ⚠️ PARTIAL | #269 |
| D2 | Structured logging (tracing) | ❌ PENDING | |
| D3 | Error codes | ❌ PENDING | |
| D4 | Progress bar with ETA | ❌ PENDING | |
| D5 | Memory usage reporting | ❌ PENDING | |
| D6 | Verification telemetry | ❌ PENDING | |
| D7 | Debug visualization | ❌ PENDING | |
| D8 | Verbose mode levels | ❌ PENDING | |
| D9 | Config file support | ❌ PENDING | |
| D10 | Shell completions | ❌ PENDING | |
| D11 | Machine-readable output | ❌ PENDING | |

---

## 4. SOUNDNESS ITEMS

### Phase 0 (Quick Wins) - ✅ COMPLETE

| ID | Item | Status | Commit |
|----|------|--------|--------|
| D1 | Version pinning | ✅ DONE | #274 |
| A5 | NaN/Inf checks | ✅ DONE | #274 |
| C3 | Edge case tests | ✅ DONE | #274 |
| D4 | Verify no fast-math | ✅ DONE | #274 |
| A4 | Denormal documentation | ✅ DONE | #274 |

### Phase 1 (Practical Soundness) - ✅ COMPLETE

| ID | Item | Status | Commit |
|----|------|--------|--------|
| A1 | Directed rounding | ✅ DONE | #275 |
| B3 | Activation validation vs PyTorch | ✅ DONE | #277 |
| C1 | Property-based testing | ✅ DONE | #276 |
| C2 | Auto-LiRPA cross-validation | ✅ DONE | #277 |
| E2 | Formal perturbation spec | ✅ DONE | #278 |

---

## 5. LITERATURE REVIEW TODO

### Papers to Review for Missing Techniques

| Paper | Topic | Priority | Implemented? |
|-------|-------|----------|--------------|
| arxiv:2103.06624 | α,β-CROWN | HIGH | ⚠️ Partial |
| arxiv:2506.06665 | SDP-CROWN | HIGH | ✅ Done |
| arxiv:2208.05740 | GCP-CROWN (cutting planes) | HIGH | ❌ No |
| arxiv:2103.03638 | PRIMA (multi-neuron) | MEDIUM | ❌ No |
| arxiv:2512.11087 | Domain clipping | DONE | ✅ Done |
| arxiv:1804.10829 | DeepPoly | LOW | ❌ No |
| VNN-COMP 2024 | Competition techniques | HIGH | ❌ Review needed |

### Techniques from VNN-COMP Winners to Investigate

- [ ] Input splitting strategies (BABSR, FSB)
- [ ] GPU-accelerated branch-and-bound
- [ ] Bound tightening via MIP
- [ ] Multi-tree search
- [ ] Learned branching heuristics

---

## 6. PRIORITY ORDER

### Immediate (Worker should do next)
1. ~~Phase 0 soundness (5 items)~~ ✅ DONE (#274)
2. ~~P3-P5 optimizations~~ P3 DONE (#279), P4 N/A, P5 pending
3. ~~CROWN support for Conv layers~~ ✅ DONE (#281) - Conv1d/Conv2d now use transposed conv

### Short Term
4. D2: Structured logging
5. GCP-CROWN implementation
6. GPU branch-and-bound

### Medium Term
7. ~~Phase 1 soundness~~ ✅ DONE (#275-278)
8. Full β-CROWN for all activations
9. PRIMA multi-neuron constraints

### Long Term
10. Phase 2: Lean 5 proofs (formal specification)
11. Phase 3: tRust compilation
12. 70B model support

---

## 7. PROGRESS SUMMARY

| Category | Done | Partial | Pending | Total |
|----------|------|---------|---------|-------|
| Methods | 8 | 2 | 5 | 15 |
| Layer IBP | 33 | 0 | 0 | 33 |
| Layer CROWN | 17 | 16 | 0 | 33 |
| Processing Opts | 3 | 2 | 7 | 12 |
| Memory Opts | 3 | 1 | 5 | 9 |
| Debug Opts | 0 | 1 | 10 | 11 |
| Soundness Ph0 | 5 | 0 | 0 | 5 |
| Soundness Ph1 | 5 | 0 | 0 | 5 |

**Overall:** ~45% complete on optimizations, ~80% on core methods, ~100% on IBP layer coverage, **100% on Phase 0-1 soundness**

---

## Commands

```bash
# Check layer coverage
grep -E "pub struct.*Layer" crates/gamma-propagate/src/layers/mod.rs | wc -l

# Check method implementations
grep -E "pub fn propagate" crates/gamma-propagate/src/network.rs | head -20

# Run all tests
cargo test --workspace

# Benchmark
cargo bench --bench propagation
```
