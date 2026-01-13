# Literature Survey: Transformer Verification

**Goal:** Comprehensive survey of existing methods for neural network and transformer verification.

---

## Core Papers (Must Read)

### Transformer-Specific Verification

| arXiv ID | Title | Year | Key Contribution |
|----------|-------|------|------------------|
| **2002.06622** | Robustness Verification for Transformers | 2020 | First transformer verification, attention bounds |
| 2401.05338 | STR-Cert: Robustness Certification for Deep Text Recognition | 2024 | Text recognition verification |
| 2503.14751 | LipShiFT: Certifiably Robust Shift-based Vision Transformer | 2025 | Vision transformer certification |
| 2405.17361 | A One-Layer Decoder-Only Transformer is a Two-Layer RNN | 2024 | Theoretical connection to RNNs |

### CROWN / α-β-CROWN Family

| arXiv ID | Title | Year | Key Contribution |
|----------|-------|------|------------------|
| **2103.06624** | Beta-CROWN: Per-neuron Split Constraints | 2021 | β-CROWN algorithm, VNN-COMP winner |
| **2506.06665** | SDP-CROWN: SDP tightness for verification | 2025 | Semidefinite programming bounds |
| 2512.11087 | Clip-and-Verify: Domain Clipping | 2025 | Acceleration via domain clipping |
| 2212.08567 | Optimized Symbolic Interval Propagation | 2022 | Improved interval bounds |

### Interval Bound Propagation (IBP)

| arXiv ID | Title | Year | Key Contribution |
|----------|-------|------|------------------|
| 1909.01492 | Verified Robustness via IBP | 2019 | IBP for NLP/symbol substitution |
| 2211.16187 | Quantization-aware IBP | 2022 | IBP for quantized networks |

---

## Papers to Fetch and Analyze

### Priority 1: Transformer Verification
```
arxiv:2002.06622 - CRITICAL - Original transformer verification
arxiv:2503.14751 - Vision transformer certification
arxiv:2401.05338 - Text recognition certification
```

### Priority 2: Bound Propagation Methods
```
arxiv:2103.06624 - Beta-CROWN (VNN-COMP winner)
arxiv:2506.06665 - SDP-CROWN (latest tightness)
arxiv:2512.11087 - Domain clipping acceleration
```

### Priority 3: Theoretical Foundations
```
arxiv:2405.17361 - Transformer-RNN equivalence
arxiv:1909.01492 - IBP foundations for NLP
```

---

## Key Findings from Literature

### Scale of Verified Models (CRITICAL)

| Method | Largest Model Verified | Notes |
|--------|----------------------|-------|
| Bonaert (2020) | Small BERT classifiers | Yelp/SST-2 sentiment, word perturbations |
| Beta-CROWN (2021) | CNNs for VNN-COMP | MNIST/CIFAR scale, not transformers |
| SDP-CROWN (2025) | **2.47M params, 65K neurons** | Latest SOTA, still small |
| γ-CROWN (us) | whisper-tiny (39M) | **BOUNDS EXPLODE** |

**Reality:** The largest verified model is ~2.5M params. We're trying to verify 39M (16x larger) and eventually 70B (28,000x larger).

### Transformer Verification (Bonaert 2020)
- **Dataset:** Yelp, SST-2 (sentiment classification)
- **Perturbation:** Word embedding L_p balls
- **NOT:** Full model, generative, large-scale
- **Code:** https://github.com/shizhouxing/Robustness-Verification-for-Transformers

### Beta-CROWN Techniques
- Per-neuron split constraints with optimizable β
- 1000x faster than LP methods
- Won VNN-COMP 2021
- Focuses on ReLU networks (CNNs)

### SDP-CROWN Innovations (2025)
- Inter-neuron coupling (not just per-neuron)
- √n tighter bounds than traditional methods
- Only 1 extra parameter per layer
- **Still limited to 2.47M params**

---

## Key Questions to Answer

1. **What is the largest transformer successfully verified?**
   - Answer: Small BERT classifiers (~110M params CLAIMED, but only word perturbations)
   - Reality: Nobody has verified full forward pass of 100M+ param transformer

2. **What bound tightening techniques exist for attention?**
   - Bonaert's method?
   - Others?

3. **What is the theoretical limit of bound propagation?**
   - Exponential blowup inherent?
   - Workarounds?

4. **What properties can be verified?**
   - Local robustness (L_inf ball)?
   - Global properties?
   - Safety specifications?

5. **What are the computational bottlenecks?**
   - Memory?
   - Time complexity?
   - Numerical precision?

---

## Research Gaps (Potential Contributions)

1. **Scale**: No one has verified 1B+ param transformers
2. **Attention bounds**: Bilinear Q@K^T still causes explosion
3. **Generative models**: Focus has been on classifiers
4. **Efficient verification**: Current methods don't scale

---

## Next Steps

1. [ ] Download and read arxiv:2002.06622 (Bonaert transformer verification)
2. [ ] Download and read arxiv:2103.06624 (Beta-CROWN)
3. [ ] Download and read arxiv:2506.06665 (SDP-CROWN)
4. [ ] Analyze what bounds they achieve on what models
5. [ ] Identify techniques we can implement in γ-CROWN
6. [ ] Identify open problems for research contribution

---

## References

- Auto-LiRPA: https://github.com/Verified-Intelligence/auto_LiRPA
- α-β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- VNN-COMP: https://sites.google.com/view/vnn2024
