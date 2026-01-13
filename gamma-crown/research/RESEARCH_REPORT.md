# Neural Network Verification: State of the Art Research Report

**Date:** 2025-12-29
**Purpose:** Comprehensive survey for γ-CROWN development
**Scope:** All major NN verification tools, algorithms, and recent research

---

## Executive Summary

Neural network verification is a rapidly evolving field focused on providing mathematical guarantees about network behavior. This report surveys all major tools, algorithms, and research to inform γ-CROWN development.

**Key Findings:**
1. **α,β-CROWN dominates**: Won VNN-COMP 2021-2025 consecutively
2. **No Rust implementations exist**: Major gap in the ecosystem
3. **Transformer verification is immature**: Limited tools, active research area
4. **GPU acceleration is essential**: All competitive tools use CUDA
5. **Standardized formats**: ONNX for models, VNN-LIB for specifications

**Recommendation:** γ-CROWN should implement the union of all SOTA techniques in Rust with Metal/MLX acceleration.

---

## Part 1: Verification Tools Survey

### 1.1 α,β-CROWN (Winner: VNN-COMP 2021-2025)

**Repository:** `research/repos/alpha-beta-CROWN/`
**Language:** Python (PyTorch)
**Lead:** Huan Zhang (UIUC)

**Architecture:**
```
alpha-beta-CROWN/
├── complete_verifier/
│   ├── abcrown.py         # Main verifier entry
│   ├── bab.py             # Branch and bound algorithm
│   ├── beta_CROWN_solver.py  # β-CROWN solver
│   ├── heuristics/        # Branching heuristics (babsr, fsb, kfsb)
│   ├── attack/            # Adversarial attack methods
│   ├── lp_mip_solver/     # LP/MIP integration
│   └── cuts/              # Cut-based improvements (BICCOS)
└── (uses auto_LiRPA as bound computation backend)
```

**Key Algorithms:**
- **IBP (Interval Bound Propagation)**: Fastest, loosest bounds
- **CROWN**: Linear relaxation of activations
- **α-CROWN**: Optimized CROWN with learnable slopes
- **β-CROWN**: Branch-and-bound with neuron splitting
- **GenBaB**: Generalized B&B for nonlinear functions
- **GCP-CROWN**: General cutting planes

**Why It Wins:**
1. GPU-accelerated bound computation
2. Efficient branch-and-bound with smart heuristics
3. Automatic batch size tuning
4. Integration with attack methods for faster falsification
5. Comprehensive activation support (ReLU, sigmoid, tanh, GELU, softmax)

---

### 1.2 Auto-LiRPA (Foundation Library)

**Repository:** `research/repos/auto_LiRPA/`
**Language:** Python (PyTorch)

**Core Structure:**
```
auto_LiRPA/
├── bound_general.py       # Main BoundedModule class (72KB)
├── backward_bound.py      # Backward bound propagation (49KB)
├── forward_bound.py       # Forward bound propagation
├── interval_bound.py      # IBP implementation
├── linear_bound.py        # LinearBound data structure
├── optimized_bounds.py    # α-CROWN optimization (50KB)
├── beta_crown.py          # β-CROWN split constraints
├── perturbations.py       # Input perturbation types
└── operators/             # Per-operator bound implementations
    ├── linear.py          # Matrix multiply, Gemm
    ├── relu.py            # ReLU with α optimization
    ├── softmax.py         # Softmax bounds
    ├── normalization.py   # LayerNorm, BatchNorm
    ├── gelu.py            # GELU activation
    ├── convolution.py     # Conv2d bounds
    └── ...                # 30+ operator implementations
```

**Key Data Structures:**
```python
# LinearBound: y = Ax + b representation
class LinearBound:
    lw: Tensor  # Lower weight matrix
    lb: Tensor  # Lower bias
    uw: Tensor  # Upper weight matrix
    ub: Tensor  # Upper bias
    x_L, x_U: Tensor  # Input bounds

# BoundedTensor: Interval representation
class BoundedTensor:
    data: Tensor
    ptb: PerturbationLp  # Perturbation specification
```

**Softmax Bounds (Key Challenge):**
```python
# From operators/softmax.py:53-61
def interval_propagate(self, *v):
    h_L, h_U = v[0]
    shift = h_U.max(dim=self.axis, keepdim=True).values
    exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
    lower = exp_L / (sum(exp_U) - exp_U + exp_L + epsilon)
    upper = exp_U / (sum(exp_L) - exp_L + exp_U + epsilon)
    return lower, upper
```

---

### 1.3 Marabou (SMT-Based)

**Repository:** `research/repos/Marabou/`
**Language:** C++ (91%) with Python bindings
**Origin:** Stanford AI Safety

**Approach:** SMT (Satisfiability Modulo Theories)
- Encodes verification as constraint satisfaction
- Uses Reluplex algorithm for ReLU networks
- Can find concrete counterexamples
- Complete but potentially slow

**Structure:**
```
Marabou/
├── src/
│   ├── engine/         # Core verification engine
│   ├── constraints/    # Constraint representations
│   └── network_parser/ # ONNX, .nnet parsing
├── maraboupy/          # Python bindings
└── deps/CVC4/          # SMT solver dependency
```

**When to Use:** When you need counterexamples or complete verification.

---

### 1.4 ERAN (ETH Zurich)

**Repository:** `research/repos/eran/`
**Language:** Python with ELINA library

**Abstract Domains:**
- **DeepZ**: Zonotope domain
- **DeepPoly**: Polyhedra domain
- **GPUPoly**: GPU-accelerated polyhedra
- **RefineZono**: Refined zonotopes
- **RefinePoly**: Refined polyhedra

**Key Innovation:** Multiple precision/speed tradeoffs via domain selection.

---

### 1.5 DeepT (Transformer-Specific)

**Repository:** `research/repos/DeepT/`
**Language:** Python
**Paper:** PLDI 2021 "Fast and precise certification of transformers"

**Key Innovation: Multi-norm Zonotopes**
- Extension of zonotopes for L1/L2 norm perturbations
- Handles softmax and dot product precisely
- 28x larger certified radii than prior SOTA

**Structure:**
```
DeepT/Robustness-Verification-for-Transformers/
├── Verifiers/
│   ├── Zonotope.py         # Multi-norm zonotope implementation
│   ├── VerifierZonotope.py # Main verifier
│   ├── relaxation.py       # Activation relaxations
│   └── Edge.py             # Computational graph edges
└── models/                  # Transformer model definitions
```

**Threat Models:**
1. L_p ball in embedding space
2. Synonym attacks (word substitution)

---

### 1.6 NeuralSAT (VNN-COMP 2024 2nd Place)

**Repository:** `research/repos/neuralsat/`
**Language:** Python

**Approach:** DPLL(T) - Combines SAT solving with DNN theory solver
- Conflict clause learning
- Polytope abstraction
- Multi-core and GPU support

**When to Use:** Alternative to α,β-CROWN for complete verification.

---

### 1.7 nnenum

**Repository:** `research/repos/nnenum/`
**Language:** Python

**Key Feature:** Only tool to verify all ACAS-Xu benchmarks in <10s each
- Combines zonotopes with star set overapproximations
- Parallelized ReLU case splitting
- Uses GLPK for linear programming

---

## Part 2: Algorithm Deep Dive

### 2.1 Interval Bound Propagation (IBP)

**Complexity:** O(n) per layer
**Precision:** Loosest

For linear layer y = Wx + b with x ∈ [l, u]:
```
W+ = max(W, 0)
W- = min(W, 0)
y_lower = W+ @ l + W- @ u + b
y_upper = W+ @ u + W- @ l + b
```

For ReLU:
```
y_lower = max(0, x_lower)
y_upper = max(0, x_upper)
```

**Implementation in Auto-LiRPA:** `interval_bound.py:26-100`

---

### 2.2 CROWN (Linear Relaxation)

**Key Idea:** Represent bounds as linear functions of input.

For output y with input x:
```
A_L @ x + b_L ≤ y ≤ A_U @ x + b_U
```

**ReLU Relaxation:**
- **Positive region (l ≥ 0):** Identity (slope = 1)
- **Negative region (u ≤ 0):** Zero (slope = 0)
- **Crossing region (l < 0 < u):**
  - Upper: Line from (l, 0) to (u, u)
  - Lower: Parametric (α-CROWN optimizes this)

```python
# Upper bound slope for crossing ReLU
upper_slope = u / (u - l)
upper_intercept = -l * upper_slope
```

---

### 2.3 α-CROWN (Optimized CROWN)

**Key Idea:** Make lower bound slope learnable.

For each unstable ReLU (l < 0 < u):
- α ∈ [0, 1] is the lower bound slope
- Optimize α via gradient descent to tighten bounds

```python
# α-CROWN optimization loop
for iteration in range(max_iterations):
    bounds = compute_bounds(alphas)
    loss = -bounds.lower.sum()  # Maximize lower bound
    loss.backward()
    optimizer.step()
    alphas.clamp_(0, 1)
```

**Implementation:** `optimized_bounds.py`

---

### 2.4 β-CROWN (Branch and Bound)

**Key Idea:** Split unstable neurons into cases.

```
For unstable neuron n:
  Case 1: n is always active (ReLU(x) = x)
  Case 2: n is always inactive (ReLU(x) = 0)

Recursively verify each case.
Prune branches that cannot violate property.
```

**Branching Heuristics:**
- **BaBSR:** Score by bound improvement
- **FSB:** Feature-based scoring
- **KFSB:** K-feature-based scoring

**Implementation:** `bab.py`, `heuristics/`

---

### 2.5 Softmax Bounds (Research Frontier)

**Challenge:** Exponentials cause bound explosion.

**Auto-LiRPA Approach:**
```python
# Shift for numerical stability
shift = h_U.max(dim=axis)
exp_L = exp(h_L - shift)
exp_U = exp(h_U - shift)

# Conservative bounds
lower = exp_L / (sum(exp_U) - exp_U + exp_L)
upper = exp_U / (sum(exp_L) - exp_L + exp_U)
```

**DeepT Approach:** Multi-norm zonotopes for tighter bounds.

**Open Problem:** Tight softmax bounds remain a research challenge.

---

### 2.6 LayerNorm Bounds

**Challenge:** Division by variance creates dependencies.

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

**Approach:**
1. Bound mean separately
2. Bound variance (always non-negative)
3. Bound division carefully
4. Apply scale and shift

---

## Part 3: Performance Comparison

### VNN-COMP Results (2024)

| Rank | Tool | Approach |
|------|------|----------|
| 1 | α,β-CROWN | Bound propagation + B&B |
| 2 | NeuralSAT | DPLL(T) |
| 3 | Marabou | SMT |
| 4 | nnenum | Zonotopes + stars |

### Benchmark Categories

1. **ACAS-Xu**: Collision avoidance (45 properties)
2. **MNIST/CIFAR**: Image classification robustness
3. **VGG/ResNet**: Large CNN verification
4. **RL Benchmarks**: Reinforcement learning networks

---

## Part 4: Key Papers and Citations

### Core Papers

1. **CROWN (NeurIPS 2018)**
   - Zhang et al. "Efficient Neural Network Robustness Certification with General Activation Functions"
   - First linear bound propagation for general activations
   - arXiv:1811.00866

2. **Auto-LiRPA (NeurIPS 2020)**
   - Xu et al. "Automatic Perturbation Analysis for Scalable Certified Robustness"
   - Automatic computation graph analysis
   - arXiv:2002.12920

3. **β-CROWN (NeurIPS 2021)**
   - Wang et al. "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints"
   - Branch and bound integration
   - arXiv:2103.06624

4. **DeepT (PLDI 2021)**
   - Bonaert et al. "Fast and Precise Certification of Transformers"
   - Multi-norm zonotopes for transformers
   - DOI:10.1145/3453483.3454056

### Recent Research (2023-2025)

5. **Certified Training with IBP (2023)**
   - Mao et al. "Understanding Certified Training with Interval Bound Propagation"
   - arXiv:2306.10426

6. **Deep RL Verification (2023)**
   - Zhi et al. "Robustness Verification of Deep Reinforcement Learning"
   - arXiv:2312.09695

7. **Text Recognition Certification (2024)**
   - Shao et al. "STR-Cert: Robustness Certification for Deep Text Recognition"
   - arXiv:2401.05338

---

## Part 5: Gap Analysis for γ-CROWN

### What Exists

| Feature | α,β-CROWN | Marabou | ERAN | DeepT |
|---------|-----------|---------|------|-------|
| IBP | ✓ | ✗ | ✓ | ✓ |
| CROWN | ✓ | ✗ | ✓ | ✓ |
| α-CROWN | ✓ | ✗ | ✗ | ✓ |
| β-CROWN | ✓ | ✗ | ✗ | ✗ |
| SMT | ✗ | ✓ | ✗ | ✗ |
| GPU | ✓ | ✗ | ✓ | ✓ |
| Transformer | Limited | ✗ | ✗ | ✓ |
| Rust | ✗ | ✗ | ✗ | ✗ |
| Metal/MLX | ✗ | ✗ | ✗ | ✗ |

### γ-CROWN Opportunities

1. **First Rust implementation** - No competition
2. **Metal/MLX acceleration** - No existing tools
3. **Unified approach** - Combine all SOTA techniques
4. **Transformer-native** - Built for Whisper-scale models
5. **Production quality** - Focus on reliability and speed

---

## Part 6: Recommended γ-CROWN Architecture

Based on this research, γ-CROWN should implement:

### Core Algorithms (Priority Order)

1. **IBP** - Foundation, implement first
2. **CROWN** - Linear relaxation
3. **α-CROWN** - Optimized bounds
4. **β-CROWN** - Complete verification

### Transformer Support

1. **Softmax bounds** - Use DeepT multi-norm zonotope approach
2. **LayerNorm bounds** - Careful variance handling
3. **Attention bounds** - Exploit structure

### Acceleration

1. **Metal compute shaders** - Primary GPU target
2. **MLX integration** - Apple ML framework
3. **SIMD (NEON)** - CPU vectorization
4. **Rayon** - CPU parallelism

### Differentiators

1. **Rust performance** - Zero-cost abstractions
2. **Apple Silicon optimization** - Metal + unified memory
3. **Whisper-native** - Built for speech models
4. **Sound verification** - Mathematical guarantees

---

## Part 7: Downloaded Resources

### Repositories (in `research/repos/`)

| Repository | Size | Language | Purpose |
|------------|------|----------|---------|
| alpha-beta-CROWN | ~50MB | Python | VNN-COMP winner, reference |
| auto_LiRPA | ~20MB | Python | Core bound propagation |
| Marabou | ~100MB | C++ | SMT-based verification |
| eran | ~30MB | Python | Abstract interpretation |
| nnenum | ~5MB | Python | Zonotope verification |
| DeepT | ~100MB | Python | Transformer verification |
| neuralsat | ~10MB | Python | SAT-based verification |

### Key Files to Study

1. `auto_LiRPA/operators/linear.py` - Linear layer bounds
2. `auto_LiRPA/operators/relu.py` - ReLU relaxation
3. `auto_LiRPA/operators/softmax.py` - Softmax bounds
4. `auto_LiRPA/interval_bound.py` - IBP implementation
5. `auto_LiRPA/optimized_bounds.py` - α-CROWN optimization
6. `alpha-beta-CROWN/complete_verifier/bab.py` - Branch and bound
7. `DeepT/.../Zonotope.py` - Multi-norm zonotopes

---

## Conclusion

γ-CROWN is uniquely positioned to become the first high-performance Rust implementation of neural network verification, combining:

1. **All SOTA algorithms** from α,β-CROWN
2. **Transformer expertise** from DeepT
3. **Apple Silicon optimization** (Metal/MLX)
4. **Production quality** Rust implementation

The field is dominated by Python tools. A well-implemented Rust solution with Metal acceleration could achieve order-of-magnitude speedups.

---

*Report generated for γ-CROWN development team*
*Total repositories downloaded: 7*
*Total source files analyzed: 100+*
