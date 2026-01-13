# γ-CROWN Design Document

## Vision

γ-CROWN (Gamma-CROWN) is a Rust-native neural network verification library designed to scale to Whisper-class transformer models. It ports and extends the state-of-the-art α,β-CROWN verification algorithms from Python to Rust, achieving maximum possible speedup through native compilation, SIMD optimization, GPU acceleration, and algorithmic improvements.

**Primary Use Case**: Verify that neural network model ports (PyTorch → Metal/CoreML) produce equivalent outputs within certified tolerance bounds.

## Goals

1. **Whisper-Scale Verification**: Verify properties of 39M+ parameter transformer models
2. **Porting Validation**: Mathematically certify that model ports are equivalent within tolerance
3. **Maximum Speedup**: Push Rust to its limits vs Python Auto-LiRPA - no arbitrary ceiling
4. **AI-Native Interface**: Structured I/O for AI-to-AI verification workflows
5. **Compositional Verification**: Verify components independently, compose guarantees

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  gamma-cli (CLI)                gamma-python (Python bindings)   │
│                    (User/AI Interfaces)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                       gamma-onnx                                 │
│              (Model Loading, Diffing & Conversion)               │
│  PyTorch/ONNX/SafeTensors/GGUF → γ-CROWN IR → Whisper extraction │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│  gamma-smt                │              gamma-transformer       │
│  (Complete SMT/z4)        │   (Transformer-Specific Propagation) │
│  DPLL(T), proofs          │   Softmax, LayerNorm, Attention      │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                    gamma-propagate                               │
│              (Core Propagation Algorithms)                       │
│            IBP → CROWN → α-CROWN → β-CROWN                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                     gamma-gpu                                    │
│        (GPU Acceleration: wgpu/Metal, Parallel CPU: rayon)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                     gamma-tensor                                 │
│              (Bounded Tensor Arithmetic)                         │
│         Interval arithmetic, linear bounds, shapes               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                      gamma-core                                  │
│                   (Core Types & Traits)                          │
│      Bound, VerificationResult, LayerType, GammaError            │
└─────────────────────────────────────────────────────────────────┘
```

## Crate Responsibilities

### gamma-core

Core types shared across all crates:

```rust
/// A bound on a scalar value: [lower, upper]
pub struct Bound { lower: f32, upper: f32 }

/// Result of verification
pub enum VerificationResult {
    Verified { output_bounds: Vec<Bound> },
    Violated { counterexample: Vec<f32>, output: Vec<f32> },
    Unknown { bounds: Vec<Bound>, reason: String },
    Timeout { partial_bounds: Option<Vec<Bound>> },
}

/// Verification specification
pub struct VerificationSpec {
    input_bounds: Vec<Bound>,
    output_bounds: Vec<Bound>,
    timeout_ms: Option<u64>,
}

/// Layer types in neural networks
pub enum LayerType {
    Linear, Conv2d, ReLU, GELU, Softmax, LayerNorm, MultiHeadAttention, ...
}
```

### gamma-tensor

Bounded tensor types with interval arithmetic:

```rust
/// Tensor where each element has certified bounds
pub struct BoundedTensor {
    lower: ArrayD<f32>,
    upper: ArrayD<f32>,
}

impl BoundedTensor {
    fn add(&self, other: &BoundedTensor) -> BoundedTensor;
    fn mul(&self, other: &BoundedTensor) -> BoundedTensor;  // Interval multiplication
    fn matmul(&self, other: &BoundedTensor) -> BoundedTensor;
    fn from_epsilon(values: ArrayD<f32>, epsilon: f32) -> BoundedTensor;
}
```

### gamma-propagate

Core bound propagation algorithms:

1. **IBP (Interval Bound Propagation)**: Fastest, loosest bounds
   - O(n) per layer
   - Good for initial screening

2. **CROWN**: Linear relaxation
   - Represents bounds as linear functions of input
   - Tighter than IBP, slower

3. **α-CROWN**: Optimized CROWN
   - Learnable relaxation parameters
   - Iteratively tightens bounds

4. **β-CROWN**: Branch and bound
   - Complete verification (can prove or find counterexample)
   - Exponential worst case, but often fast in practice

```rust
use std::borrow::Cow;

pub trait BoundPropagation {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor>;
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>>;
}

pub struct Verifier {
    fn verify(&self, layers: &[LayerType], weights: &[ArrayD<f32>], spec: &VerificationSpec)
        -> Result<VerificationResult>;
}
```

### gamma-transformer

Transformer-specific bound propagation:

**Softmax Bounds** (Research Frontier)
```rust
/// Bound propagation through softmax
/// Challenge: Exponentials cause bound explosion
/// Solution: Use logit difference bounds
pub fn softmax_bounds(input: &BoundedTensor, dim: i32) -> Result<BoundedTensor>;
```

**LayerNorm Bounds**
```rust
/// Bound propagation through LayerNorm
/// Challenge: Division creates non-linear dependencies
/// Solution: Bound mean and variance separately, propagate
pub fn layer_norm_bounds(input: &BoundedTensor, gamma: &ArrayD<f32>, beta: &ArrayD<f32>, eps: f32)
    -> Result<BoundedTensor>;
```

**Attention Bounds**
```rust
/// Bound propagation through multi-head attention
/// Challenge: Quadratic interactions (Q @ K^T), softmax, V multiplication
/// Solution: Compositional bounds, exploit structure
pub fn attention_bounds(query: &BoundedTensor, key: &BoundedTensor, value: &BoundedTensor,
    num_heads: usize, head_dim: usize) -> Result<BoundedTensor>;
```

### gamma-onnx

ONNX model loading and conversion:

```rust
/// Load ONNX model and convert to γ-CROWN IR
pub fn load_onnx<P: AsRef<Path>>(path: P) -> Result<(Network, WeightStore)>;

/// Whisper-specific loading with component extraction
pub fn load_whisper<P: AsRef<Path>>(path: P) -> Result<WhisperModel>;

impl WhisperModel {
    fn encoder(&self) -> Result<Network>;
    fn encoder_layer(&self, index: usize) -> Result<Network>;
    fn attention_head(&self, layer: usize, head: usize) -> Result<Network>;
}
```

### gamma-cli

Command-line interface for verification:

```bash
# Verify a model
gamma verify --model whisper.onnx --epsilon 0.01 --method alpha

# Compare two model implementations (porting verification)
gamma compare --reference pytorch.onnx --target metal.onnx --tolerance 0.001

# Verify Whisper component
gamma whisper --model whisper.onnx --component encoder --layer 0 --epsilon 0.01

# Generate export script
gamma export --model-type whisper --size tiny

# Run β-CROWN branch-and-bound verification
gamma beta-crown --model model.onnx --epsilon 0.01 --threshold 0.0
```

#### Compare Command (Model Porting Verification)

The `compare` command verifies that two model implementations produce equivalent outputs within a specified tolerance. This is the primary use case for validating model ports (e.g., PyTorch → Metal/CoreML).

**Usage:**
```bash
gamma compare \
  --reference <PYTORCH_ONNX> \
  --target <METAL_ONNX> \
  --epsilon 0.01 \
  --tolerance 0.001 \
  --method crown
```

**Options:**
- `--reference`: Path to reference model (e.g., PyTorch ONNX export)
- `--target`: Path to target model (e.g., Metal implementation)
- `--epsilon`: Input perturbation radius (default: 0.01)
- `--tolerance`: Maximum allowed difference in output bounds (default: 0.001)
- `--method`: Verification method - `ibp`, `crown`, or `alpha` (default: crown)
- `--verbose`: Print per-element comparison details

**Output:**
- Prints model architecture comparison
- Runs bound propagation on both models
- Reports max difference in lower/upper bounds
- Shows percentage of outputs with overlapping bounds
- Outputs JSON summary for AI workflows:
```json
{"equivalent": true, "max_lower_diff": 0.0, "max_upper_diff": 0.0, "tolerance": 0.001, "overlap_pct": 100.0, "ref_max_width": 0.05, "target_max_width": 0.05}
```

### gamma-gpu

GPU acceleration and parallel CPU backends:

```rust
/// Compute backend abstraction
pub enum Backend {
    Cpu,      // Rayon parallel CPU
    Wgpu,     // Cross-platform GPU (Metal on macOS)
    Mlx,      // Apple MLX unified memory
}

/// GPU-accelerated bound propagation
pub struct GpuDevice {
    backend: Backend,
    // Manages GPU resources, shader compilation, memory pools
}

impl GpuDevice {
    /// Propagate IBP bounds through a linear layer on GPU
    fn propagate_ibp_linear(&self, input: &BoundedTensor, weights: &Array2<f32>) -> BoundedTensor;

    /// Propagate CROWN linear bounds through network on GPU
    fn propagate_crown(&self, network: &Network, input: &BoundedTensor) -> LinearBounds;
}
```

Performance: 5-6x speedup over CPU on Whisper-scale models.

### gamma-smt

Complete SMT verification via native z4 integration:

```rust
/// SMT-based complete verifier
pub struct SmtVerifier {
    solver: z4_dpll::Solver,
}

impl SmtVerifier {
    /// Complete verification: returns SAT (counterexample) or UNSAT (verified)
    fn verify(&mut self, network: &Network, property: &Property) -> VerificationResult;

    /// Export UNSAT proof certificate
    fn export_proof(&self) -> Option<Certificate>;
}

/// Lazy verifier with incremental ReLU splitting
pub struct LazyVerifier {
    // Combines bound propagation with SMT for completeness
}
```

### gamma-python

Python bindings via PyO3:

```python
import gamma

# Model comparison
result = gamma.diff("model_a.onnx", "model_b.onnx", tolerance=1e-4)
print(result.first_bad_layer_name)  # "encoder.blocks.0.self_attn.softmax"

# Bound propagation
bounds = gamma.verify("model.onnx", epsilon=0.01, method="crown")

# Quantization safety check
result = gamma.quantize_check("model.onnx")
print(result.float16_safe)  # True/False
```

## Whisper Verification Strategy

### Challenge

Whisper-tiny has ~39M parameters across:
- 4 encoder transformer blocks
- 4 decoder transformer blocks
- Each block has: attention, layer norm, FFN

Current verification tools handle ~1M parameters well. We need 40x improvement.

### Approach

#### Phase 1: Component Verification (Current Focus)

Verify individual components in isolation:

| Component | Parameters | Verifiable? |
|-----------|------------|-------------|
| Single conv layer | ~10K | Yes |
| Single attention head | ~100K | Yes |
| Single transformer block | ~1M | Research frontier |
| Full encoder | ~20M | Not yet |

#### Phase 2: Compositional Verification

Define interface contracts between components:

```rust
struct EncoderBlockContract {
    input: BoundedTensor,   // Bounds on input activations
    output: BoundedTensor,  // Guaranteed output bounds
}

// Verify: if input satisfies contract, output satisfies contract
fn verify_block(block: &TransformerBlock, contract: &EncoderBlockContract) -> VerificationResult;
```

Compose block-level guarantees into full-model guarantees.

#### Phase 3: Specialized Relaxations

Develop tighter bounds for transformer primitives:

1. **Softmax**: Use logit-difference bounds instead of naive interval
2. **LayerNorm**: Exploit bounded variance
3. **Attention**: Low-rank structure, sparse attention patterns

#### Phase 4: Certified Training (Future)

Train Whisper variants designed for verification:
- ReLU-softmax hybrid attention
- Polynomial activations instead of GELU
- Weight magnitude constraints

## Porting Validation Use Case

**Scenario**: Port Whisper from PyTorch to Metal

**Without γ-CROWN**:
1. Run 10,000 test inputs
2. Compare outputs
3. Hope nothing was missed
4. No guarantees

**With γ-CROWN**:
1. Export both models to ONNX
2. Define input region (typical mel spectrograms ± noise)
3. Verify: `|output_pytorch - output_metal| < tolerance` for ALL inputs in region
4. Mathematical guarantee of equivalence

```bash
gamma compare \
    --reference whisper_pytorch.onnx \
    --target whisper_metal.onnx \
    --tolerance 0.001 \
    --input-spec mel_spectrogram_bounds.json
```

Output:
```json
{
    "verified": true,
    "output_difference_bounds": {
        "max": 0.00087,
        "certified": true
    },
    "method": "alpha-crown",
    "time_seconds": 45.2
}
```

## Performance Philosophy

**Goal: As fast as technically possible.** No arbitrary speedup ceiling.

Rust advantages to exploit:
- Zero-cost abstractions
- SIMD intrinsics (AVX2, AVX-512, NEON on Apple Silicon)
- Memory layout control (cache-friendly access patterns)
- **Metal GPU compute** (primary target for Apple Silicon M1/M2/M3)
- **MLX integration** (Apple's ML framework, unified memory)
- Parallelism (rayon for CPU, Metal for GPU)
- No GIL, no GC pauses
- Direct PyTorch/ONNX interop

| Operation | Python Auto-LiRPA | γ-CROWN Approach |
|-----------|-------------------|------------------|
| IBP, single layer | 10ms | Push to μs with SIMD |
| CROWN, single layer | 100ms | Fused kernels, cache-aware |
| α-CROWN, single layer | 1s | GPU acceleration |
| Full encoder (IBP) | 10s | Parallel layer propagation |
| Full encoder (α-CROWN) | 5min | GPU + algorithmic improvements |

Benchmark continuously. Optimize bottlenecks. No ceiling.

## Numerical Soundness

γ-CROWN claims **sound** bound propagation: if we report bounds [L, U], the true output is guaranteed to lie within [L, U] for all inputs in the specification. This section documents numerical behavior critical to soundness.

### IEEE 754 Compliance

**γ-CROWN requires IEEE 754 compliant floating point.** The build configuration:
- No `-ffast-math` or equivalent compiler flags
- Default rounding mode (round-to-nearest-even)
- No `--release` flags that could enable unsafe math optimizations
- Rust's default `overflow-checks` preserved

### Denormalized Numbers

**Behavior:** γ-CROWN processes denormalized (subnormal) f32 values correctly. These are numbers smaller than `f32::MIN_POSITIVE` (~1.175e-38) but greater than zero.

**Performance note:** Operations on denormals may be 10-100x slower on some hardware ("denormal stall"). This is a hardware limitation, not a correctness issue.

**Design decision:** We do NOT enable flush-to-zero (FTZ) mode because:
1. FTZ could cause soundness violations (bounds might miss values rounded to zero)
2. Denormal performance impact is negligible for neural network verification
3. Correctness is more important than performance at the margins

**Testing:** Edge case tests in `gamma-tensor` verify correct handling of:
- Denormalized values (`f32::MIN_POSITIVE / 2.0`)
- Negative zero (`-0.0`)
- Maximum finite values (`f32::MAX`)
- Minimum finite values (`f32::MIN`)

### NaN and Infinity

**Debug assertions:** In debug builds, `BoundedTensor::new()` asserts that:
- No NaN values are present in lower or upper bounds
- No infinite values are present
- Lower bounds ≤ upper bounds

**Rationale:** NaN/Inf in bounds indicates upstream numerical issues that would invalidate verification results. Catching them early aids debugging.

### Directed Rounding

**Status:** γ-CROWN provides optional directed rounding for strict soundness.

**Default behavior:** Normal propagation uses default rounding (round-to-nearest-even). This is fast and sufficient for most use cases since relaxation approximation errors dominate.

**Sound mode:** For strict interval arithmetic soundness, use these APIs:
- `BoundedTensor::round_for_soundness()` - Applies `next_down()` to lower bounds, `next_up()` to upper bounds (1 ULP widening)
- `BoundedTensor::round_for_soundness_inplace()` - In-place version
- `Network::propagate_ibp_sound()` - Applies directed rounding after each layer

**Performance:** Sound mode adds ~1 ULP width per layer. For a 100-layer network, final bounds are ~100 ULPs wider, which is negligible compared to relaxation errors (typically 10^3 - 10^6 ULPs).

**Use cases for sound mode:**
- Formal verification applications requiring mathematical proofs
- Cross-validation against reference implementations
- Debugging subtle soundness issues

### Cross-Platform Behavior

**x86_64 vs ARM64:** Both platforms implement IEEE 754 binary32 identically. Minor differences may exist in:
- Extended precision intermediate results (x86 x87 FPU, rarely used)
- SIMD NaN handling semantics (we avoid SIMD operations that could differ)
- Denormal flush-to-zero default settings (we explicitly don't enable FTZ)

**GPU backends:** wgpu/Metal/CUDA backends use GPU floating point, which is IEEE 754 compliant for f32 but may have different behavior for:
- Subnormal handling (some GPUs flush to zero by default)
- Fused multiply-add (FMA) precision

When using GPU backends, bounds may be very slightly different from CPU bounds due to operation ordering and FMA differences. Both are valid bounds.

## Formal Perturbation Specification

This section formally defines what γ-CROWN verifies and the mathematical semantics of input perturbations.

### Threat Model: L∞-Ball Perturbations

**Definition:** Given an input tensor x₀ ∈ ℝⁿ and perturbation radius ε ≥ 0, the **L∞-ball** around x₀ is:

```
B_∞(x₀, ε) = { x ∈ ℝⁿ : ||x - x₀||_∞ ≤ ε }
           = { x ∈ ℝⁿ : ∀i. |xᵢ - x₀ᵢ| ≤ ε }
           = { x ∈ ℝⁿ : ∀i. x₀ᵢ - ε ≤ xᵢ ≤ x₀ᵢ + ε }
```

This is the **primary perturbation model** implemented in γ-CROWN.

**API:** `BoundedTensor::from_epsilon(center: ArrayD<f32>, epsilon: f32)` creates bounds for all inputs in B_∞(center, ε).

### Soundness Guarantee

**Theorem (IBP Soundness):** For any neural network f: ℝⁿ → ℝᵐ with γ-CROWN network representation N, and any bounded input tensor T = (lower, upper):

```
∀x ∈ [lower, upper]: lower_out ≤ f(x) ≤ upper_out
```

where `(lower_out, upper_out) = N.propagate_ibp(T)`.

**Corollary:** If bounds are computed from `BoundedTensor::from_epsilon(x₀, ε)`, then:
```
∀x ∈ B_∞(x₀, ε): lower_out ≤ f(x) ≤ upper_out
```

### Element-wise vs Structured Perturbations

**Element-wise (default):** Each input element varies independently within its ±ε range. This is the most conservative model and captures worst-case adversarial perturbations.

**Structured (zonotope):** For cases where perturbations have correlation structure:
- `ZonotopeTensor::from_input_shared(x, ε)` - All elements perturbed by same amount (e.g., brightness shift)
- `ZonotopeTensor::from_input_per_position(x, ε)` - Each position (e.g., token) perturbed together
- `ZonotopeTensor::from_input_elementwise(x, ε)` - Full independence (reduces to IBP bounds)

### Verification Modes

| Mode | Guarantee | Use Case |
|------|-----------|----------|
| IBP | Sound, may be loose | Fast screening, large models |
| CROWN | Sound, tighter than IBP | General verification |
| α-CROWN | Sound, optimized relaxation | Production verification |
| β-CROWN | Complete (sound + complete) | Proving or finding counterexamples |

**Sound:** Reported bounds are guaranteed to contain all possible outputs.
**Complete:** Additionally, can either prove property or find counterexample (no "Unknown").

### Property Specification (VNN-LIB)

γ-CROWN supports VNN-LIB format for complex specifications:

```vnnlib
; Declare input and output variables
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Input bounds: x ∈ [-1, 1]²
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))

; Output property: y₀ > y₁ (class 0 most likely)
(assert (>= Y_0 Y_1))
```

**CLI usage:** `gamma verify --model model.onnx --property spec.vnnlib`

### Perturbation Examples

**Image robustness (pixel perturbation):**
```bash
gamma verify --model classifier.onnx --input image.npy --epsilon 0.031
# Verifies: ∀x ∈ B_∞(image, 8/255): argmax(f(x)) = original_class
```

**Audio robustness (waveform perturbation):**
```bash
gamma verify --model whisper_encoder.onnx --input mel_spec.npy --epsilon 0.01
# Verifies bounded output variation for ±0.01 mel spectrogram perturbation
```

**Model comparison (porting validation):**
```bash
gamma diff --reference pytorch.onnx --target metal.onnx --epsilon 0.001
# Verifies: ∀x ∈ B_∞(test_input, 0.001): |f_ref(x) - f_target(x)| < tolerance
```

### What γ-CROWN Does NOT Verify

**Not verified:**
- Perturbations outside the L∞-ball model (e.g., rotation, scaling)
- Properties requiring quantifier alternation (∃x ∀y ...)
- Training-time properties (only inference)
- Probabilistic guarantees (all statements are worst-case)

**Not proven formally:**
- Correctness of the Rust implementation (see Phase 2 roadmap for Lean proofs)
- Hardware correctness (assumes IEEE 754 compliance)
- Operating system correctness

### Numerical Precision in Specifications

**Input precision:** f32 (IEEE 754 binary32). All ε values are f32.

**Epsilon interpretation:** The value ε is *exact*. We compute bounds for the closed interval [x₀ - ε, x₀ + ε]. Due to floating-point representation:
- Very small ε (< f32::MIN_POSITIVE) may be denormalized
- Very small ε relative to x₀ may be absorbed (if |x₀| >> ε such that x₀ + ε = x₀)

**Recommended ε ranges:**
- Image classification: 1/255 to 8/255 (0.0039 to 0.031)
- Audio processing: 0.001 to 0.1
- Model comparison: 0.0001 to 0.001

## Benchmarking & Profiling

### Criterion Benchmarks

Run statistical benchmarks with regression detection:

```bash
# All benchmarks
cargo bench

# Specific crate
cargo bench -p gamma-propagate
cargo bench -p gamma-gpu

# Open HTML reports
make bench-html
# Or: open target/criterion/report/index.html
```

Benchmark groups:
- **IBP/**: Linear, ReLU, GELU, LayerNorm, Conv1d, Conv2d
- **CROWN/**: MLP forward pass, per-position CROWN
- **Comparison/**: CPU baseline vs accelerated (Rayon parallel)
- **Scaling/**: Sequence length and hidden dimension scaling
- **CausalAttention/**: Standard vs causal attention (decoder)

### Profiling Tools

**Quick performance check:**
```bash
make perf-check
# Or: cargo run --release --example profile -p gamma-propagate
```

**Flamegraph (requires `cargo install flamegraph`):**
```bash
make flamegraph
# Generates: flamegraph.svg
```

**samply (Firefox Profiler - recommended):**
```bash
cargo install samply
make profile-samply
# Opens Firefox Profiler with detailed trace
```

**Install all profiling tools:**
```bash
make install-tools
```

### Performance Tracking

Each worker iteration should run `cargo bench` and note any regressions:

```bash
# Compare against baseline
cargo bench -- --baseline main
cargo bench -- --save-baseline feature-x
```

Key metrics to track:
| Benchmark | Target | Current |
|-----------|--------|---------|
| Linear IBP 384→1536 seq=64 | <1ms | TBD |
| GELU IBP 1536 seq=64 | <500μs | TBD |
| Full MLP CROWN | <10ms | TBD |
| Attention IBP h=8 s=128 | <5ms | TBD |

## API Design (AI-Native)

All inputs and outputs are structured for AI-to-AI workflows:

```rust
// Input: JSON specification
{
    "model_path": "whisper_tiny.onnx",
    "input_bounds": {
        "type": "epsilon",
        "center": "mel_spectrogram.npy",
        "epsilon": 0.01
    },
    "property": {
        "type": "output_bounds",
        "bounds": [[-10, 10], [-10, 10], ...]
    },
    "method": "alpha-crown",
    "timeout_ms": 60000
}

// Output: Structured result
{
    "result": "verified",
    "output_bounds": [[0.1, 0.9], [0.2, 0.8], ...],
    "method_used": "alpha-crown",
    "iterations": 47,
    "time_ms": 2340,
    "bounds_tight": true
}
```

## Research Contributions

This project aims to contribute:

1. **First Rust NN verification library** at production quality
2. **Whisper-scale transformer verification** (39M+ params)
3. **Compositional verification framework** for large models
4. **Transformer-specific relaxations** (softmax, layernorm, attention)
5. **Model porting validation** methodology

## References

### Core Papers

1. Zhang et al. "Efficient Neural Network Robustness Certification with General Activation Functions" (NeurIPS 2018) - Original CROWN
2. Xu et al. "Automatic Perturbation Analysis for Scalable Certified Robustness" (NeurIPS 2020) - Auto-LiRPA
3. Wang et al. "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints" (NeurIPS 2021) - β-CROWN
4. Shi et al. "Certified Training for Large-Scale Neural Networks" (NeurIPS 2021)

### Transformer Verification

5. Bonaert et al. "Fast and Precise Certification of Transformers" (PLDI 2021)
6. Zhang et al. "Certified Robustness to Patches for Transformers" (2023)

### Tools

- [Auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) - Python reference implementation
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) - Competition winner

## Roadmap

### Phase 1: Foundation ✓
- [x] Workspace structure
- [x] Core types (Bound, VerificationResult, LayerType)
- [x] BoundedTensor with interval arithmetic
- [x] IBP for basic layers (Linear, ReLU, Conv2d)
- [x] ONNX loading with prost

### Phase 2: CROWN Implementation ✓
- [x] Linear bounds representation
- [x] CROWN propagation for Linear, ReLU
- [x] ReLU relaxation with α optimization (α-CROWN)
- [x] Benchmarks vs Auto-LiRPA (335-680x speedup validated)

### Phase 3: Transformer Support ✓
- [x] Softmax bounds (Auto-LiRPA algorithm, exact match)
- [x] LayerNorm bounds
- [x] GELU bounds
- [x] Attention bounds (N-D batched MatMul + Softmax CROWN, 1.85x tighter than IBP)
- [ ] Tighter attention bounds (quadratic relaxation option - future enhancement)

### Phase 4: Whisper Integration (Current)
- [x] Whisper ONNX loading (load_whisper with structure parsing)
- [x] Encoder component extraction (encoder_stem, encoder_layer, final_layer_norm)
- [x] GraphNetwork block extraction (encoder_layer_graph)
- [x] Full GraphNetwork connectivity (tensor_producer tracing through intermediate ONNX ops)
  - 17/19 nodes properly connected (2 correctly reference block input for residuals)
- [x] N-D batched LinearLayer IBP (handles [seq, hidden] inputs for transformers)
- [x] AddConstantLayer for ONNX bias addition (unary Add with constant input)
- [x] ReshapeLayer and MulConstantLayer infrastructure (IBP + CROWN)
- [x] ONNX constant folding (Pow, Sqrt, Div) and INT64 data type support
- [x] MLP subpath IBP verified (LayerNorm → Linear → GELU → Linear → Add)
  - Input [4, 384] → Output [4, 384], max width 4312.27 (loose but sound)
- [x] Attention subgraph IBP verified (Linear projections + Reshape + Transpose + attention core)
  - Input [4, 384] → Output [4, 384], max width 0.30 (tight bounds with proper shapes)
  - 15-node graph: Q/K/V projection → Reshape → Transpose → MatMul → Softmax → MatMul → Transpose → Reshape → output projection
- [x] ReshapeLayer contiguous memory fix (handles non-contiguous arrays after Transpose)
- [x] Full block IBP (GraphNetwork with inserted attention reshape/transpose)
  - `WhisperModel::encoder_layer_graph_full()` inserts Q/K/V reshape+transpose and restores `[batch, seq, hidden]` before output projection
  - IBP works but produces very loose bounds (1.28e9 width for ε=0.01)
- [x] CROWN limitation documented for N-D batched transformer blocks
  - CROWN's LinearBounds assume flattened tensors; transformer Linear layers operate per-position
  - Dimension mismatch: [1536,1536] @ [384,384] when Linear weight is [384,384] but input has 1536 elements
  - Solutions: N-D batched LinearBounds (significant refactor) or compositional verification
- [x] Compositional verification (compose attention + MLP bounds for full block)
  - `WhisperModel::attention_subgraph()` extracts attention path (without residual Add)
  - `WhisperModel::mlp_subgraph()` extracts MLP path (without residual Add)
  - `verify_block_compositional()` composes subgraph bounds with explicit residual handling
  - Result: Attention delta is tight (1.4e4), MLP delta is loose (1.28e9) → identifies MLP as bottleneck
  - Same final bounds as naive IBP, but provides diagnostic info on where bound explosion occurs

### Phase 5: GPU Acceleration
- [x] Rayon CPU parallelization (gamma-gpu crate)
  - Linear IBP: 1.2-1.4x speedup at seq≥64
  - MatMul IBP: 11-13x speedup
  - Per-position CROWN: 1.9-9.5x speedup (scales with cores)
- [x] wgpu GPU compute (WgpuDevice)
  - Cross-platform via WebGPU (Metal, Vulkan, DX12)
  - Linear IBP compute shader implemented and verified
  - MatMul IBP compute shader implemented and verified
  - Softmax IBP compute shader (two-pass: reduce + apply Auto-LiRPA formula)
  - `attention_ibp()`: Chained attention (Q@K^T → scale → softmax → probs@V)
  - Buffer pool optimization: 4.6x GPU speedup for Whisper-scale (1×1500)
  - Threshold: GPU faster when batch × seq ≥ 256 positions
- [x] Fused GPU attention kernel (`attention_ibp_fused`)
  - Single encoder submission: transpose→matmul→scale→softmax→matmul
  - Eliminates 3 intermediate host roundtrips
  - 2.8-4.6x speedup over non-fused GPU
  - Moves GPU/CPU crossover from seq~100 to seq~64
  - At Whisper scale (seq=384): 17.5x speedup vs CPU (was 6.07x non-fused)
- [x] GPU-accelerated verification pipeline (`verify_block_compositional_gpu`)
  - Integrates GPU attention into Whisper block verification
  - Adaptive dispatch: GPU for seq≥64, CPU for smaller sequences
  - Extracts Q/K/V weights and runs LayerNorm → projections → GPU attention → output projection
  - Uses parallel CPU for MLP per-position CROWN
  - Provides `GpuCompositionalDetails` with bound widths and GPU usage info
- [x] Multi-block sequential verification (`verify_encoder_sequential`)
  - Chains multiple encoder blocks: output of block N → input of block N+1
  - Supports optional encoder stem and final LayerNorm
  - GPU-accelerated with adaptive dispatch per block
  - Per-block diagnostic details (attention/MLP widths)
  - Known limitation: IBP bounds overflow after ~2 blocks with ε=0.01 (see Architecture Notes)
- [x] Configurable overflow handling (`MultiBlockConfig`)
  - `verify_encoder_sequential_with_config()` for fine-grained control
  - Early termination when bounds exceed threshold (`max_bound_width`)
  - Overflow detection with NaN/Infinity handling (`terminate_on_overflow`)
  - Diagnostic mode for analysis through overflow (`continue_after_overflow`)
  - Preset configs: `default()`, `strict()`, `diagnostic()`
- [ ] Metal compute shaders (metal-rs) - Alternative for Apple Silicon
- [ ] MLX integration for tensor operations
- [ ] ONNX Runtime Metal/CoreML execution providers

### Phase 6: Decoder/LLM Support (NEW PRIORITY)
- [ ] **Causal attention bounds** (masked softmax for autoregressive models)
- [ ] **Decoder transformer blocks** (LLaMA, GPT-style)
- [ ] **Cross-attention bounds** (encoder-decoder attention for Whisper)
- [ ] **KV cache verification** (for efficient autoregressive inference)
- [ ] **CosyVoice3 architecture research** (LLM-based TTS)

### Phase 7: Apple Silicon Optimization
- [ ] Metal compute shaders (metal-rs) - Primary GPU target
- [ ] MLX integration for tensor operations
- [ ] ONNX Runtime Metal/CoreML execution providers

### Phase 8: Production
- [x] Initial benchmarks vs Auto-LiRPA
- [ ] PyTorch model loading (tch-rs)
- [ ] Comprehensive documentation
- [ ] Integration with DashProve
- [ ] LLaMA model verification (7B scale)

### Architecture Notes
- **Network vs Verifier**: Both `Network` and `Verifier` structs are fully functional. `Network` provides direct bound propagation methods (IBP, CROWN, α-CROWN). `Verifier` provides a unified API that accepts a `Network` and `VerificationSpec`, delegating to Network's methods (and `BetaCrownVerifier` for β-CROWN).
- **GraphNetwork**: Supports DAG computation graphs with IBP, CROWN, and α-CROWN. Used for attention patterns (MatMul + Softmax).
- **N-D Batch Support**: LinearLayer, MatMul, and Softmax support N-D batched inputs for transformer verification. LinearLayer handles [batch, features] input shapes.
- **Whisper Structure**: `load_whisper()` parses block boundaries from ONNX node names (e.g., "/blocks.0/attn/"). Each block has 19 extracted layers: 2 LayerNorms, 6 Linear, 2 MatMul (attention), 1 Softmax, 1 GELU, 7 Add (5 bias + 2 residual).
- **Whisper GraphNetwork**: `encoder_layer_graph()` extracts blocks as DAG with proper connectivity (17/19 nodes connected, 2 correctly reference block input). Full block IBP fails due to intermediate ONNX ops (Reshape, Transpose, Mul) that transform shapes for attention. These ops are skipped in layer extraction, breaking the shape chain for attention MatMuls.
- **AddConstantLayer**: Handles ONNX Add where one input is a constant (bias). Converted automatically from LayerType::Add when one input is in weights.
- **ReshapeLayer**: Changes tensor shape preserving total elements. Supports -1 for dimension inference, 0 for copying input dim. IBP and CROWN implemented.
- **MulConstantLayer**: Element-wise multiplication by constant tensor. Handles sign correctly (negative constants flip bounds). IBP and CROWN implemented.
- **ONNX Constant Folding**: Pre-computes Pow, Sqrt, Div nodes when all inputs are constants. Required for attention scaling (Pow(-0.5)).
- **β-CROWN**: Implemented in `beta_crown` module. Branch-and-bound search over ReLU activation space. Features:
  - `BetaCrownVerifier`: Main verifier struct with configurable timeout, max domains, and branching heuristic
  - `SplitHistory`: Tracks neuron split decisions (active vs inactive constraints)
  - `BabDomain`: Represents a domain in the search tree with tightened bounds
  - Branching heuristics: LargestBoundWidth (default), Sequential, BoundImpact
  - Constrained CROWN backward pass with β parameters for fixed neurons
  - Priority queue ordering: domains with highest lower bound processed first
- **Conv2d CROWN**: Both IBP and CROWN linear bounds implemented. CROWN backward pass uses transposed convolution. Requires `input_shape` to be set via `with_input_shape()` or `set_input_shape()`. For pure Conv2d layers, CROWN matches IBP (both optimal for linear operations). CROWN provides benefit when combined with non-linear layers (e.g., Conv2d → ReLU).
- **Conv1d CROWN**: Both IBP and CROWN linear bounds implemented. CROWN backward pass uses 1D transposed convolution (`conv1d_transpose`). Requires `input_length` to be set via `with_input_length()` or `set_input_length()`. Same properties as Conv2d: CROWN matches IBP for pure Conv1d, provides benefit with non-linear layers.
- **N-D Batched CROWN Limitation**: CROWN/α-CROWN do not work on full transformer blocks with N-D batched inputs. LinearBounds assume fully-connected (flattened) operations where weight matrices operate on all elements. Transformer Linear layers operate per-position (last dimension only), causing dimension mismatches during backward propagation. For input [batch, seq, hidden] = [1, 4, 384] = 1536 elements with Linear weight [384, 384], CROWN backward pass expects [1536, 1536] @ [384, 384] which fails. Use IBP for full blocks (sound but loose) or compositional verification for tighter bounds.
- **Compositional Verification**: `verify_block_compositional()` bounds attention and MLP subgraphs independently, then composes with explicit residual handling. This provides diagnostic info on where bound explosion occurs. Current finding: Attention delta is tight (1.4e4 width), but MLP delta is very loose (1.28e9 width). The MLP's LayerNorm + Linear + GELU + Linear path causes the explosion due to the 4x intermediate dimension expansion and non-linear activations.
- **MLP CROWN Tightness (1-D)**: Tested CROWN vs IBP on 1-D MLP-style networks (simulating single position). Key findings:
  - **Linear → GELU → Linear (no LayerNorm)**: CROWN provides 1.2x–2x tighter bounds than IBP
  - **LayerNorm → Linear → GELU → Linear**: CROWN's advantage diminishes to ~1.2x for tight inputs, can be slightly looser (0.98x) for wider inputs
  - **Root cause**: LayerNorm's non-convex linearization introduces error that compounds through the network
  - **Implication**: For MLP bound tightening, consider bypassing LayerNorm in CROWN or using specialized LayerNorm bounds. Per-position CROWN (flattening N-D to 1-D, running CROWN, reshaping back) is mathematically sound for position-independent MLPs.
- **Per-Position CROWN for MLP**: `GraphNetwork::propagate_crown_per_position()` runs CROWN separately on each position for N-D inputs. This exploits the position-independence of transformer MLPs. For input [batch, seq, hidden], it iterates over batch×seq positions, runs CROWN on [hidden] for each, and combines results. Results on Whisper block 0 with ε=0.01:
  - **IBP MLP delta width**: 1.28e9
  - **CROWN MLP delta width**: 6.42e8
  - **Improvement**: 2.00x tighter bounds
  - `verify_block_compositional_crown()` uses this for compositional verification: IBP for attention (already tight) + per-position CROWN for MLP.
- **gamma-gpu Accelerated Operations**: The `gamma-gpu` crate provides Rayon-parallelized implementations:
  - `AcceleratedDevice::linear_ibp()`: Parallel linear layer IBP (1.2-1.4x speedup at seq≥64)
  - `AcceleratedDevice::matmul_ibp()`: Parallel batched matmul IBP (11-13x speedup)
  - `AcceleratedDevice::attention_ibp()`: Full attention IBP using Rayon (matmul + softmax chain)
  - `AcceleratedDevice::crown_per_position_parallel()`: Parallel per-position CROWN (1.9-9.5x speedup)
  - At seq=64, parallel per-position CROWN achieves 9.5x speedup (1.5s → 163ms) for MLP verification.
- **WgpuDevice GPU Compute**: Cross-platform GPU compute via WebGPU (wgpu crate v24):
  - `WgpuDevice::linear_ibp()`: WGSL compute shader for linear layer IBP
  - `WgpuDevice::matmul_ibp()`: WGSL compute shader for interval batched matmul IBP (attention QK^T and probs@V)
  - `WgpuDevice::softmax_ibp()`: Two-pass WGSL compute shader (reduce: max/exp/sum, apply: Auto-LiRPA bounds formula)
  - `WgpuDevice::attention_ibp()`: Chained attention computation (Q@K^T → scale → softmax → probs@V)
  - Backends: Metal (macOS), Vulkan (Linux/Windows), DX12 (Windows)
  - Buffer pooling: Reuses GPU buffers across calls, eliminating per-call allocation overhead
  - **Attention IBP benchmark (Q@K^T + softmax + probs@V, release mode):**
    - 1×6 heads, seq=64, dim=64: 4.3ms (~5.7k elements/ms)
    - 1×6 heads, seq=128, dim=64: 4.8ms (~10k elements/ms)
    - 1×6 heads, seq=256, dim=64: 6.8ms (~15k elements/ms)
    - 1×6 heads, seq=512, dim=64: 19.5ms (~10k elements/ms)
  - **MatMul IBP benchmark (Q @ K^T, release mode, includes readback):**
    - 2×2 heads, seq=64, dim=64: 0.22x (GPU slower)
    - 2×4 heads, seq=128, dim=64: 0.79x (GPU slower)
    - 1×6 heads, seq=256, dim=64: 2.26x
    - 1×6 heads, seq=512, dim=64: 4.11x
  - **Multi-layer IBP benchmark (MLP 384→1536→384, release mode):**
    - 4×64 (256 positions): 1.25x GPU faster
    - 4×128 (512 positions): 2.55x GPU faster
    - 4×256 (1024 positions): 4.08x GPU faster
    - 1×1500 (Whisper full sequence): **4.60x GPU faster**
  - Single-layer IBP benchmark: GPU faster at [512, 512]+ (1.19x), CPU faster for smaller matrices
  - Key insight: GPU excels when batch × seq is large (≥256 positions)
  - **GPU vs CPU Full Attention IBP benchmark (non-fused, release mode):**
    - 1×2 heads, seq=16, dim=64: GPU 3.9ms, CPU 0.1ms → 0.03x (CPU faster)
    - 1×2 heads, seq=32, dim=64: GPU 4.0ms, CPU 0.3ms → 0.07x (CPU faster)
    - 1×4 heads, seq=64, dim=64: GPU 4.1ms, CPU 1.6ms → 0.39x (CPU faster)
    - 1×6 heads, seq=128, dim=64: GPU 4.4ms, CPU 8.2ms → **1.85x** (GPU faster, crossover)
    - 1×6 heads, seq=256, dim=64: GPU 6.0ms, CPU 32ms → **5.33x**
    - 1×6 heads, seq=384, dim=64: GPU 12.3ms, CPU 74ms → **6.07x** (Whisper encoder)
    - 1×6 heads, seq=512, dim=64: GPU 17.5ms, CPU 143ms → **8.15x**
  - **Fused GPU Attention IBP** (`attention_ibp_fused`): Chains transpose→matmul→scale→softmax→matmul in single submission without intermediate host readbacks
  - **Fused vs Non-fused GPU benchmark (release mode):**
    - seq=16: 2.90x speedup (3.9ms → 1.3ms)
    - seq=64: 2.84x speedup (4.1ms → 1.4ms)
    - seq=128: 2.83x speedup (4.4ms → 1.6ms)
    - seq=384: 4.62x speedup (9.9ms → 2.1ms)
    - seq=512: 2.76x speedup (18ms → 7ms)
  - **Fused GPU vs CPU benchmark (release mode):**
    - 1×4 heads, seq=64, dim=64: GPU 1.5ms, CPU 1.6ms → **1.06x** (GPU crossover)
    - 1×6 heads, seq=128, dim=64: GPU 1.6ms, CPU 8.0ms → **5.14x**
    - 1×6 heads, seq=256, dim=64: GPU 3.0ms, CPU 31ms → **10.5x**
    - 1×6 heads, seq=384, dim=64: GPU 4.3ms, CPU 75ms → **17.5x** (Whisper encoder)
    - 1×6 heads, seq=512, dim=64: GPU 7.1ms, CPU 135ms → **19.0x**
  - **Fused attention crossover point**: seq ~64 (moved from ~100 with non-fused)
- **BoundedTensor Operations**: The gamma-tensor crate provides:
  - `scale()`: Scalar multiplication with correct bound handling for negative scalars
  - `transpose(&[usize])`: General transpose with arbitrary permutation (e.g., `[0, 2, 1, 3]` for attention reshape)
  - `transpose_last_two()`: Transposes last two dimensions for attention (K → K^T)
- **GPU Verification Pipeline**: `WhisperModel::verify_block_compositional_gpu()` integrates GPU acceleration:
  - Adaptive dispatch based on sequence length (GPU for seq≥64)
  - `attention_ibp_gpu()` extracts weights and runs: LayerNorm → Q/K/V projection → reshape → transpose → GPU fused attention → transpose → reshape → output projection
  - Uses `AcceleratedDevice::crown_per_position_parallel()` for MLP bounds
  - `get_layer_norm_weights()` and `get_linear_weights()` extract attention weights from ONNX model
- **Multi-Block Sequential Verification**: `WhisperModel::verify_encoder_sequential()` and `verify_full_encoder()`:
  - Chains encoder blocks: output of block N → input of block N+1
  - Optional stem (preprocessing) and ln_post (final LayerNorm)
  - Returns `MultiBlockDetails` with per-block timing and bound widths
  - **IBP Overflow Limitation**: IBP bounds compound exponentially through blocks. With ε=0.01 on Whisper-tiny:
    - Block 0: output width ~6.4e8 (loose but finite)
    - Block 1: output width ~1.1e19 (explosion)
    - Blocks 2+: overflow to inf/nan
  - For multi-block verification with finite bounds, requires either:
    - Much smaller epsilon (ε << 0.01)
    - Per-block CROWN instead of IBP (tighter but slower)
    - Future: β-CROWN with branch-and-bound for complete verification
