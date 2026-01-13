# γ-CROWN Feature Request from tRust

**From:** tRust (rustc fork with integrated verification)
**To:** γ-CROWN team (git@github.com:dropbox/dMATH/gamma-crown.git)
**Date:** 2024-12-31

---

## What is tRust?

tRust is a fork of rustc that integrates formal verification into the compiler. We verify Rust code meets its specifications at compile time. For neural networks embedded in Rust, we need to verify NN-specific properties: robustness, monotonicity, bounds.

**Our thesis:** AI systems need verified AI components. A classifier that's "probably robust" isn't good enough for safety-critical applications.

---

## Why γ-CROWN Matters to Us

Neural networks are increasingly embedded in Rust applications:
- Autonomous systems (robotics, drones, vehicles)
- Financial systems (fraud detection, trading)
- Medical systems (diagnosis, treatment recommendation)
- Security systems (malware detection, intrusion detection)

These systems need guarantees:
- **Adversarial robustness:** Small input perturbations don't change output
- **Monotonicity:** Increasing input → increasing output (for interpretability)
- **Output bounds:** Outputs stay in valid range
- **Fairness:** Sensitive attributes don't affect decisions

SMT solvers can't verify NNs efficiently. γ-CROWN's bound propagation can.

---

## Feature Requests (Priority Order)

### 1. Rust Native API (CRITICAL)

**Current γ-CROWN:** Python library

**Requested:** Pure Rust library callable from tRust

```rust
use gamma_crown::{Verifier, Model, Norm, RobustnessSpec};

let model = Model::from_onnx("classifier.onnx")?;
let spec = RobustnessSpec {
    epsilon: 0.1,
    norm: Norm::Linf,
};

let result = Verifier::verify_robustness(&model, &input, &spec)?;
match result {
    Verified => println!("Certified robust"),
    Counterexample(adv) => println!("Adversarial: {:?}", adv),
    Unknown(bounds) => println!("Bounds: {:?}", bounds),
}
```

**Why:**
- tRust is a Rust compiler; can't shell out to Python
- No Python runtime dependency
- Rust type safety for spec construction

---

### 2. Specification Attributes (CRITICAL)

tRust uses attributes for specs. γ-CROWN should understand:

```rust
// Adversarial robustness
#[certified_robust(epsilon = 0.1, norm = Linf)]
fn classify(model: &Model, input: &[f32; 784]) -> u8 {
    model.forward(input).argmax()
}

// Local robustness (around specific input)
#[certified_robust(epsilon = 0.1, norm = L2, at = "input")]
fn predict(model: &Model, input: Features) -> f32 {
    model.forward(input.as_slice())[0]
}

// Monotonicity
#[monotonic(input.age => output.risk)]  // Older → higher risk
fn predict_risk(model: &Model, input: PatientData) -> RiskScore {
    model.forward(input.to_tensor())
}

// Output bounds
#[output_bounded(min = 0.0, max = 1.0)]
fn predict_probability(model: &Model, input: &[f32]) -> f32 {
    model.forward(input).sigmoid()
}

// Fairness (independence from sensitive attribute)
#[independent(input.gender => output)]
#[independent(input.race => output)]
fn credit_score(model: &Model, input: Applicant) -> Score {
    model.forward(input.features())
}

// Lipschitz continuity
#[lipschitz(constant = 1.0, norm = L2)]
fn smooth_predict(model: &Model, input: &[f32]) -> f32 {
    model.forward(input)[0]
}
```

---

### 3. Model Loading (HIGH)

Support common model formats:

```rust
// ONNX (standard interchange)
let model = Model::from_onnx("model.onnx")?;

// PyTorch checkpoint (common in research)
let model = Model::from_pytorch("model.pt")?;

// TensorFlow SavedModel
let model = Model::from_tensorflow("saved_model/")?;

// Inline definition (for small models, testing)
let model = Model::sequential(&[
    Layer::linear(784, 256),
    Layer::relu(),
    Layer::linear(256, 128),
    Layer::relu(),
    Layer::linear(128, 10),
]);

// From Rust tensor library (burn, candle, tch)
let model = Model::from_burn(&burn_model)?;
```

**Architecture validation at load time:**
```rust
let model = Model::from_onnx("model.onnx")?;
// Checks:
// - Supported layer types
// - Compatible activation functions
// - Input/output shapes match spec
// - No unsupported operations
```

---

### 4. Verification Results (HIGH)

Clear, actionable results:

```rust
enum VerifyResult {
    // Property holds for all inputs in region
    Verified {
        bounds: Bounds,  // Computed output bounds
        time: Duration,
    },

    // Found adversarial example
    Counterexample {
        original_input: Tensor,
        adversarial_input: Tensor,
        perturbation_norm: f32,
        original_output: Tensor,
        adversarial_output: Tensor,
    },

    // Can't decide (bounds too loose)
    Unknown {
        lower_bound: Tensor,  // Guaranteed lower
        upper_bound: Tensor,  // Guaranteed upper
        gap: f32,            // How loose
        suggestion: String,  // "Try tighter epsilon" or "Model may be non-robust"
    },

    // Timeout
    Timeout {
        partial_bounds: Bounds,
        explored: f32,  // Fraction of input space bounded
    },
}
```

**Counterexample rendering:**
```
Adversarial example found for classify():

  Original input: [0.12, 0.45, 0.33, ...]
  Prediction: class 7 (confidence 0.94)

  Adversarial input: [0.11, 0.46, 0.32, ...]
  Prediction: class 1 (confidence 0.67)

  Perturbation: L∞ = 0.08 (threshold was 0.10)

  [Visual diff if image data]
```

---

### 5. Integration with Other Backends (MEDIUM)

γ-CROWN handles NN properties. Other properties flow to Z4/Lean:

```rust
#[requires(input.len() == 784)]                    // Z4 checks
#[requires(input.iter().all(|x| *x >= 0.0))]      // Z4 checks
#[certified_robust(epsilon = 0.1, norm = Linf)]    // γ-CROWN checks
#[ensures(result < 10)]                            // Z4 checks
fn classify(model: &Model, input: &[f32; 784]) -> u8 {
    model.forward(input).argmax()
}
```

**Verification flow:**
```
1. tRust parses all specs
2. Z4 checks: input.len() == 784 ✓
3. Z4 checks: input values non-negative ✓
4. γ-CROWN checks: robustness ✓
5. Z4 checks: result < 10 ✓ (given NN has 10 outputs)
6. All pass → compilation succeeds
```

---

### 6. Incremental Verification (MEDIUM)

Don't re-verify unchanged models:

```rust
// First compilation
verify(model_v1, spec) → Verified, cache_key=0x8f3a

// Second compilation (model unchanged, code changed elsewhere)
verify(model_v1, spec) → Cached(0x8f3a), skip verification

// Third compilation (model changed)
verify(model_v2, spec) → Re-verify...
```

**Cache key includes:**
- Model weights hash
- Model architecture hash
- Spec hash
- γ-CROWN version

---

## Supported Architectures

### Must Support (Common in deployment)
- Fully connected (Linear + activation)
- Convolutional (Conv2D, pooling)
- Residual connections (ResNet-style)
- Batch normalization
- ReLU, Sigmoid, Tanh, Softmax

### Should Support (Increasingly common)
- Attention mechanisms (Transformers)
- Recurrent (LSTM, GRU) — harder
- Skip connections (DenseNet-style)

### Stretch Goals
- Graph neural networks
- Mixture of experts
- Dynamic architectures

---

## Performance Targets

| Model Size | Verification Time | Notes |
|------------|-------------------|-------|
| Small (< 10K params) | < 1 second | Direct verification |
| Medium (10K - 1M params) | < 1 minute | May need GPU |
| Large (1M - 100M params) | < 10 minutes | Approximate bounds OK |
| Very large (> 100M params) | Best effort | Statistical guarantees |

**GPU support:**
```rust
let verifier = Verifier::new()
    .device(Device::Cuda(0))  // Use GPU
    .batch_size(1024);        // Parallel bound computation
```

---

## Integration Protocol

### Input from tRust
```rust
struct NNVerificationQuery {
    model: Model,
    spec: NNSpec,
    input_region: InputRegion,  // Where to verify

    timeout: Duration,
    precision: Precision,  // Trade speed for tightness
}

enum NNSpec {
    Robustness { epsilon: f32, norm: Norm },
    Monotonicity { input_dim: usize, output_dim: usize, direction: Direction },
    OutputBounds { min: f32, max: f32 },
    Lipschitz { constant: f32, norm: Norm },
    Independence { sensitive_dims: Vec<usize> },
}
```

### Output to tRust
```rust
struct NNVerificationResult {
    result: VerifyResult,
    bounds_computed: Bounds,
    time: Duration,

    // For caching
    model_hash: Hash,
    spec_hash: Hash,
}
```

---

## Questions for γ-CROWN Team

1. Python → Rust port: rewrite or bindings? (Rewrite preferred for tRust integration)
2. GPU support: CUDA only or also Metal/Vulkan?
3. What architectures are hardest to verify? (We can warn users)
4. Incremental: can you verify "model changed by Δ" efficiently?
5. Approximate verification: what guarantees for large models?

---

## Example: Verified MNIST Classifier

What we want to write:

```rust
use trust_nn::Model;

static MNIST_MODEL: Model = Model::include_onnx!("mnist.onnx");

#[certified_robust(epsilon = 0.1, norm = Linf)]
#[ensures(result < 10)]
pub fn classify_digit(image: &[f32; 784]) -> u8 {
    MNIST_MODEL.forward(image).argmax()
}

// At compile time:
// 1. Load MNIST model (28x28 → 10 classes)
// 2. γ-CROWN verifies: ∀ image, ∀ perturbation with ||p||∞ < 0.1,
//    classify(image) == classify(image + perturbation)
// 3. If verified: compilation succeeds
// 4. If counterexample: compilation fails with adversarial example
```

This gives us a **certified robust classifier** that's guaranteed at compile time.

---

## Contact

tRust issues: https://github.com/dropbox/tRust/issues
This request: `reports/main/feature_requests/GAMMA_CROWN_FEATURE_REQUEST.md`
