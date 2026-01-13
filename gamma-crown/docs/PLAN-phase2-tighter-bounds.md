# PLAN: Phase 2 Tighter Bounds

**Status:** COMPLETE (All tasks done, targets exceeded)
**Created:** 2026-01-01
**Updated:** 2026-01-01 (Iteration #223)
**Goal:** Get tight (non-saturated) bounds on LLMs

---

## SUMMARY OF FINDINGS (2025-01-01)

**Key Discovery:** FFN (SwiGLU) is the main bound explosion source, NOT attention.

- **Q@K^T zonotope tightening works excellently**: < 1e-4 width per block
- **Block-wise verification achieves < 1e10 per block**: Target met
- **FFN (SwiGLU) causes ~36x growth per block**: New bottleneck identified
- **Full network verification still saturates**: Due to cumulative FFN explosion

See: `reports/main/bound_analysis_2025-01-01.md` for detailed analysis.

---

## MANAGER DIRECTIVE (Worker #219+): Fix FFN Explosion

**STATUS:** Block-wise verification WORKS. Full network FAILS.

The path forward is clear: **Tighten FFN bounds via zonotope SiLU approximation.**

### TASK 4: Zonotope SiLU Approximation (NEW - HIGHEST PRIORITY)

**Goal:** Make SiLU preserve zonotope form so SwiGLU correlation is tracked.

**The math:**
SiLU(x) = x * sigmoid(x) ≈ affine approximation near center point:
- SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
- For x ∈ [l, u], bound the derivative and use linear relaxation

**Implementation:**
1. In `crates/gamma-propagate/src/lib.rs`, find `SiLULayer` zonotope propagation
2. Instead of falling back to IBP, use affine approximation:
   ```rust
   fn propagate_zonotope(&self, input: &Zonotope) -> Zonotope {
       // Current: falls back to IBP
       // New: affine approximation
       let center = input.center();
       let slope = silu_derivative(center);  // σ(c) + c·σ(c)·(1-σ(c))
       let bias = silu(center) - slope * center;
       input.affine_transform(slope, bias)
   }
   ```

**Expected impact:**
- FFN growth: 36x → ~2-5x (preserves zonotope correlation)
- Full network: ±inf → potentially finite

**Test:**
```bash
./target/release/gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --method crown --epsilon 0.001
# Target: Any finite bounds (not ±inf)
```

### Success Criteria Update

| Task | Metric | Status |
|------|--------|--------|
| Q@K^T tightening | < 1e-4 per block | ✅ DONE |
| Block-wise verification | < 1e10 per block | ✅ DONE |
| FFN SiLU tightening | Full network finite | ✅ DONE (Iter #220-222) |
| Full network | Qwen3-0.6B < ±1e10 | ✅ EXCEEDED: ~4.4e5 (Iter #223) |

---

## MANDATORY: Test ALL Models After Each Improvement

**Every improvement MUST be tested across ALL models.** Track bound tightness over time.

### Benchmark Suite (Run After Each Change)

```bash
# Run full benchmark suite
./target/release/gamma verify models/whisper-tiny/model.safetensors --native --method crown --epsilon 0.001 --json > /tmp/whisper-tiny.json
./target/release/gamma verify models/whisper-small/model.safetensors --native --method crown --epsilon 0.001 --json > /tmp/whisper-small.json
./target/release/gamma verify models/whisper-medium/model.safetensors --native --method crown --epsilon 0.001 --json > /tmp/whisper-medium.json
./target/release/gamma verify models/whisper-large-v3/model.safetensors --native --method crown --epsilon 0.001 --json > /tmp/whisper-large-v3.json
./target/release/gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --method crown --epsilon 0.001 --json > /tmp/qwen3-0.6b.json
```

### Bound Tightness Tracking

**BEFORE zonotope SiLU (baseline):**

| Model | Params | Max Bound Width | Status |
|-------|--------|-----------------|--------|
| whisper-tiny | 39M | ~12 | ✅ Finite |
| whisper-small | 244M | ~14 | ✅ Finite |
| whisper-medium | 769M | ~34 | ✅ Finite |
| whisper-large-v3 | 1.5B | ~220 | ✅ Finite |
| Qwen3-0.6B | 600M | ±inf | ❌ Infinity |

**AFTER zonotope SiLU (Iteration #223, 2026-01-01):**

| Model | Params | Actual Width | Improvement |
|-------|--------|--------------|-------------|
| whisper-tiny | 39M | ~4.2 | 3x tighter |
| whisper-small | 244M | ~7.5 | 2x tighter |
| whisper-medium | 769M | ~8.9 | 4x tighter |
| whisper-large-v3 | 1.5B | ~177 | ~1.2x tighter |
| Qwen3-0.6B | 600M | ~444,000 | ∞ → finite ✅ |

### Report Format

After each improvement, create report: `reports/main/bound_improvement_YYYY-MM-DD.md`

```markdown
# Bound Improvement Report - [Description]
Date: YYYY-MM-DD
Iteration: #NNN

## Change
[What was changed]

## Results
| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| whisper-tiny | X | Y | Z% tighter |
...

## Analysis
[Why bounds improved or didn't]
```

**DO NOT merge changes that make ANY model's bounds WORSE.**

---

## THE PROBLEM (UPDATED)

~~Qwen3-0.6B bounds saturate to ±1.7e38 because:~~
~~1. Q@K^T attention causes **1596x bound growth** per layer~~
~~2. IBP treats Q and K as independent (they're not - both come from same input)~~
~~3. 28 decoder layers × 1596x = infinity~~

**Actual root cause:**
1. **Q@K^T is now tight** (< 1e-4 per block via zonotope tightening)
2. **FFN (SwiGLU) causes ~36x growth per block** (MulBinary of two intervals)
3. **Linear projections amplify bounds** (ffn_down: ~58x growth)
4. 28 layers × cumulative FFN explosion = infinity

whisper-large-v3 works (±65 to ±111) because it's an encoder with only 32 blocks and no causal attention accumulation.

---

## YOUR TASKS (in order)

### Task 1: McCormick Bilinear CROWN ✅ COMPLETE

**Priority:** HIGHEST
**Estimated effort:** 2-3 AI commits
**Status:** DONE - McCormick implemented for both MatMul and MulBinary

Implement McCormick relaxation for bilinear Q@K^T.

**FINDING:** McCormick was already implemented. The issue was not missing McCormick - it was IBP explosion through FFN layers before McCormick could help. Q@K^T zonotope tightening achieves < 1e-4 width per block.

**The math:**
For z = x * y where x ∈ [x_l, x_u], y ∈ [y_l, y_u]:
```
Lower bounds (take max):
  z ≥ x_l·y + x·y_l - x_l·y_l
  z ≥ x_u·y + x·y_u - x_u·y_u

Upper bounds (take min):
  z ≤ x_l·y + x·y_u - x_l·y_u
  z ≤ x_u·y + x·y_l - x_u·y_l
```

This gives LINEAR bounds on z in terms of x and y → can propagate with CROWN.

**Implementation steps:**

1. **Add `BilinearLayer` to gamma-core:**
```rust
// crates/gamma-core/src/lib.rs
pub enum LayerType {
    // ... existing
    Bilinear { left_input: String, right_input: String },
}
```

2. **Add `BilinearCrownLayer` to gamma-propagate:**
```rust
// crates/gamma-propagate/src/lib.rs
pub struct BilinearCrownLayer {
    pub left_input_idx: usize,
    pub right_input_idx: usize,
}

impl BilinearCrownLayer {
    pub fn crown_backward(
        &self,
        output_bounds: &BatchedLinearBounds,
        left_bounds: &BoundedTensor,   // Q bounds
        right_bounds: &BoundedTensor,  // K bounds
    ) -> (BatchedLinearBounds, BatchedLinearBounds) {
        // McCormick relaxation here
    }
}
```

3. **Update attention generation in native.rs:**
```rust
// crates/gamma-onnx/src/native.rs
// In generate_gguf_attention(), mark Q@K^T as Bilinear instead of MatMul
```

4. **Test:**
```bash
./target/release/gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf \
  --native --method crown --epsilon 0.001

# SUCCESS: bounds < ±1e10 (not ±1.7e38)
```

### Task 2: Layer-by-Layer Verification ✅ COMPLETE

**Priority:** HIGH
**Estimated effort:** 1-2 AI commits
**Prerequisite:** Task 1 working OR as independent path
**Status:** DONE - `--layer-by-layer` and `--block-wise` flags implemented

For Qwen3-32B (64 layers), full propagation times out. Stream instead.

**FINDING:** Block-wise verification (`--block-wise`) achieves < 1e10 bounds per block (target met).

```bash
# Test command
./target/release/gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --method crown --epsilon 0.001 --block-wise

# Result: layer0 = 5.6e4, layer27 = 1.5e9 (all < 1e10 target)
```

**Implementation:**

1. **Add CLI flag:**
```rust
// crates/gamma-cli/src/main.rs
#[arg(long)]
layer_by_layer: bool,
```

2. **Add streaming verification:**
```rust
// crates/gamma-propagate/src/lib.rs
pub fn verify_layer_by_layer(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    epsilon: f32,
) -> Vec<LayerResult> {
    let mut current_bounds = input.clone();
    let mut results = vec![];

    for (i, block) in graph.transformer_blocks().enumerate() {
        let output = block.propagate_ibp(&current_bounds);
        results.push(LayerResult {
            layer: i,
            bound_width: output.width(),
        });
        current_bounds = output;
    }
    results
}
```

3. **Test:**
```bash
./target/release/gamma verify ~/Models/Qwen3-32B-Q8_0.gguf \
  --native --method ibp --epsilon 0.001 --layer-by-layer

# SUCCESS: completes (even if bounds saturate)
```

### Task 3: Zonotope Research ✅ COMPLETE

**Priority:** MEDIUM (research)
**Estimated effort:** 3-5 AI commits
**Status:** DONE - Q@K^T zonotope tightening works, FFN is the new bottleneck

Track correlations between Q and K via zonotope error symbols.

**FINDINGS:**
1. Zonotope Q@K^T tightening implemented and working (< 1e-4 width per block)
2. Zonotopes support: Linear, Add, Reshape, Tile, Transpose, LayerNorm (affine)
3. Zonotopes DON'T support: SiLU/GELU, Softmax, element-wise multiply
4. FFN (SwiGLU = silu(gate) * up) breaks zonotope form → falls back to IBP → explosion

**New research direction:** Polynomial zonotopes for element-wise multiplication in FFN.

**Start with research report:**
1. Read arxiv:2002.06622 Section 4 carefully
2. Document how Bonaert handles attention
3. Prototype zonotope type in gamma-tensor
4. Write report: `reports/main/zonotope_research_YYYY-MM-DD.md`

---

## SUCCESS CRITERIA

| Task | Metric | Current | Target | Status |
|------|--------|---------|--------|--------|
| McCormick bilinear | Qwen3-0.6B bound width (per block) | ~1e3 to ~2.5e9 | < ±1e10 | ✅ ACHIEVED |
| Layer-by-layer | Qwen3-32B completes | finishes in 7.8s | finishes | ✅ ACHIEVED |
| Zonotopes | Q@K^T tightening | < 1e-6 per block | tight | ✅ ACHIEVED |
| FFN tightening | Full network bounds | ~4.4e5 | < ±1e10 | ✅ ACHIEVED (Iter #220-223) |

**Phase 2 COMPLETE** - All targets met or exceeded. See `reports/main/verification_audit_2026-01-01_iter223.md` for details.

---

## DO NOT

- Work on CNN/MNIST/CIFAR
- Add new model formats
- Optimize performance before correctness
- Skip unit tests for new layer types

---

## COMMANDS

```bash
# Build
cargo build --release

# Test current state
./target/release/gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --method ibp --epsilon 0.001

# Run tests
cargo test -p gamma-propagate

# Check what's there
ls crates/gamma-propagate/src/
```

---

## FILES TO READ FIRST

1. `crates/gamma-propagate/src/lib.rs` - See existing CROWN implementation
2. `crates/gamma-onnx/src/native.rs` - See `generate_gguf_attention()`
3. `reports/main/attention_bounds_research_2025-12-31.md` - Q@K^T research

GO.
