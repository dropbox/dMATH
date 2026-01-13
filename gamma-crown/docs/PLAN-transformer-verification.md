# Plan: Proper Transformer Verification

**Goal:** Implement correct transformer verification following Auto-LiRPA's approach, then add improvements from Bonaert et al. (PLDI 2021).

**Principle:** Analyze models as they are (native formats), not force ONNX export.

---

## Phase 1: Fix N-D Batched Linear Bounds (Priority: CRITICAL)

**Problem:** Current CROWN assumes flattened 1-D inputs. Transformers use `[batch, seq, hidden]` shapes. LinearLayer operates per-position (last dim only), but CROWN backward pass expects `[total_elements, total_elements]` Jacobians.

**Auto-LiRPA Solution:** Maintain batch structure, broadcast linear bound coefficients.

### Tasks:

1. **Read and understand current LinearBounds implementation**
   - File: `crates/gamma-propagate/src/lib.rs` (lines 100-200 for LinearBounds struct)
   - Current: `LinearBounds` uses `Array2<f32>` shape `(num_outputs, num_inputs)` - always 2D
   - Current: `concretize()` flattens input to 1D before computing bounds
   - Problem: This flattens `[batch, seq, hidden]` to `[batch*seq*hidden]`, losing structure
   - Workaround exists: `propagate_crown_per_position()` runs CROWN independently per position
   - Why workaround fails: Attention has cross-position interactions (Q@K^T mixes positions)

2. **Implement N-D batched LinearBounds**
   - Keep shape `[batch, seq, hidden]` throughout propagation
   - Backward pass: weight matrix `[out, in]` operates on last dimension only
   - Coefficient matrices remain `[batch, seq, out, in]` not `[total, total]`

3. **Test on single Linear layer with N-D input**
   - Input: `[1, 4, 384]` (batch=1, seq=4, hidden=384)
   - Weight: `[1536, 384]`
   - Expected output shape: `[1, 4, 1536]`
   - Verify CROWN bounds match IBP (for linear-only, they should be identical)

4. **Test on Linear → GELU → Linear (MLP)**
   - This is where CROWN should be tighter than IBP
   - Compare against per-position CROWN results (should match)

---

## Phase 2: Fix Softmax CROWN Bounds (Priority: HIGH)

**Problem:** Current softmax has IBP but CROWN backward may not propagate correctly through the non-linearity.

**Auto-LiRPA Algorithm (interval_propagate):**
```python
# Shift for numerical stability
shift = h_U.max(dim=axis, keepdim=True)
exp_L = exp(h_L - shift)
exp_U = exp(h_U - shift)

# Lower bound: exp(x_L[i]) / (sum(exp_U) - exp_U[i] + exp_L[i])
lower = exp_L / (sum(exp_U) - exp_U + exp_L + eps)

# Upper bound: exp(x_U[i]) / (sum(exp_L) - exp_L[i] + exp_U[i])
upper = exp_U / (sum(exp_L) - exp_L + exp_U + eps)
```

### Tasks:

1. **Verify current softmax IBP matches Auto-LiRPA formula**
   - File: `crates/gamma-propagate/src/layers/softmax.rs`
   - Check the exact formula

2. **Implement softmax CROWN backward pass**
   - This is complex: softmax is not element-wise
   - Need Jacobian of softmax: `d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)`
   - Linear relaxation around a reference point

3. **Test softmax CROWN**
   - Input bounds `[-1, 1]^4`
   - Compare IBP vs CROWN bounds
   - CROWN should be tighter

---

## Phase 3: Fix Attention Bounds (Priority: HIGH)

**Problem:** Attention = `softmax(Q @ K^T / sqrt(d)) @ V`. Both `Q @ K^T` and `probs @ V` are bilinear (both inputs bounded).

**Auto-LiRPA Approach for Bilinear MatMul:**
- When both inputs are bounded, use interval arithmetic on products
- For `A @ B` where both have bounds:
  - Consider all 4 corner combinations: `A_L @ B_L`, `A_L @ B_U`, `A_U @ B_L`, `A_U @ B_U`
  - Take element-wise min/max

### Tasks:

1. **Implement proper bilinear MatMul IBP**
   - File: `crates/gamma-propagate/src/layers/matmul.rs`
   - Handle case where both inputs have bounds
   - Test: `[batch, heads, seq, seq] @ [batch, heads, seq, dim]`

2. **Implement attention IBP as composition**
   - Q projection → K projection → V projection
   - Q @ K^T (bilinear)
   - Scale by 1/sqrt(d)
   - Softmax
   - probs @ V (bilinear)
   - Output projection

3. **Test attention IBP**
   - Input: `[1, 4, 384]`
   - 6 heads, 64 dim per head
   - Verify bounds are sound (sample random inputs, check outputs within bounds)

---

## Phase 4: Full Transformer Block (Priority: HIGH)

### Architecture:
```
input
  │
  ├──────────────────────┐
  │                      │ (residual)
  ▼                      │
LayerNorm                │
  │                      │
Attention                │
  │                      │
  ▼                      │
Add ◄────────────────────┘
  │
  ├──────────────────────┐
  │                      │ (residual)
  ▼                      │
LayerNorm                │
  │                      │
MLP (Linear→GELU→Linear) │
  │                      │
  ▼                      │
Add ◄────────────────────┘
  │
output
```

### Tasks:

1. **Implement residual Add with bounded inputs**
   - When adding bounded tensor to itself (residual), bounds widen
   - `(x + delta).lower = x.lower + delta.lower`
   - `(x + delta).upper = x.upper + delta.upper`

2. **Implement full block IBP**
   - Chain: LayerNorm → Attention → Add → LayerNorm → MLP → Add
   - Track bound widths at each stage

3. **Identify bound explosion source**
   - Run block with ε=0.001 (small)
   - Log bound width after each operation
   - Find which operation causes explosion

4. **Implement block CROWN**
   - Use N-D batched LinearBounds from Phase 1
   - Compose CROWN through the block
   - Compare to IBP

---

## Phase 5: Multi-Block Verification (Priority: MEDIUM)

**Problem:** IBP bounds explode after 2 blocks with ε=0.01.

### Tasks:

1. **Implement tighter per-block bounds**
   - Use CROWN instead of IBP within blocks
   - This should reduce explosion rate

2. **Test sequential blocks**
   - Block 0 → Block 1 → Block 2 → ...
   - Find maximum ε that allows N blocks without overflow

3. **Compare to Auto-LiRPA results**
   - Run Auto-LiRPA on same model with same ε
   - Report bound widths
   - Our implementation should match or be close

---

## Phase 6: Native Model Loading (Priority: MEDIUM)

**Current:** Native loading works (SafeTensors, PyTorch, GGUF) but verification uses ONNX path.

### Tasks:

1. **Connect native model loading to verification pipeline**
   - `NativeModel::to_propagate_network()` exists but may not handle all layers

2. **Handle model-specific architectures**
   - Whisper: Conv1d stem + transformer blocks
   - LLaMA/GPT: Decoder with causal attention
   - BERT: Encoder with bidirectional attention

3. **Test verification on native models**
   - whisper-tiny (SafeTensors): 39M params
   - gemma-2b (GGUF): 2.5B params (may need layer-by-layer)

---

## Success Criteria

1. **N-D batched CROWN works** on `[batch, seq, hidden]` inputs
2. **Full transformer block** verifies with finite bounds at ε=0.001
3. **Bound widths match Auto-LiRPA** within 10% on same model/epsilon
4. **Native models verify** without ONNX export

---

## Files to Modify

- `crates/gamma-propagate/src/lib.rs` - N-D batched LinearBounds (lines 100-200)
- `crates/gamma-propagate/src/lib.rs` - Softmax layer (search `impl BoundPropagation for SoftmaxLayer`)
- `crates/gamma-propagate/src/lib.rs` - MatMul layer (search `MatMulLayer`)
- `crates/gamma-onnx/src/native.rs` - Native model verification
- `crates/gamma-cli/src/main.rs` - CLI integration

---

## Worker Directive: Phase 1 Implementation

**Start here.** The critical change is replacing `LinearBounds` with a batched variant.

### Current Code Pattern (broken for N-D):
```rust
pub struct LinearBounds {
    pub lower_a: Array2<f32>,  // shape (num_outputs, num_inputs) - FLAT
    pub lower_b: Array1<f32>,  // shape (num_outputs,)
    pub upper_a: Array2<f32>,
    pub upper_b: Array1<f32>,
}
```

### Target Code Pattern (batched):
```rust
pub struct BatchedLinearBounds {
    // For input shape [batch, seq, hidden_in] -> output [batch, seq, hidden_out]
    // Coefficients broadcast over batch dims, operate on last dim only
    pub lower_a: ArrayD<f32>,  // shape [...batch_dims, hidden_out, hidden_in]
    pub lower_b: ArrayD<f32>,  // shape [...batch_dims, hidden_out]
    pub upper_a: ArrayD<f32>,
    pub upper_b: ArrayD<f32>,
    pub batch_dims: Vec<usize>,  // track which dims are batch dims
}
```

### Key Auto-LiRPA Files to Study:
1. `auto_LiRPA/operators/linear.py` - `BoundLinear.bound_backward()`
2. `auto_LiRPA/bound_ops.py` - `Bound.clamp_interim_bounds()`
3. `auto_LiRPA/operators/base.py` - How `lA` (lower bound coefficients) flow backward

### Auto-LiRPA's Key Insight:
- `lA` has shape `[C, B, *output_shape, *input_shape]` where:
  - C = specification dimension (often 1)
  - B = batch dimension
  - output_shape = shape of layer output
  - input_shape = shape of layer input
- Linear layer backward: `new_lA = lA @ weight` broadcasts correctly
- Never flatten to giant matrix

---

## References

- Auto-LiRPA: https://github.com/Verified-Intelligence/auto_LiRPA
- Bonaert et al. PLDI 2021: https://doi.org/10.1145/3453483.3454056
- Current design: `docs/DESIGN.md`
