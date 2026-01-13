# Design: True Analytic α-CROWN Gradients

**Issue**: #8
**Date**: 2026-01-06
**Status**: Gradient formula fixed in #518; now matches SPSA performance but target improvement NOT achieved

## Benchmark Results (Worker #518 - After Gradient Fix)

**Test Suite** (small network, 30 iterations):
| Method | Width | Improvement |
|--------|-------|-------------|
| CROWN baseline | 0.344598 | - |
| SPSA | 0.344536 | +0.02% |
| Analytic (local) | 0.344598 | +0.00% |
| AnalyticChain | 0.344531 | +0.02% |

**cifar10_resnet 2b** (5 instances, 20s timeout):
- SPSA: 0% verified (~560 avg domains)
- AnalyticChain: 0% verified (~570 avg domains)

### What Was Fixed (Worker #518)

The gradient formula in `compute_chain_rule_gradients` and `compute_graph_chain_rule_gradients` had a bug:

**Before (wrong):**
```rust
if a_ji > 0.0 {
    grad_i += a_ji * u;  // Used upper bound u
}
```

**After (correct):**
```rust
if a_ji > 0.0 {
    grad_i += a_ji * l;  // Use lower bound l
}
```

**Explanation:** For the lower relaxation `y >= α*x` with `x ∈ [l, u]` where `l < 0 < u`:
- The contribution to the lower bound is `A[j,i] * α * min(x) = A[j,i] * α * l`
- Gradient `∂bound/∂α = A[j,i] * l` (not `u`)

### Root Cause Analysis (Updated)

The gradient formula fix brought AnalyticChain to parity with SPSA (+0.02%). However, both methods show only marginal improvement (~0.02%) which doesn't translate to verified properties on hard instances.

The fundamental limitations are:
1. α-CROWN optimization provides only marginal bound tightening on deep networks
2. For DAG models like ResNet, the optimization is even less effective
3. The >20% target requires β-CROWN with proper branching, not better gradients

### What Still Needs Work

1. The current formula uses `l` (pre-ReLU lower bound) as an approximation for the true input contribution `A_upstream @ x_lower`. This is correct for single-ReLU networks but may be imprecise for deeper networks.
2. Consider implementing full backward pass to compute exact input contribution for each ReLU
3. Focus on β-CROWN improvements rather than α-CROWN gradient methods for hard benchmarks

## Problem

Current `GradientMethod::Analytic` uses per-ReLU local gradients:
```rust
gradient[i] += la;  // where la = bounds.lower_a[[j, i]]
```

This captures only the **direct** effect of α_i on the immediately following layer's coefficients. It misses the **indirect** effects through all downstream layers.

## Solution: Full Chain-Rule Gradients

For a network with layers [Linear₁, ReLU₁, Linear₂, ReLU₂, ...], the backward pass computes:
```
A₀ = I (identity at output)
A₁ = A₀ @ W₁ᵀ
A₂ = A₁ @ diag(mask₁)  // mask depends on α₁
A₃ = A₂ @ W₂ᵀ
A₄ = A₃ @ diag(mask₂)  // mask depends on α₂
...
A_final = accumulated through all layers
```

The output lower bound is: `output_lower = A_final @ x_lower + b_final`

### Gradient Formula

For unstable neuron i in ReLU layer k:
```
∂(output_lower)/∂α_k[i] = A_downstream[*, i] · (A_upstream[i, *] @ x_lower)
```

Where:
- `A_downstream[*, i]` = column i of the accumulated A matrix from output to ReLU k (before ReLU k is applied)
- `A_upstream[i, *]` = row i of the accumulated A matrix from ReLU k to input

### Simplified Formula

For a single output dimension j:
```
∂(output_lower[j])/∂α_k[i] = A_to_relu[j, i] × input_contribution[i]
```

Where:
- `A_to_relu[j, i]` = the coefficient from output j back to neuron i (before ReLU k)
- `input_contribution[i]` = how neuron i's pre-ReLU value affects based on x_lower

## Implementation Plan

### Step 1: Store Intermediate A Matrices

Modify `propagate_alpha_crown_single_pass` to return intermediate bounds:

```rust
struct AlphaCrownIntermediate {
    /// A matrix at each ReLU layer (before ReLU applied)
    a_at_relu: Vec<Array2<f32>>,
    /// Final linear bounds
    final_bounds: LinearBounds,
}

fn propagate_alpha_crown_with_intermediates(
    &self,
    input: &BoundedTensor,
    layer_bounds: &[BoundedTensor],
    alpha_state: &AlphaState,
) -> Result<AlphaCrownIntermediate> {
    let mut a_at_relu = Vec::new();
    let mut linear_bounds = LinearBounds::identity(output_dim);

    for (i, layer) in self.layers.iter().enumerate().rev() {
        match layer {
            Layer::ReLU(r) => {
                // Store A before this ReLU
                a_at_relu.push(linear_bounds.lower_a.clone());

                // Apply ReLU
                let alpha = alpha_state.get_alpha(relu_idx).unwrap();
                let (new_bounds, _) = r.propagate_linear_with_alpha(...);
                linear_bounds = new_bounds;
            }
            // ... other layers
        }
    }

    // Reverse to get forward order
    a_at_relu.reverse();

    Ok(AlphaCrownIntermediate { a_at_relu, final_bounds: linear_bounds })
}
```

### Step 2: Compute Analytic Gradients

```rust
fn compute_chain_rule_gradients(
    &self,
    input: &BoundedTensor,
    layer_bounds: &[BoundedTensor],
    alpha_state: &AlphaState,
    intermediate: &AlphaCrownIntermediate,
) -> Vec<Array1<f32>> {
    let mut gradients = Vec::new();
    let x_lower = input.flatten().lower;

    for (relu_idx, a_at_relu) in intermediate.a_at_relu.iter().enumerate() {
        let pre_bounds = &layer_bounds[relu_layer_indices[relu_idx]];
        let alpha = &alpha_state.alphas[relu_idx];
        let n_neurons = alpha.len();

        let mut grad = Array1::zeros(n_neurons);

        // For each unstable neuron
        for i in 0..n_neurons {
            if !is_unstable(pre_bounds, i) {
                continue;
            }

            // Sum contribution from all output dimensions
            for j in 0..output_dim {
                // A_to_relu[j, i] is coefficient from output j to neuron i
                let a_ji = a_at_relu[[j, i]];

                // Only contribute when using lower relaxation (A >= 0)
                if a_ji > 0.0 {
                    // input_contribution: how x_lower affects output through this neuron
                    // This is computed by running a second backward pass from neuron i to input
                    // For simplicity, approximate with the neuron's pre-ReLU contribution
                    let contrib = compute_input_contribution(i, relu_idx, layer_bounds, x_lower);
                    grad[i] += a_ji * contrib;
                }
            }
        }

        gradients.push(grad);
    }

    gradients
}
```

### Step 3: Integrate into α-CROWN Optimization

Add new gradient method:
```rust
pub enum GradientMethod {
    Spsa,
    Fd,
    Analytic,        // Local gradients (current)
    AnalyticChain,   // Full chain-rule (new)
}
```

In the optimization loop:
```rust
GradientMethod::AnalyticChain => {
    let intermediate = self.propagate_alpha_crown_with_intermediates(
        input, &layer_bounds, &alpha_state
    )?;
    self.compute_chain_rule_gradients(
        input, &layer_bounds, &alpha_state, &intermediate
    )
}
```

## Complexity Analysis

| Method | Passes per Iteration | Complexity |
|--------|---------------------|------------|
| SPSA | 2 * samples | O(1) amortized |
| FD | 2 * num_alphas | O(N) |
| Analytic (local) | 1 | O(1) |
| **AnalyticChain** | **1 + gradient** | **O(1)** |

AnalyticChain requires:
- 1 forward pass (already done for IBP bounds)
- 1 backward pass with intermediate storage
- 1 gradient accumulation pass (linear in network depth)

## Expected Impact

True analytic gradients should:
1. Match α,β-CROWN's gradient quality (they use autograd)
2. Enable faster convergence (better gradients = fewer iterations)
3. Improve cifar10_resnet from 4.2% → target >20%

## Testing Strategy

1. **Unit test**: Small 3-layer network, verify gradient against finite differences
2. **Regression test**: Run cifar10_resnet benchmark, compare to SPSA baseline
3. **Convergence test**: Track bound improvement per iteration with each method

## Files to Modify

1. `crates/gamma-propagate/src/network.rs`:
   - Add `AlphaCrownIntermediate` struct
   - Add `propagate_alpha_crown_with_intermediates` method
   - Add `compute_chain_rule_gradients` method
   - Update `propagate_alpha_crown_with_config_and_engine` to use new method

2. `crates/gamma-propagate/src/alpha_state.rs` (if needed):
   - Add `GradientMethod::AnalyticChain` variant

3. `crates/gamma-cli/src/main.rs`:
   - Add CLI option for `analytic-chain` gradient method
