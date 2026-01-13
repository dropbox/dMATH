# Root Cause Analysis: cifar10_resnet Verification Gap

**Date:** 2026-01-07
**Worker:** #542
**Issue:** #9 - Improve CROWN bounds for ResNet verification (cifar10_resnet gap)

## Summary

The 40-percentage-point gap between γ-CROWN (35%) and α,β-CROWN (75%) on cifar10_resnet is caused by **missing specification-guided CROWN backward propagation**.

## Evidence

### Test Results

- **γ-CROWN:** 7/20 verified (props 2, 3, 4, 7, 15, 18, 19)
- **α,β-CROWN:** 15/20 verified (all except props 0, 1, 6, 10, 16)
- **Gap instances:** props 5, 8, 9, 11, 12, 13, 14, 17

### Key Observation: prop_5

**α,β-CROWN output:**
```
initial CROWN bounds: [2.81, 0.47, 3.20, 0.60, 3.84, 3.74, 2.05, 0.18, 1.41]
Verified with initial CROWN!  # ALL bounds positive
Time: 0.21s
```

**γ-CROWN output:**
```
output_bounds: [
  {lower: 2.27, upper: 5.03},   # Y_0
  {lower: -2.93, upper: 0.38},  # Y_1
  ...
]
Property Status: UNKNOWN       # Bounds cross 0
```

### Root Cause

The property requires proving `Y_0 > Y_i` for all i ∈ {1..9}.

**α,β-CROWN approach (specification-guided CROWN):**
1. Transforms the CROWN backward pass to directly compute bounds on `Y_0 - Y_i`
2. Starts backward propagation from a "C" matrix: `[[1, -1, 0, 0, ...], [1, 0, -1, 0, ...], ...]`
3. Preserves correlation information through the backward pass
4. Result: Tight bounds, often sufficient without BaB

**γ-CROWN approach (post-processing):**
1. Computes bounds on each output independently: `Y_0 ∈ [l_0, u_0]`, `Y_1 ∈ [l_1, u_1]`
2. Post-computes difference bounds: `Y_0 - Y_1 ∈ [l_0 - u_1, u_0 - l_1]`
3. Loses correlation between outputs
4. Result: Loose bounds, requires extensive BaB (which also uses loose bounds per domain)

### Code Location

`crates/gamma-propagate/src/beta_crown.rs:5336-5365` - `objective_bounds_vec` function:
```rust
// This computes bounds via interval arithmetic, losing correlations
for (idx, &c) in obj.iter().enumerate() {
    if c >= 0.0 {
        lower += c * flat.lower[[idx]];
        upper += c * flat.upper[[idx]];
    } else {
        lower += c * flat.upper[[idx]];
        upper += c * flat.lower[[idx]];
    }
}
```

## Why CROWN Bounds Match for Raw Outputs

The comparison script (`scripts/compare_crown_bounds.py`) showed γ-CROWN and auto_LiRPA produce identical CROWN bounds on raw outputs (ratio 1.0x). This is correct but misleading:

- Both compute tight bounds on individual outputs
- The difference emerges when computing bounds on **linear combinations of outputs**
- α,β-CROWN uses specification-guided propagation; γ-CROWN uses post-processing

## Impact Assessment

| Instance Type | γ-CROWN | α,β-CROWN | Gap Explanation |
|--------------|---------|-----------|-----------------|
| Easy (trivial margins) | Verified | Verified | Raw bounds sufficient |
| Medium (requires spec-guided) | UNKNOWN | Verified | Spec-guided tightens bounds |
| Hard (requires BaB) | UNKNOWN | UNKNOWN | Both timeout |

## Recommended Fix

Implement specification-guided CROWN backward propagation:

1. **Accept a specification matrix `C`** for the backward pass
2. **Initialize backward pass with `C`** instead of identity
3. **Compute bounds on `C @ outputs` directly**

This is a well-known optimization in the CROWN literature and is the primary reason α,β-CROWN achieves better results on verification benchmarks.

## References

- auto_LiRPA uses `compute_bounds(method='CROWN')` which internally handles C matrices
- α,β-CROWN config: `solver.beta-crown.lr_beta: 0.01`
- VNN-LIB properties transform to C matrices in the verifier

## Next Steps

1. Create GitHub issue for specification-guided CROWN implementation
2. Implement C matrix handling in the CROWN backward pass
3. Update `objective_bounds_vec` to use spec-guided bounds
4. Re-run benchmarks to verify improvement
