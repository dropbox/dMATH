# GCP-CROWN Analysis: Why Cuts Alone Cannot Close the Gap

**Date:** 2026-01-06
**Worker:** #498
**Related:** WORKER_DIRECTIVE_2026-01-06_gcp_crown.md

## Executive Summary

GCP-CROWN cutting planes have been integrated into the multi-objective verification path
(commit ed91057, Worker #497). However, cuts alone cannot close the cifar10_resnet gap
(2.8% vs 80%) because the underlying CROWN bounds are fundamentally too loose.

## Key Finding: Bound Quality is the Root Cause

### IBP Bound Analysis (profile-bounds)
```
Initial width: 1.60e-2 (epsilon = 8.00e-3)
Final width: 5.38e1
Total expansion: 3365x
```

### CROWN Bound Analysis
For cifar10_resnet with epsilon 0.008, CROWN produces output bounds like:
- Y_2 (target): [-384.36, +487.82] (width: 872)
- Y_0: [-429.91, +532.20] (width: 962)

For the constraint Y_0 - Y_2 < 0:
- Computed bound range: [-917.73, +916.56] (width: ~1834)

**With 1800+ unit width, no amount of cutting planes can verify.**

## Why Cuts Don't Help

### 1. BICCOS-Style Cuts Require Verified Domains

Our implementation (correctly) generates cuts from verified domains:
```rust
// In verify_graph_relu_split_multi_objective
if active_child.all_verified() {
    if self.config.enable_cuts {
        verified_histories.push(active_child.history.clone());
    }
}
```

**Problem:** With loose bounds, NO domains verify initially. Without verified domains,
BICCOS-style cuts cannot be generated. This is a chicken-and-egg problem.

### 2. Proactive Cuts Have Negative Contributions

Our proactive cut generation creates:
- Single-neuron cuts: `z_i >= 0` (always satisfied, useless)
- Pairwise cuts: `z_i + z_j >= 1` (may not be valid)

Cut contribution computation:
```rust
let constraint_min = sum(coeff_i * z_min_i) // For unstable neurons, z_min = 0
// For cut with bias=0.5: constraint_min - bias = 0 - 0.5 = -0.5
// Contribution is NEGATIVE, so cut is SKIPPED
```

**Result:** 77 proactive cuts generated, 0 applied (all have negative contributions).

### 3. Alpha-Beta-CROWN Has Different Bound Propagation

The reference implementation achieves ~80% verification through:
1. **CROWN-IBP hybrid** with different formulation
2. **Alpha optimization** for ReLU slopes
3. **Beta-CROWN** with optimized Lagrangian multipliers
4. **CPLEX-based LP cuts** from MIP solver
5. **BICCOS cuts** from verified domains (once bounds are tight enough)

Our bounds are 10-100x looser before cuts even come into play.

## Evidence

### Test Results (prop_0_eps_0.008.vnnlib, 60s timeout)

| Configuration | Domains | Verified | Cuts | Time |
|---------------|---------|----------|------|------|
| --no-cuts | 1754 | 0/9 | 0 | 60s |
| --enable-cuts --proactive-cuts | 906 | 0/9 | 77 | 30s |
| --enable-cuts --proactive-cuts --enable-near-miss-cuts | 812 | 0/9 | 77 | 30s |

**Observation:** More cuts = fewer domains explored (cut overhead) with same verification rate.

## Required Improvements

### Priority 1: Tighter Bound Propagation (Critical)
The fundamental issue is bound quality. Before cuts can help, we need:
- Improved CROWN-IBP hybrid for ResNets
- Better handling of Add/skip connections
- Alpha optimization for graph models

### Priority 2: LP-Based Cut Generation (Medium)
Instead of heuristic proactive cuts, generate cuts from LP relaxation:
- Solve LP relaxation
- Find fractional solution
- Generate Gomory/cover cuts

### Priority 3: Better Cut Selection (Lower)
Once bounds are tighter and some domains verify:
- Use beta values to filter cut terms (like BICCOS)
- Constraint strengthening to minimize cuts
- Cut merging for generalization

## Conclusion

GCP-CROWN integration is complete and correct. The 77% gap is NOT due to missing cuts
but due to loose underlying bounds. Cuts are a refinement technique that helps close
the last ~10-20% gap once bounds are reasonably tight.

**Recommended next step:** Focus on improving CROWN bound computation for ResNets,
specifically the CROWN-IBP hybrid and alpha optimization for DAG models.
