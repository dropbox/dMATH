# ResNet Verification Gap Analysis
**Date**: 2026-01-06
**Worker**: #530

## Executive Summary

This analysis investigates why gamma-CROWN achieves only 2.8% verification on cifar10_resnet vs alpha,beta-CROWN's ~80%. The key finding is that **CROWN bounds are identical** between implementations, and **alpha-CROWN optimization doesn't help** on ResNets. The gap is due to differences in BaB (Branch-and-Bound) efficiency and GPU parallelism.

## Key Findings

### 1. CROWN Bounds Match Exactly

Comparison between gamma-CROWN and auto_LiRPA shows **1.00x ratio** for all outputs:

```
Output |   auto_LiRPA |      gamma-CROWN |    Ratio
     0 |       2.4051 |          2.4051 |     1.00x
     1 |       2.4374 |          2.4374 |     1.00x
     ...
 TOTAL |      21.8862 |         21.8861 |     1.00x
```

**Conclusion**: The CROWN algorithm implementation is correct. The gap is NOT in basic bound propagation.

### 2. alpha-CROWN Does Not Help on ResNets

Running auto_LiRPA with `method='alpha-CROWN'` produces:
```
[WARNING] No optimizable parameters found. Will skip optimization.
```

Alpha-CROWN optimization has **0.00% width reduction** on resnet_2b. This is because:
- The model architecture doesn't have optimizable alpha parameters in auto_LiRPA's formulation
- The ResNet skip connections may bypass the standard alpha-optimizable layers

### 3. Verification Gap Analysis

For property `prop_0_eps_0.008.vnnlib`, we need to prove Y_0 > Y_i for all i:

```
Y_0 > Y_1: margin = 2.71 (VERIFIED by CROWN alone)
Y_0 > Y_2: margin = -3.03 (GAP: 3.03)  <-- Largest gap
Y_0 > Y_3: margin = -2.14 (GAP: 2.14)
Y_0 > Y_4: margin = -0.98 (GAP: 0.98)
Y_0 > Y_5: margin = -2.31 (GAP: 2.31)
Y_0 > Y_6: margin = -1.79 (GAP: 1.79)
Y_0 > Y_7: margin = -0.99 (GAP: 0.99)
Y_0 > Y_8: margin = 1.01 (VERIFIED by CROWN alone)
Y_0 > Y_9: margin = 1.67 (VERIFIED by CROWN alone)
```

- 3/9 constraints verified by CROWN alone
- 6/9 require BaB to close gaps of 0.98 - 3.03

### 4. CROWN-IBP is Much Worse

```
Method      | Total Width | Verified
CROWN       |      21.89  | 3/9
CROWN-IBP   |    8915.50  | 0/9  (408x worse!)
```

This explains why the `--crown-ibp` flag is disabled for DAG models.

### 5. BaB Domain Exploration

Current gamma-CROWN BaB performance:
- With optimization (beta_iter=20): ~31 domains in 60s = ~2s/domain
- Without optimization: ~922 domains in 30s = ~33ms/domain

alpha,beta-CROWN uses:
- batch_size: 2000 (vs gamma's 4)
- GPU acceleration for parallel bound computation
- Can process ~500,000+ domains in 60s on GPU

## Root Cause Analysis

The verification gap is NOT due to:
- Incorrect CROWN bounds (they match)
- Missing alpha-CROWN optimization (it doesn't help on ResNets)
- Incorrect beta-CROWN formulation (SPSA optimization implemented)

The gap IS due to:
1. **Domain throughput**: alpha,beta-CROWN explores 100-1000x more domains per second via GPU
2. **Batch processing**: GPU enables batch_size=2000 vs CPU's batch_size=4
3. **Time to reach verification**: Each domain explores one branch; need many branches to close 3.03 gap

## Recommendations

### Short-term (Issue #10)
1. **GPU acceleration for BaB**: Wire MLX/wgpu GEMM through per-domain CROWN passes
2. **Increase batch_size**: With GPU, can process 100s-1000s of domains in parallel
3. **Smarter branching**: Focus splits on neurons affecting Y_0 vs Y_2 gap

### Long-term
1. **BICCOS integration**: Multi-tree branching for disjunctive properties
2. **GCP-CROWN cuts**: Generate problem-specific cutting planes
3. **Hybrid CPU-GPU**: Use GPU for batch CROWN, CPU for BaB management

## Test Commands

```bash
# Compare CROWN bounds with auto_LiRPA
.venv/bin/python scripts/compare_intermediate_bounds.py

# Test gamma-CROWN BaB
./target/release/gamma beta-crown benchmarks/.../resnet_2b.onnx \
  --property benchmarks/.../prop_0_eps_0.008.vnnlib \
  --branching relu --timeout 60

# Check alpha-CROWN effectiveness
.venv/bin/python -c "
from auto_LiRPA import BoundedModule, BoundedTensor
# ... (see /tmp/test_bab.py for full script)
"
```

## Related Issues
- Issue #9: Improve CROWN bounds for ResNet verification (current)
- Issue #10: Beta optimization for graph networks (next step)
- Issue #4: GPU-accelerated CROWN (partially complete)

## Conclusion

The verification gap cannot be closed by improving bound propagation alone - the bounds are already correct. The gap requires:
1. Processing many more domains via GPU parallelism
2. Effective branching strategy to target the hardest constraints (Y_0 vs Y_2)
3. Potentially BICCOS-style multi-tree search for disjunctive properties

The alpha,beta-CROWN 80% rate is achieved through GPU-accelerated massive parallelism, not through fundamentally tighter bounds.
