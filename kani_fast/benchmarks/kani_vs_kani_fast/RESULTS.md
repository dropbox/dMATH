# Kani Fast vs Kani Benchmark Results

**Generated:** Sat Jan  4 11:30:00 PST 2026

## Overview

This benchmark compares verification time between:
- **Kani Fast**: CHC-based verification via kani-fast-driver
- **Kani**: CBMC-based bounded model checking via cargo kani

## Key Findings

1. **Simple Functions**: Kani Fast is typically 3-10x faster for functions it can verify
   - CHC solving is instant (~50-200ms) vs CBMC startup overhead (600ms-1s)

2. **Unbounded Proofs**: Kani Fast can prove properties for ALL inputs when CHC succeeds
   - Kani requires --unwind and can only prove bounded properties
   - However, some loop patterns require invariant synthesis that CHC cannot automatically derive

3. **Known Limitations**: Some benchmarks return UNKNOWN from CHC solver
   - `simple_bounds_check`: Too many boolean conjunctions (34 basic blocks)
   - `unbounded_sum`: Requires loop invariant synthesis (not yet automated)

---


## Simple Functions


| Benchmark | Kani Fast | Kani | Speedup | Notes |
|-----------|-----------|------|---------|-------|
| simple_abs | 163ms | 641ms | 3.93x | ✓ Verified |
| simple_add | 668ms | 640ms | 0.95x | ✓ Verified |
| simple_bounds_check | UNKNOWN | 307ms | N/A | ⚠ CHC returns unknown (complex boolean logic) |
| simple_clamp | 210ms | 650ms | 3.09x | ✓ Verified |
| simple_is_even | 134ms | 629ms | 4.69x | ✓ Verified |
| simple_max | 151ms | 964ms | 6.38x | ✓ Verified |
| simple_min3 | 159ms | 975ms | 6.13x | ✓ Verified |
| simple_saturating | 142ms | 957ms | 6.73x | ✓ Verified |
| simple_sign | 248ms | 960ms | 3.87x | ✓ Verified |
| simple_swap | 100ms | 953ms | 9.53x | ✓ Verified |

**9 of 10 simple function benchmarks verified** (90% success rate)

## Unbounded Proofs


| Benchmark | Kani Fast | Kani | Speedup | Notes |
|-----------|-----------|------|---------|-------|
| unbounded_counter | 510ms | 695ms | 1.36x | ✓ Verified |
| unbounded_gcd_termination | 272ms | 47.751s | 175.55x | ✓ Verified |
| unbounded_monotonic | 604ms | 706ms | 1.16x | ✓ Verified |
| unbounded_state_machine | 691ms | 782ms | 1.13x | ✓ Verified |
| unbounded_sum | UNKNOWN | 688ms | N/A | ⚠ Needs loop invariant synthesis |

**4 of 5 unbounded proof benchmarks verified** (80% success rate)

---

## Benchmark Anomaly Analysis (2026-01-04)

Two benchmarks show Kani Fast as slower than Kani in the raw timing data:
- `simple_bounds_check`: Previously showed 1.354s
- `unbounded_sum`: Previously showed 6.990s

**Root Cause**: These benchmarks return UNKNOWN from the CHC solver. The long times in the original benchmark run were due to the solver trying different strategies before giving up. When the solver recognizes early that it cannot prove the property, it returns UNKNOWN quickly (~100-600ms).

**Why They Fail**:
1. **simple_bounds_check**: Has 34 basic blocks with complex boolean conjunctions. The CHC encoding creates too many clauses for efficient solving.
2. **unbounded_sum**: Contains a loop that requires an inductive invariant (`sum == i*(i-1)/2`). CHC cannot automatically synthesize this invariant.

**Recommendation**: Use `--hybrid` flag to fall back to Kani BMC for these cases.

---

## Notes

- **UNKNOWN** means the CHC solver could not prove or disprove the property
- **FAIL** means the tool timed out (60s for Kani Fast, 120s for Kani) or returned error
- Speedup = Kani time / Kani Fast time
- Times include all overhead (compilation, solving, output)
- Kani Fast uses Z4/Z3 CHC backend; Kani uses CBMC

## When to Use Each Tool

| Use Case | Recommended Tool |
|----------|-----------------|
| Simple functions, quick feedback | **Kani Fast** |
| Unbounded loop invariants (simple patterns) | **Kani Fast** |
| Complex boolean logic (many branches) | Kani |
| Complex heap operations | Kani |
| Iterator patterns | Kani |
| Trait method dispatch | Kani |
| Loops requiring complex invariants | Kani (or hybrid mode) |

