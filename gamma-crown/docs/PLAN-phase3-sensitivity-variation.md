# PLAN: Phase 3 Sensitivity Variation

**Status:** COMPLETE
**Created:** 2026-01-01
**Completed:** 2026-01-01 (Iterations #225-#229)
**Goal:** Reduce per-block sensitivity variation and tighten worst-case bounds (esp. Qwen3-0.6B), without regressing any model.

---

## Context (end of Phase 2)

Phase 2 achieved **finite bounds across all benchmark models** with block-wise zonotope tightening and scaling stabilization.

Baseline (Iteration #223, 2026-01-01):
- whisper-tiny: ~4.2
- whisper-small: ~7.5
- whisper-medium: ~8.9
- whisper-large-v3: ~177
- Qwen3-0.6B: ~444,000 (∞ → finite)

Remaining issue: **Qwen3 per-block sensitivity varies ~2000x across layers**, dominating the final bound width.

Reference audit: `reports/main/verification_audit_2026-01-01_iter223.md`

---

## Phase 3 Objectives

1. **Diagnose** why later blocks are dramatically more sensitive than early blocks.
2. **Improve** worst-case blocks (not just average) while keeping bounds sound.
3. **Preserve** the “no regressions” policy across the full benchmark suite.

---

## Tasks (ordered)

### Task 1: Make block-wise JSON complete (DIAGNOSTICS) ✅ DONE (#225)

**Why:** The manager directive requires tracking bound tightness over time; `gamma verify --block-wise --json` must expose the key per-block metrics (including FFN/SwiGLU).

**Deliverables:**
- Include `swiglu_width` in per-block JSON output ✅
- Include `has_infinite` in per-node JSON output ✅

**Success:** Existing non-JSON output unchanged; JSON becomes strictly more informative.

### Task 2: Quantify sensitivity variation (METRICS) ✅ DONE (#226)

Add summarized sensitivity stats derived from `BlockWiseResult`:
- max/min/median sensitivity across blocks ✅
- worst-k blocks (name + sensitivity + widths) ✅

**Deliverables:**
- Added `min_sensitivity()`, `median_sensitivity()`, `worst_k_blocks(k)`, `sensitivity_range()` methods to `BlockWiseResult`
- JSON output now includes `summary` object with all stats and `worst_5_blocks` array

**Results (Qwen3-0.6B):**
- max_sensitivity: 1.24e12 (layer27)
- min_sensitivity: 5.21e5 (layer0)
- sensitivity_range: 2.37e6×

**Success:** One command run produces a single JSON object sufficient for regression tracking.

### Task 3: Correlate sensitivity with weight norms (ANALYSIS) ✅ DONE (#227)

Hypothesis: later blocks have larger effective linear gains (e.g., `ffn_down`, residual scaling, layernorm params).

**Deliverables:**
- Added `gamma weights norms` CLI command with Frobenius and spectral norm computation ✅
- Correlation analysis report generated ✅

**Key Results (Qwen3-0.6B):**
- **Pearson correlation (spectral norm vs sensitivity): r = 0.81** (strong positive)
- Spectral norm range: 31.5 - 631 (20x across blocks)
- Sensitivity range: 5.2e5 - 1.2e12 (2.4 million x)
- **Top 5 highest spectral norm blocks = Top 5 highest sensitivity blocks**

**Insight:** Later blocks (17-27) have spectral norms 3-20x higher than early blocks. This directly explains the bound explosion pattern. The amplification ratio (sensitivity/spectral) also grows with depth, suggesting compositional effects.

**Report:** `reports/main/weight_norm_sensitivity_correlation_2026-01-01.md`

**Success:** Strong correlation confirmed. Weight spectral norm is the primary driver of sensitivity variation.

### Task 4: Targeted tightening of worst blocks (ALGORITHMS) ✅ DONE (#228, #229)

**Deliverables:**
- Zonotope normalization for SwiGLU block-wise (#228) ✅
- Zonotope normalization for FFN-down block-wise (#228) ✅
- Analysis confirming all paths now have normalization (#229) ✅

**Results:**
- Block-wise max sensitivity: 1.24e12 → 8.86e9 (**140x tighter**)
- Block-wise sensitivity range: 2.37e6x → 33,480x (**70x tighter**)
- whisper-small: 7.5 → 5.77 (**23% better**)
- whisper-medium: 8.9 → 7.89 (**11% better**)

**Analysis (#229):** Q@K^T attention zonotope already has normalization (produces 1e-5 bounds). Full-network SwiGLU path already has normalization. Remaining bound growth is due to fundamental factors (weight spectral norms, product of large values).

**Success:** Worst-block sensitivity reduced 140x without worsening any model's bounds.

---

## Phase 3 Completion Summary

**All tasks complete.** Phase 3 achieved its objectives:

1. **Diagnosed** why later blocks are more sensitive: weight spectral norm correlation r=0.81 (#227)
2. **Improved** worst-case blocks: 140x tighter block-wise bounds via zonotope normalization (#228)
3. **Preserved** no-regressions policy: whisper models improved 11-23%, no degradation

**Final Results:**

| Metric | Phase 2 Baseline | Phase 3 Final | Improvement |
|--------|------------------|---------------|-------------|
| Qwen3 block-wise max sens | 1.24e12 | 8.86e9 | 140x tighter |
| Qwen3 block-wise sens range | 2.37e6x | 33,480x | 70x tighter |
| whisper-small | 7.5 | 5.77 | 23% better |
| whisper-medium | 8.9 | 7.89 | 11% better |

**Detailed analysis:** `reports/main/phase3_analysis_2026-01-01-13-25.md`

---

## Mandatory Regression Suite

After each improvement, re-run and record:
- `gamma verify models/whisper-tiny/model.safetensors --native --method crown --epsilon 0.001 --json`
- `gamma verify models/whisper-small/model.safetensors --native --method crown --epsilon 0.001 --json`
- `gamma verify models/whisper-medium/model.safetensors --native --method crown --epsilon 0.001 --json`
- `gamma verify models/whisper-large-v3/model.safetensors --native --method crown --epsilon 0.001 --json`
- `gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --method crown --epsilon 0.001 --json`

And for sensitivity diagnostics:
- `gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --block-wise --epsilon 0.001 --json`

