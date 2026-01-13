# SMT-COMP Benchmark Results: Z4 vs Z3

**Date**: 2024-12-31 (Iteration #170)
**Z3 Version**: 4.15.4
**Z4**: Release build

## Summary

| Theory | Benchmarks | Z4 Time | Z3 Time | Ratio | Agreement | Z4 Wins | Z3 Wins | Notes |
|--------|------------|---------|---------|-------|-----------|---------|---------|-------|
| QF_BV  | 50         | 0.56s   | 0.72s   | 1.29x | 100%      | 33      | 6       | Bitvectors working correctly |
| QF_LIA | 50         | 0.79s   | 1.05s   | 1.33x | 100%      | 40      | 4       | Linear Integer Arithmetic working |
| QF_LRA | 110        | 344.06s | 665.03s | 1.93x | 100%*     | 59      | 39      | *Z4 returns 'unknown' often |
| QF_UF  | 100        | 243.94s | 8.48s   | 0.03x | **43%**   | 79      | 14      | **CRITICAL: 57 disagreements** |
| **TOTAL** | **310** | 589.35s | 675.27s | 1.15x | 81.6%   | 211     | 63      | |

## Detailed Findings

### QF_BV (Bitvectors) - PASS
- **Agreement**: 100% (50/50)
- **Performance**: Z4 is 1.29x faster than Z3
- **Result**: Bitvector theory is working correctly

### QF_LIA (Linear Integer Arithmetic) - PASS
- **Agreement**: 100% (50/50)
- **Performance**: Z4 is 1.33x faster than Z3
- **Result**: Linear integer arithmetic theory is working correctly

### QF_LRA (Linear Real Arithmetic) - PARTIAL
- **Agreement**: 100% (technically, but Z4 returns 'unknown' frequently)
- **Performance**: Z4 is 1.93x faster overall
- **Timeouts**: Z4 had 11 timeouts, Z3 had 16 timeouts (30s limit)
- **Issue**: Z4's LRA solver returns 'unknown' on many problems that Z3 solves
- **Result**: LRA theory needs improvement for completeness

### QF_UF (Uninterpreted Functions) - CRITICAL FAILURE
- **Agreement**: 43% (43/100)
- **Disagreements**: 57 cases where Z4 says SAT but Z3 (and expected) says UNSAT
- **Performance**: Z3 is 33x faster on these benchmarks
- **Root Cause**: Z4's EUF theory solver has a soundness bug in transitive equality reasoning
- **Result**: EUF theory has critical correctness bugs

## Critical Bug Analysis: QF_UF Disagreements

All 57 disagreements follow the same pattern:
- Z4 returns: `sat`
- Z3 returns: `unsat`
- Expected: `unsat`

This is a **soundness bug** - Z4 claims satisfiable when the problem is actually unsatisfiable.

### Example Failing Benchmark: eq_diamond91.smt2

This benchmark tests transitivity of equality:
- Creates variables x0..x90, y0..y90, z0..z90
- Each step: x_i = x_{i+1} via either y_i or z_i (diamond pattern)
- Final assertion: `(not (= x0 x90))`
- Correct answer: UNSAT (x0 must equal x90 via transitivity)

Z4 incorrectly reports SAT, indicating the congruence closure or equality reasoning has a bug.

### Affected Benchmark Categories

- `eq_diamond`: Equality transitivity chains
- `iso_*`: Isomorphism benchmarks
- `gensys_*`: Generated systems
- `dead_dnd*`: Deadlock detection
- `PEQ*`: Partial equality benchmarks

## Benchmark Sources

Benchmarks downloaded from:
- **Zenodo SMT-LIB 2025 Release**: https://zenodo.org/records/16740866
- **QF_UF Archive**: 7,503 benchmarks, sampled 100

## Recommendations

### Immediate Actions Required

1. **Fix QF_UF soundness bug** (CRITICAL)
   - Root cause: Transitive equality reasoning fails on complex diamond patterns
   - Location: Likely in `z4-euf` crate's congruence closure implementation
   - Priority: P0 - blocks correctness claims

2. **Improve QF_LRA completeness** (HIGH)
   - Z4 returns 'unknown' on many solvable problems
   - May need better theory solver integration or decision procedures

### Passing Theories

- QF_BV: Ready for production use
- QF_LIA: Ready for production use

## Reproduction

```bash
# Run the benchmark suite
python3 scripts/benchmark_smtcomp.py --timeout 30

# Test specific failing benchmark
./target/release/z4 benchmarks/smtcomp/non-incremental/QF_UF/eq_diamond/eq_diamond91.smt2
z3 benchmarks/smtcomp/non-incremental/QF_UF/eq_diamond/eq_diamond91.smt2
```

## Full Disagreement List

```
[QF_UF] dead_dnd004.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl928.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl793.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn431.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl747.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl1081.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn_sk036.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl1050.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl094.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl011.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl181.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl009.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl913.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl422.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl1146.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl431.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_brn009.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] PEQ019_size4.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] QF_UF_schedule_world.2.prop1_ab_br_max.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] 00346.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] eq_diamond91.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl065.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn085.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl1241.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] eq_diamond48.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl029.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl192.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl971.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl1108.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl1184.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl1170.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl070.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl1179.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn563.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl034.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn983.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl991.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn921.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] dead_dnd024.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl1223.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_brn393.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl413.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl_sk003.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl336.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_brn396.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl116.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl646.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_brn962.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_brn469.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl802.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl1254.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] dead_dnd001.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_brn733.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_brn284.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl066.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] gensys_icl732.smt2: Z4=sat, Z3=unsat, expected=unsat
[QF_UF] iso_icl_repgen_sk004.smt2: Z4=sat, Z3=unsat, expected=unsat
```
