# SMT Benchmark Results: Z4 vs Z3

**Date**: 2024-12-31
**Iteration**: 147
**Z3 Version**: 4.15.4

## Summary

| Logic | Agreement | Z4 Time | Z3 Time | Ratio | Status |
|-------|-----------|---------|---------|-------|--------|
| QF_BV | 50/50 (100%) | 0.569s | 0.372s | 0.65x | CORRECT |
| QF_UF | 20/20 (100%) | 0.121s | 0.154s | 1.28x | CORRECT |
| QF_LIA | 26/50 (52%) | - | - | - | **PARSING BUG** |

## QF_BV (Bitvector Logic) - CORRECT

- **Agreement**: 100% (50/50 tests)
- **Performance**: Z3 is 1.54x faster overall
- **Notes**: First test has cold-start penalty (0.265s vs 0.007s). After warmup, Z4 is competitive or faster.

## QF_UF (Uninterpreted Functions) - CORRECT

- **Agreement**: 100% (20/20 tests)
- **Performance**: Z4 is 1.28x faster overall
- **Notes**: Excellent agreement and performance.

## QF_LIA (Linear Integer Arithmetic) - PARSING BUG + PERFORMANCE ISSUES

- **Agreement**: 52% (26/50 tests)
- **Fixed in Iteration 147**: LIA branch-and-bound now uses proper splitting lemmas
- **Remaining issues**:
  - **Parsing bug**: Negative integers like `-5` not recognized (24 failures)
  - **Performance**: Splits cause DpllT recreation, losing learned clauses

### Iteration 147 Fix: Splitting Lemmas

The LIA solver now returns `TheoryResult::NeedSplit` when the LRA relaxation
gives a non-integer solution. The DPLL layer creates splitting atoms:
- `x <= floor(v)` and `x >= ceil(v)`
- Adds clause: `(x <= floor) OR (x >= ceil)`
- This guides the search toward integer solutions

**New architecture**:
- `TheoryResult::NeedSplit(SplitRequest)` - new variant for branch-and-bound
- `DpllT::solve_step()` - returns split requests to caller
- `DpllT::apply_split()` - adds splitting atoms and clause
- Executor handles splits by recreating DpllT (loses learned clauses)

### Remaining Bug: Negative Number Parsing

The benchmarks use `(* -5 x)` which Z4 parses as `(* "-5" x)` (string symbol)
instead of `(* (- 5) x)` (negation of 5). This causes incorrect results on
24/50 benchmarks that use negative coefficients.

### Example: system_00.smt2 (PARSING BUG)

```smt2
(assert (<= (+ (* -5 v1) (* 2 v2)) -27))
```

- Z3: `unsat` (correct)
- Z4: `sat` (INCORRECT - `-5` parsed as undefined symbol)

### Working Benchmarks

All 26 passing tests:
- Use only positive coefficients
- Or use explicit negation `(- 5)` syntax
- LIA logic is correct when parsing succeeds

## Recommendations

1. **Priority 1**: Fix negative number parsing in z4-frontend (parser bug)
2. **Priority 2**: Optimize LIA split handling to preserve learned clauses
3. **Optional**: Consider using cuts (Gomory cuts, MIR) for better LIA performance

## Benchmark Scripts

- `scripts/generate_smt_benchmarks.py` - Generate test cases
- `scripts/benchmark_smt.py` - Compare Z4 vs Z3 on SMT-LIB files

## Test Commands

```bash
# Generate benchmarks
python3 scripts/generate_smt_benchmarks.py

# Run benchmarks
python3 scripts/benchmark_smt.py benchmarks/smt/QF_BV
python3 scripts/benchmark_smt.py benchmarks/smt/QF_LIA
python3 scripts/benchmark_smt.py benchmarks/smt/QF_UF
```
