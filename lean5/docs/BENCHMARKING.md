# Lean5 Benchmarking Methodology

This document describes the benchmarking methodology for Lean5, including how to reproduce measurements, what is measured, and interpretation guidelines.

## Hardware and Software Specification

Benchmarks should always report:

| Field | Example |
|-------|---------|
| CPU | Apple M3 Max |
| RAM | (specify) |
| OS | Darwin 24.6.0 (macOS) |
| Rust Version | rustc 1.92.0 |
| Cargo Version | cargo 1.92.0 |
| Build Profile | `--release` (optimized) |
| Date | 2026-01-01 |

## Running Benchmarks

### Kernel Benchmarks

```bash
cargo bench -p lean5-kernel --bench kernel_bench
```

### GPU/Parallel Benchmarks

```bash
cargo bench -p lean5-gpu
```

### Server Benchmarks

```bash
cargo bench -p lean5-server
```

## Current Measurements (Apple M3 Max, 2026-01-01)

These are actual measurements from running `cargo bench -p lean5-kernel --bench kernel_bench`:

### Type Inference (`infer_type`)

| Operation | Time |
|-----------|------|
| Sort_0 (Prop) | 20.4 ns |
| Sort_1 (Type) | 34.8 ns |
| lambda_simple | 102.8 ns |
| lambda_nested/2 | 228.7 ns |
| lambda_nested/4 | 674.0 ns |
| lambda_nested/8 | 2.45 µs |
| lambda_nested/16 | 10.1 µs |
| app_simple | 459.8 ns |
| app_nested/2 | 898.8 ns |
| app_nested/4 | 1.74 µs |
| app_nested/8 | 3.47 µs |

### Weak Head Normal Form (`whnf`)

| Operation | Time |
|-----------|------|
| beta_simple | 16.8 ns |
| beta_nested/2 | 30.6 ns |
| beta_nested/4 | 54.5 ns |
| beta_nested/8 | 108.9 ns |
| beta_nested/16 | 208.8 ns |
| beta_nested/32 | 422.7 ns |
| delta_unfold | 119.8 ns |
| iota_nat_zero | 520.9 ns |

### Definitional Equality (`is_def_eq`)

| Operation | Time |
|-----------|------|
| identical | 1.37 ns |
| different_sorts | 97.5 ns |
| alpha_eq | 83.8 ns |
| beta_reduce | 95.9 ns |
| structural/2 | 268.1 ns |
| structural/4 | 644.4 ns |
| structural/8 | 1.79 µs |
| structural/16 | 5.82 µs |

## Methodology

### Criterion Framework

All benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) with:

- **Warm-up time**: 3 seconds
- **Sample size**: 100 iterations
- **Statistical analysis**: Confidence intervals reported

### Expression Types Tested

1. **Simple expressions**: Sort, Prop, constants
2. **Nested lambdas**: λ (x : Sort 0). λ (y : Sort 1). ... x
3. **Nested applications**: id.{1} (id.{1} (... Prop))
4. **Beta redexes**: (λ x. x) ((λ x. x) Prop)
5. **Delta unfolding**: Definition expansion
6. **Iota reduction**: Recursor computation

### What Is NOT Measured

- **End-to-end theorem proving**: These are kernel microbenchmarks
- **Parsing time**: Only kernel operations are measured
- **Elaboration time**: Only type checking operations
- **File I/O**: Memory-only operations

## Comparison Guidelines

### Lean 4 Comparison Status

**Status**: Lean 4 is installed (v4.13.0) but not benchmarked at kernel-level granularity.

**What Lean 4 provides**: `lean --profile file.lean` reports cumulative times:
- `type checking`: Total type checking time for all definitions
- `elaboration`: Total elaboration time
- `import`: Prelude/module loading time

**Challenge**: Lean 4's profiling reports aggregate times, not per-expression microbenchmarks. A fair comparison requires:

1. Equivalent expressions tested in both systems
2. Isolation of kernel operations from elaboration
3. Same hardware, same methodology
4. Statistical analysis (multiple runs, confidence intervals)

**Lean 4 profiling example** (Apple M3 Max, 2026-01-01):
```
lean --profile benchmarks/lean4/simple_bench.lean
# type checking: 0.0875ms (cumulative for 5 definitions)
# elaboration: 13.2ms
# import: 252ms
```

This shows cumulative time for multiple definitions, not per-expression times comparable to Lean5's microbenchmarks.

### Establishing Fair Comparison

To make valid comparative claims, future work should:

1. Write Lean 4 kernel-level benchmarks using FFI to call type checker directly
2. Use identical expression structures
3. Run both on same hardware in identical conditions
4. Report raw numbers with methodology

### Self-Comparison

We can compare Lean5 against itself over time:
- Track regressions via CI
- Compare across commits
- Profile specific changes

## Correctness Evidence

Lean5 correctness is verified through:

1. **Differential testing**: 1,712 expressions tested against Lean 4
2. **Mutation testing**: 99.4% kill rate on kernel code
3. **Unit tests**: 1,225 kernel tests passing (N=252)

Performance claims are meaningless without correctness.

## Contributing Benchmarks

When adding new benchmarks:

1. Use Criterion framework
2. Document what is being measured
3. Include input sizes for scaling analysis
4. Report hardware specification
5. Include reproduction instructions

## Historical Notes

- Early development had unsubstantiated "100-500x faster" claims
- These were removed in commit #244 (2026-01-01)
- All current documentation contains only measured values
