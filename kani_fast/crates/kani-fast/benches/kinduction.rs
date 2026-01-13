//! Benchmarks for K-Induction verification
//!
//! These benchmarks measure the performance of k-induction-based
//! unbounded verification with various transition systems.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kani_fast_kinduction::{KInduction, KInductionConfigBuilder, SmtType, TransitionSystemBuilder};
use std::fmt::Write;
use std::time::Duration;

fn z3_available() -> bool {
    which::which("z3").is_ok()
}

/// Benchmark simple counter verification at different k values
fn benchmark_counter_kinduction(c: &mut Criterion) {
    if !z3_available() {
        eprintln!("Skipping k-induction benchmark: Z3 not installed");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("kinduction_counter");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Simple counter: x starts at 0, increments, prove x >= 0
    let counter_ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .init("(= x 0)")
        .transition("(= x' (+ x 1))")
        .property("p1", "non_negative", "(>= x 0)")
        .build();

    for max_k in [5, 10] {
        group.bench_with_input(BenchmarkId::new("max_k", max_k), &max_k, |b, &max_k| {
            let ts = counter_ts.clone();
            b.to_async(&runtime).iter(|| {
                let ts_inner = ts.clone();
                async move {
                    let config = KInductionConfigBuilder::new()
                        .max_k(max_k)
                        .total_timeout_secs(30)
                        .timeout_per_step_ms(5000)
                        .build();
                    let engine = KInduction::new(config);
                    engine.verify(&ts_inner).await.expect("K-induction failed")
                }
            });
        });
    }

    group.finish();
}

/// Benchmark transition system with multiple variables
fn benchmark_multivariable(c: &mut Criterion) {
    if !z3_available() {
        eprintln!("Skipping k-induction benchmark: Z3 not installed");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("kinduction_multivariable");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Two-counter system: both start at 0, increment in lockstep
    let two_counter_ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("y", SmtType::Int)
        .init("(and (= x 0) (= y 0))")
        .transition("(and (= x' (+ x 1)) (= y' (+ y 1)))")
        .property("p1", "equal", "(= x y)")
        .build();

    group.bench_function("two_counters_equal", |b| {
        let ts = two_counter_ts.clone();
        b.to_async(&runtime).iter(|| {
            let ts_inner = ts.clone();
            async move {
                let config = KInductionConfigBuilder::new()
                    .max_k(10)
                    .total_timeout_secs(30)
                    .timeout_per_step_ms(5000)
                    .build();
                let engine = KInduction::new(config);
                engine.verify(&ts_inner).await.expect("K-induction failed")
            }
        });
    });

    // Sum invariant: x + y remains constant
    let sum_ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("y", SmtType::Int)
        .variable("s", SmtType::Int)
        .init("(and (= x 5) (= y 5) (= s 10))")
        .transition("(and (= x' (+ x 1)) (= y' (- y 1)) (= s' s))")
        .property("p1", "sum_invariant", "(= (+ x y) s)")
        .build();

    group.bench_function("sum_invariant", |b| {
        let ts = sum_ts.clone();
        b.to_async(&runtime).iter(|| {
            let ts_inner = ts.clone();
            async move {
                let config = KInductionConfigBuilder::new()
                    .max_k(10)
                    .total_timeout_secs(30)
                    .timeout_per_step_ms(5000)
                    .build();
                let engine = KInduction::new(config);
                engine.verify(&ts_inner).await.expect("K-induction failed")
            }
        });
    });

    group.finish();
}

/// Benchmark transition system construction (no solver needed)
fn benchmark_ts_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("kinduction_construction");
    group.sample_size(100);

    for num_vars in [1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("variables", num_vars),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    let mut builder = TransitionSystemBuilder::new();
                    for i in 0..n {
                        builder = builder.variable(format!("x{}", i), SmtType::Int);
                    }
                    builder
                        .init("(= x0 0)")
                        .transition("(= x0' (+ x0 1))")
                        .property("p1", "prop", "(>= x0 0)")
                        .build()
                });
            },
        );
    }

    for num_props in [1, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("properties", num_props),
            &num_props,
            |b, &n| {
                b.iter(|| {
                    let mut builder = TransitionSystemBuilder::new()
                        .variable("x", SmtType::Int)
                        .init("(= x 0)")
                        .transition("(= x' (+ x 1))");

                    for i in 0..n {
                        builder =
                            builder.property(format!("p{}", i), format!("prop{}", i), "(>= x 0)");
                    }
                    builder.build()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SMT-LIB2 formula generation (no solver needed)
fn benchmark_smtlib_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kinduction_smtlib");
    group.sample_size(100);

    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("y", SmtType::Int)
        .init("(and (= x 0) (= y 0))")
        .transition("(and (= x' (+ x 1)) (= y' (+ y 1)))")
        .property("p1", "equal", "(= x y)")
        .build();

    for k in [1, 5, 10, 20] {
        group.bench_with_input(BenchmarkId::new("k_steps", k), &k, |b, &k| {
            b.iter(|| {
                // Generate base case formula
                let mut formula = String::new();
                formula.push_str("(set-logic QF_LIA)\n");
                formula.push_str("(declare-fun x () Int)\n");
                formula.push_str("(declare-fun y () Int)\n");
                for i in 0..k {
                    let _ = writeln!(formula, "(declare-fun x_{} () Int)", i);
                    let _ = writeln!(formula, "(declare-fun y_{} () Int)", i);
                }
                let _ = writeln!(formula, "(assert {})", ts.init.smt_formula);
                formula.push_str("(check-sat)\n");
                formula
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_counter_kinduction,
    benchmark_multivariable,
    benchmark_ts_construction,
    benchmark_smtlib_generation
);
criterion_main!(benches);
