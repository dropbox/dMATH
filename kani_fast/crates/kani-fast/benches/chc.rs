//! Benchmarks for CHC (Constrained Horn Clause) verification
//!
//! These benchmarks measure the performance of CHC-based verification
//! using Spacer (Z3/Z4) for invariant synthesis and property verification.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kani_fast_chc::{
    encode_mir_to_chc, encode_transition_system, verify_transition_system, ChcBackend,
    ChcSolverConfig, MirProgramBuilder,
};
use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};
use std::time::Duration;

fn spacer_available() -> bool {
    which::which("z4").or_else(|_| which::which("z3")).is_ok()
}

/// Benchmark simple counter invariant discovery
fn benchmark_counter_chc(c: &mut Criterion) {
    if !spacer_available() {
        eprintln!("Skipping CHC benchmark: Spacer backend not installed (need z4 or z3)");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(30));

    let mut group = c.benchmark_group("chc_counter");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(20));

    // Simple counter: x starts at 0, increments, prove x >= 0
    let simple_counter = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .init("(= x 0)")
        .transition("(= x' (+ x 1))")
        .property("p1", "non_negative", "(>= x 0)")
        .build();

    group.bench_function("simple_counter_x_geq_0", |b| {
        let ts = simple_counter.clone();
        let cfg = config.clone();
        b.to_async(&runtime).iter(|| {
            let ts_inner = ts.clone();
            let cfg_inner = cfg.clone();
            async move {
                verify_transition_system(&ts_inner, &cfg_inner)
                    .await
                    .expect("CHC verification failed")
            }
        });
    });

    // Counter with upper bound check (harder invariant)
    let bounded_counter = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("n", SmtType::Int)
        .init("(and (= x 0) (> n 0))")
        .transition("(and (= x' (+ x 1)) (= n' n) (< x n))")
        .property("p1", "bounded", "(and (>= x 0) (<= x n))")
        .build();

    group.bench_function("bounded_counter_x_leq_n", |b| {
        let ts = bounded_counter.clone();
        let cfg = config.clone();
        b.to_async(&runtime).iter(|| {
            let ts_inner = ts.clone();
            let cfg_inner = cfg.clone();
            async move {
                verify_transition_system(&ts_inner, &cfg_inner)
                    .await
                    .expect("CHC verification failed")
            }
        });
    });

    group.finish();
}

/// Benchmark array bounds checking
fn benchmark_array_chc(c: &mut Criterion) {
    if !spacer_available() {
        eprintln!("Skipping CHC benchmark: Spacer backend not installed (need z4 or z3)");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(30));

    let mut group = c.benchmark_group("chc_array");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(20));

    // Array index bounds: i starts at 0, increments while < len
    let array_bounds = TransitionSystemBuilder::new()
        .variable("i", SmtType::Int)
        .variable("len", SmtType::Int)
        .init("(and (= i 0) (> len 0))")
        .transition("(and (< i len) (= i' (+ i 1)) (= len' len))")
        .property("p1", "bounds", "(and (>= i 0) (<= i len))")
        .build();

    group.bench_function("array_index_bounds", |b| {
        let ts = array_bounds.clone();
        let cfg = config.clone();
        b.to_async(&runtime).iter(|| {
            let ts_inner = ts.clone();
            let cfg_inner = cfg.clone();
            async move {
                verify_transition_system(&ts_inner, &cfg_inner)
                    .await
                    .expect("CHC verification failed")
            }
        });
    });

    group.finish();
}

/// Benchmark MIR-to-CHC encoding (without solving)
fn benchmark_mir_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("chc_encoding");
    group.sample_size(100);

    // Create a simple MIR program for encoding benchmark
    let simple_program = MirProgramBuilder::new(0)
        .local("_0", SmtType::Int)
        .local("_1", SmtType::Int)
        .local("_2", SmtType::Int)
        .init("(and (= _0 0) (= _1 0) (= _2 0))")
        .finish();

    group.bench_function("simple_mir_to_chc", |b| {
        b.iter(|| encode_mir_to_chc(&simple_program));
    });

    // More complex program with more locals
    let complex_program = MirProgramBuilder::new(0)
        .local("_0", SmtType::Int)
        .local("_1", SmtType::Int)
        .local("_2", SmtType::Int)
        .local("_3", SmtType::Int)
        .local("_4", SmtType::Int)
        .local("_5", SmtType::Int)
        .local("_6", SmtType::Int)
        .local("_7", SmtType::Int)
        .local("_8", SmtType::Int)
        .local("_9", SmtType::Int)
        .init("(= _0 0)")
        .finish();

    group.bench_function("complex_mir_to_chc", |b| {
        b.iter(|| encode_mir_to_chc(&complex_program));
    });

    group.finish();
}

/// Benchmark CHC system construction and SMT-LIB2 generation
fn benchmark_chc_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("chc_construction");
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
                    let ts = builder
                        .init("(= x0 0)")
                        .transition("(= x0' (+ x0 1))")
                        .property("p1", "prop", "(>= x0 0)")
                        .build();

                    // Also benchmark encoding
                    encode_transition_system(&ts)
                });
            },
        );
    }

    // Benchmark SMT-LIB2 generation
    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("y", SmtType::Int)
        .init("(and (= x 0) (= y 0))")
        .transition("(and (= x' (+ x 1)) (= y' (+ y 1)))")
        .property("p1", "equal", "(= x y)")
        .build();

    let chc = encode_transition_system(&ts);

    group.bench_function("smt2_generation", |b| {
        b.iter(|| chc.to_smt2());
    });

    group.finish();
}

/// Benchmark bitvector verification on Z3 vs Z4 backends
fn benchmark_bitvec_backends(c: &mut Criterion) {
    let has_z4 = which::which("z4").is_ok();
    let has_z3 = which::which("z3").is_ok();

    if !has_z4 && !has_z3 {
        eprintln!("Skipping bitvec backend benchmark: need z4 or z3 in PATH");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let bitvec_ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::BitVec(32))
        .variable("count", SmtType::BitVec(32))
        .init("(= count #x00000000)")
        .transition(
            "(and \
                (= x' (ite (= x #x00000000) x (bvand x (bvsub x #x00000001)))) \
                (= count' (ite (= x #x00000000) count (bvadd count #x00000001))))",
        )
        .property("p1", "bitcount_bound", "(bvule count #x00000020)")
        .build();

    let mut group = c.benchmark_group("chc_bitvec_backends");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(20));

    if has_z4 {
        let ts = bitvec_ts.clone();
        let cfg = ChcSolverConfig::new()
            .with_backend(ChcBackend::Z4)
            .with_timeout(Duration::from_secs(30));
        group.bench_function("z4_bitvec_popcount", |b| {
            let ts = ts.clone();
            let cfg = cfg.clone();
            b.to_async(&runtime).iter(|| {
                let ts_inner = ts.clone();
                let cfg_inner = cfg.clone();
                async move {
                    verify_transition_system(&ts_inner, &cfg_inner)
                        .await
                        .expect("CHC verification failed")
                }
            });
        });
    }

    if has_z3 {
        let ts = bitvec_ts.clone();
        let cfg = ChcSolverConfig::new()
            .with_backend(ChcBackend::Z3)
            .with_timeout(Duration::from_secs(30));
        group.bench_function("z3_bitvec_popcount", |b| {
            let ts = ts.clone();
            let cfg = cfg.clone();
            b.to_async(&runtime).iter(|| {
                let ts_inner = ts.clone();
                let cfg_inner = cfg.clone();
                async move {
                    verify_transition_system(&ts_inner, &cfg_inner)
                        .await
                        .expect("CHC verification failed")
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_counter_chc,
    benchmark_array_chc,
    benchmark_mir_encoding,
    benchmark_chc_construction,
    benchmark_bitvec_backends
);
criterion_main!(benches);
