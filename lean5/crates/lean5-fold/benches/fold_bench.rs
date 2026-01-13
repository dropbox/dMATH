//! Benchmarks for Nova-style folding schemes
//!
//! These benchmarks measure performance of IVC operations and R1CS encoding.
//!
//! Run with: `cargo bench --package lean5-fold`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lean5_fold::{
    encode_cert_to_r1cs, extend_ivc_with_cert, start_ivc_from_cert, R1CSBuilder, Scalar, Transcript,
};
use lean5_kernel::{level::Level, Environment, ProofCert};
use std::hint::black_box;

/// Create a Sort certificate for benchmarking
fn sort_cert(level: u32) -> ProofCert {
    let mut l = Level::zero();
    for _ in 0..level {
        l = Level::succ(l);
    }
    ProofCert::Sort { level: l }
}

/// Create a deeply nested certificate for benchmarking
fn nested_cert(depth: u32) -> ProofCert {
    if depth == 0 {
        sort_cert(0)
    } else {
        // Create a nested App structure
        ProofCert::App {
            fn_cert: Box::new(nested_cert(depth - 1)),
            fn_type: Box::new(lean5_kernel::expr::Expr::Sort(Level::zero())),
            arg_cert: Box::new(sort_cert(0)),
            result_type: Box::new(lean5_kernel::expr::Expr::Sort(Level::zero())),
        }
    }
}

/// Benchmark R1CS encoding of certificates
fn bench_cert_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("cert_encoding");

    for depth in [1, 2, 4, 8].iter() {
        let cert = nested_cert(*depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("encode_cert", depth), depth, |b, _| {
            b.iter(|| encode_cert_to_r1cs(black_box(&cert)));
        });
    }

    group.finish();
}

/// Benchmark IVC start operation
fn bench_ivc_start(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("ivc_start");

    for depth in [1, 2, 4].iter() {
        let cert = nested_cert(*depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("start_ivc_from_cert", depth),
            depth,
            |b, _| {
                b.iter(|| start_ivc_from_cert(black_box(&cert), black_box(&env)));
            },
        );
    }

    group.finish();
}

/// Benchmark IVC extend operation
fn bench_ivc_extend(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("ivc_extend");

    for depth in [1, 2, 4].iter() {
        let cert = nested_cert(*depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("extend_ivc_with_cert", depth),
            depth,
            |b, _| {
                b.iter_batched(
                    || start_ivc_from_cert(&cert, &env).unwrap(),
                    |mut ivc| {
                        extend_ivc_with_cert(black_box(&mut ivc), black_box(&cert), black_box(&env))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark transcript operations
fn bench_transcript(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcript");

    // Benchmark appending scalars
    group.bench_function("append_scalar", |b| {
        let mut transcript = Transcript::new();
        let scalar = Scalar::from(42u64);
        b.iter(|| {
            transcript.append_scalar(b"test", black_box(&scalar));
        });
    });

    // Benchmark squeezing challenges
    group.bench_function("squeeze_challenge", |b| {
        b.iter_batched(
            || {
                let mut t = Transcript::new();
                t.append_scalar(b"test", &Scalar::from(42u64));
                t
            },
            |mut transcript| transcript.squeeze_challenge(),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark R1CS builder
fn bench_r1cs_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("r1cs_builder");

    for &num_constraints in &[10, 100, 1000] {
        group.throughput(Throughput::Elements(num_constraints as u64));
        group.bench_with_input(
            BenchmarkId::new("build", num_constraints),
            &num_constraints,
            |b, &n| {
                b.iter(|| {
                    let mut builder = R1CSBuilder::new(2);
                    // Allocate variables
                    for _ in 0..n + 5 {
                        builder.alloc_var();
                    }
                    for i in 0..n as usize {
                        let var_i = builder.var_idx(i);
                        let var_next = builder.var_idx(i + 1);
                        let a = vec![(var_i, Scalar::from(1u64))];
                        let b_vec = vec![(var_i, Scalar::from(1u64))];
                        let c_vec = vec![(var_next, Scalar::from(1u64))];
                        builder.add_constraint(a, b_vec, c_vec);
                    }
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cert_encoding,
    bench_ivc_start,
    bench_ivc_extend,
    bench_transcript,
    bench_r1cs_builder,
);

criterion_main!(benches);
