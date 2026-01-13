//! Benchmarks for polynomial commitment schemes
//!
//! These benchmarks measure performance of KZG and IPA commitment operations.
//!
//! Run with: `cargo bench --package lean5-commit`

use ark_std::rand::rngs::StdRng;
use ark_std::rand::SeedableRng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lean5_commit::{IpaScheme, KzgScheme, ProofCommitmentScheme};
use lean5_kernel::{level::Level, ProofCert};
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

/// Benchmark KZG setup
fn bench_kzg_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_setup");

    // Use smaller degrees for setup benchmark (setup is expensive)
    for &degree in &[256, 1024] {
        group.throughput(Throughput::Elements(degree as u64));
        group.bench_with_input(BenchmarkId::new("setup", degree), &degree, |b, &d| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(42);
                KzgScheme::setup(black_box(d), black_box(&mut rng))
            });
        });
    }

    group.finish();
}

/// Benchmark IPA setup
fn bench_ipa_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_setup");

    for &degree in &[256, 1024, 4096] {
        group.throughput(Throughput::Elements(degree as u64));
        group.bench_with_input(BenchmarkId::new("setup", degree), &degree, |b, &d| {
            b.iter(|| IpaScheme::setup(black_box(d)));
        });
    }

    group.finish();
}

/// Benchmark KZG commit
fn bench_kzg_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_commit");

    // Setup KZG once with a reasonable degree
    let mut rng = StdRng::seed_from_u64(42);
    let kzg = KzgScheme::setup(1024, &mut rng).unwrap();

    for &depth in &[1, 2, 4, 8] {
        let cert = nested_cert(depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("commit", depth), &depth, |b, _| {
            b.iter(|| kzg.commit(black_box(&cert)));
        });
    }

    group.finish();
}

/// Benchmark IPA commit
fn bench_ipa_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_commit");

    // Setup IPA once
    let ipa = IpaScheme::setup(1024).unwrap();

    for &depth in &[1, 2, 4, 8] {
        let cert = nested_cert(depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("commit", depth), &depth, |b, _| {
            b.iter(|| ipa.commit(black_box(&cert)));
        });
    }

    group.finish();
}

/// Benchmark KZG open (includes commit)
fn bench_kzg_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_open");

    let mut rng = StdRng::seed_from_u64(42);
    let kzg = KzgScheme::setup(1024, &mut rng).unwrap();
    let challenge = ark_bls12_381::Fr::from(42u64);

    for &depth in &[1, 2, 4] {
        let cert = nested_cert(depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("open", depth), &depth, |b, _| {
            b.iter(|| kzg.open(black_box(&cert), black_box(challenge)));
        });
    }

    group.finish();
}

/// Benchmark IPA open
fn bench_ipa_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_open");

    let ipa = IpaScheme::setup(1024).unwrap();
    let challenge = ark_bls12_381::Fr::from(42u64);

    for &depth in &[1, 2, 4] {
        let cert = nested_cert(depth);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("open", depth), &depth, |b, _| {
            b.iter(|| ipa.open(black_box(&cert), black_box(challenge)));
        });
    }

    group.finish();
}

/// Benchmark KZG verify
fn bench_kzg_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_verify");

    let mut rng = StdRng::seed_from_u64(42);
    let kzg = KzgScheme::setup(1024, &mut rng).unwrap();
    let challenge = ark_bls12_381::Fr::from(42u64);

    for &depth in &[1, 2, 4] {
        let cert = nested_cert(depth);
        let commitment = kzg.commit(&cert).unwrap();
        let (value, proof) = kzg.open(&cert, challenge).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("verify", depth), &depth, |b, _| {
            b.iter(|| {
                kzg.verify(
                    black_box(&commitment),
                    black_box(challenge),
                    black_box(value),
                    black_box(&proof),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark IPA verify
fn bench_ipa_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_verify");

    let ipa = IpaScheme::setup(1024).unwrap();
    let challenge = ark_bls12_381::Fr::from(42u64);

    for &depth in &[1, 2, 4] {
        let cert = nested_cert(depth);
        let commitment = ipa.commit(&cert).unwrap();
        let (value, proof) = ipa.open(&cert, challenge).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("verify", depth), &depth, |b, _| {
            b.iter(|| {
                ipa.verify(
                    black_box(&commitment),
                    black_box(challenge),
                    black_box(value),
                    black_box(&proof),
                )
            });
        });
    }

    group.finish();
}

/// Compare KZG vs IPA for full commit+verify workflow
fn bench_commit_verify_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_verify_comparison");

    let mut rng = StdRng::seed_from_u64(42);
    let kzg = KzgScheme::setup(1024, &mut rng).unwrap();
    let ipa = IpaScheme::setup(1024).unwrap();
    let cert = nested_cert(4);
    let challenge = ark_bls12_381::Fr::from(42u64);

    group.bench_function("kzg_full_workflow", |b| {
        b.iter(|| {
            let commitment = kzg.commit(black_box(&cert)).unwrap();
            let (value, proof) = kzg.open(black_box(&cert), black_box(challenge)).unwrap();
            kzg.verify(
                black_box(&commitment),
                black_box(challenge),
                black_box(value),
                black_box(&proof),
            )
        });
    });

    group.bench_function("ipa_full_workflow", |b| {
        b.iter(|| {
            let commitment = ipa.commit(black_box(&cert)).unwrap();
            let (value, proof) = ipa.open(black_box(&cert), black_box(challenge)).unwrap();
            ipa.verify(
                black_box(&commitment),
                black_box(challenge),
                black_box(value),
                black_box(&proof),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kzg_setup,
    bench_ipa_setup,
    bench_kzg_commit,
    bench_ipa_commit,
    bench_kzg_open,
    bench_ipa_open,
    bench_kzg_verify,
    bench_ipa_verify,
    bench_commit_verify_comparison,
);

criterion_main!(benches);
