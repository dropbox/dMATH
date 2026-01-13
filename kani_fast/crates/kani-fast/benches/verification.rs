//! Benchmarks for Kani Fast verification
//!
//! These benchmarks measure the overhead of the kani-fast wrapper
//! compared to direct cargo kani invocation.

use criterion::{criterion_group, criterion_main, Criterion};
use kani_fast::{KaniConfig, KaniWrapper};
use std::path::PathBuf;
use std::time::Duration;

fn example_project_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("examples").join("simple_proofs"))
        .expect("Failed to find workspace root")
}

fn kani_available() -> bool {
    which::which("cargo-kani").is_ok()
}

fn benchmark_simple_proof(c: &mut Criterion) {
    if !kani_available() {
        eprintln!("Skipping benchmark: Kani not installed");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let mut group = c.benchmark_group("verification");

    // Set reasonable sample size for slow benchmarks
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("proof_checked_add_safe", |b| {
        b.to_async(&runtime).iter(|| async {
            wrapper
                .verify_with_harness(&project, Some("proof_checked_add_safe"))
                .await
                .expect("Verification failed")
        });
    });

    group.bench_function("proof_abs_diff_commutative", |b| {
        b.to_async(&runtime).iter(|| async {
            wrapper
                .verify_with_harness(&project, Some("proof_abs_diff_commutative"))
                .await
                .expect("Verification failed")
        });
    });

    group.bench_function("proof_safe_div_no_panic", |b| {
        b.to_async(&runtime).iter(|| async {
            wrapper
                .verify_with_harness(&project, Some("proof_safe_div_no_panic"))
                .await
                .expect("Verification failed")
        });
    });

    group.finish();
}

fn benchmark_failing_proof(c: &mut Criterion) {
    if !kani_available() {
        eprintln!("Skipping benchmark: Kani not installed");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let mut group = c.benchmark_group("counterexample");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("proof_buggy_multiply_overflows", |b| {
        b.to_async(&runtime).iter(|| async {
            wrapper
                .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
                .await
                .expect("Verification failed")
        });
    });

    group.finish();
}

fn benchmark_with_config(c: &mut Criterion) {
    if !kani_available() {
        eprintln!("Skipping benchmark: Kani not installed");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let project = example_project_path();

    let mut group = c.benchmark_group("config_variants");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Default config
    let default_wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    group.bench_function("default_config", |b| {
        b.to_async(&runtime).iter(|| async {
            default_wrapper
                .verify_with_harness(&project, Some("proof_checked_add_safe"))
                .await
                .expect("Verification failed")
        });
    });

    // Config with explicit unwind bound
    let config_with_unwind = KaniConfig {
        default_unwind: Some(10),
        ..Default::default()
    };
    let wrapper_with_unwind =
        KaniWrapper::new(config_with_unwind).expect("Failed to create wrapper");
    group.bench_function("with_unwind_10", |b| {
        b.to_async(&runtime).iter(|| async {
            wrapper_with_unwind
                .verify_with_harness(&project, Some("proof_checked_add_safe"))
                .await
                .expect("Verification failed")
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_simple_proof,
    benchmark_failing_proof,
    benchmark_with_config
);
criterion_main!(benches);
