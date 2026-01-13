//! GPU vs CPU batch benchmarks
//!
//! Compares GPU-accelerated batch operations against CPU implementations.
//!
//! Run with: cargo bench -p lean5-gpu
//!
//! Note: GPU benchmarks require GPU hardware and will be skipped if unavailable.
//! The CPU benchmarks will always run for baseline comparison.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lean5_gpu::{
    batch::BatchWhnf, parallel_infer_type, parallel_whnf, GpuAccelerator, ParallelBatch,
    ParallelConfig,
};
use lean5_kernel::{
    env::Environment,
    expr::{BinderInfo, Expr},
    level::Level,
    tc::TypeChecker,
};
use std::hint::black_box;
use std::time::Duration;

// === Expression generators ===

/// Build a nested beta redex: (λ x. x) ((λ x. x) ((λ x. x) Prop))
fn nested_beta_redex(depth: u32) -> Expr {
    let id_lam = Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0));

    let mut result = Expr::prop();
    for _ in 0..depth {
        result = Expr::app(id_lam.clone(), result);
    }
    result
}

/// Generate batch of expressions in WHNF (no reduction needed)
fn generate_whnf_batch(count: usize) -> Vec<Expr> {
    (0..count)
        .map(|i| match i % 4 {
            0 => Expr::Sort(Level::Zero),
            1 => Expr::Sort(Level::succ(Level::Zero)),
            2 => Expr::BVar(i as u32),
            _ => Expr::prop(),
        })
        .collect()
}

/// Generate batch of beta redexes that need reduction
fn generate_beta_batch(count: usize, depth: u32) -> Vec<Expr> {
    (0..count).map(|_| nested_beta_redex(depth)).collect()
}

/// Generate mixed batch (some in WHNF, some need reduction)
fn generate_mixed_batch(count: usize) -> Vec<Expr> {
    (0..count)
        .map(|i| {
            if i % 3 == 0 {
                // Beta redex
                nested_beta_redex(2)
            } else {
                // Already in WHNF
                match i % 4 {
                    0 => Expr::Sort(Level::Zero),
                    1 => Expr::BVar(i as u32),
                    2 => Expr::prop(),
                    _ => Expr::type_(),
                }
            }
        })
        .collect()
}

// === CPU Benchmarks (always run) ===

fn bench_cpu_whnf_batch_whnf_exprs(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("cpu_whnf/already_whnf");
    group.warm_up_time(Duration::from_millis(500));

    for count in [16, 64, 256, 1024, 4096] {
        let batch = generate_whnf_batch(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| BatchWhnf::cpu_batch_whnf(&env, black_box(batch)));
        });
    }
    group.finish();
}

fn bench_cpu_whnf_batch_beta(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("cpu_whnf/beta_depth_2");
    group.warm_up_time(Duration::from_millis(500));

    for count in [16, 64, 256, 1024] {
        let batch = generate_beta_batch(count, 2);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| BatchWhnf::cpu_batch_whnf(&env, black_box(batch)));
        });
    }
    group.finish();
}

fn bench_cpu_whnf_batch_mixed(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("cpu_whnf/mixed");
    group.warm_up_time(Duration::from_millis(500));

    for count in [16, 64, 256, 1024] {
        let batch = generate_mixed_batch(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| BatchWhnf::cpu_batch_whnf(&env, black_box(batch)));
        });
    }
    group.finish();
}

fn bench_cpu_whnf_single_vs_batch(c: &mut Criterion) {
    let env = Environment::new();
    let batch = generate_mixed_batch(256);

    let mut group = c.benchmark_group("cpu_whnf/single_vs_batch");

    // Single expression at a time
    group.bench_function("single_256", |b| {
        b.iter(|| {
            let tc = TypeChecker::new(&env);
            batch
                .iter()
                .map(|e| tc.whnf(black_box(e)))
                .collect::<Vec<_>>()
        });
    });

    // Batched (uses same TypeChecker)
    group.bench_function("batch_256", |b| {
        b.iter(|| BatchWhnf::cpu_batch_whnf(&env, black_box(&batch)));
    });

    group.finish();
}

// === GPU Benchmarks (require GPU) ===

/// Try to create GPU accelerator, return None if unavailable
fn try_gpu_accelerator() -> Option<GpuAccelerator> {
    // Use tokio runtime to run async GPU init
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    rt.block_on(async {
        match GpuAccelerator::new().await {
            Ok(mut gpu) => {
                // Try to initialize pipelines
                if gpu.init_pipelines().is_ok() {
                    Some(gpu)
                } else {
                    eprintln!("GPU available but shader compilation failed");
                    None
                }
            }
            Err(e) => {
                eprintln!("GPU unavailable: {e}");
                None
            }
        }
    })
}

fn bench_gpu_whnf_batch_whnf_exprs(c: &mut Criterion) {
    let Some(mut gpu) = try_gpu_accelerator() else {
        eprintln!("Skipping GPU benchmark: no GPU available");
        return;
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let env = Environment::new();
    let mut group = c.benchmark_group("gpu_whnf/already_whnf");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for count in [16, 64, 256, 1024, 4096] {
        let batch = generate_whnf_batch(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| rt.block_on(async { gpu.batch_whnf(&env, black_box(batch)).await.unwrap() }));
        });
    }
    group.finish();
}

fn bench_gpu_whnf_batch_beta(c: &mut Criterion) {
    let Some(mut gpu) = try_gpu_accelerator() else {
        eprintln!("Skipping GPU benchmark: no GPU available");
        return;
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let env = Environment::new();
    let mut group = c.benchmark_group("gpu_whnf/beta_depth_2");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for count in [16, 64, 256, 1024] {
        let batch = generate_beta_batch(count, 2);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| rt.block_on(async { gpu.batch_whnf(&env, black_box(batch)).await.unwrap() }));
        });
    }
    group.finish();
}

fn bench_gpu_whnf_batch_mixed(c: &mut Criterion) {
    let Some(mut gpu) = try_gpu_accelerator() else {
        eprintln!("Skipping GPU benchmark: no GPU available");
        return;
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let env = Environment::new();
    let mut group = c.benchmark_group("gpu_whnf/mixed");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for count in [16, 64, 256, 1024] {
        let batch = generate_mixed_batch(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| rt.block_on(async { gpu.batch_whnf(&env, black_box(batch)).await.unwrap() }));
        });
    }
    group.finish();
}

/// Compare GPU vs CPU side by side for same workload
fn bench_gpu_vs_cpu_comparison(c: &mut Criterion) {
    let gpu_available = try_gpu_accelerator().is_some();

    let env = Environment::new();
    let mut group = c.benchmark_group("gpu_vs_cpu");
    group.warm_up_time(Duration::from_millis(500));

    for count in [64, 256, 1024] {
        let batch = generate_mixed_batch(count);
        group.throughput(Throughput::Elements(count as u64));

        // CPU baseline
        group.bench_with_input(BenchmarkId::new("cpu", count), &batch, |b, batch| {
            b.iter(|| BatchWhnf::cpu_batch_whnf(&env, black_box(batch)));
        });

        // GPU (if available)
        if gpu_available {
            let mut gpu = try_gpu_accelerator().unwrap();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            group.bench_with_input(BenchmarkId::new("gpu", count), &batch, |b, batch| {
                b.iter(|| {
                    rt.block_on(async { gpu.batch_whnf(&env, black_box(batch)).await.unwrap() })
                });
            });
        }
    }
    group.finish();
}

// === Serialization benchmarks (CPU only, measures overhead) ===

fn bench_arena_serialization(c: &mut Criterion) {
    use lean5_gpu::GpuExprArena;

    let mut group = c.benchmark_group("arena/serialize");

    for count in [64, 256, 1024] {
        let batch = generate_mixed_batch(count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(count), &batch, |b, batch| {
            b.iter(|| {
                let mut arena = GpuExprArena::new();
                for expr in batch {
                    arena.add_expr(black_box(expr));
                }
                arena
            });
        });
    }
    group.finish();
}

fn bench_arena_roundtrip(c: &mut Criterion) {
    use lean5_gpu::GpuExprArena;

    let mut group = c.benchmark_group("arena/roundtrip");

    for count in [64, 256, 1024] {
        let batch = generate_mixed_batch(count);
        group.throughput(Throughput::Elements(count as u64));

        // Pre-serialize
        let mut arena = GpuExprArena::new();
        let indices: Vec<u32> = batch.iter().map(|e| arena.add_expr(e)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &(&arena, &indices),
            |b, (arena, indices)| {
                b.iter(|| {
                    indices
                        .iter()
                        .map(|&idx| arena.to_expr(black_box(idx)))
                        .collect::<Vec<_>>()
                });
            },
        );
    }
    group.finish();
}

// === Parallel (Rayon) Benchmarks ===

/// Generate type-checkable expressions (lambdas)
fn generate_lambda_batch(count: usize) -> Vec<Expr> {
    (0..count)
        .map(|_| {
            // λ (x : Prop). x - simple identity function
            Expr::lam(BinderInfo::Default, Expr::prop(), Expr::bvar(0))
        })
        .collect()
}

fn bench_parallel_infer_type(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("parallel/infer_type");
    group.warm_up_time(Duration::from_millis(500));

    for count in [16, 64, 256, 1024] {
        let batch = generate_lambda_batch(count);
        group.throughput(Throughput::Elements(count as u64));

        // Sequential baseline
        group.bench_with_input(BenchmarkId::new("sequential", count), &batch, |b, batch| {
            b.iter(|| {
                batch
                    .iter()
                    .map(|e| {
                        let mut tc = TypeChecker::new(&env);
                        tc.infer_type(black_box(e))
                    })
                    .collect::<Vec<_>>()
            });
        });

        // Parallel with Rayon
        group.bench_with_input(BenchmarkId::new("parallel", count), &batch, |b, batch| {
            b.iter(|| parallel_infer_type(&env, black_box(batch)));
        });
    }
    group.finish();
}

fn bench_parallel_whnf(c: &mut Criterion) {
    let env = Environment::new();
    let mut group = c.benchmark_group("parallel/whnf");
    group.warm_up_time(Duration::from_millis(500));

    for count in [16, 64, 256, 1024] {
        let batch = generate_beta_batch(count, 2);
        group.throughput(Throughput::Elements(count as u64));

        // Sequential baseline
        group.bench_with_input(BenchmarkId::new("sequential", count), &batch, |b, batch| {
            b.iter(|| {
                let tc = TypeChecker::new(&env);
                batch
                    .iter()
                    .map(|e| tc.whnf(black_box(e)))
                    .collect::<Vec<_>>()
            });
        });

        // Parallel with Rayon
        group.bench_with_input(BenchmarkId::new("parallel", count), &batch, |b, batch| {
            b.iter(|| parallel_whnf(&env, black_box(batch)));
        });
    }
    group.finish();
}

fn bench_parallel_config_comparison(c: &mut Criterion) {
    let env = Environment::new();
    let batch = generate_lambda_batch(256);
    let mut group = c.benchmark_group("parallel/config_comparison");
    group.throughput(Throughput::Elements(256));

    // Default config
    let default_batch = ParallelBatch::new(&env);
    group.bench_function("default", |b| {
        b.iter(|| default_batch.batch_infer_type(black_box(&batch)));
    });

    // Low latency config
    let low_latency = ParallelBatch::with_config(&env, ParallelConfig::low_latency());
    group.bench_function("low_latency", |b| {
        b.iter(|| low_latency.batch_infer_type(black_box(&batch)));
    });

    // High throughput config
    let high_throughput = ParallelBatch::with_config(&env, ParallelConfig::high_throughput());
    group.bench_function("high_throughput", |b| {
        b.iter(|| high_throughput.batch_infer_type(black_box(&batch)));
    });

    // Force sequential (very high threshold)
    let sequential = ParallelBatch::with_config(
        &env,
        ParallelConfig {
            min_parallel_batch: 10000,
            num_threads: None,
        },
    );
    group.bench_function("force_sequential", |b| {
        b.iter(|| sequential.batch_infer_type(black_box(&batch)));
    });

    group.finish();
}

// === Criterion groups ===

criterion_group!(
    cpu_benches,
    bench_cpu_whnf_batch_whnf_exprs,
    bench_cpu_whnf_batch_beta,
    bench_cpu_whnf_batch_mixed,
    bench_cpu_whnf_single_vs_batch,
);

criterion_group!(
    gpu_benches,
    bench_gpu_whnf_batch_whnf_exprs,
    bench_gpu_whnf_batch_beta,
    bench_gpu_whnf_batch_mixed,
    bench_gpu_vs_cpu_comparison,
);

criterion_group!(
    arena_benches,
    bench_arena_serialization,
    bench_arena_roundtrip,
);

criterion_group!(
    parallel_benches,
    bench_parallel_infer_type,
    bench_parallel_whnf,
    bench_parallel_config_comparison,
);

criterion_main!(cpu_benches, gpu_benches, arena_benches, parallel_benches);
