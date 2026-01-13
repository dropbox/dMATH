//! SIMD benchmarks for gamma-tensor interval arithmetic.
//!
//! Run with: cargo bench -p gamma-tensor

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gamma_tensor::{simd, BoundedTensor};
use ndarray::ArrayD;

/// Benchmark interval multiplication using direct SIMD functions.
fn bench_interval_mul_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD/IntervalMul");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000] {
        // Create test data
        let a_lower: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let a_upper: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001 + 0.5).collect();
        let b_lower: Vec<f32> = (0..size)
            .map(|i| ((size - i) as f32) * 0.001 - 0.3)
            .collect();
        let b_upper: Vec<f32> = (0..size)
            .map(|i| ((size - i) as f32) * 0.001 + 0.3)
            .collect();

        let mut out_lower = vec![0.0f32; size];
        let mut out_upper = vec![0.0f32; size];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |b, _| {
            b.iter(|| {
                simd::interval_mul(
                    black_box(&a_lower),
                    black_box(&a_upper),
                    black_box(&b_lower),
                    black_box(&b_upper),
                    &mut out_lower,
                    &mut out_upper,
                );
                black_box(&out_lower);
            })
        });
    }
    group.finish();
}

/// Benchmark BoundedTensor::mul which uses SIMD internally.
fn bench_bounded_tensor_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("BoundedTensor/mul");

    for size in [100, 1_000, 10_000, 100_000] {
        let a_lower: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let a_upper: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001 + 0.5).collect();
        let b_lower: Vec<f32> = (0..size)
            .map(|i| ((size - i) as f32) * 0.001 - 0.3)
            .collect();
        let b_upper: Vec<f32> = (0..size)
            .map(|i| ((size - i) as f32) * 0.001 + 0.3)
            .collect();

        let a = BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[size]), a_lower).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[size]), a_upper).unwrap(),
        )
        .unwrap();

        let b = BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[size]), b_lower).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[size]), b_upper).unwrap(),
        )
        .unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("mul", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(a.mul(black_box(b)).unwrap()))
        });
    }
    group.finish();
}

/// Benchmark pos/neg split.
fn bench_pos_neg_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD/PosNegSplit");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let x: Vec<f32> = (0..size)
            .map(|i| (i as f32) - (size as f32 / 2.0))
            .collect();
        let mut pos = vec![0.0f32; size];
        let mut neg = vec![0.0f32; size];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |b, _| {
            b.iter(|| {
                simd::pos_neg_split(black_box(&x), &mut pos, &mut neg);
                black_box(&pos);
            })
        });

        // Compare to separate mapv calls
        let x_array = ArrayD::from_shape_vec(ndarray::IxDyn(&[size]), x.clone()).unwrap();
        group.bench_with_input(BenchmarkId::new("mapv_separate", size), &x_array, |b, x| {
            b.iter(|| {
                let pos = x.mapv(|v| v.max(0.0));
                let neg = x.mapv(|v| v.min(0.0));
                black_box((pos, neg))
            })
        });
    }
    group.finish();
}

/// Benchmark dot product.
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD/Dot");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.001).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| black_box(simd::dot(black_box(&a), black_box(&b))))
        });

        // Compare to iterator
        group.bench_with_input(BenchmarkId::new("iter", size), &size, |bench, _| {
            bench.iter(|| {
                let result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                black_box(result)
            })
        });
    }
    group.finish();
}

/// Benchmark sum.
fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD/Sum");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let x: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| black_box(simd::sum(black_box(&x))))
        });

        // Compare to iterator
        group.bench_with_input(BenchmarkId::new("iter", size), &size, |bench, _| {
            bench.iter(|| {
                let result: f32 = x.iter().sum();
                black_box(result)
            })
        });
    }
    group.finish();
}

criterion_group!(
    simd_benches,
    bench_interval_mul_simd,
    bench_bounded_tensor_mul,
    bench_pos_neg_split,
    bench_dot_product,
    bench_sum,
);

criterion_main!(simd_benches);
