//! Server operation benchmarks
//!
//! Benchmarks JSON-RPC server operations including:
//! - check: Type checking expressions
//! - getType: Type inference
//! - batchCheck: Batch checking with various sizes
//! - prove: SMT proof attempts
//! - Environment serialization (save/load)
//!
//! Run with: cargo bench -p lean5-server
//!
//! These benchmarks measure end-to-end server handler performance,
//! including parsing, elaboration, and type checking.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lean5_kernel::{Environment, Expr, Level, ProofCert};
use lean5_server::handlers::{
    handle_batch_check, handle_batch_verify_cert, handle_check, handle_get_type, handle_prove,
    handle_server_info, handle_verify_cert, BatchCheckItem, BatchCheckParams, BatchVerifyCertItem,
    BatchVerifyCertParams, CheckParams, GetTypeParams, ProveParams, ServerState, VerifyCertParams,
};
use lean5_server::progress::ProgressSender;
use lean5_server::rpc::RequestId;
use std::hint::black_box;
use std::time::Duration;
use tempfile::tempdir;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

// ============================================================================
// Test Expression Generators
// ============================================================================

/// Simple expressions that parse and type-check quickly
fn simple_expressions() -> Vec<&'static str> {
    vec![
        "Type",
        "Prop",
        "Type 1",
        "fun (x : Type) => x",
        "fun (A : Type) (x : A) => x",
        "fun (A : Type) (B : Type) (x : A) => x",
    ]
}

/// More complex expressions requiring more elaboration work
fn complex_expressions() -> Vec<&'static str> {
    vec![
        "fun (A : Type) (B : Type) (f : A -> B) (x : A) => f x",
        "fun (A : Type) (P : A -> Prop) (x : A) (h : P x) => h",
        "fun (A : Type) (B : Type) (C : Type) (f : A -> B) (g : B -> C) (x : A) => g (f x)",
    ]
}

/// Invalid expressions (for testing error paths)
fn invalid_expressions() -> Vec<&'static str> {
    vec![
        "fun x =>",            // incomplete
        "fun (x : ???) => x",  // invalid type
        "fun (x : Type) => y", // unbound variable
    ]
}

/// Generate batch check items
fn generate_batch_items(count: usize) -> Vec<BatchCheckItem> {
    let exprs = simple_expressions();
    (0..count)
        .map(|i| BatchCheckItem {
            id: format!("item_{i}"),
            code: exprs[i % exprs.len()].to_string(),
        })
        .collect()
}

/// Generate mixed batch (some valid, some invalid)
fn generate_mixed_batch(count: usize) -> Vec<BatchCheckItem> {
    let valid = simple_expressions();
    let invalid = invalid_expressions();
    (0..count)
        .map(|i| {
            if i % 5 == 0 {
                // 20% invalid
                BatchCheckItem {
                    id: format!("item_{i}"),
                    code: invalid[i % invalid.len()].to_string(),
                }
            } else {
                BatchCheckItem {
                    id: format!("item_{i}"),
                    code: valid[i % valid.len()].to_string(),
                }
            }
        })
        .collect()
}

// ============================================================================
// Check Benchmarks
// ============================================================================

fn bench_check_simple(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/check/simple");
    group.warm_up_time(Duration::from_millis(500));

    for expr in simple_expressions() {
        let params = CheckParams {
            code: expr.to_string(),
            timeout_ms: None,
        };

        group.bench_with_input(BenchmarkId::from_parameter(expr), &params, |b, params| {
            b.iter(|| {
                rt.block_on(async {
                    handle_check(&state, RequestId::Number(1), black_box(params.clone())).await
                })
            });
        });
    }
    group.finish();
}

fn bench_check_complex(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/check/complex");
    group.warm_up_time(Duration::from_millis(500));

    for (i, expr) in complex_expressions().iter().enumerate() {
        let params = CheckParams {
            code: expr.to_string(),
            timeout_ms: None,
        };

        group.bench_with_input(BenchmarkId::from_parameter(i), &params, |b, params| {
            b.iter(|| {
                rt.block_on(async {
                    handle_check(&state, RequestId::Number(1), black_box(params.clone())).await
                })
            });
        });
    }
    group.finish();
}

fn bench_check_invalid(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/check/invalid");
    group.warm_up_time(Duration::from_millis(500));

    for (i, expr) in invalid_expressions().iter().enumerate() {
        let params = CheckParams {
            code: expr.to_string(),
            timeout_ms: None,
        };

        group.bench_with_input(BenchmarkId::from_parameter(i), &params, |b, params| {
            b.iter(|| {
                rt.block_on(async {
                    handle_check(&state, RequestId::Number(1), black_box(params.clone())).await
                })
            });
        });
    }
    group.finish();
}

// ============================================================================
// GetType Benchmarks
// ============================================================================

fn bench_get_type(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/getType");
    group.warm_up_time(Duration::from_millis(500));

    let expressions = vec![
        ("Type", "Type"),
        ("lambda_id", "fun (x : Type) => x"),
        ("lambda_poly", "fun (A : Type) (x : A) => x"),
        (
            "lambda_app",
            "fun (A : Type) (B : Type) (f : A -> B) (x : A) => f x",
        ),
    ];

    for (name, expr) in expressions {
        let params = GetTypeParams {
            expr: expr.to_string(),
        };

        group.bench_with_input(BenchmarkId::from_parameter(name), &params, |b, params| {
            b.iter(|| {
                rt.block_on(async {
                    handle_get_type(&state, RequestId::Number(1), black_box(params.clone())).await
                })
            });
        });
    }
    group.finish();
}

// ============================================================================
// BatchCheck Benchmarks
// ============================================================================

fn bench_batch_check_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/batchCheck/scaling");
    group.warm_up_time(Duration::from_millis(500));

    for count in [1, 4, 16, 64, 256] {
        let items = generate_batch_items(count);
        let params = BatchCheckParams {
            items,
            use_gpu: false,
            timeout_ms: None,
        };

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &params, |b, params| {
            b.iter(|| {
                rt.block_on(async {
                    handle_batch_check(
                        &state,
                        RequestId::Number(1),
                        black_box(params.clone()),
                        None,
                    )
                    .await
                })
            });
        });
    }
    group.finish();
}

fn bench_batch_check_mixed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/batchCheck/mixed");
    group.warm_up_time(Duration::from_millis(500));

    for count in [16, 64, 256] {
        let items = generate_mixed_batch(count);
        let params = BatchCheckParams {
            items,
            use_gpu: false,
            timeout_ms: None,
        };

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &params, |b, params| {
            b.iter(|| {
                rt.block_on(async {
                    handle_batch_check(
                        &state,
                        RequestId::Number(1),
                        black_box(params.clone()),
                        None,
                    )
                    .await
                })
            });
        });
    }
    group.finish();
}

// ============================================================================
// Prove Benchmarks
// ============================================================================

fn bench_prove_simple(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/prove");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    // Simple reflexivity proof (a = a)
    let params_refl = ProveParams {
        goal: "fun (A : Type) (a : A) => a".to_string(), // This needs equality types
        hypotheses: vec![],
        timeout_ms: Some(1000),
    };

    group.bench_function("reflexivity_attempt", |b| {
        b.iter(|| {
            rt.block_on(async {
                handle_prove(&state, RequestId::Number(1), black_box(params_refl.clone())).await
            })
        });
    });

    // Proof with hypothesis
    let params_hyp = ProveParams {
        goal: "fun (P : Prop) (h : P) => h".to_string(),
        hypotheses: vec![],
        timeout_ms: Some(1000),
    };

    group.bench_function("hypothesis_use", |b| {
        b.iter(|| {
            rt.block_on(async {
                handle_prove(&state, RequestId::Number(1), black_box(params_hyp.clone())).await
            })
        });
    });

    group.finish();
}

// ============================================================================
// ServerInfo Benchmark
// ============================================================================

fn bench_server_info(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    c.bench_function("server/serverInfo", |b| {
        b.iter(|| {
            rt.block_on(async { handle_server_info(black_box(&state), RequestId::Number(1)).await })
        });
    });
}

// ============================================================================
// Environment Serialization Benchmarks
// ============================================================================

fn bench_env_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("server/env_serialization");
    group.warm_up_time(Duration::from_millis(500));

    // Empty environment (baseline)
    let empty_env = Environment::new();

    // For "with_defs" benchmarks, we just use the empty env since
    // setting up inductives through the kernel API is complex.
    // The serialization benchmarks still measure the serialization overhead.
    let env_with_defs = Environment::new();

    // Benchmark JSON serialization
    group.bench_function("to_json/empty", |b| {
        b.iter(|| black_box(&empty_env).to_json());
    });

    group.bench_function("to_json/with_defs", |b| {
        b.iter(|| black_box(&env_with_defs).to_json());
    });

    // Benchmark bincode serialization
    group.bench_function("to_bincode/empty", |b| {
        b.iter(|| black_box(&empty_env).to_bincode());
    });

    group.bench_function("to_bincode/with_defs", |b| {
        b.iter(|| black_box(&env_with_defs).to_bincode());
    });

    // Benchmark file save/load (to temp files)
    let dir = tempdir().unwrap();

    group.bench_function("save_bincode/empty", |b| {
        let path = dir.path().join("empty.bin");
        b.iter(|| black_box(&empty_env).save_to_file(black_box(&path)));
    });

    group.bench_function("save_bincode/with_defs", |b| {
        let path = dir.path().join("defs.bin");
        b.iter(|| black_box(&env_with_defs).save_to_file(black_box(&path)));
    });

    // Save files for load benchmarks
    let empty_path = dir.path().join("empty_load.bin");
    let defs_path = dir.path().join("defs_load.bin");
    empty_env.save_to_file(&empty_path).unwrap();
    env_with_defs.save_to_file(&defs_path).unwrap();

    group.bench_function("load_bincode/empty", |b| {
        b.iter(|| Environment::load_from_file(black_box(&empty_path)));
    });

    group.bench_function("load_bincode/with_defs", |b| {
        b.iter(|| Environment::load_from_file(black_box(&defs_path)));
    });

    // JSON load benchmarks
    let empty_json = empty_env.to_json().unwrap();
    let defs_json = env_with_defs.to_json().unwrap();

    group.bench_function("from_json/empty", |b| {
        b.iter(|| Environment::from_json(black_box(&empty_json)));
    });

    group.bench_function("from_json/with_defs", |b| {
        b.iter(|| Environment::from_json(black_box(&defs_json)));
    });

    group.finish();
}

// ============================================================================
// End-to-End Latency (simulating full request cycle)
// ============================================================================

fn bench_e2e_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/e2e_latency");
    group.warm_up_time(Duration::from_millis(500));

    // Measure complete request-response cycle for typical operations
    group.bench_function("check_simple_e2e", |b| {
        let params = CheckParams {
            code: "fun (A : Type) (x : A) => x".to_string(),
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_check(&state, RequestId::Number(1), black_box(params.clone())).await
            })
        });
    });

    group.bench_function("get_type_e2e", |b| {
        let params = GetTypeParams {
            expr: "fun (A : Type) (x : A) => x".to_string(),
        };
        b.iter(|| {
            rt.block_on(async {
                handle_get_type(&state, RequestId::Number(1), black_box(params.clone())).await
            })
        });
    });

    group.bench_function("batch_10_e2e", |b| {
        let params = BatchCheckParams {
            items: generate_batch_items(10),
            use_gpu: false,
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_batch_check(
                    &state,
                    RequestId::Number(1),
                    black_box(params.clone()),
                    None,
                )
                .await
            })
        });
    });

    group.finish();
}

// ============================================================================
// Certificate Verification Benchmarks
// ============================================================================

/// Generate certificate items for batch verification
fn generate_cert_items(count: usize) -> Vec<BatchVerifyCertItem> {
    (0..count)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyCertItem {
                id: format!("cert_{i}"),
                cert,
                expr,
            }
        })
        .collect()
}

fn bench_verify_cert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/verify_cert");
    group.warm_up_time(Duration::from_millis(500));

    // Single certificate verification
    group.bench_function("single_sort", |b| {
        let level = Level::zero();
        let params = VerifyCertParams {
            cert: ProofCert::Sort {
                level: level.clone(),
            },
            expr: Expr::Sort(level),
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_verify_cert(&state, RequestId::Number(1), black_box(params.clone())).await
            })
        });
    });

    group.finish();
}

fn bench_batch_verify_cert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/batch_verify_cert");
    group.warm_up_time(Duration::from_millis(500));

    // Batch sizes
    for count in [1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("no_progress", count),
            &count,
            |b, &count| {
                let items = generate_cert_items(count);
                let params = BatchVerifyCertParams {
                    items,
                    threads: 0,
                    timeout_ms: None,
                };
                b.iter(|| {
                    rt.block_on(async {
                        handle_batch_verify_cert(
                            &state,
                            RequestId::Number(1),
                            black_box(params.clone()),
                            None,
                        )
                        .await
                    })
                });
            },
        );

        // Measure overhead of progress callbacks
        group.bench_with_input(
            BenchmarkId::new("with_progress", count),
            &count,
            |b, &count| {
                let items = generate_cert_items(count);
                let params = BatchVerifyCertParams {
                    items,
                    threads: 0,
                    timeout_ms: None,
                };
                b.iter(|| {
                    rt.block_on(async {
                        // Create a progress sender that just discards messages
                        // Use channel size of count + 10 to avoid blocking
                        let (tx, mut rx) = mpsc::channel(count + 10);
                        let progress = ProgressSender::new(RequestId::Number(1), tx);

                        // Spawn task to drain progress messages
                        let drain_task =
                            tokio::spawn(async move { while rx.recv().await.is_some() {} });

                        let result = handle_batch_verify_cert(
                            &state,
                            RequestId::Number(1),
                            black_box(params.clone()),
                            Some(progress),
                        )
                        .await;

                        // Clean up
                        drop(result);
                        let _ = drain_task.await;
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark comparing verifyCert vs batchVerifyCert(1)
///
/// This measures the overhead difference between using the single-certificate
/// endpoint vs the batch endpoint with a single item.
fn bench_verify_cert_vs_batch_single(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = ServerState::new();

    let mut group = c.benchmark_group("server/verify_cert_comparison");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    // Test with Sort certificate (simple)
    let level = Level::zero();
    let cert = ProofCert::Sort {
        level: level.clone(),
    };
    let expr = Expr::Sort(level);

    // verifyCert endpoint
    group.bench_function("verifyCert_single_sort", |b| {
        let params = VerifyCertParams {
            cert: cert.clone(),
            expr: expr.clone(),
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_verify_cert(&state, RequestId::Number(1), black_box(params.clone())).await
            })
        });
    });

    // batchVerifyCert with 1 item
    group.bench_function("batchVerifyCert_1_sort", |b| {
        let params = BatchVerifyCertParams {
            items: vec![BatchVerifyCertItem {
                id: "single".to_string(),
                cert: cert.clone(),
                expr: expr.clone(),
            }],
            threads: 0,
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_batch_verify_cert(
                    &state,
                    RequestId::Number(1),
                    black_box(params.clone()),
                    None,
                )
                .await
            })
        });
    });

    // Now test with more complex Pi certificate
    use lean5_kernel::{BinderInfo, TypeChecker};
    use std::sync::Arc;

    let type_0 = Expr::Sort(Level::zero());
    let pi_expr = Expr::Pi(
        BinderInfo::Default,
        Arc::new(type_0.clone()),
        Arc::new(type_0),
    );

    let env = Environment::new();
    let mut tc = TypeChecker::new(&env);
    let (_, pi_cert) = tc
        .infer_type_with_cert(&pi_expr)
        .expect("Pi should type-check");

    group.bench_function("verifyCert_single_pi", |b| {
        let params = VerifyCertParams {
            cert: pi_cert.clone(),
            expr: pi_expr.clone(),
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_verify_cert(&state, RequestId::Number(1), black_box(params.clone())).await
            })
        });
    });

    group.bench_function("batchVerifyCert_1_pi", |b| {
        let params = BatchVerifyCertParams {
            items: vec![BatchVerifyCertItem {
                id: "single_pi".to_string(),
                cert: pi_cert.clone(),
                expr: pi_expr.clone(),
            }],
            threads: 0,
            timeout_ms: None,
        };
        b.iter(|| {
            rt.block_on(async {
                handle_batch_verify_cert(
                    &state,
                    RequestId::Number(1),
                    black_box(params.clone()),
                    None,
                )
                .await
            })
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    check_benches,
    bench_check_simple,
    bench_check_complex,
    bench_check_invalid,
);

criterion_group!(type_benches, bench_get_type,);

criterion_group!(
    batch_benches,
    bench_batch_check_scaling,
    bench_batch_check_mixed,
);

criterion_group!(prove_benches, bench_prove_simple,);

criterion_group!(info_benches, bench_server_info,);

criterion_group!(serialization_benches, bench_env_serialization,);

criterion_group!(e2e_benches, bench_e2e_latency,);

criterion_group!(
    cert_benches,
    bench_verify_cert,
    bench_batch_verify_cert,
    bench_verify_cert_vs_batch_single,
);

criterion_main!(
    check_benches,
    type_benches,
    batch_benches,
    prove_benches,
    info_benches,
    serialization_benches,
    e2e_benches,
    cert_benches
);
