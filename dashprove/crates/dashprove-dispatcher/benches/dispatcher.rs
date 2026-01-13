//! Performance benchmarks for the DashProve dispatcher
//!
//! Run with: cargo bench -p dashprove-dispatcher
//!
//! These benchmarks measure the performance of dispatcher operations:
//! - Backend registry operations
//! - Backend selector logic
//! - Result merger strategies

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use dashprove_backends::{
    BackendId, BackendResult, HealthStatus, PropertyType, VerificationStatus,
};
use dashprove_dispatcher::{
    BackendRegistry, BackendSelector, MergeStrategy, ResultMerger, SelectionStrategy, TaskResult,
};
use dashprove_usl::ast::{Expr, Property, Theorem};
use std::collections::HashMap;
use std::time::Duration;

// Helper to create test properties
fn make_properties(count: usize) -> Vec<Property> {
    (0..count)
        .map(|i| {
            Property::Theorem(Theorem {
                name: format!("theorem_{i}"),
                body: Expr::Bool(true),
            })
        })
        .collect()
}

// Helper to create a registry with given number of backends
fn make_registry(backend_count: usize) -> BackendRegistry {
    let mut registry = BackendRegistry::new();

    // Use actual BackendId variants
    let all_backends = [
        BackendId::Lean4,
        BackendId::TlaPlus,
        BackendId::Kani,
        BackendId::Alloy,
        BackendId::Coq,
        BackendId::Dafny,
        BackendId::Isabelle,
        BackendId::Z3,
        BackendId::Cvc5,
        BackendId::Verus,
        BackendId::Creusot,
        BackendId::Prusti,
        BackendId::Storm,
        BackendId::Prism,
        BackendId::Tamarin,
        BackendId::ProVerif,
        BackendId::Verifpal,
        BackendId::Marabou,
        BackendId::AlphaBetaCrown,
        BackendId::Eran,
    ];

    for &backend_id in all_backends.iter().take(backend_count) {
        // Create a mock backend (we'll use our test mock)
        let backend = MockBackend::new(backend_id);
        registry.register(std::sync::Arc::new(backend));
    }

    registry
}

// Mock backend for benchmarks
struct MockBackend {
    id: BackendId,
    supported: Vec<PropertyType>,
}

impl MockBackend {
    fn new(id: BackendId) -> Self {
        // Assign property types based on backend
        let supported = match id {
            BackendId::Lean4 | BackendId::Coq | BackendId::Isabelle => {
                vec![PropertyType::Theorem, PropertyType::Invariant]
            }
            BackendId::TlaPlus | BackendId::Apalache => {
                vec![PropertyType::Temporal, PropertyType::Invariant]
            }
            BackendId::Kani | BackendId::Verus | BackendId::Creusot | BackendId::Prusti => {
                vec![PropertyType::Contract]
            }
            BackendId::Alloy => vec![PropertyType::Invariant, PropertyType::Theorem],
            BackendId::Dafny => {
                vec![PropertyType::Contract, PropertyType::Theorem]
            }
            BackendId::Z3 | BackendId::Cvc5 => {
                vec![PropertyType::Theorem, PropertyType::Invariant]
            }
            BackendId::Storm | BackendId::Prism => vec![PropertyType::Probabilistic],
            BackendId::Tamarin | BackendId::ProVerif | BackendId::Verifpal => {
                vec![PropertyType::SecurityProtocol]
            }
            BackendId::Marabou | BackendId::AlphaBetaCrown | BackendId::Eran => {
                vec![PropertyType::NeuralRobustness]
            }
            BackendId::PlatformApi => vec![PropertyType::PlatformApi],
            // Phase 12 backends - assign reasonable property types for benchmarks
            _ => vec![PropertyType::Contract],
        };
        MockBackend { id, supported }
    }
}

#[async_trait::async_trait]
impl dashprove_backends::VerificationBackend for MockBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn supports(&self) -> Vec<PropertyType> {
        self.supported.clone()
    }

    async fn verify(
        &self,
        _spec: &dashprove_usl::typecheck::TypedSpec,
    ) -> Result<BackendResult, dashprove_backends::BackendError> {
        Ok(BackendResult {
            backend: self.id,
            status: VerificationStatus::Proven,
            proof: Some("mock proof".into()),
            counterexample: None,
            diagnostics: vec![],
            time_taken: Duration::from_millis(10),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

// Helper to create task results for merger benchmarks
fn make_execution_results(
    property_count: usize,
    backends_per_property: usize,
) -> dashprove_dispatcher::ExecutionResults {
    let all_backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::Alloy,
        BackendId::Dafny,
    ];

    let mut by_property: HashMap<usize, Vec<TaskResult>> = HashMap::new();
    let mut property_types: HashMap<usize, PropertyType> = HashMap::new();
    let mut successful = 0;
    let failed = 0;

    for prop_idx in 0..property_count {
        property_types.insert(prop_idx, PropertyType::Theorem);
        let results: Vec<TaskResult> = (0..backends_per_property)
            .map(|backend_idx| {
                let backend_id = all_backends[backend_idx % all_backends.len()];
                successful += 1;
                TaskResult {
                    property_index: prop_idx,
                    backend: backend_id,
                    result: Ok(BackendResult {
                        backend: backend_id,
                        status: if prop_idx % 3 == 0 {
                            VerificationStatus::Proven
                        } else if prop_idx % 3 == 1 {
                            VerificationStatus::Disproven
                        } else {
                            VerificationStatus::Unknown {
                                reason: "timeout".to_string(),
                            }
                        },
                        proof: Some("proof".to_string()),
                        counterexample: None,
                        diagnostics: vec![],
                        time_taken: Duration::from_millis(100),
                    }),
                }
            })
            .collect();
        by_property.insert(prop_idx, results);
    }

    dashprove_dispatcher::ExecutionResults {
        by_property,
        property_types,
        total_time: Duration::from_secs(1),
        successful,
        failed,
    }
}

fn bench_registry_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry");

    // Registration benchmark
    group.bench_function("register_5_backends", |b| {
        b.iter(|| {
            let mut registry = BackendRegistry::new();
            for &backend_id in &[
                BackendId::Lean4,
                BackendId::TlaPlus,
                BackendId::Kani,
                BackendId::Alloy,
                BackendId::Coq,
            ] {
                registry.register(std::sync::Arc::new(MockBackend::new(backend_id)));
            }
            registry
        })
    });

    // Lookup benchmarks
    let registry = make_registry(10);

    group.bench_with_input(
        BenchmarkId::new("backends_for_type", "Theorem"),
        &(&registry, PropertyType::Theorem),
        |b, (reg, pt)| b.iter(|| reg.backends_for_type(black_box(*pt))),
    );

    group.bench_with_input(
        BenchmarkId::new("backends_for_type", "Contract"),
        &(&registry, PropertyType::Contract),
        |b, (reg, pt)| b.iter(|| reg.backends_for_type(black_box(*pt))),
    );

    group.bench_with_input(
        BenchmarkId::new("healthy_backends_for_type", "Theorem"),
        &(&registry, PropertyType::Theorem),
        |b, (reg, pt)| b.iter(|| reg.healthy_backends_for_type(black_box(*pt))),
    );

    group.bench_with_input(
        BenchmarkId::new("get", "Lean4"),
        &(&registry, BackendId::Lean4),
        |b, (reg, id)| b.iter(|| reg.get(black_box(*id))),
    );

    group.bench_with_input(
        BenchmarkId::new("get_info", "Lean4"),
        &(&registry, BackendId::Lean4),
        |b, (reg, id)| b.iter(|| reg.get_info(black_box(*id))),
    );

    group.bench_function("all_backends/10", |b| {
        let reg = make_registry(10);
        b.iter(|| black_box(&reg).all_backends())
    });

    group.finish();
}

fn bench_selector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("selector");

    // Build registries of different sizes
    let registry_5 = make_registry(5);
    let registry_10 = make_registry(10);
    let registry_20 = make_registry(20);

    // Build property lists of different sizes
    let props_5 = make_properties(5);
    let props_10 = make_properties(10);
    let props_50 = make_properties(50);

    // Single strategy selection
    group.bench_with_input(
        BenchmarkId::new("select/Single", "5 props, 5 backends"),
        &(&registry_5, &props_5),
        |b, (reg, props)| {
            let selector = BackendSelector::new(reg, SelectionStrategy::Single);
            b.iter(|| selector.select(black_box(props)))
        },
    );

    group.bench_with_input(
        BenchmarkId::new("select/Single", "10 props, 10 backends"),
        &(&registry_10, &props_10),
        |b, (reg, props)| {
            let selector = BackendSelector::new(reg, SelectionStrategy::Single);
            b.iter(|| selector.select(black_box(props)))
        },
    );

    group.bench_with_input(
        BenchmarkId::new("select/Single", "50 props, 20 backends"),
        &(&registry_20, &props_50),
        |b, (reg, props)| {
            let selector = BackendSelector::new(reg, SelectionStrategy::Single);
            b.iter(|| selector.select(black_box(props)))
        },
    );

    // All strategy selection
    group.bench_with_input(
        BenchmarkId::new("select/All", "10 props, 10 backends"),
        &(&registry_10, &props_10),
        |b, (reg, props)| {
            let selector = BackendSelector::new(reg, SelectionStrategy::All);
            b.iter(|| selector.select(black_box(props)))
        },
    );

    // Redundant strategy selection
    group.bench_with_input(
        BenchmarkId::new("select/Redundant", "10 props, 10 backends"),
        &(&registry_10, &props_10),
        |b, (reg, props)| {
            let selector =
                BackendSelector::new(reg, SelectionStrategy::Redundant { min_backends: 2 });
            b.iter(|| selector.select(black_box(props)))
        },
    );

    // Specific backend strategy
    group.bench_with_input(
        BenchmarkId::new("select/Specific", "10 props"),
        &(&registry_10, &props_10),
        |b, (reg, props)| {
            let selector = BackendSelector::new(reg, SelectionStrategy::Specific(BackendId::Lean4));
            b.iter(|| selector.select(black_box(props)))
        },
    );

    group.finish();
}

fn bench_merger_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("merger");

    // FirstSuccess strategy - use iter_batched since ExecutionResults doesn't impl Clone
    group.bench_function("merge/FirstSuccess/5 props, 1 backend", |b| {
        let merger = ResultMerger::new(MergeStrategy::FirstSuccess);
        b.iter_batched(
            || make_execution_results(5, 1),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("merge/FirstSuccess/10 props, 1 backend", |b| {
        let merger = ResultMerger::new(MergeStrategy::FirstSuccess);
        b.iter_batched(
            || make_execution_results(10, 1),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("merge/FirstSuccess/50 props, 1 backend", |b| {
        let merger = ResultMerger::new(MergeStrategy::FirstSuccess);
        b.iter_batched(
            || make_execution_results(50, 1),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    // Unanimous strategy (requires multiple backends)
    group.bench_function("merge/Unanimous/10 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::Unanimous);
        b.iter_batched(
            || make_execution_results(10, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("merge/Unanimous/50 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::Unanimous);
        b.iter_batched(
            || make_execution_results(50, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    // Majority strategy
    group.bench_function("merge/Majority/10 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::Majority);
        b.iter_batched(
            || make_execution_results(10, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("merge/Majority/50 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::Majority);
        b.iter_batched(
            || make_execution_results(50, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    // MostConfident strategy
    group.bench_function("merge/MostConfident/10 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::MostConfident);
        b.iter_batched(
            || make_execution_results(10, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    // Pessimistic and Optimistic strategies
    group.bench_function("merge/Pessimistic/10 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::Pessimistic);
        b.iter_batched(
            || make_execution_results(10, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("merge/Optimistic/10 props, 3 backends", |b| {
        let merger = ResultMerger::new(MergeStrategy::Optimistic);
        b.iter_batched(
            || make_execution_results(10, 3),
            |results| merger.merge(black_box(results)),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_registry_operations,
    bench_selector_operations,
    bench_merger_operations,
);

criterion_main!(benches);
