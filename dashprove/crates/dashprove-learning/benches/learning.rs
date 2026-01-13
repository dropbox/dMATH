//! Benchmarks for the learning system
//!
//! Run with: `cargo bench -p dashprove-learning`
//!
//! Measures the performance of:
//! - Feature extraction from properties
//! - Embedding computation (structural and keyword)
//! - Similarity computation (cosine, L2, feature-based)
//! - Embedding index operations (insert, nearest neighbor search)
//! - Proof corpus operations (insert, find_similar, search_by_keywords)
//! - Tactic database operations

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use dashprove_backends::traits::{BackendId, CounterexampleValue, FailedCheck, TraceState};
use dashprove_backends::CounterexampleClusters;
use dashprove_learning::{
    corpus::{
        BatchedHnswBuilder, BatchedHnswConfig, BatchedHnswIndex, StreamingIvfBuilder,
        StreamingIvfConfig, StreamingIvfIndex,
    },
    embedder::{Embedding, EmbeddingIndex, EmbeddingIndexBuilder, PropertyEmbedder},
    euclidean_distance_sq, euclidean_distance_sq_scalar,
    similarity::{compute_similarity, extract_features},
    CounterexampleCorpus, LearnableResult, ProofCorpus, ProofId, ProofLearningSystem,
    TacticContext, TacticDatabase,
};
use dashprove_usl::ast::{
    BinaryOp, ComparisonOp, Contract, Expr, Invariant, Property, Temporal, TemporalExpr, Theorem,
    Type,
};
use std::time::Duration;

// ============================================================================
// Helper functions to create test data
// ============================================================================

fn make_simple_theorem(name: &str) -> Property {
    Property::Theorem(Theorem {
        name: name.to_string(),
        body: Expr::Bool(true),
    })
}

fn make_quantified_theorem(name: &str, depth: usize) -> Property {
    let mut body: Expr = Expr::Bool(true);
    for i in 0..depth {
        body = Expr::ForAll {
            var: format!("x{}", i),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(body),
        };
    }
    Property::Theorem(Theorem {
        name: name.to_string(),
        body,
    })
}

fn make_complex_theorem(name: &str) -> Property {
    let body = Expr::ForAll {
        var: "n".to_string(),
        ty: Some(Type::Named("Int".to_string())),
        body: Box::new(Expr::Implies(
            Box::new(Expr::Compare(
                Box::new(Expr::Var("n".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Int(0)),
            )),
            Box::new(Expr::Compare(
                Box::new(Expr::Binary(
                    Box::new(Expr::Var("n".to_string())),
                    BinaryOp::Mul,
                    Box::new(Expr::Var("n".to_string())),
                )),
                ComparisonOp::Ge,
                Box::new(Expr::Int(0)),
            )),
        )),
    };
    Property::Theorem(Theorem {
        name: name.to_string(),
        body,
    })
}

fn make_temporal_property(name: &str) -> Property {
    Property::Temporal(Temporal {
        name: name.to_string(),
        body: TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
            TemporalExpr::Atom(Expr::Var("done".to_string())),
        )))),
        fairness: vec![],
    })
}

fn make_invariant(name: &str) -> Property {
    Property::Invariant(Invariant {
        name: name.to_string(),
        body: Expr::And(
            Box::new(Expr::Compare(
                Box::new(Expr::Var("count".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Int(0)),
            )),
            Box::new(Expr::Compare(
                Box::new(Expr::Var("count".to_string())),
                ComparisonOp::Le,
                Box::new(Expr::Var("capacity".to_string())),
            )),
        ),
    })
}

fn make_contract(name: &str) -> Property {
    Property::Contract(Contract {
        type_path: vec![name.to_string()],
        params: vec![],
        return_type: None,
        requires: vec![Expr::Compare(
            Box::new(Expr::Var("len".to_string())),
            ComparisonOp::Gt,
            Box::new(Expr::Int(0)),
        )],
        ensures: vec![Expr::Compare(
            Box::new(Expr::Var("result".to_string())),
            ComparisonOp::Eq,
            Box::new(Expr::Bool(true)),
        )],
        ensures_err: vec![],
        assigns: vec![],
        allocates: vec![],
        frees: vec![],
        terminates: None,
        decreases: None,
        behaviors: vec![],
        complete_behaviors: false,
        disjoint_behaviors: false,
    })
}

fn make_learnable_result(property: Property, backend: BackendId) -> LearnableResult {
    LearnableResult {
        property,
        backend,
        status: dashprove_backends::traits::VerificationStatus::Proven,
        tactics: vec!["simp".to_string(), "decide".to_string()],
        time_taken: Duration::from_millis(100),
        proof_output: Some("proof term".to_string()),
    }
}

fn make_counterexample(
    witness_vars: &[(&str, i128)],
    check_desc: &str,
) -> dashprove_backends::traits::StructuredCounterexample {
    let mut cx = dashprove_backends::traits::StructuredCounterexample::new();

    for (name, value) in witness_vars {
        cx.witness.insert(
            name.to_string(),
            CounterexampleValue::Int {
                value: *value,
                type_hint: None,
            },
        );
    }

    if !check_desc.is_empty() {
        cx.failed_checks.push(FailedCheck {
            check_id: "test_check".to_string(),
            description: check_desc.to_string(),
            location: None,
            function: None,
        });
    }

    let mut state = TraceState::new(1);
    state.action = Some("Init".to_string());
    state.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    cx.trace.push(state);

    cx
}

// ============================================================================
// Feature Extraction Benchmarks
// ============================================================================

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let simple = make_simple_theorem("simple");
    let quantified = make_quantified_theorem("quantified", 3);
    let complex = make_complex_theorem("complex");
    let temporal = make_temporal_property("temporal");
    let invariant = make_invariant("invariant");
    let contract = make_contract("contract");

    group.bench_function("simple_theorem", |b| {
        b.iter(|| extract_features(black_box(&simple)))
    });

    group.bench_function("quantified_theorem", |b| {
        b.iter(|| extract_features(black_box(&quantified)))
    });

    group.bench_function("complex_theorem", |b| {
        b.iter(|| extract_features(black_box(&complex)))
    });

    group.bench_function("temporal_property", |b| {
        b.iter(|| extract_features(black_box(&temporal)))
    });

    group.bench_function("invariant", |b| {
        b.iter(|| extract_features(black_box(&invariant)))
    });

    group.bench_function("contract", |b| {
        b.iter(|| extract_features(black_box(&contract)))
    });

    group.finish();
}

// ============================================================================
// Embedding Benchmarks
// ============================================================================

fn bench_embedding_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

    // Pre-extract features for embedding benchmarks
    let simple_features = extract_features(&make_simple_theorem("simple"));
    let complex_features = extract_features(&make_complex_theorem("complex"));

    // Embedding creation
    group.bench_function("create_zeros", |b| {
        b.iter(|| Embedding::zeros(black_box(96)))
    });

    group.bench_function("create_from_vec", |b| {
        let vec = vec![0.5f32; 96];
        b.iter(|| Embedding::new(black_box(vec.clone())))
    });

    // Property embedding
    group.bench_function("embed_simple_property", |b| {
        b.iter_batched(
            PropertyEmbedder::new,
            |mut embedder| embedder.embed(black_box(&make_simple_theorem("test"))),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("embed_complex_property", |b| {
        b.iter_batched(
            PropertyEmbedder::new,
            |mut embedder| embedder.embed(black_box(&make_complex_theorem("test"))),
            BatchSize::SmallInput,
        )
    });

    // Feature-based embedding
    group.bench_function("embed_features_simple", |b| {
        b.iter_batched(
            PropertyEmbedder::new,
            |mut embedder| embedder.embed_features(black_box(&simple_features)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("embed_features_complex", |b| {
        b.iter_batched(
            PropertyEmbedder::new,
            |mut embedder| embedder.embed_features(black_box(&complex_features)),
            BatchSize::SmallInput,
        )
    });

    // Query embedding (readonly)
    let mut trained_embedder = PropertyEmbedder::new();
    for i in 0..50 {
        trained_embedder.embed(&make_simple_theorem(&format!("train_{}", i)));
    }

    group.bench_function("embed_query_readonly", |b| {
        let query = make_simple_theorem("query");
        b.iter(|| trained_embedder.embed_query(black_box(&query)))
    });

    group.finish();
}

fn bench_similarity_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    // Create embeddings for similarity benchmarks
    let emb_a = Embedding::new(vec![0.5f32; 96]);
    let emb_b = Embedding::new(vec![0.3f32; 96]);

    group.bench_function("l2_distance", |b| {
        b.iter(|| black_box(&emb_a).l2_distance(black_box(&emb_b)))
    });

    group.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(&emb_a).cosine_similarity(black_box(&emb_b)))
    });

    group.bench_function("normalized_similarity", |b| {
        b.iter(|| black_box(&emb_a).normalized_similarity(black_box(&emb_b)))
    });

    group.bench_function("normalize_embedding", |b| {
        b.iter_batched(
            || Embedding::new(vec![0.5f32; 96]),
            |mut emb| {
                emb.normalize();
                emb
            },
            BatchSize::SmallInput,
        )
    });

    // Feature-based similarity
    let features_a = extract_features(&make_simple_theorem("a"));
    let features_b = extract_features(&make_complex_theorem("b"));

    group.bench_function("compute_feature_similarity", |b| {
        b.iter(|| compute_similarity(black_box(&features_a), black_box(&features_b)))
    });

    group.finish();
}

// ============================================================================
// Embedding Index Benchmarks
// ============================================================================

fn bench_embedding_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_index");

    // Build index of various sizes
    let properties_100: Vec<_> = (0..100)
        .map(|i| match i % 4 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            _ => make_invariant(&format!("prop_{}", i)),
        })
        .collect();

    let properties_500: Vec<_> = (0..500)
        .map(|i| match i % 4 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            _ => make_invariant(&format!("prop_{}", i)),
        })
        .collect();

    // Index building benchmarks
    group.bench_function(BenchmarkId::new("build_index", "100"), |b| {
        b.iter(|| {
            let mut builder = EmbeddingIndexBuilder::new();
            for (i, prop) in properties_100.iter().enumerate() {
                builder.add(format!("id_{}", i), black_box(prop));
            }
            builder.build()
        })
    });

    // Pre-build indices for search benchmarks
    let mut builder_100 = EmbeddingIndexBuilder::new();
    for (i, prop) in properties_100.iter().enumerate() {
        builder_100.add(format!("id_{}", i), prop);
    }
    let index_100 = builder_100.build();

    let mut builder_500 = EmbeddingIndexBuilder::new();
    for (i, prop) in properties_500.iter().enumerate() {
        builder_500.add(format!("id_{}", i), prop);
    }
    let index_500 = builder_500.build();

    let query = make_complex_theorem("query");

    // Nearest neighbor search benchmarks
    group.bench_function(BenchmarkId::new("find_nearest", "100_k5"), |b| {
        b.iter(|| index_100.find_nearest(black_box(&query), 5))
    });

    group.bench_function(BenchmarkId::new("find_nearest", "100_k10"), |b| {
        b.iter(|| index_100.find_nearest(black_box(&query), 10))
    });

    group.bench_function(BenchmarkId::new("find_nearest", "500_k5"), |b| {
        b.iter(|| index_500.find_nearest(black_box(&query), 5))
    });

    group.bench_function(BenchmarkId::new("find_nearest", "500_k10"), |b| {
        b.iter(|| index_500.find_nearest(black_box(&query), 10))
    });

    // Insert benchmarks
    group.bench_function("insert_single", |b| {
        let prop = make_simple_theorem("new_prop");
        b.iter_batched(
            EmbeddingIndex::new,
            |mut index| {
                index.insert("new_id".to_string(), black_box(&prop));
                index
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Proof Corpus Benchmarks
// ============================================================================

fn bench_proof_corpus(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_corpus");

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Insert benchmarks
    group.bench_function("insert_single", |b| {
        let result = make_learnable_result(make_simple_theorem("test"), BackendId::Lean4);
        b.iter_batched(
            ProofCorpus::new,
            |mut corpus| {
                corpus.insert(black_box(&result));
                corpus
            },
            BatchSize::SmallInput,
        )
    });

    // Build corpora of various sizes for search benchmarks
    let mut corpus_100 = ProofCorpus::new();
    for i in 0..100 {
        let prop = match i % 4 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            _ => make_invariant(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop, backends[i % backends.len()]);
        corpus_100.insert(&result);
    }

    let mut corpus_500 = ProofCorpus::new();
    for i in 0..500 {
        let prop = match i % 4 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            _ => make_invariant(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop, backends[i % backends.len()]);
        corpus_500.insert(&result);
    }

    let mut corpus_2000 = ProofCorpus::new();
    for i in 0..2000 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop, backends[i % backends.len()]);
        corpus_2000.insert(&result);
    }

    let query = make_complex_theorem("query");

    // Similarity search benchmarks
    group.bench_function(BenchmarkId::new("find_similar", "100_k5"), |b| {
        b.iter(|| corpus_100.find_similar(black_box(&query), 5))
    });

    group.bench_function(BenchmarkId::new("find_similar", "100_k10"), |b| {
        b.iter(|| corpus_100.find_similar(black_box(&query), 10))
    });

    group.bench_function(BenchmarkId::new("find_similar", "500_k5"), |b| {
        b.iter(|| corpus_500.find_similar(black_box(&query), 5))
    });

    group.bench_function(BenchmarkId::new("find_similar", "500_k10"), |b| {
        b.iter(|| corpus_500.find_similar(black_box(&query), 10))
    });

    group.bench_function(BenchmarkId::new("find_similar", "2000_k5"), |b| {
        b.iter(|| corpus_2000.find_similar(black_box(&query), 5))
    });

    group.bench_function(BenchmarkId::new("find_similar", "2000_k25"), |b| {
        b.iter(|| corpus_2000.find_similar(black_box(&query), 25))
    });

    // Keyword search benchmarks
    group.bench_function(BenchmarkId::new("search_keywords", "100"), |b| {
        b.iter(|| corpus_100.search_by_keywords(black_box("theorem"), 5))
    });

    group.bench_function(BenchmarkId::new("search_keywords", "500"), |b| {
        b.iter(|| corpus_500.search_by_keywords(black_box("theorem"), 5))
    });

    group.bench_function(BenchmarkId::new("search_keywords", "2000"), |b| {
        b.iter(|| corpus_2000.search_by_keywords(black_box("theorem"), 5))
    });

    group.finish();
}

// ============================================================================
// Counterexample Corpus Benchmarks
// ============================================================================

fn bench_counterexample_corpus(c: &mut Criterion) {
    let mut group = c.benchmark_group("counterexample_corpus");

    // Insert benchmarks
    group.bench_function("insert_single", |b| {
        let cx = make_counterexample(&[("n", 42)], "division by zero");
        b.iter_batched(
            CounterexampleCorpus::new,
            |mut corpus| {
                corpus.insert("test_prop", BackendId::TlaPlus, black_box(cx.clone()), None);
                corpus
            },
            BatchSize::SmallInput,
        )
    });

    // Build corpus for search benchmarks
    let mut corpus_100 = CounterexampleCorpus::new();
    for i in 0..100 {
        let cx = make_counterexample(&[("n", i as i128)], &format!("check_{}", i % 5));
        corpus_100.insert(&format!("prop_{}", i), BackendId::TlaPlus, cx, None);
    }

    let query_cx = make_counterexample(&[("n", 50)], "check_0");

    // Similarity search benchmarks
    group.bench_function(BenchmarkId::new("find_similar", "100_k5"), |b| {
        b.iter(|| corpus_100.find_similar(black_box(&query_cx), 5))
    });

    group.bench_function(BenchmarkId::new("find_similar", "100_k10"), |b| {
        b.iter(|| corpus_100.find_similar(black_box(&query_cx), 10))
    });

    // Keyword search benchmarks
    group.bench_function(BenchmarkId::new("search_keywords", "100"), |b| {
        b.iter(|| corpus_100.search_by_keywords(black_box("division"), 5))
    });

    // Classification benchmarks
    let cx1 = make_counterexample(&[("x", 1)], "overflow");
    let cx2 = make_counterexample(&[("x", 2)], "overflow");
    let clusters = CounterexampleClusters::from_counterexamples(vec![cx1, cx2], 0.5);
    let mut corpus_with_patterns = CounterexampleCorpus::new();
    corpus_with_patterns.record_clusters(&clusters);

    let classify_query = make_counterexample(&[("x", 3)], "overflow");

    group.bench_function("classify_counterexample", |b| {
        b.iter(|| corpus_with_patterns.classify(black_box(&classify_query)))
    });

    group.finish();
}

// ============================================================================
// Tactic Database Benchmarks
// ============================================================================

fn bench_tactic_database(c: &mut Criterion) {
    let mut group = c.benchmark_group("tactic_database");

    let tactics = ["simp", "decide", "ring", "omega", "aesop", "norm_num"];

    // Helper to create a populated database
    fn create_populated_db() -> TacticDatabase {
        let mut db = TacticDatabase::new();
        let tactics = ["simp", "decide", "ring", "omega", "aesop", "norm_num"];

        for i in 0..100 {
            let features = extract_features(&match i % 4 {
                0 => make_simple_theorem(&format!("prop_{}", i)),
                1 => make_complex_theorem(&format!("prop_{}", i)),
                2 => make_temporal_property(&format!("prop_{}", i)),
                _ => make_invariant(&format!("prop_{}", i)),
            });
            let context = TacticContext::from_features(&features);

            for (j, tactic) in tactics.iter().enumerate() {
                if (i + j) % 3 == 0 {
                    db.record_success(&context, tactic);
                } else {
                    db.record_failure(&context, tactic);
                }
            }
        }
        db
    }

    // Record operations - use iter_batched to create fresh db each time
    let test_features = extract_features(&make_simple_theorem("test"));
    let test_context = TacticContext::from_features(&test_features);

    group.bench_function("record_success", |b| {
        b.iter_batched(
            create_populated_db,
            |mut db| {
                db.record_success(black_box(&test_context), "test_tactic");
                db
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("record_failure", |b| {
        b.iter_batched(
            create_populated_db,
            |mut db| {
                db.record_failure(black_box(&test_context), "test_tactic");
                db
            },
            BatchSize::LargeInput,
        )
    });

    // Query operations - create db once for read-only benchmarks
    let db = create_populated_db();

    group.bench_function("best_for_context_5", |b| {
        b.iter(|| db.best_for_context(black_box(&test_context), 5))
    });

    group.bench_function("best_for_context_10", |b| {
        b.iter(|| db.best_for_context(black_box(&test_context), 10))
    });

    // Also bench context creation from features
    let complex_prop = make_complex_theorem("complex");
    let complex_features = extract_features(&complex_prop);

    group.bench_function("context_from_features", |b| {
        b.iter(|| TacticContext::from_features(black_box(&complex_features)))
    });

    // Drop unused variable warning fix
    let _ = tactics;

    group.finish();
}

// ============================================================================
// ProofLearningSystem Integration Benchmarks
// ============================================================================

fn bench_learning_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("learning_system");
    group.sample_size(50);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Helper to build a populated learning system
    fn create_populated_system() -> ProofLearningSystem {
        let backends = [
            BackendId::Lean4,
            BackendId::Coq,
            BackendId::TlaPlus,
            BackendId::Kani,
        ];

        let mut system = ProofLearningSystem::new();
        for i in 0..200 {
            let prop = match i % 5 {
                0 => make_simple_theorem(&format!("prop_{}", i)),
                1 => make_complex_theorem(&format!("prop_{}", i)),
                2 => make_temporal_property(&format!("prop_{}", i)),
                3 => make_invariant(&format!("prop_{}", i)),
                _ => make_contract(&format!("prop_{}", i)),
            };
            let result = make_learnable_result(prop, backends[i % backends.len()]);
            system.record(&result);
        }
        system
    }

    // Record operation - create fresh system for mutation benchmarks
    group.bench_function("record_proof", |b| {
        let result = make_learnable_result(make_simple_theorem("new"), BackendId::Lean4);
        b.iter_batched(
            create_populated_system,
            |mut system| {
                system.record(black_box(&result));
                system
            },
            BatchSize::LargeInput,
        )
    });

    // Build system once for read-only benchmarks
    let system = create_populated_system();

    // Search operations
    let query = make_complex_theorem("query");

    group.bench_function("find_similar_5", |b| {
        b.iter(|| system.find_similar(black_box(&query), 5))
    });

    group.bench_function("find_similar_10", |b| {
        b.iter(|| system.find_similar(black_box(&query), 10))
    });

    group.bench_function("suggest_tactics_5", |b| {
        b.iter(|| system.suggest_tactics(black_box(&query), 5))
    });

    group.bench_function("search_by_keywords", |b| {
        b.iter(|| system.search_by_keywords(black_box("theorem"), 5))
    });

    // Drop unused variable
    let _ = backends;

    group.finish();
}

// ============================================================================
// OrderedF64 and Top-K Selection Microbenchmarks
// ============================================================================

fn bench_ordered_float(c: &mut Criterion) {
    use dashprove_learning::ordered_float::OrderedF64;
    use std::collections::BinaryHeap;

    let mut group = c.benchmark_group("ordered_float");

    // Benchmark OrderedF64 comparisons
    group.bench_function("compare_normal", |b| {
        let a = OrderedF64(0.5);
        let z = OrderedF64(0.8);
        b.iter(|| black_box(&a).cmp(black_box(&z)))
    });

    group.bench_function("compare_with_nan", |b| {
        let a = OrderedF64(0.5);
        let nan = OrderedF64(f64::NAN);
        b.iter(|| black_box(&a).cmp(black_box(&nan)))
    });

    group.bench_function("compare_two_nans", |b| {
        let nan1 = OrderedF64(f64::NAN);
        let nan2 = OrderedF64(f64::NAN);
        b.iter(|| black_box(&nan1).cmp(black_box(&nan2)))
    });

    // Benchmark BinaryHeap operations with OrderedF64
    group.bench_function("heap_push_100", |b| {
        b.iter_batched(
            || BinaryHeap::with_capacity(100),
            |mut heap| {
                for i in 0..100 {
                    heap.push(OrderedF64(i as f64 / 100.0));
                }
                heap
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("heap_push_pop_maintain_k10", |b| {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.001) % 1.0).collect();
        b.iter(|| {
            let mut heap: BinaryHeap<std::cmp::Reverse<OrderedF64>> = BinaryHeap::with_capacity(11);
            for &score in &data {
                let ordered = OrderedF64(score);
                if heap.len() < 10 {
                    heap.push(std::cmp::Reverse(ordered));
                } else if let Some(std::cmp::Reverse(min)) = heap.peek() {
                    if ordered > *min {
                        heap.pop();
                        heap.push(std::cmp::Reverse(ordered));
                    }
                }
            }
            heap
        })
    });

    group.bench_function("heap_push_pop_maintain_k50", |b| {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.001) % 1.0).collect();
        b.iter(|| {
            let mut heap: BinaryHeap<std::cmp::Reverse<OrderedF64>> = BinaryHeap::with_capacity(51);
            for &score in &data {
                let ordered = OrderedF64(score);
                if heap.len() < 50 {
                    heap.push(std::cmp::Reverse(ordered));
                } else if let Some(std::cmp::Reverse(min)) = heap.peek() {
                    if ordered > *min {
                        heap.pop();
                        heap.push(std::cmp::Reverse(ordered));
                    }
                }
            }
            heap
        })
    });

    group.finish();
}

fn bench_top_k_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_scaling");
    group.sample_size(50);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpora of various sizes to measure scaling behavior
    let mut corpus_1000 = ProofCorpus::new();
    for i in 0..1000 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop, backends[i % backends.len()]);
        corpus_1000.insert(&result);
    }

    let query = make_complex_theorem("query");

    // Benchmark find_similar with different k values to measure O(n log k) behavior
    for k in [5, 10, 25, 50, 100] {
        group.bench_function(BenchmarkId::new("1000_entries", format!("k{}", k)), |b| {
            b.iter(|| corpus_1000.find_similar(black_box(&query), k))
        });
    }

    // Benchmark keyword search scaling
    for k in [5, 10, 25, 50, 100] {
        group.bench_function(BenchmarkId::new("keywords_1000", format!("k{}", k)), |b| {
            b.iter(|| corpus_1000.search_by_keywords(black_box("theorem"), k))
        });
    }

    group.finish();
}

// ============================================================================
// LSH vs Exact Search Benchmarks
// ============================================================================

fn bench_lsh_vs_exact(c: &mut Criterion) {
    use dashprove_learning::embedder::{Embedding, PropertyEmbedder, EMBEDDING_DIM};
    use dashprove_learning::lsh::{LshConfig, LshIndex};
    use dashprove_learning::ProofCorpusLsh;

    let mut group = c.benchmark_group("lsh_vs_exact");
    group.sample_size(30);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Simple deterministic random embedding generator
    fn random_embedding(seed: u64) -> Embedding {
        let mut state = seed;
        let mut vector = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32);
            vector.push(f * 2.0 - 1.0);
        }
        Embedding::new(vector)
    }

    // Build corpus with embeddings for LSH testing
    let mut corpus_1000 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..1000 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_1000.insert_with_embedding(&result, embedding);
    }

    // Build LSH index
    let lsh_1000 = ProofCorpusLsh::build_auto_config(&corpus_1000).unwrap();

    // Create query embedding
    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    // Benchmark exact embedding search vs LSH approximate search
    group.bench_function(BenchmarkId::new("exact_embedding", "1000_k10"), |b| {
        b.iter(|| corpus_1000.find_similar_embedding(black_box(&query_embedding), 10))
    });

    group.bench_function(BenchmarkId::new("lsh_approximate", "1000_k10"), |b| {
        b.iter(|| {
            lsh_1000.find_similar_approximate(
                black_box(&corpus_1000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    group.bench_function(BenchmarkId::new("lsh_exact_via_index", "1000_k10"), |b| {
        b.iter(|| {
            lsh_1000.find_similar_exact(black_box(&corpus_1000), black_box(&query_embedding), 10)
        })
    });

    // Build larger corpus for more dramatic speedup
    let mut corpus_5000 = ProofCorpus::new();
    for i in 0..5000 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_5000.insert_with_embedding(&result, embedding);
    }

    let lsh_5000 = ProofCorpusLsh::build_auto_config(&corpus_5000).unwrap();

    group.bench_function(BenchmarkId::new("exact_embedding", "5000_k10"), |b| {
        b.iter(|| corpus_5000.find_similar_embedding(black_box(&query_embedding), 10))
    });

    group.bench_function(BenchmarkId::new("lsh_approximate", "5000_k10"), |b| {
        b.iter(|| {
            lsh_5000.find_similar_approximate(
                black_box(&corpus_5000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    // LSH index construction benchmark
    group.bench_function(BenchmarkId::new("lsh_build", "1000"), |b| {
        b.iter(|| ProofCorpusLsh::build_auto_config(black_box(&corpus_1000)))
    });

    // Standalone LshIndex benchmarks
    let config = LshConfig::for_medium_corpus();
    let mut lsh_standalone = LshIndex::new(config);
    for i in 0..1000 {
        lsh_standalone.insert(format!("id_{}", i), random_embedding(i as u64));
    }

    let query_emb = random_embedding(99999);

    group.bench_function(BenchmarkId::new("lsh_standalone_query", "1000_k10"), |b| {
        b.iter(|| lsh_standalone.query(black_box(&query_emb), 10))
    });

    group.bench_function(BenchmarkId::new("lsh_standalone_exact", "1000_k10"), |b| {
        b.iter(|| lsh_standalone.exact_search(black_box(&query_emb), 10))
    });

    group.finish();
}

// ============================================================================
// LSH Incremental Insert vs Full Rebuild Benchmarks
// ============================================================================

fn bench_lsh_incremental_vs_rebuild(c: &mut Criterion) {
    use dashprove_learning::embedder::{Embedding, PropertyEmbedder, EMBEDDING_DIM};
    use dashprove_learning::lsh::LshConfig;
    use dashprove_learning::ProofCorpusLsh;

    let mut group = c.benchmark_group("lsh_incremental_vs_rebuild");
    group.sample_size(20);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Helper to create random embedding
    fn random_embedding(seed: u64) -> Embedding {
        let mut state = seed;
        let mut vector = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32);
            vector.push(f * 2.0 - 1.0);
        }
        Embedding::new(vector)
    }

    // Helper to create corpus with n proofs
    fn create_corpus(
        n: usize,
        embedder: &mut PropertyEmbedder,
        backends: &[BackendId; 4],
    ) -> ProofCorpus {
        let mut corpus = ProofCorpus::new();
        for i in 0..n {
            let prop = match i % 5 {
                0 => make_simple_theorem(&format!("prop_{}", i)),
                1 => make_complex_theorem(&format!("prop_{}", i)),
                2 => make_temporal_property(&format!("prop_{}", i)),
                3 => make_invariant(&format!("prop_{}", i)),
                _ => make_contract(&format!("prop_{}", i)),
            };
            let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
            let embedding = embedder.embed(&prop);
            corpus.insert_with_embedding(&result, embedding);
        }
        corpus
    }

    // Create a base corpus with 1000 proofs
    let mut embedder = PropertyEmbedder::new();
    let base_corpus = create_corpus(1000, &mut embedder, &backends);

    // Benchmark: Incremental insert of 10 new proofs
    group.bench_function(BenchmarkId::new("incremental_insert", "10_proofs"), |b| {
        // Pre-generate new embeddings for insertion
        let new_embeddings: Vec<_> = (1000..1010).map(|i| random_embedding(i as u64)).collect();

        b.iter_batched(
            || {
                // Setup: build fresh LSH index
                ProofCorpusLsh::build_auto_config(&base_corpus).unwrap()
            },
            |mut lsh| {
                // Insert 10 new proofs incrementally
                for (i, embedding) in new_embeddings.iter().enumerate() {
                    let id = ProofId(format!("new_prop_{}", i));
                    lsh.insert(id, embedding.clone());
                }
                lsh
            },
            BatchSize::LargeInput,
        )
    });

    // Benchmark: Full rebuild after adding 10 proofs
    // Pre-build the extended corpus
    let corpus_1010 = create_corpus(1010, &mut embedder, &backends);
    group.bench_function(BenchmarkId::new("full_rebuild", "10_proofs"), |b| {
        b.iter(|| ProofCorpusLsh::build_auto_config(black_box(&corpus_1010)))
    });

    // Benchmark: Incremental insert of 50 new proofs
    group.bench_function(BenchmarkId::new("incremental_insert", "50_proofs"), |b| {
        let new_embeddings: Vec<_> = (1000..1050).map(|i| random_embedding(i as u64)).collect();

        b.iter_batched(
            || ProofCorpusLsh::build_auto_config(&base_corpus).unwrap(),
            |mut lsh| {
                for (i, embedding) in new_embeddings.iter().enumerate() {
                    let id = ProofId(format!("new_prop_{}", i));
                    lsh.insert(id, embedding.clone());
                }
                lsh
            },
            BatchSize::LargeInput,
        )
    });

    // Benchmark: Full rebuild after adding 50 proofs
    let corpus_1050 = create_corpus(1050, &mut embedder, &backends);
    group.bench_function(BenchmarkId::new("full_rebuild", "50_proofs"), |b| {
        b.iter(|| ProofCorpusLsh::build_auto_config(black_box(&corpus_1050)))
    });

    // Benchmark: Incremental insert of 100 new proofs
    group.bench_function(BenchmarkId::new("incremental_insert", "100_proofs"), |b| {
        let new_embeddings: Vec<_> = (1000..1100).map(|i| random_embedding(i as u64)).collect();

        b.iter_batched(
            || ProofCorpusLsh::build_auto_config(&base_corpus).unwrap(),
            |mut lsh| {
                for (i, embedding) in new_embeddings.iter().enumerate() {
                    let id = ProofId(format!("new_prop_{}", i));
                    lsh.insert(id, embedding.clone());
                }
                lsh
            },
            BatchSize::LargeInput,
        )
    });

    // Benchmark: Full rebuild after adding 100 proofs
    let corpus_1100 = create_corpus(1100, &mut embedder, &backends);
    group.bench_function(BenchmarkId::new("full_rebuild", "100_proofs"), |b| {
        b.iter(|| ProofCorpusLsh::build_auto_config(black_box(&corpus_1100)))
    });

    // Benchmark: sync_with_corpus batch operation (uses corpus_1050)
    group.bench_function(BenchmarkId::new("sync_with_corpus", "50_proofs"), |b| {
        b.iter_batched(
            || ProofCorpusLsh::build_auto_config(&base_corpus).unwrap(),
            |mut lsh| {
                lsh.sync_with_corpus(black_box(&corpus_1050));
                lsh
            },
            BatchSize::LargeInput,
        )
    });

    // Benchmark: Single proof insert (amortized cost)
    group.bench_function(BenchmarkId::new("single_insert", "per_proof"), |b| {
        let embedding = random_embedding(99999);

        b.iter_batched(
            || ProofCorpusLsh::build_auto_config(&base_corpus).unwrap(),
            |mut lsh| {
                let id = ProofId("single_new_proof".to_string());
                lsh.insert(id, black_box(embedding.clone()));
                lsh
            },
            BatchSize::LargeInput,
        )
    });

    // Suppress unused warning
    let _ = LshConfig::default();

    group.finish();
}

// ============================================================================
// PQ-LSH vs LSH-only Benchmarks
// ============================================================================

fn bench_pq_lsh_vs_lsh(c: &mut Criterion) {
    use dashprove_learning::corpus::{PqLshConfig, ProofCorpusPqLsh};
    use dashprove_learning::embedder::PropertyEmbedder;
    use dashprove_learning::ProofCorpusLsh;

    let mut group = c.benchmark_group("pq_lsh_vs_lsh");
    group.sample_size(30);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings for testing
    let mut corpus_1000 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..1000 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_1000.insert_with_embedding(&result, embedding);
    }

    // Build both indexes
    let lsh_1000 = ProofCorpusLsh::build_auto_config(&corpus_1000).unwrap();
    let pq_lsh_1000 = ProofCorpusPqLsh::build_with_config(&corpus_1000, PqLshConfig::default())
        .unwrap()
        .unwrap();

    // Create query embedding
    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    // Benchmark LSH-only approximate search
    group.bench_function(BenchmarkId::new("lsh_approximate", "1000_k10"), |b| {
        b.iter(|| {
            lsh_1000.find_similar_approximate(
                black_box(&corpus_1000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    // Benchmark PQ-LSH approximate search
    group.bench_function(BenchmarkId::new("pq_lsh_approximate", "1000_k10"), |b| {
        b.iter(|| {
            pq_lsh_1000.find_similar_approximate(
                black_box(&corpus_1000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    // Benchmark PQ-LSH exact (using only PQ, no LSH filtering)
    group.bench_function(BenchmarkId::new("pq_lsh_exact_pq", "1000_k10"), |b| {
        b.iter(|| {
            pq_lsh_1000.find_similar_exact_pq(
                black_box(&corpus_1000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    // Build larger corpus (5000 entries) to see scaling
    let mut corpus_5000 = ProofCorpus::new();
    for i in 0..5000 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_5000.insert_with_embedding(&result, embedding);
    }

    let lsh_5000 = ProofCorpusLsh::build_auto_config(&corpus_5000).unwrap();
    let pq_lsh_5000 =
        ProofCorpusPqLsh::build_with_config(&corpus_5000, PqLshConfig::for_medium_corpus())
            .unwrap()
            .unwrap();

    group.bench_function(BenchmarkId::new("lsh_approximate", "5000_k10"), |b| {
        b.iter(|| {
            lsh_5000.find_similar_approximate(
                black_box(&corpus_5000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    group.bench_function(BenchmarkId::new("pq_lsh_approximate", "5000_k10"), |b| {
        b.iter(|| {
            pq_lsh_5000.find_similar_approximate(
                black_box(&corpus_5000),
                black_box(&query_embedding),
                10,
            )
        })
    });

    // Index build time comparison
    group.bench_function(BenchmarkId::new("lsh_build", "1000"), |b| {
        b.iter(|| ProofCorpusLsh::build_auto_config(black_box(&corpus_1000)))
    });

    group.bench_function(BenchmarkId::new("pq_lsh_build", "1000"), |b| {
        b.iter(|| {
            ProofCorpusPqLsh::build_with_config(black_box(&corpus_1000), PqLshConfig::default())
        })
    });

    // Memory comparison
    let lsh_1000 = ProofCorpusLsh::build_auto_config(&corpus_1000).unwrap();
    let pq_lsh_1000 = ProofCorpusPqLsh::build_with_config(&corpus_1000, PqLshConfig::default())
        .unwrap()
        .unwrap();

    let pq_stats = pq_lsh_1000.memory_stats();
    println!("\nMemory comparison for 1000 entries:");
    println!(
        "  PQ-LSH: {} bytes (compression ratio: {:.1}x)",
        pq_stats.total_bytes, pq_stats.compression_ratio
    );
    println!("  Raw equivalent: {} bytes", pq_stats.raw_equivalent_bytes);

    // Recall comparison (measure quality vs LSH baseline)
    let recall_pq_lsh = pq_lsh_1000.measure_recall(&corpus_1000, 10, 50);
    println!("\nRecall@10 for PQ-LSH: {:.2}%", recall_pq_lsh * 100.0);

    // Suppress unused warning
    let _ = lsh_1000.len();

    group.finish();
}

// ============================================================================
// OPQ vs PQ Reconstruction Error Benchmarks
// ============================================================================

fn bench_opq_vs_pq_error(c: &mut Criterion) {
    use dashprove_learning::corpus::{
        OpqLshConfig, PqLshConfig, ProofCorpusOpqLsh, ProofCorpusPqLsh,
    };
    use dashprove_learning::embedder::PropertyEmbedder;

    let mut group = c.benchmark_group("opq_vs_pq_error");
    group.sample_size(20);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings for testing
    let mut corpus_500 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..500 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_500.insert_with_embedding(&result, embedding);
    }

    // Build both indexes
    let pq_lsh = ProofCorpusPqLsh::build_with_config(&corpus_500, PqLshConfig::default())
        .unwrap()
        .unwrap();
    let opq_lsh = ProofCorpusOpqLsh::build_with_config(&corpus_500, OpqLshConfig::default())
        .unwrap()
        .unwrap();

    // Measure quantization error (lower is better)
    // For PQ, calculate error manually
    let embeddings: Vec<_> = corpus_500
        .entries()
        .filter_map(|e| e.embedding.clone())
        .collect();
    let pq = pq_lsh.quantizer();
    let pq_error: f32 = embeddings
        .iter()
        .map(|e| {
            let codes = pq.encode(e);
            let decoded = pq.decode(&codes);
            e.vector
                .iter()
                .zip(decoded.vector.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        })
        .sum::<f32>()
        / embeddings.len() as f32;
    let opq_error = opq_lsh.quantization_error(&corpus_500);

    println!("\nQuantization Error Comparison (500 entries):");
    println!("  PQ:  {:.6}", pq_error);
    println!("  OPQ: {:.6}", opq_error);
    println!(
        "  Improvement: {:.1}%",
        (1.0 - opq_error / pq_error) * 100.0
    );

    // Create query embedding
    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    // Benchmark PQ-LSH build time
    group.bench_function(BenchmarkId::new("build", "pq_lsh_500"), |b| {
        b.iter(|| {
            ProofCorpusPqLsh::build_with_config(black_box(&corpus_500), PqLshConfig::default())
        })
    });

    // Benchmark OPQ-LSH build time (includes rotation matrix training)
    group.bench_function(BenchmarkId::new("build", "opq_lsh_500"), |b| {
        b.iter(|| {
            ProofCorpusOpqLsh::build_with_config(black_box(&corpus_500), OpqLshConfig::default())
        })
    });

    // Benchmark PQ-LSH approximate search
    group.bench_function(BenchmarkId::new("search", "pq_lsh_500_k10"), |b| {
        b.iter(|| {
            pq_lsh.find_similar_approximate(black_box(&corpus_500), black_box(&query_embedding), 10)
        })
    });

    // Benchmark OPQ-LSH approximate search
    group.bench_function(BenchmarkId::new("search", "opq_lsh_500_k10"), |b| {
        b.iter(|| {
            opq_lsh.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
            )
        })
    });

    // Memory comparison
    let pq_stats = pq_lsh.memory_stats();
    let opq_stats = opq_lsh.memory_stats();

    println!("\nMemory Comparison (500 entries):");
    println!(
        "  PQ-LSH:  {} bytes (compression: {:.1}x)",
        pq_stats.total_bytes, pq_stats.compression_ratio
    );
    println!(
        "  OPQ-LSH: {} bytes (compression: {:.1}x)",
        opq_stats.total_bytes, opq_stats.compression_ratio
    );
    println!(
        "  Overhead from rotation: {} bytes",
        opq_stats.codebook_bytes - pq_stats.codebook_bytes
    );

    // Recall comparison
    let pq_recall = pq_lsh.measure_recall(&corpus_500, 10, 50);
    let opq_recall = opq_lsh.measure_recall(&corpus_500, 10, 50);

    println!("\nRecall@10 Comparison:");
    println!("  PQ-LSH:  {:.2}%", pq_recall * 100.0);
    println!("  OPQ-LSH: {:.2}%", opq_recall * 100.0);

    group.finish();
}

// ============================================================================
// IVFPQ vs IVFOPQ Benchmarks
// ============================================================================

fn bench_ivfpq_vs_ivfopq(c: &mut Criterion) {
    use dashprove_learning::corpus::{
        IvfOpqConfig, IvfPqConfig, ProofCorpusIvfOpq, ProofCorpusIvfPq,
    };
    use dashprove_learning::embedder::PropertyEmbedder;

    let mut group = c.benchmark_group("ivfpq_vs_ivfopq");
    group.sample_size(10); // Lower sample size due to build time

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings for testing
    let mut corpus_500 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..500 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_500.insert_with_embedding(&result, embedding);
    }

    // Configuration for comparable comparison
    let ivfpq_config = IvfPqConfig {
        nlist: 32,
        nprobe: 8,
        pq: dashprove_learning::pq::PqConfig::fast(),
        coarse_kmeans_iterations: 10,
        seed: 42,
    };

    let ivfopq_config = IvfOpqConfig {
        nlist: 32,
        nprobe: 8,
        opq: dashprove_learning::pq::OpqConfig::fast(),
        coarse_kmeans_iterations: 10,
        seed: 42,
    };

    // Build both indexes
    let ivfpq = ProofCorpusIvfPq::build_with_config(&corpus_500, ivfpq_config.clone())
        .unwrap()
        .unwrap();
    let ivfopq = ProofCorpusIvfOpq::build_with_config(&corpus_500, ivfopq_config.clone())
        .unwrap()
        .unwrap();

    // Create query embedding
    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    // ========== Build Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("build", "ivfpq_500"), |b| {
        b.iter(|| ProofCorpusIvfPq::build_with_config(black_box(&corpus_500), ivfpq_config.clone()))
    });

    group.bench_function(BenchmarkId::new("build", "ivfopq_500"), |b| {
        b.iter(|| {
            ProofCorpusIvfOpq::build_with_config(black_box(&corpus_500), ivfopq_config.clone())
        })
    });

    // ========== Search Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("search_nprobe4", "ivfpq_500"), |b| {
        b.iter(|| {
            ivfpq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                4,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_nprobe4", "ivfopq_500"), |b| {
        b.iter(|| {
            ivfopq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                4,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_nprobe16", "ivfpq_500"), |b| {
        b.iter(|| {
            ivfpq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                16,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_nprobe16", "ivfopq_500"), |b| {
        b.iter(|| {
            ivfopq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                16,
            )
        })
    });

    // ========== Quality Metrics ==========

    // Measure quantization error for both (compares original vs reconstructed residuals)
    let ivfpq_quant_error = ivfpq.measure_quantization_error(&corpus_500, 100);
    let ivfopq_quant_error = ivfopq.measure_quantization_error(&corpus_500, 100);

    // Measure recall at different nprobe levels
    let ivfpq_recall_nprobe4 = ivfpq.measure_recall(&corpus_500, 10, 4, 50);
    let ivfpq_recall_nprobe8 = ivfpq.measure_recall(&corpus_500, 10, 8, 50);
    let ivfpq_recall_nprobe16 = ivfpq.measure_recall(&corpus_500, 10, 16, 50);

    let ivfopq_recall_nprobe4 = ivfopq.measure_recall(&corpus_500, 10, 4, 50);
    let ivfopq_recall_nprobe8 = ivfopq.measure_recall(&corpus_500, 10, 8, 50);
    let ivfopq_recall_nprobe16 = ivfopq.measure_recall(&corpus_500, 10, 16, 50);

    println!("\n============================================================");
    println!("IVFPQ vs IVFOPQ Comparison (500 entries, nlist=32)");
    println!("============================================================");

    println!("\nQuantization Error (mean squared residual reconstruction error):");
    println!("  IVFPQ:  {:.6}", ivfpq_quant_error);
    println!("  IVFOPQ: {:.6}", ivfopq_quant_error);
    if ivfpq_quant_error > 0.0 {
        let improvement = (1.0 - ivfopq_quant_error / ivfpq_quant_error) * 100.0;
        println!("  Improvement: {:.1}% lower error with IVFOPQ", improvement);
    }

    println!("\nRecall@10 (percentage of exact top-10 found by approximate search):");
    println!(
        "  nprobe=4:  IVFPQ={:.1}%, IVFOPQ={:.1}%",
        ivfpq_recall_nprobe4 * 100.0,
        ivfopq_recall_nprobe4 * 100.0
    );
    println!(
        "  nprobe=8:  IVFPQ={:.1}%, IVFOPQ={:.1}%",
        ivfpq_recall_nprobe8 * 100.0,
        ivfopq_recall_nprobe8 * 100.0
    );
    println!(
        "  nprobe=16: IVFPQ={:.1}%, IVFOPQ={:.1}%",
        ivfpq_recall_nprobe16 * 100.0,
        ivfopq_recall_nprobe16 * 100.0
    );

    // ========== Memory Comparison ==========

    let ivfpq_stats = ivfpq.memory_stats();
    let ivfopq_stats = ivfopq.memory_stats();

    println!("\nMemory Usage:");
    println!(
        "  IVFPQ:  {} bytes (compression: {:.1}x)",
        ivfpq_stats.total_bytes, ivfpq_stats.compression_ratio
    );
    println!(
        "  IVFOPQ: {} bytes (compression: {:.1}x)",
        ivfopq_stats.total_bytes, ivfopq_stats.compression_ratio
    );
    println!(
        "  Rotation matrix overhead: {} bytes",
        ivfopq_stats.rotation_bytes
    );

    // ========== List Distribution ==========

    let ivfpq_list_stats = ivfpq.list_stats();
    let ivfopq_list_stats = ivfopq.list_stats();

    println!("\nInverted List Distribution:");
    println!(
        "  IVFPQ:  non_empty={}/{}, min={}, max={}, mean={:.1}",
        ivfpq_list_stats.non_empty_lists,
        ivfpq_list_stats.nlist,
        ivfpq_list_stats.min_list_size,
        ivfpq_list_stats.max_list_size,
        ivfpq_list_stats.mean_list_size
    );
    println!(
        "  IVFOPQ: non_empty={}/{}, min={}, max={}, mean={:.1}",
        ivfopq_list_stats.non_empty_lists,
        ivfopq_list_stats.nlist,
        ivfopq_list_stats.min_list_size,
        ivfopq_list_stats.max_list_size,
        ivfopq_list_stats.mean_list_size
    );

    println!("\n============================================================\n");

    group.finish();
}

// ============================================================================
// HNSW vs IVFPQ vs IVFOPQ Benchmarks
// ============================================================================

fn bench_hnsw_vs_ivf(c: &mut Criterion) {
    use dashprove_learning::corpus::{
        HnswConfig, IvfOpqConfig, IvfPqConfig, ProofCorpusHnsw, ProofCorpusIvfOpq, ProofCorpusIvfPq,
    };
    use dashprove_learning::embedder::PropertyEmbedder;

    let mut group = c.benchmark_group("hnsw_vs_ivf");
    group.sample_size(10); // Lower sample size due to build time

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings for testing
    let mut corpus_500 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..500 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_500.insert_with_embedding(&result, embedding);
    }

    // Configuration for comparable comparison
    let hnsw_config = HnswConfig::default();

    let ivfpq_config = IvfPqConfig {
        nlist: 32,
        nprobe: 8,
        pq: dashprove_learning::pq::PqConfig::fast(),
        coarse_kmeans_iterations: 10,
        seed: 42,
    };

    let ivfopq_config = IvfOpqConfig {
        nlist: 32,
        nprobe: 8,
        opq: dashprove_learning::pq::OpqConfig::fast(),
        coarse_kmeans_iterations: 10,
        seed: 42,
    };

    // Build all indexes
    let hnsw = ProofCorpusHnsw::build_with_config(&corpus_500, hnsw_config.clone())
        .unwrap()
        .unwrap();
    let ivfpq = ProofCorpusIvfPq::build_with_config(&corpus_500, ivfpq_config.clone())
        .unwrap()
        .unwrap();
    let ivfopq = ProofCorpusIvfOpq::build_with_config(&corpus_500, ivfopq_config.clone())
        .unwrap()
        .unwrap();

    // Create query embedding
    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    println!("\n============================================================");
    println!("HNSW vs IVFPQ vs IVFOPQ Comparison (500 embeddings)");
    println!("============================================================\n");

    // ========== Build Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("build", "hnsw_500"), |b| {
        b.iter(|| ProofCorpusHnsw::build_with_config(black_box(&corpus_500), hnsw_config.clone()))
    });

    group.bench_function(BenchmarkId::new("build", "ivfpq_500"), |b| {
        b.iter(|| ProofCorpusIvfPq::build_with_config(black_box(&corpus_500), ivfpq_config.clone()))
    });

    group.bench_function(BenchmarkId::new("build", "ivfopq_500"), |b| {
        b.iter(|| {
            ProofCorpusIvfOpq::build_with_config(black_box(&corpus_500), ivfopq_config.clone())
        })
    });

    // ========== Search Time Benchmarks ==========

    // HNSW with different ef values
    group.bench_function(BenchmarkId::new("search_ef50", "hnsw_500"), |b| {
        b.iter(|| {
            hnsw.find_similar_with_ef(black_box(&corpus_500), black_box(&query_embedding), 10, 50)
        })
    });

    group.bench_function(BenchmarkId::new("search_ef100", "hnsw_500"), |b| {
        b.iter(|| {
            hnsw.find_similar_with_ef(black_box(&corpus_500), black_box(&query_embedding), 10, 100)
        })
    });

    // IVF with different nprobe values
    group.bench_function(BenchmarkId::new("search_nprobe4", "ivfpq_500"), |b| {
        b.iter(|| {
            ivfpq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                4,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_nprobe8", "ivfpq_500"), |b| {
        b.iter(|| {
            ivfpq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                8,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_nprobe4", "ivfopq_500"), |b| {
        b.iter(|| {
            ivfopq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                4,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_nprobe8", "ivfopq_500"), |b| {
        b.iter(|| {
            ivfopq.find_similar_approximate(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                8,
            )
        })
    });

    // ========== Recall Measurement ==========

    let hnsw_recall_50 = hnsw.measure_recall(&corpus_500, 10, 50, 50);
    let hnsw_recall_100 = hnsw.measure_recall(&corpus_500, 10, 100, 50);
    let hnsw_recall_200 = hnsw.measure_recall(&corpus_500, 10, 200, 50);

    // For IVF indices, measure recall by comparing to exact search
    let exact_results = hnsw.find_similar_exact(&corpus_500, &query_embedding, 10);
    let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| &r.id).collect();

    let ivfpq_4 = ivfpq.find_similar_approximate(&corpus_500, &query_embedding, 10, 4);
    let ivfpq_8 = ivfpq.find_similar_approximate(&corpus_500, &query_embedding, 10, 8);
    let ivfpq_16 = ivfpq.find_similar_approximate(&corpus_500, &query_embedding, 10, 16);

    let ivfopq_4 = ivfopq.find_similar_approximate(&corpus_500, &query_embedding, 10, 4);
    let ivfopq_8 = ivfopq.find_similar_approximate(&corpus_500, &query_embedding, 10, 8);
    let ivfopq_16 = ivfopq.find_similar_approximate(&corpus_500, &query_embedding, 10, 16);

    let ivfpq_recall_4 = ivfpq_4.iter().filter(|r| exact_ids.contains(&r.id)).count() as f64 / 10.0;
    let ivfpq_recall_8 = ivfpq_8.iter().filter(|r| exact_ids.contains(&r.id)).count() as f64 / 10.0;
    let ivfpq_recall_16 = ivfpq_16
        .iter()
        .filter(|r| exact_ids.contains(&r.id))
        .count() as f64
        / 10.0;

    let ivfopq_recall_4 = ivfopq_4
        .iter()
        .filter(|r| exact_ids.contains(&r.id))
        .count() as f64
        / 10.0;
    let ivfopq_recall_8 = ivfopq_8
        .iter()
        .filter(|r| exact_ids.contains(&r.id))
        .count() as f64
        / 10.0;
    let ivfopq_recall_16 = ivfopq_16
        .iter()
        .filter(|r| exact_ids.contains(&r.id))
        .count() as f64
        / 10.0;

    println!("Recall@10 (averaged over samples):");
    println!(
        "  HNSW:   ef=50: {:.1}%,  ef=100: {:.1}%,  ef=200: {:.1}%",
        hnsw_recall_50 * 100.0,
        hnsw_recall_100 * 100.0,
        hnsw_recall_200 * 100.0
    );
    println!(
        "  IVFPQ:  nprobe=4: {:.1}%,  nprobe=8: {:.1}%,  nprobe=16: {:.1}%",
        ivfpq_recall_4 * 100.0,
        ivfpq_recall_8 * 100.0,
        ivfpq_recall_16 * 100.0
    );
    println!(
        "  IVFOPQ: nprobe=4: {:.1}%,  nprobe=8: {:.1}%,  nprobe=16: {:.1}%",
        ivfopq_recall_4 * 100.0,
        ivfopq_recall_8 * 100.0,
        ivfopq_recall_16 * 100.0
    );

    // ========== Memory Statistics ==========

    let hnsw_mem = hnsw.memory_stats();
    let ivfpq_mem = ivfpq.memory_stats();
    let ivfopq_mem = ivfopq.memory_stats();

    println!("\nMemory Usage:");
    println!(
        "  HNSW:   {} bytes (embeddings: {}, graph: {})",
        hnsw_mem.total_bytes, hnsw_mem.embeddings_bytes, hnsw_mem.graph_bytes
    );
    println!(
        "  IVFPQ:  {} bytes (compression: {:.1}x)",
        ivfpq_mem.total_bytes, ivfpq_mem.compression_ratio
    );
    println!(
        "  IVFOPQ: {} bytes (compression: {:.1}x)",
        ivfopq_mem.total_bytes, ivfopq_mem.compression_ratio
    );

    // ========== Graph/Index Statistics ==========

    let hnsw_graph = hnsw.graph_stats();

    println!("\nIndex Structure:");
    println!(
        "  HNSW:   {} layers, {:.1} mean edges/node",
        hnsw_graph.nodes_per_layer.len(),
        hnsw_graph.mean_edges_per_node
    );
    println!("          Layer nodes: {:?}", hnsw_graph.nodes_per_layer);

    let ivfpq_list = ivfpq.list_stats();
    let ivfopq_list = ivfopq.list_stats();

    println!(
        "  IVFPQ:  {} lists, {:.1} mean entries/list",
        ivfpq_list.nlist, ivfpq_list.mean_list_size
    );
    println!(
        "  IVFOPQ: {} lists, {:.1} mean entries/list",
        ivfopq_list.nlist, ivfopq_list.mean_list_size
    );

    println!("\n============================================================\n");

    group.finish();
}

fn bench_hnsw_vs_hnsw_pq(c: &mut Criterion) {
    use dashprove_learning::corpus::{
        HnswConfig, HnswPqConfig, ProofCorpusHnsw, ProofCorpusHnswPq,
    };
    use dashprove_learning::embedder::PropertyEmbedder;

    let mut group = c.benchmark_group("hnsw_vs_hnsw_pq");
    group.sample_size(10);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings
    let mut corpus_500 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..500 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_500.insert_with_embedding(&result, embedding);
    }

    let hnsw_config = HnswConfig::default();
    let hnsw_pq_config = HnswPqConfig::default();

    // Build indexes
    let hnsw = ProofCorpusHnsw::build_with_config(&corpus_500, hnsw_config.clone())
        .unwrap()
        .unwrap();
    let hnsw_pq = ProofCorpusHnswPq::build_with_config(&corpus_500, hnsw_pq_config.clone())
        .unwrap()
        .unwrap();

    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    println!("\n============================================================");
    println!("HNSW vs HNSW-PQ Comparison (500 embeddings)");
    println!("============================================================\n");

    // ========== Build Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("build", "hnsw_500"), |b| {
        b.iter(|| ProofCorpusHnsw::build_with_config(black_box(&corpus_500), hnsw_config.clone()))
    });

    group.bench_function(BenchmarkId::new("build", "hnsw_pq_500"), |b| {
        b.iter(|| {
            ProofCorpusHnswPq::build_with_config(black_box(&corpus_500), hnsw_pq_config.clone())
        })
    });

    // ========== Search Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("search_ef50", "hnsw_500"), |b| {
        b.iter(|| {
            hnsw.find_similar_with_ef(black_box(&corpus_500), black_box(&query_embedding), 10, 50)
        })
    });

    group.bench_function(BenchmarkId::new("search_ef50", "hnsw_pq_500"), |b| {
        b.iter(|| {
            hnsw_pq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                50,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_ef100", "hnsw_500"), |b| {
        b.iter(|| {
            hnsw.find_similar_with_ef(black_box(&corpus_500), black_box(&query_embedding), 10, 100)
        })
    });

    group.bench_function(BenchmarkId::new("search_ef100", "hnsw_pq_500"), |b| {
        b.iter(|| {
            hnsw_pq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                100,
            )
        })
    });

    // ========== Recall Measurement ==========

    let hnsw_recall_50 = hnsw.measure_recall(&corpus_500, 10, 50, 50);
    let hnsw_recall_100 = hnsw.measure_recall(&corpus_500, 10, 100, 50);
    let hnsw_pq_recall_50 = hnsw_pq.measure_recall(&corpus_500, 10, 50, 50);
    let hnsw_pq_recall_100 = hnsw_pq.measure_recall(&corpus_500, 10, 100, 50);

    println!("Recall@10 (averaged over 50 samples):");
    println!(
        "  HNSW:    ef=50: {:.1}%,  ef=100: {:.1}%",
        hnsw_recall_50 * 100.0,
        hnsw_recall_100 * 100.0
    );
    println!(
        "  HNSW-PQ: ef=50: {:.1}%,  ef=100: {:.1}%",
        hnsw_pq_recall_50 * 100.0,
        hnsw_pq_recall_100 * 100.0
    );

    // ========== Memory Statistics ==========

    let hnsw_mem = hnsw.memory_stats();
    let hnsw_pq_mem = hnsw_pq.memory_stats();

    println!("\nMemory Usage:");
    println!(
        "  HNSW:    {} bytes total (embeddings: {}, graph: {})",
        hnsw_mem.total_bytes, hnsw_mem.embeddings_bytes, hnsw_mem.graph_bytes
    );
    println!(
        "  HNSW-PQ: {} bytes total (codes: {}, codebook: {}, graph: {})",
        hnsw_pq_mem.total_bytes,
        hnsw_pq_mem.codes_bytes,
        hnsw_pq_mem.codebook_bytes,
        hnsw_pq_mem.graph_bytes
    );
    println!(
        "  HNSW-PQ embedding compression: {:.1}x (full would be {} bytes)",
        hnsw_pq_mem.compression_ratio, hnsw_pq_mem.full_embedding_bytes
    );
    println!(
        "  Total memory ratio (HNSW/HNSW-PQ): {:.2}x",
        hnsw_mem.total_bytes as f64 / hnsw_pq_mem.total_bytes as f64
    );

    // ========== Quantization Error ==========

    let pq_error = hnsw_pq.measure_quantization_error(&corpus_500, 50);
    println!("\nQuantization Error (MSE):");
    println!("  HNSW-PQ: {:.6}", pq_error);

    // ========== Graph Statistics ==========

    let hnsw_graph = hnsw.graph_stats();
    let hnsw_pq_graph = hnsw_pq.graph_stats();

    println!("\nGraph Structure:");
    println!(
        "  HNSW:    {} layers, {:.1} mean edges/node",
        hnsw_graph.nodes_per_layer.len(),
        hnsw_graph.mean_edges_per_node
    );
    println!(
        "  HNSW-PQ: {} layers, {:.1} mean edges/node",
        hnsw_pq_graph.nodes_per_layer.len(),
        hnsw_pq_graph.mean_edges_per_node
    );

    println!("\n============================================================\n");

    group.finish();
}

fn bench_hnsw_pq_vs_hnsw_opq(c: &mut Criterion) {
    use dashprove_learning::corpus::{
        HnswOpqConfig, HnswPqConfig, ProofCorpusHnswOpq, ProofCorpusHnswPq,
    };
    use dashprove_learning::embedder::PropertyEmbedder;

    let mut group = c.benchmark_group("hnsw_pq_vs_hnsw_opq");
    group.sample_size(10);

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings
    let mut corpus_500 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();

    for i in 0..500 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        corpus_500.insert_with_embedding(&result, embedding);
    }

    let hnsw_pq_config = HnswPqConfig::fast();
    let hnsw_opq_config = HnswOpqConfig::fast();

    // Build indexes
    let hnsw_pq = ProofCorpusHnswPq::build_with_config(&corpus_500, hnsw_pq_config.clone())
        .unwrap()
        .unwrap();
    let hnsw_opq = ProofCorpusHnswOpq::build_with_config(&corpus_500, hnsw_opq_config.clone())
        .unwrap()
        .unwrap();

    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    println!("\n============================================================");
    println!("HNSW-PQ vs HNSW-OPQ Comparison (500 embeddings)");
    println!("============================================================\n");

    // ========== Build Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("build", "hnsw_pq_500"), |b| {
        b.iter(|| {
            ProofCorpusHnswPq::build_with_config(black_box(&corpus_500), hnsw_pq_config.clone())
        })
    });

    group.bench_function(BenchmarkId::new("build", "hnsw_opq_500"), |b| {
        b.iter(|| {
            ProofCorpusHnswOpq::build_with_config(black_box(&corpus_500), hnsw_opq_config.clone())
        })
    });

    // ========== Search Time Benchmarks ==========

    group.bench_function(BenchmarkId::new("search_ef50", "hnsw_pq_500"), |b| {
        b.iter(|| {
            hnsw_pq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                50,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_ef50", "hnsw_opq_500"), |b| {
        b.iter(|| {
            hnsw_opq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                50,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_ef100", "hnsw_pq_500"), |b| {
        b.iter(|| {
            hnsw_pq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                100,
            )
        })
    });

    group.bench_function(BenchmarkId::new("search_ef100", "hnsw_opq_500"), |b| {
        b.iter(|| {
            hnsw_opq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                100,
            )
        })
    });

    // ========== Recall Measurement ==========

    let hnsw_pq_recall_50 = hnsw_pq.measure_recall(&corpus_500, 10, 50, 50);
    let hnsw_pq_recall_100 = hnsw_pq.measure_recall(&corpus_500, 10, 100, 50);
    let hnsw_opq_recall_50 = hnsw_opq.measure_recall(&corpus_500, 10, 50, 50);
    let hnsw_opq_recall_100 = hnsw_opq.measure_recall(&corpus_500, 10, 100, 50);

    println!("Recall@10 (averaged over 50 samples):");
    println!(
        "  HNSW-PQ:  ef=50: {:.1}%,  ef=100: {:.1}%",
        hnsw_pq_recall_50 * 100.0,
        hnsw_pq_recall_100 * 100.0
    );
    println!(
        "  HNSW-OPQ: ef=50: {:.1}%,  ef=100: {:.1}%",
        hnsw_opq_recall_50 * 100.0,
        hnsw_opq_recall_100 * 100.0
    );

    // ========== Memory Statistics ==========

    let hnsw_pq_mem = hnsw_pq.memory_stats();
    let hnsw_opq_mem = hnsw_opq.memory_stats();

    println!("\nMemory Usage:");
    println!(
        "  HNSW-PQ:  {} bytes total (codes: {}, codebook: {}, graph: {})",
        hnsw_pq_mem.total_bytes,
        hnsw_pq_mem.codes_bytes,
        hnsw_pq_mem.codebook_bytes,
        hnsw_pq_mem.graph_bytes
    );
    println!(
        "  HNSW-OPQ: {} bytes total (codes: {}, codebook: {}, rotation: {}, graph: {})",
        hnsw_opq_mem.total_bytes,
        hnsw_opq_mem.codes_bytes,
        hnsw_opq_mem.codebook_bytes,
        hnsw_opq_mem.rotation_bytes,
        hnsw_opq_mem.graph_bytes
    );
    println!(
        "  Rotation matrix overhead: {} bytes",
        hnsw_opq_mem.rotation_bytes
    );
    println!(
        "  Total memory ratio (HNSW-OPQ/HNSW-PQ): {:.2}x",
        hnsw_opq_mem.total_bytes as f64 / hnsw_pq_mem.total_bytes as f64
    );

    // ========== Quantization Error ==========

    let pq_error = hnsw_pq.measure_quantization_error(&corpus_500, 50);
    let opq_error = hnsw_opq.measure_quantization_error(&corpus_500, 50);
    let improvement = (pq_error - opq_error) / pq_error * 100.0;

    println!("\nQuantization Error (MSE):");
    println!("  HNSW-PQ:  {:.6}", pq_error);
    println!("  HNSW-OPQ: {:.6}", opq_error);
    println!("  OPQ improvement: {:.1}%", improvement);

    // ========== Graph Statistics ==========

    let hnsw_pq_graph = hnsw_pq.graph_stats();
    let hnsw_opq_graph = hnsw_opq.graph_stats();

    println!("\nGraph Structure:");
    println!(
        "  HNSW-PQ:  {} layers, {:.1} mean edges/node",
        hnsw_pq_graph.nodes_per_layer.len(),
        hnsw_pq_graph.mean_edges_per_node
    );
    println!(
        "  HNSW-OPQ: {} layers, {:.1} mean edges/node",
        hnsw_opq_graph.nodes_per_layer.len(),
        hnsw_opq_graph.mean_edges_per_node
    );

    println!("\n============================================================\n");

    group.finish();
}

// ============================================================================
// Binary vs PQ vs OPQ Compression Comparison
// ============================================================================

fn bench_binary_vs_pq_vs_opq(c: &mut Criterion) {
    use dashprove_learning::binary::{BinaryConfig, BinaryQuantizer};
    use dashprove_learning::corpus::{
        HnswBinaryConfig, HnswOpqConfig, HnswPqConfig, ProofCorpusHnswBinary, ProofCorpusHnswOpq,
        ProofCorpusHnswPq,
    };
    use dashprove_learning::embedder::PropertyEmbedder;
    use dashprove_learning::pq::{
        OpqConfig, OptimizedProductQuantizer, PqConfig, ProductQuantizer,
    };

    let mut group = c.benchmark_group("binary_vs_pq_vs_opq");
    group.sample_size(10); // Lower sample size due to build time

    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::TlaPlus,
        BackendId::Kani,
    ];

    // Build corpus with embeddings for testing
    let mut corpus_500 = ProofCorpus::new();
    let mut embedder = PropertyEmbedder::new();
    let mut embeddings_vec = Vec::with_capacity(500);

    for i in 0..500 {
        let prop = match i % 5 {
            0 => make_simple_theorem(&format!("prop_{}", i)),
            1 => make_complex_theorem(&format!("prop_{}", i)),
            2 => make_temporal_property(&format!("prop_{}", i)),
            3 => make_invariant(&format!("prop_{}", i)),
            _ => make_contract(&format!("prop_{}", i)),
        };
        let result = make_learnable_result(prop.clone(), backends[i % backends.len()]);
        let embedding = embedder.embed(&prop);
        embeddings_vec.push(embedding.clone());
        corpus_500.insert_with_embedding(&result, embedding);
    }

    println!("\n============================================================");
    println!("Binary vs PQ vs OPQ Quantization Comparison (500 embeddings)");
    println!("============================================================\n");

    // ========== Quantizer Training/Setup Time ==========

    println!("Quantizer Setup Time (no training for Binary):\n");

    // Binary quantizer - instant, no training
    let binary_config = BinaryConfig::default();
    let binary_quantizer = BinaryQuantizer::new(96, binary_config.clone());
    println!("  Binary:  Instant (no training required)");

    // PQ quantizer - requires k-means training
    group.bench_function(BenchmarkId::new("train", "pq"), |b| {
        b.iter(|| ProductQuantizer::train(black_box(&embeddings_vec), PqConfig::fast()))
    });

    // OPQ quantizer - requires k-means + rotation training
    group.bench_function(BenchmarkId::new("train", "opq"), |b| {
        b.iter(|| OptimizedProductQuantizer::train(black_box(&embeddings_vec), OpqConfig::fast()))
    });

    let pq = ProductQuantizer::train(&embeddings_vec, PqConfig::fast()).unwrap();
    let opq = OptimizedProductQuantizer::train(&embeddings_vec, OpqConfig::fast()).unwrap();

    // ========== Encoding Speed ==========

    println!("\nEncoding Speed (per embedding):\n");

    let sample_embedding = &embeddings_vec[0];

    group.bench_function(BenchmarkId::new("encode", "binary"), |b| {
        b.iter(|| binary_quantizer.encode(black_box(sample_embedding)))
    });

    group.bench_function(BenchmarkId::new("encode", "pq"), |b| {
        b.iter(|| pq.encode(black_box(sample_embedding)))
    });

    group.bench_function(BenchmarkId::new("encode", "opq"), |b| {
        b.iter(|| opq.encode(black_box(sample_embedding)))
    });

    // ========== Memory Usage ==========

    // Binary: 96 bits = 12 bytes per embedding
    // PQ: 8 bytes per embedding (M=8 subspaces) + codebook (~25KB)
    // OPQ: 8 bytes per embedding + codebook (~25KB) + rotation matrix (~37KB)
    let n = 500;
    let binary_mem_per_entry = 12;
    let pq_mem_per_entry = 8;
    let pq_codebook = pq.codebook_size_bytes();
    let opq_codebook = opq.codebook_size_bytes();
    let opq_total = opq.total_size_bytes();
    let opq_rotation = opq_total - opq_codebook;
    let raw_per_entry = 96 * 4; // 384 bytes

    println!("\nMemory Usage for {} embeddings:", n);
    println!(
        "  Raw f32:  {} bytes ({} bytes/entry)",
        n * raw_per_entry,
        raw_per_entry
    );
    println!(
        "  Binary:   {} bytes ({} bytes/entry) - {:.1}x compression",
        n * binary_mem_per_entry,
        binary_mem_per_entry,
        raw_per_entry as f64 / binary_mem_per_entry as f64
    );
    println!(
        "  PQ:       {} bytes ({} bytes/entry + {} codebook) - {:.1}x compression",
        n * pq_mem_per_entry + pq_codebook,
        pq_mem_per_entry,
        pq_codebook,
        (n * raw_per_entry) as f64 / (n * pq_mem_per_entry + pq_codebook) as f64
    );
    println!(
        "  OPQ:      {} bytes ({} bytes/entry + {} codebook + {} rotation) - {:.1}x compression",
        n * pq_mem_per_entry + opq_codebook + opq_rotation,
        pq_mem_per_entry,
        opq_codebook,
        opq_rotation,
        (n * raw_per_entry) as f64 / (n * pq_mem_per_entry + opq_codebook + opq_rotation) as f64
    );

    // ========== Quantization Error ==========

    let binary_error: f32 = binary_quantizer
        .quantization_error(&embeddings_vec)
        .unwrap();

    // OPQ has quantization_error method. PQ doesn't so compute manually:
    let mut pq_error_sum = 0.0f32;
    for emb in &embeddings_vec {
        let codes = pq.encode(emb);
        let reconstructed = pq.decode(&codes);
        for (a, b) in emb.vector.iter().zip(reconstructed.vector.iter()) {
            let diff = a - b;
            pq_error_sum += diff * diff;
        }
    }
    let pq_error = pq_error_sum / embeddings_vec.len() as f32;
    let opq_error = opq.quantization_error(&embeddings_vec);

    // Normalize binary error to total (not per-dim) for comparison
    let binary_error_total = binary_error * 96.0;

    println!("\nQuantization Error (Total MSE):");
    println!("  Binary:  {:.6}", binary_error_total);
    println!("  PQ:      {:.6}", pq_error);
    println!("  OPQ:     {:.6}", opq_error);
    if binary_error_total > pq_error {
        println!(
            "  PQ vs Binary improvement: {:.1}%",
            (binary_error_total - pq_error) / binary_error_total * 100.0
        );
    }
    if pq_error > opq_error {
        println!(
            "  OPQ vs PQ improvement:    {:.1}%",
            (pq_error - opq_error) / pq_error * 100.0
        );
    }

    // ========== HNSW Index Comparison ==========

    println!("\nBuilding HNSW indexes with different quantization...\n");

    let hnsw_binary_config = HnswBinaryConfig::fast();
    let hnsw_pq_config = HnswPqConfig::fast();
    let hnsw_opq_config = HnswOpqConfig::fast();

    group.bench_function(BenchmarkId::new("hnsw_build", "binary"), |b| {
        b.iter(|| {
            ProofCorpusHnswBinary::build_with_config(
                black_box(&corpus_500),
                hnsw_binary_config.clone(),
            )
        })
    });

    group.bench_function(BenchmarkId::new("hnsw_build", "pq"), |b| {
        b.iter(|| {
            ProofCorpusHnswPq::build_with_config(black_box(&corpus_500), hnsw_pq_config.clone())
        })
    });

    group.bench_function(BenchmarkId::new("hnsw_build", "opq"), |b| {
        b.iter(|| {
            ProofCorpusHnswOpq::build_with_config(black_box(&corpus_500), hnsw_opq_config.clone())
        })
    });

    let hnsw_binary =
        ProofCorpusHnswBinary::build_with_config(&corpus_500, hnsw_binary_config.clone())
            .unwrap()
            .unwrap();
    let hnsw_pq = ProofCorpusHnswPq::build_with_config(&corpus_500, hnsw_pq_config.clone())
        .unwrap()
        .unwrap();
    let hnsw_opq = ProofCorpusHnswOpq::build_with_config(&corpus_500, hnsw_opq_config.clone())
        .unwrap()
        .unwrap();

    let query_prop = make_complex_theorem("query");
    let query_embedding = embedder.embed_query(&query_prop);

    // ========== Search Benchmarks ==========

    group.bench_function(BenchmarkId::new("hnsw_search", "binary"), |b| {
        b.iter(|| {
            hnsw_binary.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                50,
            )
        })
    });

    group.bench_function(BenchmarkId::new("hnsw_search", "pq"), |b| {
        b.iter(|| {
            hnsw_pq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                50,
            )
        })
    });

    group.bench_function(BenchmarkId::new("hnsw_search", "opq"), |b| {
        b.iter(|| {
            hnsw_opq.find_similar_with_ef(
                black_box(&corpus_500),
                black_box(&query_embedding),
                10,
                50,
            )
        })
    });

    // ========== HNSW Memory Stats ==========

    let binary_mem = hnsw_binary.memory_stats();
    let pq_mem = hnsw_pq.memory_stats();
    let opq_mem = hnsw_opq.memory_stats();

    println!("\nHNSW Index Memory:");
    println!(
        "  HNSW-Binary: {} bytes total (codes: {}, graph: {})",
        binary_mem.total_bytes, binary_mem.code_bytes, binary_mem.graph_bytes
    );
    println!(
        "  HNSW-PQ:     {} bytes total (codes: {}, codebook: {}, graph: {})",
        pq_mem.total_bytes, pq_mem.codes_bytes, pq_mem.codebook_bytes, pq_mem.graph_bytes
    );
    println!(
        "  HNSW-OPQ:    {} bytes total (codes: {}, codebook: {}, rotation: {}, graph: {})",
        opq_mem.total_bytes,
        opq_mem.codes_bytes,
        opq_mem.codebook_bytes,
        opq_mem.rotation_bytes,
        opq_mem.graph_bytes
    );

    println!("\n============================================================\n");

    group.finish();
}

// ============================================================================
// Distance kernel benchmarks (SIMD vs scalar)
// ============================================================================

fn bench_distance_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_kernels");
    let dim = 384;

    fn make_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state % 65_537) as f32 / 1024.0
            })
            .collect()
    }

    let query = make_vector(dim, 11);
    let centroids: Vec<Vec<f32>> = (0..64).map(|i| make_vector(dim, 200 + i)).collect();

    // Raw distance throughput for a single pair
    group.bench_function("euclidean_distance_sq_simd", |b| {
        b.iter(|| euclidean_distance_sq(black_box(&query), black_box(&centroids[0])))
    });

    group.bench_function("euclidean_distance_sq_scalar", |b| {
        b.iter(|| euclidean_distance_sq_scalar(black_box(&query), black_box(&centroids[0])))
    });

    // Cell scan style workload (matches IVF coarse search)
    group.bench_function("cell_scan_simd", |b| {
        b.iter(|| {
            let mut best = f32::MAX;
            for c_vec in &centroids {
                let dist = euclidean_distance_sq(black_box(&query), black_box(c_vec));
                if dist < best {
                    best = dist;
                }
            }
            best
        })
    });

    group.bench_function("cell_scan_scalar", |b| {
        b.iter(|| {
            let mut best = f32::MAX;
            for c_vec in &centroids {
                let dist = euclidean_distance_sq_scalar(black_box(&query), black_box(c_vec));
                if dist < best {
                    best = dist;
                }
            }
            best
        })
    });

    group.finish();
}

// ============================================================================
// Streaming Index Persistence Benchmarks
// ============================================================================

#[allow(clippy::field_reassign_with_default)]
fn bench_streaming_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_persistence");
    group.sample_size(10); // Lower sample size due to I/O

    // Helper to create random embedding
    fn random_embedding(seed: u64) -> Embedding {
        let mut state = seed;
        let vector: Vec<f32> = (0..96)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f64 / (1u64 << 31) as f64) as f32
            })
            .collect();
        Embedding::new(vector)
    }

    // ========== StreamingIvfIndex Persistence ==========

    // Build a StreamingIvfIndex
    let mut ivf_config = StreamingIvfConfig::default();
    ivf_config.min_training_samples = 100;
    ivf_config.nlist = 32;
    ivf_config.nprobe = 8;
    ivf_config.pq.num_subspaces = 12;
    ivf_config.pq.num_centroids = 64;

    let mut ivf_builder = StreamingIvfBuilder::new(ivf_config.clone(), 96).unwrap();

    // Add training samples
    for i in 0..200 {
        ivf_builder
            .add_training_sample(ProofId(format!("train_{}", i)), random_embedding(i as u64));
    }
    ivf_builder.train().unwrap();

    // Add entries
    for i in 0..500 {
        ivf_builder
            .add_entry(
                ProofId(format!("entry_{}", i)),
                random_embedding((i + 1000) as u64),
            )
            .unwrap();
    }

    let ivf_index = ivf_builder.build().unwrap();

    // Benchmark save
    let temp_dir = std::env::temp_dir();
    let ivf_path = temp_dir.join("bench_streaming_ivf_index.json");

    group.bench_function("streaming_ivf_save_500", |b| {
        b.iter(|| ivf_index.save_to_file(black_box(&ivf_path)))
    });

    // Save for load benchmark
    ivf_index.save_to_file(&ivf_path).unwrap();

    // Benchmark load
    group.bench_function("streaming_ivf_load_500", |b| {
        b.iter(|| StreamingIvfIndex::load_from_file(black_box(&ivf_path)))
    });

    // ========== BatchedHnswIndex Persistence ==========

    let mut hnsw_config = BatchedHnswConfig::default();
    hnsw_config.batch_size = 100;
    hnsw_config.m = 12;
    hnsw_config.m0 = 24;
    hnsw_config.ef_construction = 100;

    let mut hnsw_builder = BatchedHnswBuilder::new(hnsw_config.clone()).unwrap();

    for i in 0..500 {
        hnsw_builder
            .add_entry(ProofId(format!("hnsw_{}", i)), random_embedding(i as u64))
            .unwrap();
    }

    let hnsw_index = hnsw_builder.build().unwrap();

    let hnsw_path = temp_dir.join("bench_batched_hnsw_index.json");

    group.bench_function("batched_hnsw_save_500", |b| {
        b.iter(|| hnsw_index.save_to_file(black_box(&hnsw_path)))
    });

    // Save for load benchmark
    hnsw_index.save_to_file(&hnsw_path).unwrap();

    group.bench_function("batched_hnsw_load_500", |b| {
        b.iter(|| BatchedHnswIndex::load_from_file(black_box(&hnsw_path)))
    });

    // ========== Parallel Batch Query Benchmarks ==========

    let queries: Vec<Embedding> = (0..100)
        .map(|i| random_embedding((i + 2000) as u64))
        .collect();

    // StreamingIvf batch query: single vs multi-threaded
    group.bench_function("streaming_ivf_batch_query_100_1thread", |b| {
        b.iter(|| ivf_index.find_similar_batch(black_box(&queries), 10, 8, 1))
    });

    group.bench_function("streaming_ivf_batch_query_100_4threads", |b| {
        b.iter(|| ivf_index.find_similar_batch(black_box(&queries), 10, 8, 4))
    });

    // BatchedHnsw batch query: single vs multi-threaded
    group.bench_function("batched_hnsw_batch_query_100_1thread", |b| {
        b.iter(|| hnsw_index.find_similar_batch(black_box(&queries), 10, 1))
    });

    group.bench_function("batched_hnsw_batch_query_100_4threads", |b| {
        b.iter(|| hnsw_index.find_similar_batch(black_box(&queries), 10, 4))
    });

    // ========== Report Index Stats ==========

    let ivf_stats = ivf_index.memory_stats();
    let hnsw_stats = hnsw_index.memory_stats();

    println!("\nStreaming Index Statistics (500 entries):");
    println!(
        "  StreamingIvf: {} total bytes ({} codebook, {} codes, {:.1}x compression)",
        ivf_stats.total_bytes,
        ivf_stats.codebook_bytes,
        ivf_stats.codes_bytes,
        ivf_stats.compression_ratio
    );
    println!(
        "  BatchedHnsw:  {} total bytes ({} embeddings, {} graph, {} edges)",
        hnsw_stats.total_bytes,
        hnsw_stats.embedding_bytes,
        hnsw_stats.graph_bytes,
        hnsw_stats.graph_edges
    );

    // Get file sizes
    let ivf_file_size = std::fs::metadata(&ivf_path).map(|m| m.len()).unwrap_or(0);
    let hnsw_file_size = std::fs::metadata(&hnsw_path).map(|m| m.len()).unwrap_or(0);

    println!("\nPersisted File Sizes:");
    println!(
        "  StreamingIvf: {} bytes ({:.1} KB)",
        ivf_file_size,
        ivf_file_size as f64 / 1024.0
    );
    println!(
        "  BatchedHnsw:  {} bytes ({:.1} KB)",
        hnsw_file_size,
        hnsw_file_size as f64 / 1024.0
    );

    // Clean up
    std::fs::remove_file(&ivf_path).ok();
    std::fs::remove_file(&hnsw_path).ok();

    println!("\n============================================================\n");

    group.finish();
}

criterion_group!(
    benches,
    bench_feature_extraction,
    bench_embedding_operations,
    bench_similarity_computation,
    bench_embedding_index,
    bench_proof_corpus,
    bench_counterexample_corpus,
    bench_tactic_database,
    bench_learning_system,
    bench_ordered_float,
    bench_top_k_scaling,
    bench_lsh_vs_exact,
    bench_lsh_incremental_vs_rebuild,
    bench_pq_lsh_vs_lsh,
    bench_opq_vs_pq_error,
    bench_ivfpq_vs_ivfopq,
    bench_hnsw_vs_ivf,
    bench_hnsw_vs_hnsw_pq,
    bench_hnsw_pq_vs_hnsw_opq,
    bench_binary_vs_pq_vs_opq,
    bench_distance_kernels,
    bench_streaming_persistence,
);

criterion_main!(benches);
