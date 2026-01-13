//! Benchmarks for ML strategy prediction
//!
//! Run with: `cargo bench -p dashprove-ai`
//!
//! Measures the performance of:
//! - Feature extraction from properties
//! - Neural network inference (backend prediction, tactic prediction, time prediction)
//! - Training iterations (single epoch)
//! - Cross-validation fold computation

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use dashprove_ai::{
    PropertyFeatureVector, StrategyPredictor, TrainingDataGenerator, TrainingExample,
};
use dashprove_backends::BackendId;
use dashprove_usl::ast::{
    BinaryOp, ComparisonOp, Contract, Expr, Invariant, Property, Temporal, TemporalExpr, Theorem,
    Type,
};

// Helper functions to create test properties

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
    // A theorem with implications, conjunctions, arithmetic
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

// Generate synthetic training data for benchmark
fn make_training_data(count: usize) -> Vec<TrainingExample> {
    let backends = [
        BackendId::Lean4,
        BackendId::Coq,
        BackendId::Isabelle,
        BackendId::TlaPlus,
        BackendId::Kani,
        BackendId::Alloy,
        BackendId::Dafny,
        BackendId::Z3,
    ];

    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        // Vary property types
        let property = match i % 5 {
            0 => make_simple_theorem(&format!("thm_{}", i)),
            1 => make_quantified_theorem(&format!("qthm_{}", i), (i % 3) + 1),
            2 => make_complex_theorem(&format!("cthm_{}", i)),
            3 => make_temporal_property(&format!("temp_{}", i)),
            _ => make_invariant(&format!("inv_{}", i)),
        };

        let features = PropertyFeatureVector::from_property(&property);
        let backend = backends[i % backends.len()];
        let tactics = vec!["simp".to_string(), "decide".to_string()];
        let time_seconds = 0.1 + (i % 10) as f64 * 0.05;

        data.push(TrainingExample {
            features,
            backend,
            tactics,
            time_seconds,
            success: true,
        });
    }
    data
}

// Benchmarks for feature extraction

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let simple = make_simple_theorem("simple");
    let quantified = make_quantified_theorem("quantified", 3);
    let complex = make_complex_theorem("complex");
    let temporal = make_temporal_property("temporal");
    let invariant = make_invariant("invariant");
    let contract = make_contract("contract");

    group.bench_function("simple_theorem", |b| {
        b.iter(|| PropertyFeatureVector::from_property(black_box(&simple)))
    });

    group.bench_function("quantified_theorem", |b| {
        b.iter(|| PropertyFeatureVector::from_property(black_box(&quantified)))
    });

    group.bench_function("complex_theorem", |b| {
        b.iter(|| PropertyFeatureVector::from_property(black_box(&complex)))
    });

    group.bench_function("temporal_property", |b| {
        b.iter(|| PropertyFeatureVector::from_property(black_box(&temporal)))
    });

    group.bench_function("invariant", |b| {
        b.iter(|| PropertyFeatureVector::from_property(black_box(&invariant)))
    });

    group.bench_function("contract", |b| {
        b.iter(|| PropertyFeatureVector::from_property(black_box(&contract)))
    });

    group.finish();
}

// Benchmarks for neural network inference

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    let predictor = StrategyPredictor::new();
    let simple_features = PropertyFeatureVector::from_property(&make_simple_theorem("test"));
    let complex_features = PropertyFeatureVector::from_property(&make_complex_theorem("test"));
    let temporal_features = PropertyFeatureVector::from_property(&make_temporal_property("test"));

    // Backend prediction benchmarks
    group.bench_function("predict_backend/simple", |b| {
        b.iter(|| predictor.predict_backend(black_box(&simple_features)))
    });

    group.bench_function("predict_backend/complex", |b| {
        b.iter(|| predictor.predict_backend(black_box(&complex_features)))
    });

    group.bench_function("predict_backend/temporal", |b| {
        b.iter(|| predictor.predict_backend(black_box(&temporal_features)))
    });

    // Tactic prediction benchmarks
    group.bench_function("predict_tactics/simple", |b| {
        b.iter(|| predictor.predict_tactics(black_box(&simple_features), 5))
    });

    group.bench_function("predict_tactics/complex", |b| {
        b.iter(|| predictor.predict_tactics(black_box(&complex_features), 5))
    });

    // Time prediction benchmarks
    group.bench_function("predict_time/simple", |b| {
        b.iter(|| predictor.predict_time(black_box(&simple_features)))
    });

    group.bench_function("predict_time/complex", |b| {
        b.iter(|| predictor.predict_time(black_box(&complex_features)))
    });

    // Full strategy prediction
    let simple_prop = make_simple_theorem("bench");
    let complex_prop = make_complex_theorem("bench");

    group.bench_function("predict_strategy/simple", |b| {
        b.iter(|| predictor.predict_strategy(black_box(&simple_prop)))
    });

    group.bench_function("predict_strategy/complex", |b| {
        b.iter(|| predictor.predict_strategy(black_box(&complex_prop)))
    });

    group.finish();
}

// Benchmarks for batch inference

fn bench_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference");

    let predictor = StrategyPredictor::new();

    // Create batch of features
    let batch_10: Vec<_> = (0..10)
        .map(|i| {
            let prop = make_quantified_theorem(&format!("batch_{}", i), i % 3 + 1);
            PropertyFeatureVector::from_property(&prop)
        })
        .collect();

    let batch_100: Vec<_> = (0..100)
        .map(|i| {
            let prop = match i % 3 {
                0 => make_simple_theorem(&format!("batch_{}", i)),
                1 => make_complex_theorem(&format!("batch_{}", i)),
                _ => make_invariant(&format!("batch_{}", i)),
            };
            PropertyFeatureVector::from_property(&prop)
        })
        .collect();

    group.bench_function(BenchmarkId::new("backend_predictions", "10"), |b| {
        b.iter(|| {
            for features in black_box(&batch_10) {
                let _ = predictor.predict_backend(features);
            }
        })
    });

    group.bench_function(BenchmarkId::new("backend_predictions", "100"), |b| {
        b.iter(|| {
            for features in black_box(&batch_100) {
                let _ = predictor.predict_backend(features);
            }
        })
    });

    group.finish();
}

// Benchmarks for training

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    // Training benchmarks can be slow, so limit sample size
    group.sample_size(20);

    let train_data_50 = make_training_data(50);
    let train_data_200 = make_training_data(200);

    // Single epoch training benchmark
    group.bench_function(BenchmarkId::new("single_epoch", "50_examples"), |b| {
        b.iter_batched(
            StrategyPredictor::new,
            |mut predictor| {
                predictor.train(black_box(&train_data_50), 0.01, 1);
                predictor
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function(BenchmarkId::new("single_epoch", "200_examples"), |b| {
        b.iter_batched(
            StrategyPredictor::new,
            |mut predictor| {
                predictor.train(black_box(&train_data_200), 0.01, 1);
                predictor
            },
            BatchSize::SmallInput,
        )
    });

    // Multiple epochs (5) benchmark
    group.bench_function(BenchmarkId::new("five_epochs", "50_examples"), |b| {
        b.iter_batched(
            StrategyPredictor::new,
            |mut predictor| {
                predictor.train(black_box(&train_data_50), 0.01, 5);
                predictor
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// Benchmarks for evaluation

fn bench_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation");
    group.sample_size(20);

    // Train a predictor first
    let train_data = make_training_data(100);
    let mut predictor = StrategyPredictor::new();
    predictor.train(&train_data, 0.01, 10);

    let test_data_20 = make_training_data(20);
    let test_data_100 = make_training_data(100);

    group.bench_function(BenchmarkId::new("evaluate", "20_examples"), |b| {
        b.iter(|| predictor.evaluate(black_box(&test_data_20)))
    });

    group.bench_function(BenchmarkId::new("evaluate", "100_examples"), |b| {
        b.iter(|| predictor.evaluate(black_box(&test_data_100)))
    });

    group.finish();
}

// Benchmarks for training data generation

fn bench_training_data_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_data_generation");

    // Create properties for training data
    let properties: Vec<_> = (0..50)
        .map(|i| match i % 4 {
            0 => make_simple_theorem(&format!("gen_{}", i)),
            1 => make_complex_theorem(&format!("gen_{}", i)),
            2 => make_temporal_property(&format!("gen_{}", i)),
            _ => make_invariant(&format!("gen_{}", i)),
        })
        .collect();

    group.bench_function("from_properties/50", |b| {
        b.iter(|| {
            let mut generator = TrainingDataGenerator::new();
            for prop in black_box(&properties) {
                generator.add_success(prop, BackendId::Lean4, vec!["simp".to_string()], 0.1);
            }
            generator.get_training_data()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_feature_extraction,
    bench_inference,
    bench_batch_inference,
    bench_training,
    bench_evaluation,
    bench_training_data_generation,
);

criterion_main!(benches);
