//! Performance benchmarks for DashProve
//!
//! Run with: cargo bench -p dashprove
//!
//! These benchmarks measure the performance of core operations:
//! - USL parsing
//! - Type checking
//! - Backend compilation (code generation for all 8 USL compilers)
//!   - Theorem provers: LEAN 4, Coq, Isabelle
//!   - Model checkers: TLA+, Alloy
//!   - Verifying compilers: Kani, Dafny
//!   - SMT solvers: SMT-LIB2 (Z3/CVC5)
//! - Runtime monitor generation (Rust, TypeScript, Python)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dashprove::monitor::MonitorTarget;
use dashprove::{MonitorConfig, RuntimeMonitor};
use dashprove_usl::{
    compile_to_alloy, compile_to_coq, compile_to_dafny, compile_to_isabelle, compile_to_kani,
    compile_to_lean, compile_to_smtlib2, compile_to_tlaplus,
};
use dashprove_usl::{parse, typecheck};

// Test specifications of varying complexity
const SIMPLE_SPEC: &str = "theorem test { forall x: Bool . x or not x }";

const MEDIUM_SPEC: &str = r#"
theorem excluded_middle {
    forall x: Bool . x or not x
}

theorem implication {
    forall p: Bool, q: Bool . (p and (p implies q)) implies q
}

theorem de_morgan {
    forall a: Bool, b: Bool . not (a and b) == (not a or not b)
}

invariant positive_square {
    forall n: Int . n >= 0 implies n * n >= 0
}
"#;

const COMPLEX_SPEC: &str = r#"
theorem excluded_middle {
    forall x: Bool . x or not x
}

theorem implication {
    forall p: Bool, q: Bool . (p and (p implies q)) implies q
}

theorem de_morgan {
    forall a: Bool, b: Bool . not (a and b) == (not a or not b)
}

theorem contraposition {
    forall p: Bool, q: Bool . (p implies q) == (not q implies not p)
}

theorem double_negation {
    forall x: Bool . not (not x) == x
}

theorem distribution {
    forall a: Bool, b: Bool, c: Bool . (a and (b or c)) == ((a and b) or (a and c))
}

invariant positive_square {
    forall n: Int . n >= 0 implies n * n >= 0
}

invariant zero_identity {
    forall x: Int . x + 0 == x
}

invariant multiplication_identity {
    forall x: Int . x * 1 == x
}
"#;

fn bench_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");

    group.throughput(Throughput::Bytes(SIMPLE_SPEC.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("simple", "1 theorem"),
        SIMPLE_SPEC,
        |b, spec| b.iter(|| parse(black_box(spec)).unwrap()),
    );

    group.throughput(Throughput::Bytes(MEDIUM_SPEC.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("medium", "4 properties"),
        MEDIUM_SPEC,
        |b, spec| b.iter(|| parse(black_box(spec)).unwrap()),
    );

    group.throughput(Throughput::Bytes(COMPLEX_SPEC.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("complex", "9 properties"),
        COMPLEX_SPEC,
        |b, spec| b.iter(|| parse(black_box(spec)).unwrap()),
    );

    group.finish();
}

fn bench_typecheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("typecheck");

    // Pre-parse the specs
    let simple_ast = parse(SIMPLE_SPEC).unwrap();
    let medium_ast = parse(MEDIUM_SPEC).unwrap();
    let complex_ast = parse(COMPLEX_SPEC).unwrap();

    group.bench_function("simple", |b| {
        b.iter(|| typecheck(black_box(simple_ast.clone())).unwrap())
    });

    group.bench_function("medium", |b| {
        b.iter(|| typecheck(black_box(medium_ast.clone())).unwrap())
    });

    group.bench_function("complex", |b| {
        b.iter(|| typecheck(black_box(complex_ast.clone())).unwrap())
    });

    group.finish();
}

fn bench_backend_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_compilation");

    // Pre-parse and typecheck
    let simple_typed = typecheck(parse(SIMPLE_SPEC).unwrap()).unwrap();
    let medium_typed = typecheck(parse(MEDIUM_SPEC).unwrap()).unwrap();

    // LEAN 4 compilation
    group.bench_with_input(
        BenchmarkId::new("lean4", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_lean(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("lean4", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_lean(black_box(spec))),
    );

    // TLA+ compilation
    group.bench_with_input(
        BenchmarkId::new("tlaplus", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_tlaplus(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("tlaplus", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_tlaplus(black_box(spec))),
    );

    // Alloy compilation
    group.bench_with_input(
        BenchmarkId::new("alloy", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_alloy(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("alloy", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_alloy(black_box(spec))),
    );

    // Kani compilation
    group.bench_with_input(
        BenchmarkId::new("kani", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_kani(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("kani", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_kani(black_box(spec))),
    );

    // Coq compilation
    group.bench_with_input(
        BenchmarkId::new("coq", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_coq(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("coq", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_coq(black_box(spec))),
    );

    // Isabelle compilation
    group.bench_with_input(
        BenchmarkId::new("isabelle", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_isabelle(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("isabelle", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_isabelle(black_box(spec))),
    );

    // Dafny compilation
    group.bench_with_input(
        BenchmarkId::new("dafny", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_dafny(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("dafny", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_dafny(black_box(spec))),
    );

    // SMT-LIB2 compilation (Z3/CVC5)
    group.bench_with_input(
        BenchmarkId::new("smtlib2", "simple"),
        &simple_typed,
        |b, spec| b.iter(|| compile_to_smtlib2(black_box(spec))),
    );
    group.bench_with_input(
        BenchmarkId::new("smtlib2", "medium"),
        &medium_typed,
        |b, spec| b.iter(|| compile_to_smtlib2(black_box(spec))),
    );

    group.finish();
}

fn bench_monitor_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("monitor_generation");

    // Pre-parse and typecheck
    let simple_typed = typecheck(parse(SIMPLE_SPEC).unwrap()).unwrap();
    let medium_typed = typecheck(parse(MEDIUM_SPEC).unwrap()).unwrap();

    let default_config = MonitorConfig::default();
    let full_config = MonitorConfig {
        generate_assertions: true,
        generate_logging: true,
        generate_metrics: true,
        ..Default::default()
    };

    group.bench_function("simple/rust", |b| {
        b.iter(|| RuntimeMonitor::from_spec(black_box(&simple_typed), black_box(&default_config)))
    });

    group.bench_function("medium/rust", |b| {
        b.iter(|| RuntimeMonitor::from_spec(black_box(&medium_typed), black_box(&default_config)))
    });

    group.bench_function("medium/rust_full_features", |b| {
        b.iter(|| RuntimeMonitor::from_spec(black_box(&medium_typed), black_box(&full_config)))
    });

    // TypeScript target
    let ts_config = MonitorConfig {
        target: MonitorTarget::TypeScript,
        ..Default::default()
    };
    group.bench_function("medium/typescript", |b| {
        b.iter(|| RuntimeMonitor::from_spec(black_box(&medium_typed), black_box(&ts_config)))
    });

    // Python target
    let py_config = MonitorConfig {
        target: MonitorTarget::Python,
        ..Default::default()
    };
    group.bench_function("medium/python", |b| {
        b.iter(|| RuntimeMonitor::from_spec(black_box(&medium_typed), black_box(&py_config)))
    });

    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");

    // Measure full pipeline: parse -> typecheck -> compile
    // Lean4 (theorem prover)
    group.bench_function("simple_to_lean4", |b| {
        b.iter(|| {
            let ast = parse(black_box(SIMPLE_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_lean(&typed)
        })
    });

    group.bench_function("medium_to_lean4", |b| {
        b.iter(|| {
            let ast = parse(black_box(MEDIUM_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_lean(&typed)
        })
    });

    group.bench_function("complex_to_lean4", |b| {
        b.iter(|| {
            let ast = parse(black_box(COMPLEX_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_lean(&typed)
        })
    });

    // Coq (theorem prover)
    group.bench_function("medium_to_coq", |b| {
        b.iter(|| {
            let ast = parse(black_box(MEDIUM_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_coq(&typed)
        })
    });

    // Isabelle (theorem prover)
    group.bench_function("medium_to_isabelle", |b| {
        b.iter(|| {
            let ast = parse(black_box(MEDIUM_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_isabelle(&typed)
        })
    });

    // SMT-LIB2 (Z3/CVC5)
    group.bench_function("medium_to_smtlib2", |b| {
        b.iter(|| {
            let ast = parse(black_box(MEDIUM_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_smtlib2(&typed)
        })
    });

    // Dafny (verifying compiler)
    group.bench_function("medium_to_dafny", |b| {
        b.iter(|| {
            let ast = parse(black_box(MEDIUM_SPEC)).unwrap();
            let typed = typecheck(ast).unwrap();
            compile_to_dafny(&typed)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parsing,
    bench_typecheck,
    bench_backend_compilation,
    bench_monitor_generation,
    bench_end_to_end,
);

criterion_main!(benches);
