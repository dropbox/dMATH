//! Kernel benchmarks
//!
//! Benchmarks for core kernel operations to measure performance against targets.
//!
//! Target metrics from DESIGN.md:
//! - infer_type (simple): 10μs (vs Lean 4: 100μs)
//! - is_def_eq (simple): 5μs (vs Lean 4: 50μs)
//! - whnf (simple): 2μs (vs Lean 4: 20μs)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lean5_kernel::{
    env::{Declaration, Environment},
    expr::{BinderInfo, Expr},
    inductive::{Constructor, InductiveDecl, InductiveType},
    level::Level,
    name::Name,
    tc::TypeChecker,
};
use std::hint::black_box;

/// Create a simple environment with basic types
fn simple_env() -> Environment {
    let mut env = Environment::new();

    // Add a simple axiom
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("P"),
        level_params: vec![],
        type_: Expr::prop(),
    })
    .unwrap();

    // Add identity function: id : {A : Sort u} → A → A
    let u = Name::from_string("u");
    let id_type = Expr::pi(
        BinderInfo::Implicit,
        Expr::Sort(Level::param(u.clone())),
        Expr::pi(BinderInfo::Default, Expr::bvar(0), Expr::bvar(1)),
    );
    let id_value = Expr::lam(
        BinderInfo::Implicit,
        Expr::Sort(Level::param(u.clone())),
        Expr::lam(BinderInfo::Default, Expr::bvar(0), Expr::bvar(0)),
    );
    env.add_decl(Declaration::Definition {
        name: Name::from_string("id"),
        level_params: vec![u],
        type_: id_type,
        value: id_value,
        is_reducible: true,
    })
    .unwrap();

    env
}

/// Create environment with Nat inductive type
fn nat_env() -> Environment {
    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();
    env
}

/// Build a Nat literal using successor constructor
fn build_nat(env: &Environment, n: u32) -> Expr {
    let _ = env; // env used to ensure Nat is defined
    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    let succ = Expr::const_(Name::from_string("Nat.succ"), vec![]);

    let mut result = zero;
    for _ in 0..n {
        result = Expr::app(succ.clone(), result);
    }
    result
}

/// Build a nested identity application: id (id (id ... P)) where P : Prop
fn nested_id_app(env: &Environment, depth: u32) -> Expr {
    let _ = env; // ensure id is defined
    let one = Level::succ(Level::zero());
    let id = Expr::const_(Name::from_string("id"), vec![one]);
    // P is a constant of type Prop
    let p = Expr::const_(Name::from_string("P"), vec![]);

    let mut result = p;
    for _ in 0..depth {
        // id.{1} Prop : Prop → Prop (since Prop : Type = Sort 1)
        let id_prop = Expr::app(id.clone(), Expr::prop());
        // id.{1} Prop result : Prop
        result = Expr::app(id_prop, result);
    }
    result
}

/// Build a nested lambda: λ x. λ y. λ z. ... x
fn nested_lambda(depth: u32) -> Expr {
    let mut body = Expr::bvar(depth - 1);
    for i in 0..depth {
        body = Expr::lam(
            BinderInfo::Default,
            Expr::Sort(Level::zero().add_offset(i)),
            body,
        );
    }
    body
}

/// Build a nested beta redex: (λ x. x) ((λ x. x) ((λ x. x) Prop))
fn nested_beta_redex(depth: u32) -> Expr {
    let id_lam = Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0));

    let mut result = Expr::prop();
    for _ in 0..depth {
        result = Expr::app(id_lam.clone(), result);
    }
    result
}

// === Benchmarks ===

fn bench_infer_type_sort(c: &mut Criterion) {
    let env = Environment::new();

    c.bench_function("infer_type/Sort_0", |b| {
        b.iter(|| {
            let mut tc = TypeChecker::new(&env);
            tc.infer_type(black_box(&Expr::prop())).unwrap()
        });
    });

    c.bench_function("infer_type/Sort_1", |b| {
        b.iter(|| {
            let mut tc = TypeChecker::new(&env);
            tc.infer_type(black_box(&Expr::type_())).unwrap()
        });
    });
}

fn bench_infer_type_lambda(c: &mut Criterion) {
    let env = Environment::new();

    // λ (x : Prop). x
    let simple_lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::bvar(0));

    c.bench_function("infer_type/lambda_simple", |b| {
        b.iter(|| {
            let mut tc = TypeChecker::new(&env);
            tc.infer_type(black_box(&simple_lam)).unwrap()
        });
    });

    let mut group = c.benchmark_group("infer_type/lambda_nested");
    for depth in [2, 4, 8, 16] {
        let nested = nested_lambda(depth);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &nested, |b, expr| {
            b.iter(|| {
                let mut tc = TypeChecker::new(&env);
                tc.infer_type(black_box(expr)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_infer_type_app(c: &mut Criterion) {
    let env = simple_env();

    // id.{1} Prop P
    let one = Level::succ(Level::zero());
    let id_app = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("id"), vec![one]),
            Expr::prop(),
        ),
        Expr::const_(Name::from_string("P"), vec![]),
    );

    c.bench_function("infer_type/app_simple", |b| {
        b.iter(|| {
            let mut tc = TypeChecker::new(&env);
            tc.infer_type(black_box(&id_app)).unwrap()
        });
    });

    let mut group = c.benchmark_group("infer_type/app_nested");
    for depth in [2, 4, 8, 16] {
        let nested = nested_id_app(&env, depth);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &nested, |b, expr| {
            b.iter(|| {
                let mut tc = TypeChecker::new(&env);
                tc.infer_type(black_box(expr)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_whnf_beta(c: &mut Criterion) {
    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    // (λ x. x) Prop
    let simple_beta = Expr::app(
        Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0)),
        Expr::prop(),
    );

    c.bench_function("whnf/beta_simple", |b| {
        b.iter(|| tc.whnf(black_box(&simple_beta)));
    });

    let mut group = c.benchmark_group("whnf/beta_nested");
    for depth in [2, 4, 8, 16, 32] {
        let nested = nested_beta_redex(depth);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &nested, |b, expr| {
            b.iter(|| tc.whnf(black_box(expr)));
        });
    }
    group.finish();
}

fn bench_whnf_delta(c: &mut Criterion) {
    let env = simple_env();
    let tc = TypeChecker::new(&env);

    // id.{1} - should unfold the definition
    let one = Level::succ(Level::zero());
    let id_const = Expr::const_(Name::from_string("id"), vec![one]);

    c.bench_function("whnf/delta_unfold", |b| {
        b.iter(|| tc.whnf(black_box(&id_const)));
    });
}

fn bench_whnf_iota(c: &mut Criterion) {
    let env = nat_env();
    let tc = TypeChecker::new(&env);

    // Build Nat.rec application on Nat.zero
    let nat = Expr::const_(Name::from_string("Nat"), vec![]);
    let motive = Expr::lam(BinderInfo::Default, nat.clone(), Expr::prop());
    let zero_case = Expr::prop();
    let succ_case = Expr::lam(
        BinderInfo::Default,
        nat.clone(),
        Expr::lam(BinderInfo::Default, Expr::prop(), Expr::prop()),
    );
    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);

    let rec = Expr::const_(Name::from_string("Nat.rec"), vec![Level::zero()]);
    let rec_app = Expr::app(
        Expr::app(Expr::app(Expr::app(rec, motive), zero_case), succ_case),
        zero,
    );

    c.bench_function("whnf/iota_nat_zero", |b| {
        b.iter(|| tc.whnf(black_box(&rec_app)));
    });
}

fn bench_is_def_eq_simple(c: &mut Criterion) {
    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    c.bench_function("is_def_eq/identical", |b| {
        let prop = Expr::prop();
        b.iter(|| tc.is_def_eq(black_box(&prop), black_box(&prop)));
    });

    c.bench_function("is_def_eq/different_sorts", |b| {
        let prop = Expr::prop();
        let type_ = Expr::type_();
        b.iter(|| tc.is_def_eq(black_box(&prop), black_box(&type_)));
    });

    // max(0, 0) == 0
    c.bench_function("is_def_eq/level_normalize", |b| {
        let max_00 = Expr::Sort(Level::max(Level::zero(), Level::zero()));
        let zero = Expr::prop();
        b.iter(|| tc.is_def_eq(black_box(&max_00), black_box(&zero)));
    });
}

fn bench_is_def_eq_beta(c: &mut Criterion) {
    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    // (λ x. x) Prop == Prop
    let beta_lhs = Expr::app(
        Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0)),
        Expr::prop(),
    );
    let beta_rhs = Expr::prop();

    c.bench_function("is_def_eq/beta_reduce", |b| {
        b.iter(|| tc.is_def_eq(black_box(&beta_lhs), black_box(&beta_rhs)));
    });
}

fn bench_is_def_eq_structural(c: &mut Criterion) {
    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    let mut group = c.benchmark_group("is_def_eq/structural");
    for depth in [2, 4, 8, 16] {
        let lam1 = nested_lambda(depth);
        let lam2 = nested_lambda(depth);
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            &(lam1, lam2),
            |b, (l1, l2)| {
                b.iter(|| tc.is_def_eq(black_box(l1), black_box(l2)));
            },
        );
    }
    group.finish();
}

fn bench_nat_operations(c: &mut Criterion) {
    let env = nat_env();

    let mut group = c.benchmark_group("nat/build");
    for n in [1, 5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_nat(&env, black_box(n)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("nat/infer_type");
    for n in [1, 5, 10, 20] {
        let nat_n = build_nat(&env, n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &nat_n, |b, expr| {
            b.iter(|| {
                let mut tc = TypeChecker::new(&env);
                tc.infer_type(black_box(expr)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_environment_lookup(c: &mut Criterion) {
    let mut env = Environment::new();

    // Add many declarations
    for i in 0..100 {
        env.add_decl(Declaration::Axiom {
            name: Name::from_string(&format!("decl_{i}")),
            level_params: vec![],
            type_: Expr::prop(),
        })
        .unwrap();
    }

    c.bench_function("env/lookup_first", |b| {
        b.iter(|| env.get_const(black_box(&Name::from_string("decl_0"))));
    });

    c.bench_function("env/lookup_middle", |b| {
        b.iter(|| env.get_const(black_box(&Name::from_string("decl_50"))));
    });

    c.bench_function("env/lookup_last", |b| {
        b.iter(|| env.get_const(black_box(&Name::from_string("decl_99"))));
    });

    c.bench_function("env/lookup_missing", |b| {
        b.iter(|| env.get_const(black_box(&Name::from_string("nonexistent"))));
    });
}

criterion_group!(
    benches,
    bench_infer_type_sort,
    bench_infer_type_lambda,
    bench_infer_type_app,
    bench_whnf_beta,
    bench_whnf_delta,
    bench_whnf_iota,
    bench_is_def_eq_simple,
    bench_is_def_eq_beta,
    bench_is_def_eq_structural,
    bench_nat_operations,
    bench_environment_lookup,
);

criterion_main!(benches);
