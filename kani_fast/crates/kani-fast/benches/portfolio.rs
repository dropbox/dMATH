//! Benchmarks for Portfolio solver management
//!
//! These benchmarks measure the performance of portfolio solving
//! with multiple SAT/SMT solvers running in parallel.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kani_fast_portfolio::{PortfolioBuilder, PortfolioConfig, PortfolioStrategy, SolverConfig};
use std::fmt::Write;
use std::time::Duration;

fn solvers_available() -> bool {
    which::which("cadical").is_ok() || which::which("kissat").is_ok() || which::which("z3").is_ok()
}

/// Generate a simple satisfiable CNF problem
fn generate_sat_cnf(num_vars: usize, num_clauses: usize) -> String {
    let mut cnf = format!("p cnf {} {}\n", num_vars, num_clauses);
    for i in 0..num_clauses {
        // Simple clauses: (x_i OR x_{i+1})
        let v1 = (i % num_vars) + 1;
        let v2 = ((i + 1) % num_vars) + 1;
        let _ = writeln!(cnf, "{} {} 0", v1, v2);
    }
    cnf
}

/// Generate an unsatisfiable CNF (pigeonhole principle)
fn generate_unsat_cnf(n: usize) -> String {
    // PHP_{n,n-1}: n pigeons, n-1 holes
    let mut cnf = String::new();
    let holes = n - 1;

    // Variables: x_{i,j} = pigeon i in hole j
    // Variable number: i * holes + j + 1

    let mut clauses = Vec::new();

    // At least one hole per pigeon
    for i in 0..n {
        let clause: Vec<i32> = (0..holes).map(|j| (i * holes + j + 1) as i32).collect();
        clauses.push(clause);
    }

    // At most one pigeon per hole
    for j in 0..holes {
        for i1 in 0..n {
            for i2 in (i1 + 1)..n {
                let v1 = (i1 * holes + j + 1) as i32;
                let v2 = (i2 * holes + j + 1) as i32;
                clauses.push(vec![-v1, -v2]);
            }
        }
    }

    let num_vars = n * holes;
    let _ = writeln!(cnf, "p cnf {} {}", num_vars, clauses.len());
    for clause in clauses {
        for lit in clause {
            let _ = write!(cnf, "{} ", lit);
        }
        cnf.push_str("0\n");
    }
    cnf
}

/// Benchmark portfolio auto-detection time
fn benchmark_portfolio_detection(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("portfolio_detection");
    group.sample_size(20);

    group.bench_function("auto_detect_solvers", |b| {
        b.to_async(&runtime)
            .iter(|| async { PortfolioBuilder::new().auto_detect().await.build() });
    });

    group.finish();
}

/// Benchmark portfolio solving with different numbers of solvers
fn benchmark_portfolio_parallel(c: &mut Criterion) {
    if !solvers_available() {
        eprintln!("Skipping portfolio benchmark: no solvers available");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let portfolio = runtime.block_on(async { PortfolioBuilder::new().auto_detect().await.build() });

    if portfolio.is_empty() {
        eprintln!("No solvers detected, skipping portfolio benchmarks");
        return;
    }

    let mut group = c.benchmark_group("portfolio_parallel");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Create test CNF file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("portfolio_bench.cnf");

    // Simple SAT problem
    let cnf = generate_sat_cnf(50, 100);
    std::fs::write(&temp_file, &cnf).unwrap();

    for max_concurrent in [1, 2, 4] {
        if max_concurrent > portfolio.len() {
            continue;
        }

        group.bench_with_input(
            BenchmarkId::new("concurrent", max_concurrent),
            &max_concurrent,
            |b, &max_concurrent| {
                let config = PortfolioConfig {
                    strategy: PortfolioStrategy::All,
                    solver_config: SolverConfig {
                        timeout: Duration::from_secs(10),
                        ..Default::default()
                    },
                    max_concurrent,
                    cancel_on_first: true,
                };

                b.to_async(&runtime).iter(|| {
                    let p = &portfolio;
                    let f = &temp_file;
                    let cfg = config.clone();
                    async move { p.solve_dimacs(f, &cfg).await.expect("Portfolio failed") }
                });
            },
        );
    }

    std::fs::remove_file(&temp_file).ok();
    group.finish();
}

/// Benchmark different problem sizes
fn benchmark_problem_scaling(c: &mut Criterion) {
    if !solvers_available() {
        eprintln!("Skipping problem scaling benchmark: no solvers available");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let portfolio = runtime.block_on(async { PortfolioBuilder::new().auto_detect().await.build() });

    if portfolio.is_empty() {
        eprintln!("No solvers detected, skipping problem scaling benchmarks");
        return;
    }

    let mut group = c.benchmark_group("portfolio_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let config = PortfolioConfig {
        strategy: PortfolioStrategy::All,
        solver_config: SolverConfig {
            timeout: Duration::from_secs(30),
            ..Default::default()
        },
        max_concurrent: 4,
        cancel_on_first: true,
    };

    let temp_dir = std::env::temp_dir();

    // SAT problems of increasing size
    for (num_vars, num_clauses) in [(10, 20), (50, 100), (100, 200)] {
        let temp_file = temp_dir.join(format!("scaling_{}_{}.cnf", num_vars, num_clauses));
        let cnf = generate_sat_cnf(num_vars, num_clauses);
        std::fs::write(&temp_file, &cnf).unwrap();

        group.bench_with_input(
            BenchmarkId::new("sat_vars_clauses", format!("{}_{}", num_vars, num_clauses)),
            &temp_file,
            |b, temp_file| {
                let cfg = config.clone();
                b.to_async(&runtime).iter(|| {
                    let p = &portfolio;
                    let f = temp_file;
                    let cfg_inner = cfg.clone();
                    async move {
                        p.solve_dimacs(f, &cfg_inner)
                            .await
                            .expect("Portfolio failed")
                    }
                });
            },
        );

        std::fs::remove_file(&temp_file).ok();
    }

    // UNSAT problems (pigeonhole) - much harder
    for n in [4, 5] {
        let temp_file = temp_dir.join(format!("php_{}.cnf", n));
        let cnf = generate_unsat_cnf(n);
        std::fs::write(&temp_file, &cnf).unwrap();

        group.bench_with_input(
            BenchmarkId::new("unsat_php", n),
            &temp_file,
            |b, temp_file| {
                let cfg = config.clone();
                b.to_async(&runtime).iter(|| {
                    let p = &portfolio;
                    let f = temp_file;
                    let cfg_inner = cfg.clone();
                    async move {
                        p.solve_dimacs(f, &cfg_inner)
                            .await
                            .expect("Portfolio failed")
                    }
                });
            },
        );

        std::fs::remove_file(&temp_file).ok();
    }

    group.finish();
}

/// Benchmark adaptive strategy
fn benchmark_adaptive_strategy(c: &mut Criterion) {
    if !solvers_available() {
        eprintln!("Skipping adaptive benchmark: no solvers available");
        return;
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let portfolio = runtime.block_on(async { PortfolioBuilder::new().auto_detect().await.build() });

    if portfolio.len() < 2 {
        eprintln!("Need at least 2 solvers for adaptive benchmark");
        return;
    }

    let mut group = c.benchmark_group("portfolio_adaptive");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("adaptive_bench.cnf");
    let cnf = generate_sat_cnf(50, 100);
    std::fs::write(&temp_file, &cnf).unwrap();

    // Get first solver ID for adaptive strategy
    let first_solver = portfolio.solver_infos()[0].id.clone();

    // All solvers at once (baseline)
    let all_config = PortfolioConfig {
        strategy: PortfolioStrategy::All,
        solver_config: SolverConfig {
            timeout: Duration::from_secs(10),
            ..Default::default()
        },
        max_concurrent: 4,
        cancel_on_first: true,
    };

    group.bench_function("all_at_once", |b| {
        let cfg = all_config.clone();
        b.to_async(&runtime).iter(|| {
            let p = &portfolio;
            let f = &temp_file;
            let cfg_inner = cfg.clone();
            async move {
                p.solve_dimacs(f, &cfg_inner)
                    .await
                    .expect("Portfolio failed")
            }
        });
    });

    // Adaptive: start with first solver, add others after 100ms
    let adaptive_config = PortfolioConfig {
        strategy: PortfolioStrategy::Adaptive {
            initial: vec![first_solver],
            delay: Duration::from_millis(100),
        },
        solver_config: SolverConfig {
            timeout: Duration::from_secs(10),
            ..Default::default()
        },
        max_concurrent: 4,
        cancel_on_first: true,
    };

    group.bench_function("adaptive_100ms", |b| {
        let cfg = adaptive_config.clone();
        b.to_async(&runtime).iter(|| {
            let p = &portfolio;
            let f = &temp_file;
            let cfg_inner = cfg.clone();
            async move {
                p.solve_dimacs(f, &cfg_inner)
                    .await
                    .expect("Portfolio failed")
            }
        });
    });

    std::fs::remove_file(&temp_file).ok();
    group.finish();
}

/// Benchmark CNF generation (no solver needed)
fn benchmark_cnf_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnf_generation");
    group.sample_size(100);

    for (vars, clauses) in [(10, 20), (50, 100), (100, 200), (500, 1000)] {
        group.bench_with_input(
            BenchmarkId::new("sat", format!("{}_{}", vars, clauses)),
            &(vars, clauses),
            |b, &(vars, clauses)| {
                b.iter(|| generate_sat_cnf(vars, clauses));
            },
        );
    }

    for n in [4, 5, 6, 7] {
        group.bench_with_input(BenchmarkId::new("php", n), &n, |b, &n| {
            b.iter(|| generate_unsat_cnf(n));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_portfolio_detection,
    benchmark_cnf_generation,
    benchmark_portfolio_parallel,
    benchmark_problem_scaling,
    benchmark_adaptive_strategy
);
criterion_main!(benches);
