//! Benchmarks for CLI argument parsing and configuration
//!
//! Run with: `cargo bench -p dashprove-cli`
//!
//! Measures the performance of:
//! - Clap argument parsing for various command structures
//! - Backend name parsing and validation
//! - Output format parsing
//! - Configuration struct construction

use clap::Parser;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashprove_cli::cli::{AnalyzeAction, Cli, Commands, CorpusAction};

// Helper to parse CLI from args slice
fn parse_cli(args: &[&str]) -> Result<Cli, clap::Error> {
    let full_args: Vec<&str> = std::iter::once("dashprove")
        .chain(args.iter().copied())
        .collect();
    Cli::try_parse_from(full_args)
}

// Benchmarks for command parsing

fn bench_command_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("command_parsing");

    // Verify command - minimal
    group.bench_function("verify/minimal", |b| {
        b.iter(|| parse_cli(black_box(&["verify", "spec.usl"])))
    });

    // Verify command - with backends
    group.bench_function("verify/with_backends", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "verify",
                "spec.usl",
                "--backends",
                "lean,tla+,kani",
            ]))
        })
    });

    // Verify command - full options
    group.bench_function("verify/full_options", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "verify",
                "spec.usl",
                "--backends",
                "lean,tla+",
                "--timeout",
                "300",
                "--learn",
                "--suggest",
                "--verbose",
                "--ml",
                "--ml-confidence",
                "0.7",
            ]))
        })
    });

    // Verify command - incremental
    group.bench_function("verify/incremental", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "verify",
                "spec.usl",
                "--incremental",
                "--since",
                "HEAD~1",
            ]))
        })
    });

    // Export command
    group.bench_function("export/minimal", |b| {
        b.iter(|| parse_cli(black_box(&["export", "spec.usl", "--target", "lean"])))
    });

    group.bench_function("export/with_output", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "export",
                "spec.usl",
                "--target",
                "lean",
                "-o",
                "output.lean",
            ]))
        })
    });

    // Train command - minimal
    group.bench_function("train/minimal", |b| {
        b.iter(|| parse_cli(black_box(&["train"])))
    });

    // Train command - full options
    group.bench_function("train/full_options", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "train",
                "--learning-rate",
                "0.01",
                "--epochs",
                "50",
                "--early-stopping",
                "--patience",
                "5",
                "--validation-split",
                "0.2",
                "--lr-scheduler",
                "cosine",
                "--verbose",
            ]))
        })
    });

    // Train command with checkpointing
    group.bench_function("train/with_checkpointing", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "train",
                "--checkpoint",
                "--checkpoint-interval",
                "10",
                "--keep-best",
                "3",
            ]))
        })
    });

    // Tune command
    group.bench_function("tune/minimal", |b| {
        b.iter(|| parse_cli(black_box(&["tune"])))
    });

    group.bench_function("tune/grid_search", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "tune",
                "--method",
                "grid",
                "--lr-values",
                "0.001,0.01,0.1",
                "--epoch-values",
                "50,100",
            ]))
        })
    });

    group.bench_function("tune/bayesian", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "tune",
                "--method",
                "bayesian",
                "--iterations",
                "25",
                "--kappa",
                "2.5",
                "--cv-folds",
                "5",
            ]))
        })
    });

    group.finish();
}

// Benchmarks for subcommand parsing

fn bench_subcommand_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("subcommand_parsing");

    // Corpus subcommands
    group.bench_function("corpus/stats", |b| {
        b.iter(|| parse_cli(black_box(&["corpus", "stats"])))
    });

    group.bench_function("corpus/search", |b| {
        b.iter(|| parse_cli(black_box(&["corpus", "search", "spec.usl", "-n", "10"])))
    });

    group.bench_function("corpus/history", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "corpus",
                "history",
                "--corpus",
                "counterexamples",
                "--period",
                "week",
                "--format",
                "html",
            ]))
        })
    });

    group.bench_function("corpus/compare", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "corpus",
                "compare",
                "--baseline-from",
                "2024-01-01",
                "--baseline-to",
                "2024-01-31",
                "--compare-from",
                "2024-02-01",
                "--compare-to",
                "2024-02-28",
            ]))
        })
    });

    // Analyze subcommands
    group.bench_function("analyze/suggest", |b| {
        b.iter(|| parse_cli(black_box(&["analyze", "cx.json", "suggest"])))
    });

    group.bench_function("analyze/compress", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "analyze", "cx.json", "compress", "--format", "html",
            ]))
        })
    });

    group.bench_function("analyze/diff", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "analyze", "cx1.json", "diff", "cx2.json", "--format", "html",
            ]))
        })
    });

    group.finish();
}

// Benchmarks for multi-argument parsing

fn bench_multi_argument_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_argument_parsing");

    // Cluster command with multiple files
    group.bench_function(BenchmarkId::new("cluster", "2_files"), |b| {
        b.iter(|| parse_cli(black_box(&["cluster", "cx1.json", "cx2.json"])))
    });

    group.bench_function(BenchmarkId::new("cluster", "5_files"), |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "cluster", "cx1.json", "cx2.json", "cx3.json", "cx4.json", "cx5.json",
            ]))
        })
    });

    group.bench_function(BenchmarkId::new("cluster", "10_files"), |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "cluster",
                "cx1.json",
                "cx2.json",
                "cx3.json",
                "cx4.json",
                "cx5.json",
                "cx6.json",
                "cx7.json",
                "cx8.json",
                "cx9.json",
                "cx10.json",
            ]))
        })
    });

    // Ensemble command with multiple models
    group.bench_function(BenchmarkId::new("ensemble", "2_models"), |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "ensemble",
                "--models",
                "model1.json,model2.json",
            ]))
        })
    });

    group.bench_function(BenchmarkId::new("ensemble", "5_models"), |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "ensemble",
                "--models",
                "m1.json,m2.json,m3.json,m4.json,m5.json",
                "--weights",
                "0.3,0.2,0.2,0.2,0.1",
            ]))
        })
    });

    group.finish();
}

// Benchmarks for option validation (post-parse)

fn bench_option_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("option_validation");

    // Monitor command options
    group.bench_function("monitor/rust_assertions", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "monitor",
                "spec.usl",
                "--target",
                "rust",
                "--assertions",
            ]))
        })
    });

    group.bench_function("monitor/all_options", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "monitor",
                "spec.usl",
                "--target",
                "typescript",
                "--assertions",
                "--logging",
                "--metrics",
                "-o",
                "monitor.ts",
            ]))
        })
    });

    // Visualize command formats
    group.bench_function("visualize/html", |b| {
        b.iter(|| parse_cli(black_box(&["visualize", "cx.json", "--format", "html"])))
    });

    group.bench_function("visualize/mermaid", |b| {
        b.iter(|| parse_cli(black_box(&["visualize", "cx.json", "--format", "mermaid"])))
    });

    group.bench_function("visualize/dot", |b| {
        b.iter(|| parse_cli(black_box(&["visualize", "cx.json", "--format", "dot"])))
    });

    // Search command
    group.bench_function("search/basic", |b| {
        b.iter(|| parse_cli(black_box(&["search", "termination proof"])))
    });

    group.bench_function("search/with_options", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "search",
                "mutex invariant",
                "-n",
                "20",
                "--data-dir",
                "/tmp/dashprove",
            ]))
        })
    });

    group.finish();
}

// Benchmarks for simple commands

fn bench_simple_commands(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_commands");

    group.bench_function("backends", |b| {
        b.iter(|| parse_cli(black_box(&["backends"])))
    });

    group.bench_function("topics", |b| b.iter(|| parse_cli(black_box(&["topics"]))));

    group.bench_function("topics/specific", |b| {
        b.iter(|| parse_cli(black_box(&["topics", "usl"])))
    });

    group.bench_function("explain", |b| {
        b.iter(|| parse_cli(black_box(&["explain", "counterexample.json"])))
    });

    group.bench_function("prove", |b| {
        b.iter(|| parse_cli(black_box(&["prove", "spec.usl", "--hints"])))
    });

    group.bench_function("verify-code", |b| {
        b.iter(|| {
            parse_cli(black_box(&[
                "verify-code",
                "--code",
                "src/lib.rs",
                "--spec",
                "contracts.usl",
            ]))
        })
    });

    group.finish();
}

// Benchmarks for command matching after parsing

fn bench_command_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("command_matching");

    // Pre-parse the commands
    let verify_cmd = parse_cli(&["verify", "spec.usl"]).unwrap();
    let train_cmd = parse_cli(&["train", "--verbose"]).unwrap();
    let corpus_stats_cmd = parse_cli(&["corpus", "stats"]).unwrap();
    let analyze_suggest_cmd = parse_cli(&["analyze", "cx.json", "suggest"]).unwrap();

    group.bench_function("match_verify", |b| {
        b.iter(|| {
            matches!(
                black_box(&verify_cmd.command),
                Commands::Verify { path, .. } if path == "spec.usl"
            )
        })
    });

    group.bench_function("match_train", |b| {
        b.iter(|| {
            matches!(
                black_box(&train_cmd.command),
                Commands::Train { verbose: true, .. }
            )
        })
    });

    group.bench_function("match_corpus_action", |b| {
        b.iter(|| {
            matches!(
                black_box(&corpus_stats_cmd.command),
                Commands::Corpus {
                    action: CorpusAction::Stats { .. }
                }
            )
        })
    });

    group.bench_function("match_analyze_action", |b| {
        b.iter(|| {
            matches!(
                black_box(&analyze_suggest_cmd.command),
                Commands::Analyze {
                    action: AnalyzeAction::Suggest { .. },
                    ..
                }
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_command_parsing,
    bench_subcommand_parsing,
    bench_multi_argument_parsing,
    bench_option_validation,
    bench_simple_commands,
    bench_command_matching,
);

criterion_main!(benches);
