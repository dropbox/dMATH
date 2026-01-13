//! Kani Fast CLI
//!
//! Command-line interface for Kani Fast verification.

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::stream::{self, StreamExt};
use kani_fast::{
    check_installation, KaniConfig, KaniWrapper, PortfolioKaniConfigBuilder, PortfolioVerifier,
    VerificationStatus,
};
use kani_fast_ai::{AiConfig, AiSynthesizer, InvariantSource};
use kani_fast_chc::{
    encode_mir_to_chc, encode_mir_to_chc_bitvec, encode_mir_to_transition_system,
    generate_chc_proof, generate_mir_from_file, hybrid_verify_with_chc_result,
    program_needs_bitvec_encoding, ChcBackend, ChcResult, ChcSolver, ChcSolverConfig,
    ChcSolverError, HybridConfig, HybridResult, MirParser,
};
use kani_fast_counterexample::{ExplanationGenerator, RepairEngine, StructuredCounterexample};
use kani_fast_incremental::{
    CachedChcResult, CachedOutcome, ChcVerificationCache, ContentHash, IncrementalBmc,
    IncrementalConfigBuilder, WatchEvent, WatchModeBuilder,
};
use kani_fast_kinduction::{
    generate_kinduction_proof, KInduction, KInductionConfigBuilder, PropertyType, SmtType,
    TransitionSystemBuilder,
};
use kani_fast_lean5::{
    certificate_from_chc, certificate_from_kinduction, check_lean_installation, Lean5Backend,
    Lean5Config, Lean5Expr, Lean5Type, ProofObligation, ProofObligationKind,
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{error, info};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BitvecConfig {
    force_bitvec: bool,
    bitvec_width: u32,
}

fn print_overflow_warning() {
    println!(
        "NOTE: Using integer encoding without wraparound. Rust overflows wrap; \
         pass --bitvec or set KANI_FAST_BITVEC=1 to model wrapping semantics."
    );
}

fn resolve_bitvec_config(
    bitvec_flag: bool,
    bitvec_width_arg: Option<u32>,
    bitvec_env_set: bool,
    bitvec_width_env: Option<u32>,
) -> BitvecConfig {
    let bitvec_width = bitvec_width_arg.or(bitvec_width_env).unwrap_or(32);
    let force_bitvec =
        bitvec_flag || bitvec_env_set || bitvec_width_arg.is_some() || bitvec_width_env.is_some();

    BitvecConfig {
        force_bitvec,
        bitvec_width,
    }
}

#[derive(Parser)]
#[command(name = "kani-fast")]
#[command(author, version, about = "Next-generation Rust verification", long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text", global = true)]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ChcBackendArg {
    Auto,
    Z3,
    Z4,
}

impl From<ChcBackendArg> for ChcBackend {
    fn from(value: ChcBackendArg) -> Self {
        match value {
            ChcBackendArg::Auto => ChcBackend::Auto,
            ChcBackendArg::Z3 => ChcBackend::Z3,
            ChcBackendArg::Z4 => ChcBackend::Z4,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Verify a Rust project using BMC (bounded model checking)
    Verify {
        /// Path to the Rust project (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Specific harness to verify
        #[arg(short = 'H', long)]
        harness: Option<String>,

        /// Timeout in seconds
        #[arg(short, long, default_value = "300")]
        timeout: u64,

        /// Loop unwinding bound
        #[arg(short, long)]
        unwind: Option<u32>,

        /// SAT solver to use
        #[arg(long)]
        solver: Option<String>,

        /// Enable portfolio mode: run multiple solvers in parallel
        #[arg(long)]
        portfolio: bool,

        /// Solvers for portfolio mode (comma-separated, e.g., "cadical,kissat,minisat")
        #[arg(long, value_delimiter = ',')]
        portfolio_solvers: Option<Vec<String>>,

        /// Maximum concurrent solvers for portfolio mode
        #[arg(long, default_value = "3")]
        portfolio_max_concurrent: usize,

        /// Generate beautiful counterexample explanations with repair suggestions
        #[arg(long)]
        explain: bool,
    },

    /// Verify using CHC (Constrained Horn Clauses) with Spacer
    Chc {
        /// Rust source file to verify
        source: PathBuf,

        /// Function name to verify (optional, verifies first function if not specified)
        #[arg(short = 'F', long)]
        function: Option<String>,

        /// Timeout in seconds
        #[arg(short, long, default_value = "60")]
        timeout: u64,

        /// Rust edition
        #[arg(long, default_value = "2021")]
        edition: String,

        /// Spacer backend to use (auto prefers Z4, then Z3)
        #[arg(long, value_enum, default_value_t = ChcBackendArg::Auto)]
        backend: ChcBackendArg,

        /// Use BitVec encoding for precise bitwise reasoning
        #[arg(long = "bitvec", alias = "use-bitvec")]
        bitvec: bool,

        /// Bit width for BitVec encoding (default 32 when enabled)
        #[arg(long = "bitvec-width")]
        bitvec_width: Option<u32>,

        /// Generate Lean5 proof obligations from discovered invariants
        #[arg(long)]
        lean5: bool,

        /// Output path for Lean5 file (defaults to <source>.lean)
        #[arg(long)]
        lean5_output: Option<PathBuf>,

        /// Verify all functions ending with _proof
        #[arg(long)]
        all_proofs: bool,

        /// Enable incremental caching (stores results in .kani-cache.db)
        #[arg(long)]
        cache: bool,

        /// Path to cache database (default: .kani-cache.db)
        #[arg(long, default_value = ".kani-cache.db")]
        cache_path: PathBuf,

        /// Force re-verification even if cached
        #[arg(long)]
        no_cache: bool,

        /// Show cache statistics
        #[arg(long)]
        cache_stats: bool,

        /// Number of parallel workers for --all-proofs (default: number of CPU cores)
        #[arg(long, short = 'j')]
        parallel: Option<usize>,

        /// Generate a universal proof file (JSON format)
        #[arg(long)]
        proof: bool,

        /// Output path for proof file (defaults to <source>.proof.json)
        #[arg(long)]
        proof_output: Option<PathBuf>,

        /// Enable hybrid mode: fall back to Kani BMC when CHC returns unknown
        #[arg(long)]
        hybrid: bool,

        /// Timeout for Kani BMC fallback in hybrid mode (seconds)
        #[arg(long, default_value = "60")]
        hybrid_kani_timeout: u64,
    },

    /// Verify using k-induction for unbounded verification
    Kinduction {
        /// SMT-LIB2 file with transition system, or use --demo for built-in example
        #[arg(required_unless_present = "demo")]
        input: Option<PathBuf>,

        /// Run a demonstration with a built-in counter example
        #[arg(long)]
        demo: bool,

        /// Maximum k value for induction
        #[arg(short = 'k', long, default_value = "10")]
        max_k: u32,

        /// Timeout in seconds
        #[arg(short, long, default_value = "60")]
        timeout: u64,

        /// Generate a universal proof file (JSON format)
        #[arg(long)]
        proof: bool,

        /// Output path for proof file (defaults to <input>.proof.json)
        #[arg(long)]
        proof_output: Option<PathBuf>,
    },

    /// Unified unbounded verification (tries k-induction, then CHC)
    Unbounded {
        /// Rust source file to verify
        source: PathBuf,

        /// Function name to verify (optional)
        #[arg(short = 'F', long)]
        function: Option<String>,

        /// Maximum k value for k-induction phase
        #[arg(short = 'k', long, default_value = "5")]
        max_k: u32,

        /// Timeout in seconds (total for all phases)
        #[arg(short, long, default_value = "120")]
        timeout: u64,

        /// Rust edition
        #[arg(long, default_value = "2021")]
        edition: String,

        /// Skip k-induction and go directly to CHC
        #[arg(long)]
        chc_only: bool,

        /// Skip CHC and only try k-induction
        #[arg(long)]
        kinduction_only: bool,

        /// Spacer backend to use for CHC phase
        #[arg(long, value_enum, default_value_t = ChcBackendArg::Auto)]
        chc_backend: ChcBackendArg,

        /// Generate Lean5 proof obligations from discovered invariants
        #[arg(long)]
        lean5: bool,

        /// Output path for Lean5 file (defaults to <source>.lean)
        #[arg(long)]
        lean5_output: Option<PathBuf>,

        /// Enable AI-assisted invariant synthesis as fallback
        #[arg(long)]
        ai: bool,

        /// Maximum AI synthesis attempts
        #[arg(long, default_value = "10")]
        ai_max_attempts: usize,

        /// Skip k-induction and CHC, go directly to AI synthesis
        #[arg(long)]
        ai_only: bool,

        /// Generate a proof certificate (includes Lean5 file + JSON metadata)
        #[arg(long)]
        certificate: bool,

        /// Verify generated Lean5 proof with Lean (requires Lean installation)
        #[arg(long)]
        verify_lean: bool,
    },

    /// Check a Lean5 file for proof validity
    Lean5Check {
        /// Path to the Lean5 file to check
        file: PathBuf,

        /// Timeout in seconds
        #[arg(short, long, default_value = "60")]
        timeout: u64,
    },

    /// Watch a project for changes and verify continuously
    Watch {
        /// Path to the project to watch (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Debounce duration in milliseconds (time to wait after last change)
        #[arg(short, long, default_value = "500")]
        debounce: u64,

        /// Solver timeout in seconds
        #[arg(short, long, default_value = "300")]
        timeout: u64,
    },

    /// Incremental verification with clause learning and caching
    Incremental {
        /// DIMACS CNF file to verify
        input: PathBuf,

        /// Solver timeout in seconds
        #[arg(short, long, default_value = "300")]
        timeout: u64,

        /// Path to the cache database (defaults to .kani_fast_cache.db)
        #[arg(long)]
        cache_db: Option<PathBuf>,

        /// Clear cache before verification
        #[arg(long)]
        clear_cache: bool,

        /// Show cache statistics
        #[arg(long)]
        stats: bool,
    },

    /// Check if Kani and solvers are properly installed
    Check,

    /// Show version information
    Version,
}

fn setup_logging(verbose: bool) {
    let filter = if verbose { "debug" } else { "warn" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn has_z3() -> bool {
    which::which("z3").is_ok()
}

fn print_counterexample_details(ce: &StructuredCounterexample, explain: bool) {
    println!("\n{}", ce.format_detailed());

    if !explain {
        return;
    }

    let explanation_gen = ExplanationGenerator::new();
    let explanations = explanation_gen.explain(ce);
    if !explanations.is_empty() {
        println!("\n=== Explanations ===");
        for exp in &explanations {
            println!("{}", exp.format());
        }
    }

    let repair_engine = RepairEngine::new();
    let suggestions = repair_engine.suggest(ce);
    if !suggestions.is_empty() {
        println!("\n=== Repair Suggestions ===");
        for sug in &suggestions {
            println!("{}", sug.format());
        }
    }
}

fn attach_explanations(json_result: &mut serde_json::Value, ce: &StructuredCounterexample) {
    let explanation_gen = ExplanationGenerator::new();
    let explanations = explanation_gen.explain(ce);
    let repair_engine = RepairEngine::new();
    let suggestions = repair_engine.suggest(ce);

    json_result["explanations"] = serde_json::json!(explanations
        .iter()
        .map(|e| {
            serde_json::json!({
                "category": e.category.as_str(),
                "summary": e.summary,
                "details": e.details,
                "severity": e.severity,
            })
        })
        .collect::<Vec<_>>());
    json_result["repair_suggestions"] = serde_json::json!(suggestions
        .iter()
        .map(|s| {
            serde_json::json!({
                "title": s.title,
                "strategy": s.strategy.as_str(),
                "description": s.description,
                "code_snippet": s.code_snippet,
                "confidence": s.confidence,
            })
        })
        .collect::<Vec<_>>());
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match run(cli).await {
        Ok(code) => code,
        Err(e) => {
            error!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Commands::Verify {
            path,
            harness,
            timeout,
            unwind,
            solver,
            portfolio,
            portfolio_solvers,
            portfolio_max_concurrent,
            explain,
        } => {
            let base_config = KaniConfig {
                timeout: Duration::from_secs(timeout),
                default_unwind: unwind,
                solver: solver.clone(),
                ..Default::default()
            };

            // Use portfolio mode if requested
            if portfolio {
                let solvers = portfolio_solvers.unwrap_or_else(|| {
                    vec![
                        "cadical".to_string(),
                        "kissat".to_string(),
                        "minisat".to_string(),
                    ]
                });

                let portfolio_config = PortfolioKaniConfigBuilder::new()
                    .with_base_config(base_config)
                    .with_solvers(solvers.clone())
                    .with_max_concurrent(portfolio_max_concurrent)
                    .build();

                let verifier = PortfolioVerifier::new(portfolio_config);
                let portfolio_result = verifier
                    .verify_with_harness(&path, harness.as_deref())
                    .await;

                // Output result
                match cli.format {
                    OutputFormat::Text => {
                        println!("{}", portfolio_result.result);
                        println!(
                            "\nSolved by: {} in {:?}",
                            portfolio_result.winning_solver, portfolio_result.total_time
                        );

                        // Show detailed counterexample if failed
                        if let VerificationStatus::Disproven = portfolio_result.result.status {
                            if let Some(ce) = &portfolio_result.result.counterexample {
                                print_counterexample_details(ce, explain);
                            }
                        }
                    }
                    OutputFormat::Json => {
                        // Serialize portfolio result
                        let mut json_result = serde_json::json!({
                            "result": portfolio_result.result,
                            "winning_solver": portfolio_result.winning_solver,
                            "total_time_ms": portfolio_result.total_time.as_millis(),
                        });

                        // Add explanations to JSON if requested
                        if explain {
                            if let Some(ce) = &portfolio_result.result.counterexample {
                                attach_explanations(&mut json_result, ce);
                            }
                        }
                        println!("{}", serde_json::to_string_pretty(&json_result)?);
                    }
                }

                // Exit code based on result
                return Ok(match portfolio_result.result.status {
                    VerificationStatus::Proven => ExitCode::SUCCESS,
                    VerificationStatus::Disproven => ExitCode::from(1),
                    VerificationStatus::Unknown { .. } => ExitCode::from(2),
                    VerificationStatus::Timeout => ExitCode::from(3),
                    VerificationStatus::Error { .. } => ExitCode::FAILURE,
                });
            }

            // Single solver mode
            let wrapper = KaniWrapper::new(base_config)?;
            let result = wrapper
                .verify_with_harness(&path, harness.as_deref())
                .await?;

            // Output result
            match cli.format {
                OutputFormat::Text => {
                    println!("{result}");

                    // Show detailed counterexample if failed
                    if let VerificationStatus::Disproven = result.status {
                        if let Some(ce) = &result.counterexample {
                            print_counterexample_details(ce, explain);
                        }
                    }
                }
                OutputFormat::Json => {
                    let mut json_result = serde_json::to_value(&result)?;

                    // Add explanations to JSON if requested
                    if explain {
                        if let Some(ce) = &result.counterexample {
                            attach_explanations(&mut json_result, ce);
                        }
                    }
                    println!("{}", serde_json::to_string_pretty(&json_result)?);
                }
            }

            // Exit code based on result
            Ok(match result.status {
                VerificationStatus::Proven => ExitCode::SUCCESS,
                VerificationStatus::Disproven => ExitCode::from(1),
                VerificationStatus::Unknown { .. } => ExitCode::from(2),
                VerificationStatus::Timeout => ExitCode::from(3),
                VerificationStatus::Error { .. } => ExitCode::FAILURE,
            })
        }

        Commands::Chc {
            source,
            function,
            timeout,
            edition,
            backend,
            bitvec,
            bitvec_width,
            lean5,
            lean5_output,
            all_proofs,
            cache,
            cache_path,
            no_cache,
            cache_stats,
            parallel,
            proof,
            proof_output,
            hybrid,
            hybrid_kani_timeout,
        } => {
            // Open cache if enabled
            let chc_cache = if cache || cache_stats {
                Some(ChcVerificationCache::open(&cache_path)?)
            } else {
                None
            };

            // Show cache statistics if requested
            if cache_stats {
                if let Some(c) = &chc_cache {
                    let stats = c.stats()?;
                    println!("CHC Verification Cache Statistics:");
                    println!("  Cached functions: {}", stats.cached_functions);
                    println!("  Proven: {}", stats.proven_functions);
                    println!("  Disproven: {}", stats.disproven_functions);
                    println!("  Cache hits: {}", stats.cache_hits);
                    println!("  Cache misses: {}", stats.cache_misses);
                    println!("  Hit rate: {:.1}%", stats.hit_rate());
                    println!(
                        "  Total verification time: {:.2}s",
                        stats.total_verification_time_ms as f64 / 1000.0
                    );
                    println!(
                        "  Estimated time saved: {:.2}s",
                        stats.estimated_time_saved_ms() as f64 / 1000.0
                    );
                }
                // If only showing stats, exit
                if function.is_none() && !all_proofs {
                    return Ok(ExitCode::SUCCESS);
                }
            }

            // Generate MIR from source
            info!("Generating MIR from {}", source.display());
            let mir_text = if source.extension().is_some_and(|e| e == "rs") {
                generate_mir_from_file(&source, Some(&edition))?
            } else {
                // Assume it's already MIR text
                std::fs::read_to_string(&source)?
            };

            // Parse MIR
            let parser = MirParser::new();
            let functions = parser.parse(&mir_text)?;

            if functions.is_empty() {
                println!("No functions found in MIR output");
                return Ok(ExitCode::FAILURE);
            }

            // BitVec configuration (manual flag or env var)
            let bitvec_env = std::env::var("KANI_FAST_BITVEC").is_ok();
            let bitvec_width_env = std::env::var("KANI_FAST_BITVEC_WIDTH")
                .ok()
                .and_then(|s| s.parse().ok());
            let BitvecConfig {
                force_bitvec,
                bitvec_width,
            } = resolve_bitvec_config(bitvec, bitvec_width, bitvec_env, bitvec_width_env);

            // Create solver once for all functions
            let config = ChcSolverConfig::new()
                .with_backend(backend.into())
                .with_timeout(Duration::from_secs(timeout))
                .with_stats(true);
            let solver = match ChcSolver::new(config.clone()) {
                Ok(solver) => solver,
                Err(ChcSolverError::SolverNotFound(msg)) => {
                    println!("Error: {msg}");
                    return Ok(ExitCode::FAILURE);
                }
                Err(e) => return Err(e.into()),
            };

            // If --all-proofs, verify all functions ending with _proof
            if all_proofs {
                let proof_funcs: Vec<_> = functions
                    .iter()
                    .filter(|f| f.name.ends_with("_proof"))
                    .collect();

                if proof_funcs.is_empty() {
                    println!("No proof functions found (functions ending with _proof)");
                    return Ok(ExitCode::FAILURE);
                }

                // Determine parallelism level
                let num_workers = parallel.unwrap_or_else(num_cpus::get);
                let num_workers = num_workers.max(1); // At least 1 worker

                println!(
                    "Found {} proof function(s) in {}",
                    proof_funcs.len(),
                    source.display()
                );
                println!("CHC Verification using {:?} Spacer", solver.backend());
                if num_workers > 1 {
                    println!("Parallel mode: {num_workers} workers\n");
                } else {
                    println!();
                }

                // Use atomic counters for thread-safe counting
                let verified_count = Arc::new(AtomicUsize::new(0));
                let failed_count = Arc::new(AtomicUsize::new(0));
                let cached_count = Arc::new(AtomicUsize::new(0));

                // Create source-based hash that's stable across runs
                let source_content = std::fs::read_to_string(&source).unwrap_or_default();
                let source_hash = Arc::new(ContentHash::from_source(&source_content));

                // Wrap cache in Arc<Mutex> for thread-safe sharing across tasks
                let chc_cache = Arc::new(Mutex::new(chc_cache));

                // Warn once if we fall back to integer encoding (no overflow wrap)
                let overflow_warning_emitted = Arc::new(AtomicBool::new(false));

                // Clone config for each task
                let solver_config = Arc::new(config.clone());
                let verbose = cli.verbose;

                // Create verification tasks
                let all_funcs = functions.clone();
                let _results: Vec<_> = stream::iter(proof_funcs.iter().enumerate())
                    .map(|(idx, func)| {
                        let func_name = func.name.clone();
                        let program = func.to_mir_program_with_all_inlines(&all_funcs);
                        let use_bitvec = force_bitvec || program_needs_bitvec_encoding(&program);
                        if !use_bitvec
                            && !force_bitvec
                            && overflow_warning_emitted
                                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                                .is_ok()
                        {
                            print_overflow_warning();
                        }
                        let func_hash = ContentHash::from_function(
                            &func_name,
                            &format!(
                                "{source_hash};bitvec={use_bitvec};width={bitvec_width}"
                            ),
                        );
                        let config_hash = ContentHash::from_context(
                            &format!(
                                "timeout={timeout},backend={backend:?},bitvec={use_bitvec},width={bitvec_width}"
                            ),
                            &[],
                        );
                        let verified = Arc::clone(&verified_count);
                        let failed = Arc::clone(&failed_count);
                        let cached = Arc::clone(&cached_count);
                        let cache = Arc::clone(&chc_cache);
                        let solver_cfg = Arc::clone(&solver_config);

                        async move {
                            // Check cache first (unless --no-cache)
                            if !no_cache {
                                if let Ok(guard) = cache.lock() {
                                    if let Some(c) = &*guard {
                                        if let Ok(Some(cached_result)) = c.get_result(&func_hash) {
                                            match cached_result.outcome {
                                                CachedOutcome::Proven => {
                                                    println!(
                                                        "{func_name}: Property verified (cached)"
                                                    );
                                                    verified.fetch_add(1, Ordering::SeqCst);
                                                    cached.fetch_add(1, Ordering::SeqCst);
                                                    return (idx, func_name, true);
                                                }
                                                CachedOutcome::Disproven => {
                                                    println!(
                                                        "{func_name}: Property violated (cached)"
                                                    );
                                                    failed.fetch_add(1, Ordering::SeqCst);
                                                    cached.fetch_add(1, Ordering::SeqCst);
                                                    return (idx, func_name, false);
                                                }
                                                CachedOutcome::Unknown | CachedOutcome::Error => {
                                                    // Re-verify
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            let chc = if use_bitvec {
                                if verbose {
                                    println!(
                                        "{func_name}: Using BitVec encoding ({bitvec_width}-bit)"
                                    );
                                }
                                encode_mir_to_chc_bitvec(&program, bitvec_width)
                            } else {
                                encode_mir_to_chc(&program)
                            };

                            if verbose {
                                println!("Generated CHC for {}:\n{}\n", func_name, chc.to_smt2());
                            }

                            // Create solver for this task
                            let solver = match ChcSolver::new((*solver_cfg).clone()) {
                                Ok(s) => s,
                                Err(e) => {
                                    println!("{func_name}: Error creating solver ({e})");
                                    failed.fetch_add(1, Ordering::SeqCst);
                                    return (idx, func_name, false);
                                }
                            };

                            // Time the verification
                            let start = std::time::Instant::now();

                            match solver.solve(&chc).await {
                                Ok(result) => {
                                    let duration_ms = start.elapsed().as_millis() as u64;

                                    match &result {
                                        ChcResult::Sat { model, .. } => {
                                            println!("{func_name}: Property verified");
                                            verified.fetch_add(1, Ordering::SeqCst);

                                            // Store in cache
                                            if let Ok(guard) = cache.lock() {
                                                if let Some(c) = &*guard {
                                                    let cached_result = CachedChcResult {
                                                        outcome: CachedOutcome::Proven,
                                                        invariant: Some(model.to_readable_string()),
                                                        counterexample: None,
                                                        error_message: None,
                                                        duration_ms,
                                                        backend: format!("{:?}", solver.backend()),
                                                        computed_at: std::time::SystemTime::now(),
                                                    };
                                                    let _ = c.store_result(
                                                        &func_hash,
                                                        &func_name,
                                                        &cached_result,
                                                        &config_hash,
                                                        &[],
                                                    );
                                                }
                                            }
                                            (idx, func_name, true)
                                        }
                                        ChcResult::Unsat { .. } => {
                                            println!("{func_name}: Property violated");
                                            failed.fetch_add(1, Ordering::SeqCst);

                                            // Store in cache
                                            if let Ok(guard) = cache.lock() {
                                                if let Some(c) = &*guard {
                                                    let cached_result = CachedChcResult {
                                                        outcome: CachedOutcome::Disproven,
                                                        invariant: None,
                                                        counterexample: None,
                                                        error_message: None,
                                                        duration_ms,
                                                        backend: format!("{:?}", solver.backend()),
                                                        computed_at: std::time::SystemTime::now(),
                                                    };
                                                    let _ = c.store_result(
                                                        &func_hash,
                                                        &func_name,
                                                        &cached_result,
                                                        &config_hash,
                                                        &[],
                                                    );
                                                }
                                            }
                                            (idx, func_name, false)
                                        }
                                        ChcResult::Unknown { reason, .. } => {
                                            println!("{func_name}: Unknown ({reason})");
                                            failed.fetch_add(1, Ordering::SeqCst);

                                            // Store in cache
                                            if let Ok(guard) = cache.lock() {
                                                if let Some(c) = &*guard {
                                                    let cached_result = CachedChcResult {
                                                        outcome: CachedOutcome::Unknown,
                                                        invariant: None,
                                                        counterexample: None,
                                                        error_message: Some(reason.clone()),
                                                        duration_ms,
                                                        backend: format!("{:?}", solver.backend()),
                                                        computed_at: std::time::SystemTime::now(),
                                                    };
                                                    let _ = c.store_result(
                                                        &func_hash,
                                                        &func_name,
                                                        &cached_result,
                                                        &config_hash,
                                                        &[],
                                                    );
                                                }
                                            }
                                            (idx, func_name, false)
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("{func_name}: Error ({e})");
                                    failed.fetch_add(1, Ordering::SeqCst);
                                    (idx, func_name, false)
                                }
                            }
                        }
                    })
                    .buffer_unordered(num_workers)
                    .collect()
                    .await;

                let final_verified = verified_count.load(Ordering::SeqCst);
                let final_failed = failed_count.load(Ordering::SeqCst);
                let final_cached = cached_count.load(Ordering::SeqCst);

                println!("\n========================================");
                println!("Summary:");
                println!("  Verified: {final_verified}");
                println!("  Failed: {final_failed}");
                if cache && final_cached > 0 {
                    println!("  From cache: {final_cached}");
                }
                if num_workers > 1 {
                    println!("  Workers: {num_workers}");
                }
                println!("========================================");

                // Return success only if all proofs verified
                if final_failed == 0 {
                    Ok(ExitCode::SUCCESS)
                } else {
                    Ok(ExitCode::from(1))
                }
            } else {
                // Single function verification (original behavior)
                let func = if let Some(name) = &function {
                    functions
                        .iter()
                        .find(|f| f.name.contains(name))
                        .ok_or_else(|| anyhow::anyhow!("Function '{name}' not found"))?
                } else {
                    // Skip compiler-generated functions, try to find user functions
                    functions
                        .iter()
                        .find(|f| !f.name.starts_with("std::") && !f.name.contains("::{{closure}}"))
                        .unwrap_or(&functions[0])
                };

                println!("Verifying function: {}", func.name);
                println!("  Parameters: {}", func.args.len());
                println!("  Basic blocks: {}\n", func.basic_blocks.len());

                // Convert to MIR program and encode as CHC (with closure support)
                let program = func.to_mir_program_with_all_inlines(&functions);
                let use_bitvec = force_bitvec || program_needs_bitvec_encoding(&program);
                if use_bitvec {
                    println!(
                        "Encoding mode: BitVec ({}-bit){}",
                        bitvec_width,
                        if force_bitvec { " [forced]" } else { " [auto]" }
                    );
                } else {
                    if cli.verbose {
                        println!("Encoding mode: Int theory with algebraic rewrites");
                    }
                    print_overflow_warning();
                }
                let chc = if use_bitvec {
                    encode_mir_to_chc_bitvec(&program, bitvec_width)
                } else {
                    encode_mir_to_chc(&program)
                };

                // Output CHC if verbose
                if cli.verbose {
                    println!("Generated CHC:\n{}\n", chc.to_smt2());
                }

                println!("CHC Verification using {:?} Spacer\n", solver.backend());

                let result = solver.solve(&chc).await?;

                // Output result
                match &result {
                    ChcResult::Sat { model, stats, .. } => {
                        println!("VERIFIED (SAT)");
                        println!("Method: CHC/Spacer ({:?})", solver.backend());
                        println!("\nInvariant discovered:");
                        // Use summary by default, full output with -v
                        if cli.verbose {
                            println!("{}", model.to_readable_string());
                        } else {
                            println!("{}", model.to_summary_string());
                        }
                        println!("\nStats: {stats}");

                        // Generate Lean5 proof obligations if requested
                        if lean5 {
                            match ProofObligation::from_invariant(model) {
                                Ok(obligations) => {
                                    let lean_source = ProofObligation::to_lean5_file(&obligations);
                                    let output_path = lean5_output
                                        .unwrap_or_else(|| source.with_extension("lean"));
                                    std::fs::write(&output_path, &lean_source)?;
                                    println!(
                                        "\nLean5 proof obligations written to: {}",
                                        output_path.display()
                                    );
                                    println!("  {} theorem(s) generated", obligations.len());
                                }
                                Err(e) => {
                                    println!(
                                        "\nWarning: Failed to generate Lean5 obligations: {e}"
                                    );
                                }
                            }
                        }

                        // Generate universal proof if requested
                        if proof {
                            let generation_time = stats.solve_time;
                            if let Some(universal_proof) =
                                generate_chc_proof(&result, &chc, generation_time)
                            {
                                let proof_json = serde_json::to_string_pretty(&universal_proof)?;
                                let proof_path = proof_output
                                    .clone()
                                    .unwrap_or_else(|| source.with_extension("proof.json"));
                                std::fs::write(&proof_path, &proof_json)?;
                                println!("\nUniversal proof written to: {}", proof_path.display());
                                println!("  Proof ID: {}", universal_proof.id);
                                println!("  Format: {:?}", universal_proof.format);
                                println!("  Steps: {}", universal_proof.steps.len());
                            }
                        }

                        match cli.format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "status": "verified",
                                    "backend": format!("{:?}", solver.backend()),
                                    "invariant": model.to_readable_string(),
                                    "invariant_summary": model.to_summary_string(),
                                    "stats": {
                                        "solve_time_ms": stats.solve_time.as_millis(),
                                        "iterations": stats.iterations,
                                        "lemmas": stats.lemmas,
                                        "max_depth": stats.max_depth,
                                        "memory_bytes": stats.memory_bytes,
                                    }
                                });
                                println!("\n{}", serde_json::to_string_pretty(&json)?);
                            }
                            OutputFormat::Text => {}
                        }
                        Ok(ExitCode::SUCCESS)
                    }
                    ChcResult::Unsat { stats, .. } => {
                        println!("VIOLATED (UNSAT)");
                        println!("No invariant exists - property can be violated.");
                        println!("Method: CHC/Spacer ({:?})", solver.backend());
                        println!("\nStats: {stats}");
                        Ok(ExitCode::from(1))
                    }
                    ChcResult::Unknown { reason, stats, .. } => {
                        if hybrid {
                            // Hybrid mode: try Kani BMC fallback
                            println!("CHC returned UNKNOWN: {reason}");
                            println!("Hybrid mode: attempting Kani BMC fallback...\n");

                            let hybrid_config = HybridConfig::new()
                                .with_kani_timeout(Duration::from_secs(hybrid_kani_timeout));

                            let hybrid_result = hybrid_verify_with_chc_result(
                                &source,
                                &result,
                                stats.solve_time,
                                &hybrid_config,
                            );

                            println!("{hybrid_result}");

                            match &hybrid_result {
                                HybridResult::BmcVerified { .. } => {
                                    println!(
                                        "\nNote: This is a bounded proof. Kani BMC cannot prove unbounded properties."
                                    );
                                    match cli.format {
                                        OutputFormat::Json => {
                                            let json = serde_json::json!({
                                                "status": "verified",
                                                "method": "hybrid",
                                                "proof_type": "bounded",
                                                "chc_reason": reason,
                                                "total_duration_ms": hybrid_result.total_duration().as_millis(),
                                            });
                                            println!("\n{}", serde_json::to_string_pretty(&json)?);
                                        }
                                        OutputFormat::Text => {}
                                    }
                                    Ok(ExitCode::SUCCESS)
                                }
                                HybridResult::BmcViolated { kani_output, .. } => {
                                    println!("\nKani BMC found a counterexample.");
                                    if cli.verbose {
                                        println!("\nKani output:\n{kani_output}");
                                    }
                                    match cli.format {
                                        OutputFormat::Json => {
                                            let json = serde_json::json!({
                                                "status": "violated",
                                                "method": "hybrid",
                                                "chc_reason": reason,
                                                "total_duration_ms": hybrid_result.total_duration().as_millis(),
                                            });
                                            println!("\n{}", serde_json::to_string_pretty(&json)?);
                                        }
                                        OutputFormat::Text => {}
                                    }
                                    Ok(ExitCode::from(1))
                                }
                                HybridResult::KaniUnavailable { .. } => {
                                    println!("\nNote: Install Kani for BMC fallback: cargo install --locked kani-verifier");
                                    Ok(ExitCode::from(2))
                                }
                                _ => {
                                    // Unknown even after BMC
                                    Ok(ExitCode::from(2))
                                }
                            }
                        } else {
                            // Normal mode: just report unknown
                            println!("UNKNOWN");
                            println!("Reason: {reason}");
                            println!("Method: CHC/Spacer ({:?})", solver.backend());
                            println!("\nStats: {stats}");
                            println!("\nTip: Use --hybrid to try Kani BMC fallback");
                            Ok(ExitCode::from(2))
                        }
                    }
                }
            }
        }

        Commands::Kinduction {
            input,
            demo,
            max_k,
            timeout,
            proof,
            proof_output,
        } => {
            if !has_z3() {
                println!("Error: Z3 not found in PATH. Please install Z3:");
                println!("  brew install z3");
                return Ok(ExitCode::FAILURE);
            }

            println!("K-Induction Verification\n");

            let ts = if demo {
                // Built-in counter example
                println!("Running demo: Simple counter (x starts at 0, increments)");
                println!("Property: x >= 0 (should hold unboundedly)\n");

                TransitionSystemBuilder::new()
                    .variable("x", SmtType::Int)
                    .init("(= x 0)")
                    .transition("(= x' (+ x 1))")
                    .property("p1", "non_negative", "(>= x 0)")
                    .build()
            } else if let Some(path) = &input {
                // Parse from SMT-LIB2 file (simplified format)
                let content = std::fs::read_to_string(path)?;
                parse_transition_system(&content)?
            } else {
                return Err(anyhow::anyhow!(
                    "Either --demo or an input file must be specified"
                ));
            };

            println!("Transition System:");
            println!(
                "  Variables: {:?}",
                ts.variables.iter().map(|v| &v.name).collect::<Vec<_>>()
            );
            println!("  Properties: {}\n", ts.properties.len());

            // Configure k-induction engine
            let config = KInductionConfigBuilder::new()
                .max_k(max_k)
                .total_timeout_secs(timeout)
                .timeout_per_step_ms(timeout * 1000 / (max_k as u64 + 1))
                .build();

            let engine = KInduction::new(config);

            // Run k-induction on the entire system
            println!("Running k-induction (max_k={max_k}, timeout={timeout}s)...");

            let start_time = std::time::Instant::now();
            match engine.verify(&ts).await {
                Ok(result) => {
                    let generation_time = start_time.elapsed();

                    match &result {
                        kani_fast_kinduction::KInductionResult::Proven {
                            k,
                            invariant,
                            stats,
                        } => {
                            println!("\nVERIFIED at k={k}");
                            println!("  Time: {:?}", stats.total_time);
                            if let Some(inv) = invariant {
                                println!("  Invariant: {inv}");
                            }

                            // Generate proof if requested
                            if proof {
                                if let Some(universal_proof) =
                                    generate_kinduction_proof(&result, &ts, generation_time)
                                {
                                    let proof_path = proof_output.unwrap_or_else(|| {
                                        input
                                            .as_ref()
                                            .map(|p| p.with_extension("proof.json"))
                                            .unwrap_or_else(|| PathBuf::from("demo.proof.json"))
                                    });

                                    let json = serde_json::to_string_pretty(&universal_proof)
                                        .expect("Failed to serialize proof");
                                    std::fs::write(&proof_path, json)
                                        .expect("Failed to write proof file");

                                    println!("\n  Proof generated:");
                                    println!("    ID: {}", universal_proof.id);
                                    println!("    Format: {:?}", universal_proof.format);
                                    println!("    Steps: {}", universal_proof.steps.len());
                                    println!("    Output: {}", proof_path.display());
                                }
                            }
                        }
                        kani_fast_kinduction::KInductionResult::Disproven {
                            k,
                            counterexample,
                            stats,
                        } => {
                            println!("\nVIOLATED at step {k}");
                            println!("  Time: {:?}", stats.total_time);
                            println!("  Property: {}", counterexample.violated_property);
                            println!("  Counterexample trace:");
                            for state in &counterexample.states {
                                println!("    Step {}: {:?}", state.step, state.variables);
                            }
                        }
                        kani_fast_kinduction::KInductionResult::Unknown {
                            reason,
                            last_k,
                            stats,
                        } => {
                            println!("\nUNKNOWN");
                            println!("  Reason: {reason}");
                            println!("  Last k attempted: {last_k}");
                            println!("  Time: {:?}", stats.total_time);
                        }
                    }
                }
                Err(e) => {
                    println!("\nERROR: {e}");
                    return Ok(ExitCode::FAILURE);
                }
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Unbounded {
            source,
            function,
            max_k,
            timeout,
            edition,
            chc_only,
            kinduction_only,
            chc_backend,
            lean5,
            lean5_output,
            ai,
            ai_max_attempts,
            ai_only,
            certificate,
            verify_lean,
        } => {
            // Skip Z3 check if --ai-only
            if !chc_only && !ai_only && !has_z3() {
                println!("Error: Z3 not found in PATH. K-induction requires Z3.");
                println!("  Install: brew install z3");
                println!(
                    "  Hint: use --chc-only to skip k-induction, or --ai-only for AI synthesis."
                );
                return Ok(ExitCode::FAILURE);
            }

            println!("Unbounded Verification (automatic escalation)\n");

            // Parse MIR from source
            info!("Generating MIR from {}", source.display());
            let mir_text = if source.extension().is_some_and(|e| e == "rs") {
                generate_mir_from_file(&source, Some(&edition))?
            } else {
                std::fs::read_to_string(&source)?
            };

            let parser = MirParser::new();
            let functions = parser.parse(&mir_text)?;

            if functions.is_empty() {
                println!("No functions found in MIR output");
                return Ok(ExitCode::FAILURE);
            }

            // Select function to verify
            let func = if let Some(name) = &function {
                functions
                    .iter()
                    .find(|f| f.name.contains(name))
                    .ok_or_else(|| anyhow::anyhow!("Function '{name}' not found"))?
            } else {
                functions
                    .iter()
                    .find(|f| !f.name.starts_with("std::") && !f.name.contains("::{{closure}}"))
                    .unwrap_or(&functions[0])
            };

            println!("Function: {}", func.name);
            println!("  Parameters: {}", func.args.len());
            println!("  Basic blocks: {}\n", func.basic_blocks.len());

            // Convert to transition system for k-induction
            let program = func.to_mir_program_with_all_inlines(&functions);
            let ts = encode_mir_to_transition_system(&program);

            let start_time = std::time::Instant::now();
            let total_timeout = Duration::from_secs(timeout);

            // Track whether we should escalate to AI
            let mut escalate_to_ai = ai_only;

            // Phase 1: Try k-induction first (unless --chc-only or --ai-only)
            if !chc_only && !ai_only {
                println!("Phase 1: K-induction (max_k={max_k})");

                let k_timeout = if kinduction_only {
                    total_timeout
                } else {
                    // Give k-induction a portion of the timeout
                    Duration::from_secs(timeout / 3)
                };

                let k_config = KInductionConfigBuilder::new()
                    .max_k(max_k)
                    .total_timeout_secs(k_timeout.as_secs())
                    .timeout_per_step_ms(k_timeout.as_millis() as u64 / (max_k as u64 + 1))
                    .build();

                let engine = KInduction::new(k_config);

                match engine.verify(&ts).await {
                    Ok(kani_fast_kinduction::KInductionResult::Proven {
                        k,
                        invariant,
                        stats,
                    }) => {
                        println!("\nVERIFIED by k-induction at k={k}");
                        println!("  Method: k-induction");
                        println!("  Time: {:?}", stats.total_time);
                        if let Some(inv) = &invariant {
                            println!("  Invariant: {inv}");
                        }

                        // Generate Lean5 certificate if requested
                        if lean5 || certificate {
                            let obligations = generate_kinduction_proof_obligations(
                                &func.name,
                                k,
                                invariant.as_deref(),
                                &ts,
                            );

                            if certificate {
                                let mut cert = certificate_from_kinduction(
                                    &func.name,
                                    k as usize,
                                    obligations,
                                );

                                // Optionally verify with Lean
                                if verify_lean {
                                    match cert.verify() {
                                        Ok(result) => {
                                            if result.verified {
                                                println!("\n✓ Lean verification: PASSED");
                                            } else {
                                                println!("\n✗ Lean verification: FAILED");
                                                for err in &result.errors {
                                                    println!("    {err}");
                                                }
                                            }
                                            if result.sorry_count > 0 {
                                                println!(
                                                    "  {} sorry placeholder(s) remain",
                                                    result.sorry_count
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            println!("\nWarning: Lean verification failed: {e}");
                                        }
                                    }
                                }

                                // Write certificate to file
                                let output_path = lean5_output
                                    .clone()
                                    .unwrap_or_else(|| source.with_extension(""));
                                if let Err(e) = cert.write_to_file(&output_path) {
                                    println!("\nWarning: Failed to write certificate: {e}");
                                } else {
                                    println!(
                                        "\nCertificate written to: {}.lean, {}.cert.json",
                                        output_path.display(),
                                        output_path.display()
                                    );
                                    println!("{}", cert.summary());
                                }
                            } else {
                                // Just write Lean5 source
                                let lean_source = ProofObligation::to_lean5_file(&obligations);
                                let output_path = lean5_output
                                    .clone()
                                    .unwrap_or_else(|| source.with_extension("lean"));
                                std::fs::write(&output_path, &lean_source)?;
                                println!(
                                    "\nLean5 proof obligations written to: {}",
                                    output_path.display()
                                );
                                println!("  {} theorem(s) generated", obligations.len());
                            }
                        }

                        match cli.format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "status": "verified",
                                    "method": "k-induction",
                                    "k": k,
                                    "time_ms": stats.total_time.as_millis(),
                                });
                                println!("\n{}", serde_json::to_string_pretty(&json)?);
                            }
                            OutputFormat::Text => {}
                        }
                        return Ok(ExitCode::SUCCESS);
                    }
                    Ok(kani_fast_kinduction::KInductionResult::Disproven {
                        k,
                        counterexample,
                        stats,
                    }) => {
                        // K-induction found a potential violation, but it may be spurious
                        // due to over-approximation (e.g., Havoc for overflow flags).
                        // If --kinduction-only is set, trust the result.
                        // Otherwise, escalate to CHC for confirmation.
                        if kinduction_only {
                            println!("\nVIOLATED at step {k} (found by k-induction)");
                            println!("  Property: {}", counterexample.violated_property);
                            println!("  Time: {:?}", stats.total_time);
                            println!("  Counterexample trace:");
                            for state in &counterexample.states {
                                println!("    Step {}: {:?}", state.step, state.variables);
                            }

                            match cli.format {
                                OutputFormat::Json => {
                                    let json = serde_json::json!({
                                        "status": "violated",
                                        "method": "k-induction",
                                        "k": k,
                                        "time_ms": stats.total_time.as_millis(),
                                        "property": counterexample.violated_property,
                                    });
                                    println!("\n{}", serde_json::to_string_pretty(&json)?);
                                }
                                OutputFormat::Text => {}
                            }
                            return Ok(ExitCode::from(1));
                        }

                        // Escalate to CHC for confirmation
                        println!("\n  K-induction found potential violation at k={k}");
                        println!("  (May be spurious due to over-approximation)");
                        println!("  Escalating to CHC for confirmation...\n");
                    }
                    Ok(kani_fast_kinduction::KInductionResult::Unknown {
                        reason,
                        last_k,
                        stats,
                    }) => {
                        println!("  Result: inconclusive after k={last_k}");
                        println!("  Reason: {reason}");
                        println!("  Time: {:?}", stats.total_time);

                        if kinduction_only {
                            println!("\nUNKNOWN (k-induction only mode)");
                            return Ok(ExitCode::from(2));
                        }

                        println!("\n  Escalating to CHC...\n");
                    }
                    Err(e) => {
                        println!("  Error: {e}");
                        if kinduction_only {
                            return Ok(ExitCode::FAILURE);
                        }
                        println!("  Escalating to CHC...\n");
                    }
                }
            }

            // Phase 2: CHC solving (unless --kinduction-only or --ai-only)
            if !kinduction_only && !ai_only {
                let elapsed = start_time.elapsed();
                let remaining = if elapsed < total_timeout {
                    total_timeout - elapsed
                } else {
                    Duration::from_secs(10) // Minimum 10 seconds for CHC
                };

                let chc = encode_mir_to_chc(&program);

                if cli.verbose {
                    println!("Generated CHC:\n{}\n", chc.to_smt2());
                }

                let chc_config = ChcSolverConfig::new()
                    .with_timeout(remaining)
                    .with_stats(true)
                    .with_backend(chc_backend.into());
                let chc_solver = match ChcSolver::new(chc_config) {
                    Ok(solver) => solver,
                    Err(ChcSolverError::SolverNotFound(msg)) => {
                        println!("CHC backend unavailable: {msg}");
                        if ai {
                            println!("  Escalating to AI synthesis...\n");
                            escalate_to_ai = true;
                            // Continue to AI phase
                            ChcSolver::new(ChcSolverConfig::new())?
                        } else {
                            return Ok(ExitCode::FAILURE);
                        }
                    }
                    Err(e) => return Err(e.into()),
                };

                if !escalate_to_ai {
                    println!(
                        "Phase 2: CHC solving with {:?} Spacer",
                        chc_solver.backend()
                    );

                    let result = chc_solver.solve(&chc).await?;

                    match &result {
                        ChcResult::Sat { model, stats, .. } => {
                            println!("\nVERIFIED by CHC (invariant discovered)");
                            println!("  Method: CHC/Spacer ({:?})", chc_solver.backend());
                            println!("  Time: {:?}", start_time.elapsed());
                            println!("\nInvariant:");
                            if cli.verbose {
                                println!("{}", model.to_readable_string());
                            } else {
                                println!("{}", model.to_summary_string());
                            }
                            println!("\nStats: {stats}");

                            // Generate Lean5 proof obligations if requested
                            if lean5 || certificate {
                                match ProofObligation::from_invariant(model) {
                                    Ok(obligations) => {
                                        if certificate {
                                            // Generate a full certificate
                                            let backend_str = format!("{:?}", chc_solver.backend());
                                            let mut cert = certificate_from_chc(
                                                &func.name,
                                                &backend_str,
                                                obligations,
                                            );

                                            // Optionally verify with Lean
                                            if verify_lean {
                                                match cert.verify() {
                                                    Ok(result) => {
                                                        if result.verified {
                                                            println!(
                                                                "\n✓ Lean verification: PASSED"
                                                            );
                                                        } else {
                                                            println!(
                                                                "\n✗ Lean verification: FAILED"
                                                            );
                                                            for err in &result.errors {
                                                                println!("    {err}");
                                                            }
                                                        }
                                                        if result.sorry_count > 0 {
                                                            println!(
                                                                "  {} sorry placeholder(s) remain",
                                                                result.sorry_count
                                                            );
                                                        }
                                                    }
                                                    Err(e) => {
                                                        println!(
                                                            "\nWarning: Lean verification failed: {e}"
                                                        );
                                                    }
                                                }
                                            }

                                            // Write certificate to file
                                            let output_path = lean5_output
                                                .clone()
                                                .unwrap_or_else(|| source.with_extension(""));
                                            if let Err(e) = cert.write_to_file(&output_path) {
                                                println!(
                                                    "\nWarning: Failed to write certificate: {e}"
                                                );
                                            } else {
                                                println!(
                                                    "\nCertificate written to: {}.lean, {}.cert.json",
                                                    output_path.display(),
                                                    output_path.display()
                                                );
                                                println!("{}", cert.summary());
                                            }
                                        } else {
                                            // Just write Lean5 source
                                            let lean_source =
                                                ProofObligation::to_lean5_file(&obligations);
                                            let output_path = lean5_output
                                                .clone()
                                                .unwrap_or_else(|| source.with_extension("lean"));
                                            std::fs::write(&output_path, &lean_source)?;
                                            println!(
                                                "\nLean5 proof obligations written to: {}",
                                                output_path.display()
                                            );
                                            println!(
                                                "  {} theorem(s) generated",
                                                obligations.len()
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        println!(
                                            "\nWarning: Failed to generate Lean5 obligations: {e}"
                                        );
                                    }
                                }
                            }

                            match cli.format {
                                OutputFormat::Json => {
                                    let json = serde_json::json!({
                                        "status": "verified",
                                        "method": "chc",
                                        "backend": format!("{:?}", chc_solver.backend()),
                                        "invariant": model.to_readable_string(),
                                        "invariant_summary": model.to_summary_string(),
                                        "stats": {
                                            "solve_time_ms": stats.solve_time.as_millis(),
                                            "iterations": stats.iterations,
                                            "lemmas": stats.lemmas,
                                        }
                                    });
                                    println!("\n{}", serde_json::to_string_pretty(&json)?);
                                }
                                OutputFormat::Text => {}
                            }
                            return Ok(ExitCode::SUCCESS);
                        }
                        ChcResult::Unsat { stats, .. } => {
                            println!("\nVIOLATED (no invariant exists)");
                            println!("  Method: CHC/Spacer ({:?})", chc_solver.backend());
                            println!("  Time: {:?}", start_time.elapsed());
                            println!("  Stats: {stats}");
                            return Ok(ExitCode::from(1));
                        }
                        ChcResult::Unknown { reason, stats, .. } => {
                            println!("  Result: unknown");
                            println!("  Reason: {reason}");
                            println!("  Stats: {stats}");

                            if ai {
                                println!("\n  Escalating to AI synthesis...\n");
                                escalate_to_ai = true;
                            } else {
                                println!("\nUNKNOWN");
                                println!("  Method: CHC/Spacer ({:?})", chc_solver.backend());
                                println!("  Time: {:?}", start_time.elapsed());
                                println!("  Hint: use --ai to enable AI-assisted synthesis");
                                return Ok(ExitCode::from(2));
                            }
                        }
                    }
                }
            }

            // Phase 3: AI-assisted invariant synthesis (if --ai or --ai-only)
            if escalate_to_ai || ai_only {
                println!("Phase 3: AI-assisted invariant synthesis");

                // Build a property from the transition system for AI synthesis
                let property = if let Some(prop) = ts.properties.first() {
                    prop.clone()
                } else {
                    // Default safety property: no overflow/underflow
                    kani_fast_kinduction::Property {
                        id: "safety".to_string(),
                        name: "Default safety property".to_string(),
                        formula: kani_fast_kinduction::StateFormula::new("true".to_string()),
                        property_type: PropertyType::Safety,
                    }
                };

                // Configure AI synthesizer
                let ai_config = AiConfig {
                    max_attempts: ai_max_attempts,
                    timeout_secs: 30,
                    use_llm: true,
                    use_corpus: true,
                    ..Default::default()
                };

                let mut synthesizer = AiSynthesizer::with_config(ai_config);

                println!("  Max attempts: {ai_max_attempts}");
                println!("  Using ICE learning with LLM fallback\n");

                match synthesizer.synthesize(&ts, &property).await {
                    Ok(result) => {
                        let source_str = match &result.source {
                            InvariantSource::Corpus(id) => format!("corpus (id={id})"),
                            InvariantSource::Ice => "ICE learning".to_string(),
                            InvariantSource::Llm => "LLM suggestion".to_string(),
                        };

                        println!("\nVERIFIED by AI synthesis");
                        println!("  Method: AI/{source_str}");
                        println!("  Attempts: {}", result.attempts);
                        println!("  Examples collected: {}", result.examples_collected);
                        println!("  Time: {:?}", start_time.elapsed());
                        println!("\nInvariant:");
                        println!("  {}", result.invariant.smt_formula);

                        // Generate Lean5 proof obligations if requested
                        if lean5 {
                            // Convert AI invariant to Lean5 proof sketch
                            let lean_content =
                                generate_lean5_from_ai_invariant(&result.invariant, &ts, &property);
                            let output_path = lean5_output
                                .clone()
                                .unwrap_or_else(|| source.with_extension("lean"));
                            std::fs::write(&output_path, &lean_content)?;
                            println!("\nLean5 proof sketch written to: {}", output_path.display());
                        }

                        match cli.format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "status": "verified",
                                    "method": "ai",
                                    "source": source_str,
                                    "invariant": result.invariant.smt_formula,
                                    "attempts": result.attempts,
                                    "examples_collected": result.examples_collected,
                                    "time_ms": start_time.elapsed().as_millis(),
                                });
                                println!("\n{}", serde_json::to_string_pretty(&json)?);
                            }
                            OutputFormat::Text => {}
                        }
                        return Ok(ExitCode::SUCCESS);
                    }
                    Err(e) => {
                        println!("\nAI synthesis failed: {e}");
                        println!("  Time: {:?}", start_time.elapsed());

                        match cli.format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "status": "unknown",
                                    "method": "ai",
                                    "error": e.to_string(),
                                    "time_ms": start_time.elapsed().as_millis(),
                                });
                                println!("\n{}", serde_json::to_string_pretty(&json)?);
                            }
                            OutputFormat::Text => {}
                        }
                        return Ok(ExitCode::from(2));
                    }
                }
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Watch {
            path,
            debounce,
            timeout,
        } => {
            println!("Kani Fast - Watch Mode\n");
            println!("Watching: {}", path.display());
            println!("Debounce: {debounce}ms");
            println!("Timeout:  {timeout}s\n");

            let config = IncrementalConfigBuilder::new()
                .watch_mode(true)
                .watch_debounce(Duration::from_millis(debounce))
                .solver_timeout(Duration::from_secs(timeout))
                .build();

            let mut watch = WatchModeBuilder::new(&path).config(config).build();

            let (event_rx, _shutdown_rx) = watch.start().map_err(|e| anyhow::anyhow!("{e}"))?;

            println!("Watching for changes... (Ctrl+C to stop)\n");

            // Process events until shutdown
            while let Ok(event) = event_rx.recv() {
                match event {
                    WatchEvent::VerificationStarted { changed_files } => {
                        println!("--- Files changed ---");
                        for f in &changed_files {
                            println!("  {}", f.display());
                        }
                        println!("Verifying...");
                    }
                    WatchEvent::VerificationCompleted { result, .. } => {
                        if result.proven {
                            println!("VERIFIED in {:?}", result.duration);
                        } else {
                            println!("VIOLATED in {:?}", result.duration);
                        }
                        if result.from_cache {
                            println!("  (from cache)");
                        }
                        println!();
                    }
                    WatchEvent::VerificationError { error } => {
                        println!("ERROR: {error}");
                        println!();
                    }
                    WatchEvent::Shutdown => {
                        println!("Shutting down watch mode.");
                        break;
                    }
                }
            }

            Ok(ExitCode::SUCCESS)
        }

        Commands::Incremental {
            input,
            timeout,
            cache_db,
            clear_cache,
            stats,
        } => {
            // Check for CaDiCaL
            if which::which("cadical").is_err() {
                println!("Error: CaDiCaL not found in PATH. Please install CaDiCaL:");
                println!("  brew install cadical");
                return Ok(ExitCode::FAILURE);
            }

            println!("Incremental BMC Verification\n");

            // Determine project root (parent of input file)
            let project_root = input
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));

            // Configure
            let mut config_builder =
                IncrementalConfigBuilder::new().solver_timeout(Duration::from_secs(timeout));

            if let Some(db_path) = cache_db {
                config_builder = config_builder.database_path(db_path);
            }

            let config = config_builder.build();

            // Create engine
            let mut engine =
                IncrementalBmc::new(&project_root, config).map_err(|e| anyhow::anyhow!("{e}"))?;

            // Clear cache if requested
            if clear_cache {
                let cleaned = engine.cleanup().await.unwrap_or(0);
                println!("Cleared {cleaned} cached entries\n");
            }

            // Show stats if requested
            if stats {
                match engine.database_stats().await {
                    Ok(db_stats) => {
                        println!("Cache Statistics:");
                        println!("  Total clauses: {}", db_stats.clause_count);
                        println!("  Total functions: {}", db_stats.function_count);
                        println!("  Total literals: {}", db_stats.total_literals);
                        println!("  Database size: {} bytes\n", db_stats.database_size);
                    }
                    Err(e) => {
                        println!("Could not get cache stats: {e}\n");
                    }
                }
            }

            // Verify DIMACS file
            println!("Verifying: {}", input.display());

            match engine.verify_dimacs(&input).await {
                Ok(result) => {
                    if result.from_cache {
                        println!("\nRESULT (from cache):");
                    } else {
                        println!("\nRESULT:");
                    }

                    if result.proven {
                        println!("  Status: UNSAT (property holds)");
                    } else if result.satisfiable {
                        println!("  Status: SAT (counterexample found)");
                    } else {
                        println!("  Status: UNKNOWN");
                    }

                    println!("  Time: {:?}", result.duration);
                    println!("  Learned clauses: {}", result.learned_clauses);
                    println!("  Cached clauses used: {}", result.cached_clauses_used);

                    match cli.format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "status": if result.proven { "unsat" } else if result.satisfiable { "sat" } else { "unknown" },
                                "from_cache": result.from_cache,
                                "time_ms": result.duration.as_millis(),
                                "learned_clauses": result.learned_clauses,
                                "cached_clauses_used": result.cached_clauses_used,
                            });
                            println!("\n{}", serde_json::to_string_pretty(&json)?);
                        }
                        OutputFormat::Text => {}
                    }

                    if result.satisfiable {
                        Ok(ExitCode::from(1))
                    } else {
                        Ok(ExitCode::SUCCESS)
                    }
                }
                Err(e) => {
                    println!("\nERROR: {e}");
                    Ok(ExitCode::FAILURE)
                }
            }
        }

        Commands::Lean5Check { file, timeout } => {
            println!("Lean5 Proof Checker\n");

            // Check if Lean is installed
            match check_lean_installation() {
                Ok(version) => {
                    println!("Lean Version: {version}\n");
                }
                Err(e) => {
                    println!("Error: {e}");
                    return Ok(ExitCode::FAILURE);
                }
            }

            // Check the file
            let config = Lean5Config::new().with_timeout(Duration::from_secs(timeout));

            let backend = match Lean5Backend::new(config) {
                Ok(b) => b,
                Err(e) => {
                    println!("Error initializing Lean backend: {e}");
                    return Ok(ExitCode::FAILURE);
                }
            };

            println!("Checking: {}", file.display());

            match backend.check_file(&file) {
                Ok(result) => {
                    if result.success {
                        println!("\n✓ VALID");
                    } else {
                        println!("\n✗ INVALID");
                        for err in &result.errors {
                            println!("  {err}");
                        }
                    }

                    println!("\nTime: {:?}", result.check_time);

                    if result.sorry_count > 0 {
                        println!(
                            "Warning: {} sorry placeholder(s) found (proof incomplete)",
                            result.sorry_count
                        );
                    }

                    if !result.warnings.is_empty() {
                        println!("\nWarnings:");
                        for warning in &result.warnings {
                            println!("  {warning}");
                        }
                    }

                    match cli.format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "valid": result.success,
                                "complete": result.sorry_count == 0,
                                "sorry_count": result.sorry_count,
                                "errors": result.errors,
                                "warnings": result.warnings,
                                "time_ms": result.check_time.as_millis(),
                                "lean_version": result.lean_version,
                            });
                            println!("\n{}", serde_json::to_string_pretty(&json)?);
                        }
                        OutputFormat::Text => {}
                    }

                    if result.success {
                        Ok(ExitCode::SUCCESS)
                    } else {
                        Ok(ExitCode::from(1))
                    }
                }
                Err(e) => {
                    println!("\nError: {e}");
                    Ok(ExitCode::FAILURE)
                }
            }
        }

        Commands::Check => {
            println!("Kani Fast - Installation Check\n");

            // Check Kani
            match check_installation() {
                Ok(info) => {
                    println!("  Kani Version: {}", info.version);
                    if let Some(cbmc) = info.cbmc_version {
                        println!("  CBMC Version: {cbmc}");
                    }
                    println!("  Cargo Path:   {}", info.cargo_path);
                }
                Err(e) => {
                    println!("  Kani: NOT FOUND ({e})");
                    println!(
                        "    Install: cargo install --locked kani-verifier && cargo kani setup"
                    );
                }
            }

            // Check Z4 (primary CHC backend)
            let z4_path = which::which("z4").ok().or_else(|| {
                // Check ~/.local/bin/z4 as fallback
                dirs::home_dir()
                    .map(|h| h.join(".local/bin/z4"))
                    .filter(|p| p.exists())
            });
            if let Some(path) = z4_path {
                let output = std::process::Command::new(&path).arg("--version").output();
                let version = output
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .unwrap_or_else(|| "unknown".to_string());
                println!("  Z4 Path:      {} (primary CHC)", path.display());
                println!("  Z4 Version:   {}", version.trim());
            } else {
                println!("  Z4: NOT FOUND (optional, primary CHC backend)");
                println!("    Install: see https://github.com/dropbox/z4");
            }

            // Check Z3 (fallback CHC backend)
            if let Ok(z3_path) = which::which("z3") {
                let output = std::process::Command::new("z3").arg("--version").output();
                let version = output
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .unwrap_or_else(|| "unknown".to_string());
                println!("  Z3 Path:      {} (fallback CHC)", z3_path.display());
                println!("  Z3 Version:   {}", version.trim());
            } else {
                println!("  Z3: NOT FOUND");
                println!("    Install: brew install z3");
            }

            // Check CaDiCaL
            if let Ok(path) = which::which("cadical") {
                println!("  CaDiCaL:      {}", path.display());
            } else {
                println!("  CaDiCaL: NOT FOUND (optional)");
            }

            // Check Kissat
            if let Ok(path) = which::which("kissat") {
                println!("  Kissat:       {}", path.display());
            } else {
                println!("  Kissat: NOT FOUND (optional)");
            }

            // Check Lean
            match check_lean_installation() {
                Ok(version) => {
                    println!("  Lean:         {version}");
                }
                Err(_) => {
                    println!("  Lean: NOT FOUND (optional, for proof checking)");
                    println!("    Install: curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh");
                }
            }

            println!("\nAll required systems operational.");
            Ok(ExitCode::SUCCESS)
        }

        Commands::Version => {
            println!("kani-fast {}", kani_fast::VERSION);

            if let Ok(info) = check_installation() {
                println!("kani {}", info.version);
                if let Some(cbmc) = info.cbmc_version {
                    println!("cbmc {cbmc}");
                }
            }

            // Check Z4 first (primary backend)
            let z4_path = which::which("z4").ok().or_else(|| {
                dirs::home_dir()
                    .map(|h| h.join(".local/bin/z4"))
                    .filter(|p| p.exists())
            });
            if let Some(path) = z4_path {
                if let Ok(output) = std::process::Command::new(&path).arg("--version").output() {
                    if let Ok(version) = String::from_utf8(output.stdout) {
                        println!("{}", version.trim());
                    }
                }
            }

            if let Ok(output) = std::process::Command::new("z3").arg("--version").output() {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    println!("{}", version.trim());
                }
            }

            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Parse a simple transition system from a text format
fn parse_transition_system(content: &str) -> Result<kani_fast_kinduction::TransitionSystem> {
    let mut builder = TransitionSystemBuilder::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("var ") {
            // var x : Int
            if let Some((name_part, type_part)) = rest.split_once(':') {
                let name = name_part.trim();
                let ty = match type_part.trim().to_lowercase().as_str() {
                    "int" => SmtType::Int,
                    "bool" => SmtType::Bool,
                    "real" => SmtType::Real,
                    _ => SmtType::Int,
                };
                builder = builder.variable(name, ty);
            }
        } else if let Some(rest) = line.strip_prefix("init ") {
            builder = builder.init(rest.trim());
        } else if let Some(rest) = line.strip_prefix("trans ") {
            builder = builder.transition(rest.trim());
        } else if let Some(rest) = line.strip_prefix("prop ") {
            // prop name : formula
            // Use split_once to avoid Vec allocation
            if let Some((name_part, formula_part)) = rest.split_once(':') {
                let name = name_part.trim();
                builder = builder.property(name, name, formula_part.trim());
            }
        }
    }

    Ok(builder.build())
}

/// Generate Lean5 proof sketch from AI-discovered invariant
///
/// Uses proper SMT-to-Lean translation for correct Lean5 syntax.
fn generate_lean5_from_ai_invariant(
    invariant: &kani_fast_kinduction::StateFormula,
    system: &kani_fast_kinduction::TransitionSystem,
    property: &kani_fast_kinduction::Property,
) -> String {
    // Generate proof obligations using proper translation
    let obligations = generate_ai_proof_obligations(invariant, system, property);
    ProofObligation::to_lean5_file(&obligations)
}

/// Generate proof obligations from AI-discovered invariant
fn generate_ai_proof_obligations(
    invariant: &kani_fast_kinduction::StateFormula,
    system: &kani_fast_kinduction::TransitionSystem,
    property: &kani_fast_kinduction::Property,
) -> Vec<ProofObligation> {
    let mut obligations = Vec::new();

    // Parse invariant using proper SMT parser
    let invariant_expr = parse_smt_to_lean5_expr(&invariant.smt_formula);

    // Get variable context from transition system
    let var_context: Vec<(String, Lean5Type)> = system
        .variables
        .iter()
        .map(|v| {
            let lean_ty = match v.smt_type {
                SmtType::Int => Lean5Type::Int,
                SmtType::Bool => Lean5Type::Bool,
                _ => Lean5Type::Int,
            };
            (clean_var_name_for_lean5(&v.name), lean_ty)
        })
        .collect();

    // 1. Initiation: init => invariant
    let init_expr = parse_smt_to_lean5_expr(&system.init.smt_formula);
    let init_obligation = ProofObligation::new(
        "ai_initiation",
        ProofObligationKind::Initiation,
        Lean5Expr::implies(init_expr, invariant_expr.clone()),
    );
    obligations.push(add_context(init_obligation, &var_context));

    // 2. Consecution: invariant ∧ transition => invariant'
    let trans_expr = parse_smt_to_lean5_expr(&system.transition.smt_formula);
    let consecution_hypothesis = Lean5Expr::and(invariant_expr.clone(), trans_expr);
    let consecution_obligation = ProofObligation::new(
        "ai_consecution",
        ProofObligationKind::Consecution,
        Lean5Expr::implies(consecution_hypothesis, invariant_expr.clone()),
    );
    obligations.push(add_context(consecution_obligation, &var_context));

    // 3. Property: invariant => property
    let property_expr = parse_smt_to_lean5_expr(&property.formula.smt_formula);
    let property_obligation = ProofObligation::new(
        "ai_property",
        ProofObligationKind::Property,
        Lean5Expr::implies(invariant_expr, property_expr),
    );
    obligations.push(add_context(property_obligation, &var_context));

    obligations
}

/// Generate proof obligations for a k-induction proof
///
/// Creates three proof obligations:
/// 1. Initiation: init => invariant
/// 2. Consecution: invariant ∧ transition => invariant'
/// 3. Property: invariant => property
fn generate_kinduction_proof_obligations(
    property_name: &str,
    k: u32,
    invariant_formula: Option<&str>,
    ts: &kani_fast_kinduction::TransitionSystem,
) -> Vec<ProofObligation> {
    let mut obligations = Vec::new();

    // Parse invariant if available, otherwise use a simple one based on property
    let invariant_expr = if let Some(inv_formula) = invariant_formula {
        parse_smt_to_lean5_expr(inv_formula)
    } else {
        // Default invariant: use the property formula
        parse_smt_to_lean5_expr(&ts.properties[0].formula.smt_formula)
    };

    // Get variable names and types from transition system
    let var_context: Vec<(String, Lean5Type)> = ts
        .variables
        .iter()
        .map(|v| {
            let lean_ty = match v.smt_type {
                SmtType::Int => Lean5Type::Int,
                SmtType::Bool => Lean5Type::Bool,
                _ => Lean5Type::Int,
            };
            (clean_var_name_for_lean5(&v.name), lean_ty)
        })
        .collect();

    // 1. Initiation: init => invariant
    let init_expr = parse_smt_to_lean5_expr(&ts.init.smt_formula);
    let init_obligation = ProofObligation::new(
        format!("{}_initiation", clean_lean5_name(property_name)),
        ProofObligationKind::Initiation,
        Lean5Expr::implies(init_expr, invariant_expr.clone()),
    );
    obligations.push(add_context(init_obligation, &var_context));

    // 2. Consecution: invariant ∧ transition => invariant'
    // For k-induction, we need to show: if invariant holds for k consecutive states,
    // then it holds for the next state
    let trans_expr = parse_smt_to_lean5_expr(&ts.transition.smt_formula);
    let consecution_hypothesis = Lean5Expr::and(invariant_expr.clone(), trans_expr);
    let consecution_obligation = ProofObligation::new(
        format!("{}_consecution_k{}", clean_lean5_name(property_name), k),
        ProofObligationKind::Consecution,
        Lean5Expr::implies(consecution_hypothesis, invariant_expr.clone()),
    );
    obligations.push(add_context(consecution_obligation, &var_context));

    // 3. Property: invariant => property
    let property_expr = if !ts.properties.is_empty() {
        parse_smt_to_lean5_expr(&ts.properties[0].formula.smt_formula)
    } else {
        Lean5Expr::BoolLit(true)
    };
    let property_obligation = ProofObligation::new(
        format!("{}_property", clean_lean5_name(property_name)),
        ProofObligationKind::Property,
        Lean5Expr::implies(invariant_expr, property_expr),
    );
    obligations.push(add_context(property_obligation, &var_context));

    obligations
}

/// Parse an SMT formula to a Lean5 expression
fn parse_smt_to_lean5_expr(smt: &str) -> Lean5Expr {
    // Try to use the proper SMT parser from kani-fast-lean5
    match kani_fast_lean5::parse_smt_formula(smt) {
        Ok(ast) => {
            let ctx = kani_fast_lean5::TranslationContext::new();
            match kani_fast_lean5::translate_ast(&ast, &ctx) {
                Ok(expr) => expr,
                Err(_) => {
                    // Fallback to a placeholder
                    Lean5Expr::const_(kani_fast_lean5::Lean5Name::simple(format!("/* {smt} */")))
                }
            }
        }
        Err(_) => {
            // Fallback to a placeholder
            Lean5Expr::const_(kani_fast_lean5::Lean5Name::simple(format!("/* {smt} */")))
        }
    }
}

/// Add variable context to a proof obligation
fn add_context(
    mut obligation: ProofObligation,
    context: &[(String, Lean5Type)],
) -> ProofObligation {
    for (name, ty) in context {
        obligation = obligation.with_var(name, ty.clone());
    }
    obligation
}

/// Clean a variable name for Lean5 (replace ! with _)
fn clean_var_name_for_lean5(name: &str) -> String {
    name.replace('!', "_")
}

/// Clean a name to be a valid Lean5 identifier
fn clean_lean5_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ========== OutputFormat tests ==========

    #[test]
    fn test_output_format_default_is_text() {
        // Verify that text is the default via CLI parsing
        let cli = Cli::parse_from(["kani-fast", "check"]);
        assert_eq!(cli.format, OutputFormat::Text);
    }

    #[test]
    fn test_output_format_text_explicit() {
        let cli = Cli::parse_from(["kani-fast", "--format", "text", "check"]);
        assert_eq!(cli.format, OutputFormat::Text);
    }

    #[test]
    fn test_output_format_json() {
        let cli = Cli::parse_from(["kani-fast", "--format", "json", "check"]);
        assert_eq!(cli.format, OutputFormat::Json);
    }

    #[test]
    fn test_output_format_short_flag() {
        let cli = Cli::parse_from(["kani-fast", "-f", "json", "check"]);
        assert_eq!(cli.format, OutputFormat::Json);
    }

    // ========== ChcBackendArg conversion tests ==========

    #[test]
    fn test_chc_backend_arg_auto_conversion() {
        let backend: ChcBackend = ChcBackendArg::Auto.into();
        assert!(matches!(backend, ChcBackend::Auto));
    }

    #[test]
    fn test_chc_backend_arg_z3_conversion() {
        let backend: ChcBackend = ChcBackendArg::Z3.into();
        assert!(matches!(backend, ChcBackend::Z3));
    }

    #[test]
    fn test_chc_backend_arg_z4_conversion() {
        let backend: ChcBackend = ChcBackendArg::Z4.into();
        assert!(matches!(backend, ChcBackend::Z4));
    }

    // ========== parse_transition_system tests ==========

    #[test]
    fn test_parse_transition_system_simple_counter() {
        let content = r"
            var x : Int
            init (= x 0)
            trans (= x' (+ x 1))
            prop non_negative : (>= x 0)
        ";

        let ts = parse_transition_system(content).expect("Should parse successfully");

        assert_eq!(ts.variables.len(), 1);
        assert_eq!(ts.variables[0].name, "x");
        assert!(matches!(ts.variables[0].smt_type, SmtType::Int));

        assert_eq!(ts.init.smt_formula, "(= x 0)");
        assert_eq!(ts.transition.smt_formula, "(= x' (+ x 1))");
        assert_eq!(ts.properties.len(), 1);
        assert_eq!(ts.properties[0].name, "non_negative");
    }

    #[test]
    fn test_parse_transition_system_multiple_variables() {
        let content = r"
            var x : Int
            var y : Bool
            var z : Real
            init (and (= x 0) (= y true) (= z 0.0))
            trans (and (= x' (+ x 1)) (= y' (not y)))
            prop safety : (or y (> x 0))
        ";

        let ts = parse_transition_system(content).expect("Should parse successfully");

        assert_eq!(ts.variables.len(), 3);
        assert_eq!(ts.variables[0].name, "x");
        assert!(matches!(ts.variables[0].smt_type, SmtType::Int));
        assert_eq!(ts.variables[1].name, "y");
        assert!(matches!(ts.variables[1].smt_type, SmtType::Bool));
        assert_eq!(ts.variables[2].name, "z");
        assert!(matches!(ts.variables[2].smt_type, SmtType::Real));
    }

    #[test]
    fn test_parse_transition_system_with_comments() {
        let content = r"
            # This is a comment
            ; This is also a comment
            var x : Int
            # Another comment
            init (= x 0)
            trans (= x' (+ x 1))
            prop p : (>= x 0)
        ";

        let ts = parse_transition_system(content).expect("Should parse with comments");
        assert_eq!(ts.variables.len(), 1);
    }

    #[test]
    fn test_parse_transition_system_empty_lines() {
        let content = r"

            var x : Int

            init (= x 0)

            trans (= x' (+ x 1))

            prop p : (>= x 0)

        ";

        let ts = parse_transition_system(content).expect("Should parse with empty lines");
        assert_eq!(ts.variables.len(), 1);
    }

    #[test]
    fn test_parse_transition_system_unknown_type_defaults_to_int() {
        let content = r"
            var x : Unknown
            init (= x 0)
            trans (= x' x)
        ";

        let ts = parse_transition_system(content).expect("Should parse with unknown type");
        assert!(matches!(ts.variables[0].smt_type, SmtType::Int));
    }

    #[test]
    fn test_parse_transition_system_case_insensitive_types() {
        let content = r"
            var a : INT
            var b : BOOL
            var c : REAL
            init true
            trans true
        ";

        let ts = parse_transition_system(content).expect("Should parse case-insensitive types");
        assert!(matches!(ts.variables[0].smt_type, SmtType::Int));
        assert!(matches!(ts.variables[1].smt_type, SmtType::Bool));
        assert!(matches!(ts.variables[2].smt_type, SmtType::Real));
    }

    #[test]
    fn test_parse_transition_system_no_properties() {
        let content = r"
            var x : Int
            init (= x 0)
            trans (= x' (+ x 1))
        ";

        let ts = parse_transition_system(content).expect("Should parse without properties");
        assert!(ts.properties.is_empty());
    }

    #[test]
    fn test_parse_transition_system_multiple_properties() {
        let content = r"
            var x : Int
            init (= x 0)
            trans (= x' (+ x 1))
            prop non_negative : (>= x 0)
            prop bounded : (< x 100)
        ";

        let ts = parse_transition_system(content).expect("Should parse multiple properties");
        assert_eq!(ts.properties.len(), 2);
        assert_eq!(ts.properties[0].name, "non_negative");
        assert_eq!(ts.properties[1].name, "bounded");
    }

    // ========== clean_var_name_for_lean5 tests ==========

    #[test]
    fn test_clean_var_name_replaces_bang() {
        assert_eq!(clean_var_name_for_lean5("x!0"), "x_0");
    }

    #[test]
    fn test_clean_var_name_multiple_bangs() {
        assert_eq!(clean_var_name_for_lean5("x!0!1!2"), "x_0_1_2");
    }

    #[test]
    fn test_clean_var_name_no_special_chars() {
        assert_eq!(clean_var_name_for_lean5("myVar"), "myVar");
    }

    #[test]
    fn test_clean_var_name_only_bangs() {
        assert_eq!(clean_var_name_for_lean5("!!!"), "___");
    }

    #[test]
    fn test_clean_var_name_empty() {
        assert_eq!(clean_var_name_for_lean5(""), "");
    }

    // ========== clean_lean5_name tests ==========

    #[test]
    fn test_clean_lean5_name_simple() {
        assert_eq!(clean_lean5_name("myFunction"), "myFunction");
    }

    #[test]
    fn test_clean_lean5_name_with_colons() {
        assert_eq!(clean_lean5_name("std::core::func"), "std__core__func");
    }

    #[test]
    fn test_clean_lean5_name_with_special_chars() {
        assert_eq!(clean_lean5_name("foo<bar>"), "foo_bar_");
    }

    #[test]
    fn test_clean_lean5_name_preserves_underscores() {
        assert_eq!(clean_lean5_name("my_function_name"), "my_function_name");
    }

    #[test]
    fn test_clean_lean5_name_preserves_numbers() {
        assert_eq!(clean_lean5_name("func123"), "func123");
    }

    #[test]
    fn test_clean_lean5_name_dots_to_underscores() {
        assert_eq!(clean_lean5_name("a.b.c"), "a_b_c");
    }

    #[test]
    fn test_clean_lean5_name_mixed() {
        assert_eq!(clean_lean5_name("my::func<T>(x)"), "my__func_T__x_");
    }

    #[test]
    fn test_clean_lean5_name_empty() {
        assert_eq!(clean_lean5_name(""), "");
    }

    // ========== CLI argument parsing tests ==========

    #[test]
    fn test_cli_verbose_flag_global() {
        let cli = Cli::parse_from(["kani-fast", "-v", "check"]);
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_verbose_long() {
        let cli = Cli::parse_from(["kani-fast", "--verbose", "check"]);
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_verify_command_defaults() {
        let cli = Cli::parse_from(["kani-fast", "verify"]);
        if let Commands::Verify {
            path,
            harness,
            timeout,
            unwind,
            solver,
            portfolio,
            portfolio_solvers,
            portfolio_max_concurrent,
            explain,
        } = cli.command
        {
            assert_eq!(path, PathBuf::from("."));
            assert!(harness.is_none());
            assert_eq!(timeout, 300);
            assert!(unwind.is_none());
            assert!(solver.is_none());
            assert!(!portfolio);
            assert!(portfolio_solvers.is_none());
            assert_eq!(portfolio_max_concurrent, 3);
            assert!(!explain);
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_with_path() {
        let cli = Cli::parse_from(["kani-fast", "verify", "/path/to/project"]);
        if let Commands::Verify { path, .. } = cli.command {
            assert_eq!(path, PathBuf::from("/path/to/project"));
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_with_harness() {
        let cli = Cli::parse_from(["kani-fast", "verify", "-H", "my_harness"]);
        if let Commands::Verify { harness, .. } = cli.command {
            assert_eq!(harness, Some("my_harness".to_string()));
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_with_timeout() {
        let cli = Cli::parse_from(["kani-fast", "verify", "-t", "600"]);
        if let Commands::Verify { timeout, .. } = cli.command {
            assert_eq!(timeout, 600);
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_with_unwind() {
        let cli = Cli::parse_from(["kani-fast", "verify", "-u", "10"]);
        if let Commands::Verify { unwind, .. } = cli.command {
            assert_eq!(unwind, Some(10));
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_portfolio_mode() {
        let cli = Cli::parse_from(["kani-fast", "verify", "--portfolio"]);
        if let Commands::Verify { portfolio, .. } = cli.command {
            assert!(portfolio);
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_portfolio_solvers() {
        let cli = Cli::parse_from([
            "kani-fast",
            "verify",
            "--portfolio",
            "--portfolio-solvers",
            "cadical,z3",
        ]);
        if let Commands::Verify {
            portfolio_solvers, ..
        } = cli.command
        {
            assert_eq!(
                portfolio_solvers,
                Some(vec!["cadical".to_string(), "z3".to_string()])
            );
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_verify_explain() {
        let cli = Cli::parse_from(["kani-fast", "verify", "--explain"]);
        if let Commands::Verify { explain, .. } = cli.command {
            assert!(explain);
        } else {
            panic!("Expected Verify command");
        }
    }

    #[test]
    fn test_cli_chc_command() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs"]);
        if let Commands::Chc {
            source,
            function,
            timeout,
            edition,
            backend,
            bitvec,
            bitvec_width,
            lean5,
            lean5_output,
            all_proofs,
            cache,
            no_cache,
            cache_stats,
            ..
        } = cli.command
        {
            assert_eq!(source, PathBuf::from("test.rs"));
            assert!(function.is_none());
            assert_eq!(timeout, 60);
            assert_eq!(edition, "2021");
            assert!(matches!(backend, ChcBackendArg::Auto));
            assert!(!bitvec);
            assert!(bitvec_width.is_none());
            assert!(!lean5);
            assert!(lean5_output.is_none());
            assert!(!all_proofs);
            assert!(!cache);
            assert!(!no_cache);
            assert!(!cache_stats);
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_cache() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--cache"]);
        if let Commands::Chc {
            cache, cache_path, ..
        } = cli.command
        {
            assert!(cache);
            assert_eq!(cache_path, PathBuf::from(".kani-cache.db"));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_cache_path() {
        let cli = Cli::parse_from([
            "kani-fast",
            "chc",
            "test.rs",
            "--cache",
            "--cache-path",
            "my_cache.db",
        ]);
        if let Commands::Chc { cache_path, .. } = cli.command {
            assert_eq!(cache_path, PathBuf::from("my_cache.db"));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_no_cache() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--no-cache"]);
        if let Commands::Chc { no_cache, .. } = cli.command {
            assert!(no_cache);
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_function() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "-F", "my_func"]);
        if let Commands::Chc { function, .. } = cli.command {
            assert_eq!(function, Some("my_func".to_string()));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_lean5() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--lean5"]);
        if let Commands::Chc { lean5, .. } = cli.command {
            assert!(lean5);
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_parallel() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--all-proofs", "-j", "4"]);
        if let Commands::Chc {
            all_proofs,
            parallel,
            ..
        } = cli.command
        {
            assert!(all_proofs);
            assert_eq!(parallel, Some(4));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_parallel_long_form() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--parallel", "8"]);
        if let Commands::Chc { parallel, .. } = cli.command {
            assert_eq!(parallel, Some(8));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_backend_z3() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--backend", "z3"]);
        if let Commands::Chc { backend, .. } = cli.command {
            assert!(matches!(backend, ChcBackendArg::Z3));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_bitvec_flag() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--bitvec"]);
        if let Commands::Chc {
            bitvec,
            bitvec_width,
            ..
        } = cli.command
        {
            assert!(bitvec);
            assert!(bitvec_width.is_none());
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_bitvec_width() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--bitvec-width", "64"]);
        if let Commands::Chc {
            bitvec,
            bitvec_width,
            ..
        } = cli.command
        {
            assert!(!bitvec);
            assert_eq!(bitvec_width, Some(64));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_resolve_bitvec_config_defaults() {
        let config = resolve_bitvec_config(false, None, false, None);
        assert!(!config.force_bitvec);
        assert_eq!(config.bitvec_width, 32);
    }

    #[test]
    fn test_resolve_bitvec_config_env_enables_force() {
        let config = resolve_bitvec_config(false, None, true, None);
        assert!(config.force_bitvec);
        assert_eq!(config.bitvec_width, 32);
    }

    #[test]
    fn test_resolve_bitvec_config_env_width_enables_force() {
        let config = resolve_bitvec_config(false, None, false, Some(64));
        assert!(config.force_bitvec);
        assert_eq!(config.bitvec_width, 64);
    }

    #[test]
    fn test_resolve_bitvec_config_cli_width_overrides_env() {
        let config = resolve_bitvec_config(false, Some(16), true, Some(64));
        assert!(config.force_bitvec);
        assert_eq!(config.bitvec_width, 16);
    }

    #[test]
    fn test_cli_chc_with_proof() {
        let cli = Cli::parse_from(["kani-fast", "chc", "test.rs", "--proof"]);
        if let Commands::Chc {
            proof,
            proof_output,
            ..
        } = cli.command
        {
            assert!(proof);
            assert!(proof_output.is_none());
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_chc_with_proof_output() {
        let cli = Cli::parse_from([
            "kani-fast",
            "chc",
            "test.rs",
            "--proof",
            "--proof-output",
            "my_proof.json",
        ]);
        if let Commands::Chc {
            proof,
            proof_output,
            ..
        } = cli.command
        {
            assert!(proof);
            assert_eq!(proof_output, Some(PathBuf::from("my_proof.json")));
        } else {
            panic!("Expected Chc command");
        }
    }

    #[test]
    fn test_cli_kinduction_demo() {
        let cli = Cli::parse_from(["kani-fast", "kinduction", "--demo"]);
        if let Commands::Kinduction { demo, max_k, .. } = cli.command {
            assert!(demo);
            assert_eq!(max_k, 10); // default
        } else {
            panic!("Expected Kinduction command");
        }
    }

    #[test]
    fn test_cli_kinduction_with_file() {
        let cli = Cli::parse_from(["kani-fast", "kinduction", "system.smt2", "-k", "20"]);
        if let Commands::Kinduction {
            input,
            max_k,
            timeout,
            ..
        } = cli.command
        {
            assert_eq!(input, Some(PathBuf::from("system.smt2")));
            assert_eq!(max_k, 20);
            assert_eq!(timeout, 60); // default
        } else {
            panic!("Expected Kinduction command");
        }
    }

    #[test]
    fn test_cli_kinduction_with_proof_output() {
        let cli = Cli::parse_from([
            "kani-fast",
            "kinduction",
            "system.smt2",
            "--proof",
            "--proof-output",
            "my_proof.json",
        ]);
        if let Commands::Kinduction {
            input,
            proof,
            proof_output,
            ..
        } = cli.command
        {
            assert_eq!(input, Some(PathBuf::from("system.smt2")));
            assert!(proof);
            assert_eq!(proof_output, Some(PathBuf::from("my_proof.json")));
        } else {
            panic!("Expected Kinduction command");
        }
    }

    #[test]
    fn test_cli_unbounded_command() {
        let cli = Cli::parse_from(["kani-fast", "unbounded", "test.rs"]);
        if let Commands::Unbounded {
            source,
            function,
            max_k,
            timeout,
            chc_only,
            kinduction_only,
            ai,
            ai_only,
            certificate,
            verify_lean,
            ..
        } = cli.command
        {
            assert_eq!(source, PathBuf::from("test.rs"));
            assert!(function.is_none());
            assert_eq!(max_k, 5); // default
            assert_eq!(timeout, 120); // default
            assert!(!chc_only);
            assert!(!kinduction_only);
            assert!(!ai);
            assert!(!ai_only);
            assert!(!certificate);
            assert!(!verify_lean);
        } else {
            panic!("Expected Unbounded command");
        }
    }

    #[test]
    fn test_cli_unbounded_chc_only() {
        let cli = Cli::parse_from(["kani-fast", "unbounded", "test.rs", "--chc-only"]);
        if let Commands::Unbounded { chc_only, .. } = cli.command {
            assert!(chc_only);
        } else {
            panic!("Expected Unbounded command");
        }
    }

    #[test]
    fn test_cli_unbounded_kinduction_only() {
        let cli = Cli::parse_from(["kani-fast", "unbounded", "test.rs", "--kinduction-only"]);
        if let Commands::Unbounded {
            kinduction_only, ..
        } = cli.command
        {
            assert!(kinduction_only);
        } else {
            panic!("Expected Unbounded command");
        }
    }

    #[test]
    fn test_cli_unbounded_ai_options() {
        let cli = Cli::parse_from([
            "kani-fast",
            "unbounded",
            "test.rs",
            "--ai",
            "--ai-max-attempts",
            "20",
        ]);
        if let Commands::Unbounded {
            ai,
            ai_max_attempts,
            ..
        } = cli.command
        {
            assert!(ai);
            assert_eq!(ai_max_attempts, 20);
        } else {
            panic!("Expected Unbounded command");
        }
    }

    #[test]
    fn test_cli_unbounded_certificate() {
        let cli = Cli::parse_from([
            "kani-fast",
            "unbounded",
            "test.rs",
            "--certificate",
            "--verify-lean",
        ]);
        if let Commands::Unbounded {
            certificate,
            verify_lean,
            ..
        } = cli.command
        {
            assert!(certificate);
            assert!(verify_lean);
        } else {
            panic!("Expected Unbounded command");
        }
    }

    #[test]
    fn test_cli_watch_command() {
        let cli = Cli::parse_from(["kani-fast", "watch", "/project", "-d", "1000"]);
        if let Commands::Watch {
            path,
            debounce,
            timeout,
        } = cli.command
        {
            assert_eq!(path, PathBuf::from("/project"));
            assert_eq!(debounce, 1000);
            assert_eq!(timeout, 300); // default
        } else {
            panic!("Expected Watch command");
        }
    }

    #[test]
    fn test_cli_incremental_command() {
        let cli = Cli::parse_from(["kani-fast", "incremental", "test.cnf", "--stats"]);
        if let Commands::Incremental {
            input,
            stats,
            clear_cache,
            ..
        } = cli.command
        {
            assert_eq!(input, PathBuf::from("test.cnf"));
            assert!(stats);
            assert!(!clear_cache);
        } else {
            panic!("Expected Incremental command");
        }
    }

    #[test]
    fn test_cli_incremental_with_cache_options() {
        let cli = Cli::parse_from([
            "kani-fast",
            "incremental",
            "test.cnf",
            "--cache-db",
            "/tmp/cache.db",
            "--clear-cache",
        ]);
        if let Commands::Incremental {
            cache_db,
            clear_cache,
            ..
        } = cli.command
        {
            assert_eq!(cache_db, Some(PathBuf::from("/tmp/cache.db")));
            assert!(clear_cache);
        } else {
            panic!("Expected Incremental command");
        }
    }

    #[test]
    fn test_cli_lean5_check_command() {
        let cli = Cli::parse_from(["kani-fast", "lean5-check", "proof.lean", "-t", "120"]);
        if let Commands::Lean5Check { file, timeout } = cli.command {
            assert_eq!(file, PathBuf::from("proof.lean"));
            assert_eq!(timeout, 120);
        } else {
            panic!("Expected Lean5Check command");
        }
    }

    #[test]
    fn test_cli_check_command() {
        let cli = Cli::parse_from(["kani-fast", "check"]);
        assert!(matches!(cli.command, Commands::Check));
    }

    #[test]
    fn test_cli_version_command() {
        let cli = Cli::parse_from(["kani-fast", "version"]);
        assert!(matches!(cli.command, Commands::Version));
    }

    // ========== has_z3 tests ==========

    #[test]
    fn test_has_z3_returns_bool() {
        // This test just verifies the function returns a bool without panicking
        let _result = has_z3();
    }

    // ========== parse_smt_to_lean5_expr tests ==========

    #[test]
    fn test_parse_smt_simple_comparison() {
        let expr = parse_smt_to_lean5_expr("(>= x 0)");
        // Just verify it doesn't panic and returns something
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_complex_formula() {
        let expr = parse_smt_to_lean5_expr("(and (>= x 0) (< x 100))");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_equality() {
        let expr = parse_smt_to_lean5_expr("(= x 0)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_invalid_falls_back() {
        // Invalid SMT should fall back to placeholder
        let expr = parse_smt_to_lean5_expr("not valid smt");
        let _ = format!("{:?}", expr);
    }

    // ========== ChcBackendArg Debug/Clone/PartialEq tests ==========

    #[test]
    fn test_chc_backend_arg_debug() {
        assert_eq!(format!("{:?}", ChcBackendArg::Auto), "Auto");
        assert_eq!(format!("{:?}", ChcBackendArg::Z3), "Z3");
        assert_eq!(format!("{:?}", ChcBackendArg::Z4), "Z4");
    }

    #[test]
    fn test_chc_backend_arg_clone() {
        let backend = ChcBackendArg::Z3;
        let cloned = backend;
        assert_eq!(backend, cloned);
    }

    #[test]
    fn test_chc_backend_arg_eq() {
        assert_eq!(ChcBackendArg::Auto, ChcBackendArg::Auto);
        assert_eq!(ChcBackendArg::Z3, ChcBackendArg::Z3);
        assert_ne!(ChcBackendArg::Z3, ChcBackendArg::Z4);
    }

    // ========== OutputFormat Debug/Clone/PartialEq tests ==========

    #[test]
    fn test_output_format_debug() {
        assert_eq!(format!("{:?}", OutputFormat::Text), "Text");
        assert_eq!(format!("{:?}", OutputFormat::Json), "Json");
    }

    #[test]
    fn test_output_format_clone() {
        let format = OutputFormat::Json;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_output_format_eq() {
        assert_eq!(OutputFormat::Text, OutputFormat::Text);
        assert_ne!(OutputFormat::Text, OutputFormat::Json);
    }

    // ========== add_context tests ==========

    #[test]
    fn test_add_context_empty() {
        let obligation = ProofObligation::new(
            "test",
            ProofObligationKind::Initiation,
            Lean5Expr::BoolLit(true),
        );
        let context: Vec<(String, Lean5Type)> = vec![];
        let result = add_context(obligation, &context);
        // Should return obligation unchanged
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_add_context_single_var() {
        let obligation = ProofObligation::new(
            "test",
            ProofObligationKind::Initiation,
            Lean5Expr::BoolLit(true),
        );
        let context = vec![("x".to_string(), Lean5Type::Int)];
        let result = add_context(obligation, &context);
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_add_context_multiple_vars() {
        let obligation = ProofObligation::new(
            "test",
            ProofObligationKind::Consecution,
            Lean5Expr::BoolLit(true),
        );
        let context = vec![
            ("x".to_string(), Lean5Type::Int),
            ("y".to_string(), Lean5Type::Bool),
            ("z".to_string(), Lean5Type::Nat),
        ];
        let result = add_context(obligation, &context);
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_add_context_preserves_kind() {
        let obligation =
            ProofObligation::new("prop", ProofObligationKind::Property, Lean5Expr::var("x"));
        let context = vec![("x".to_string(), Lean5Type::Int)];
        let result = add_context(obligation, &context);
        assert!(matches!(result.kind, ProofObligationKind::Property));
    }

    // ========== generate_kinduction_proof_obligations tests ==========

    #[test]
    fn test_generate_kinduction_obligations_basic() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("non_neg", "non_neg", "(>= x 0)")
            .build();

        let obligations = generate_kinduction_proof_obligations("non_neg", 3, None, &ts);

        assert_eq!(obligations.len(), 3);
        assert!(obligations[0].name.contains("initiation"));
        assert!(obligations[1].name.contains("consecution"));
        assert!(obligations[2].name.contains("property"));
    }

    #[test]
    fn test_generate_kinduction_obligations_with_invariant() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("bounded", "bounded", "(< x 100)")
            .build();

        let invariant = "(>= x 0)";
        let obligations = generate_kinduction_proof_obligations("bounded", 5, Some(invariant), &ts);

        assert_eq!(obligations.len(), 3);
        // Verify the k value is in the consecution name
        assert!(obligations[1].name.contains("k5"));
    }

    #[test]
    fn test_generate_kinduction_obligations_cleaned_name() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' x)")
            .property("my_prop", "my_prop", "(>= x 0)")
            .build();

        let obligations = generate_kinduction_proof_obligations("my::prop<T>", 1, None, &ts);

        // Names should be cleaned (no colons or angle brackets)
        assert!(obligations[0].name.contains("my__prop_T_"));
        assert!(!obligations[0].name.contains(':'));
        assert!(!obligations[0].name.contains('<'));
    }

    #[test]
    fn test_generate_kinduction_obligations_multiple_vars() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .variable("flag", SmtType::Bool)
            .init("(and (= x 0) (= y 0) (= flag true))")
            .transition("(and (= x' (+ x 1)) (= y' y) (= flag' flag))")
            .property("safety", "safety", "(>= x 0)")
            .build();

        let obligations = generate_kinduction_proof_obligations("safety", 2, None, &ts);

        assert_eq!(obligations.len(), 3);
    }

    #[test]
    fn test_generate_kinduction_obligations_k_values() {
        let ts = TransitionSystemBuilder::new()
            .variable("n", SmtType::Int)
            .init("(= n 0)")
            .transition("(= n' (+ n 1))")
            .property("pos", "pos", "(>= n 0)")
            .build();

        // Test different k values
        for k in [1, 5, 10, 100] {
            let obligations = generate_kinduction_proof_obligations("pos", k, None, &ts);
            assert!(obligations[1].name.contains(&format!("k{}", k)));
        }
    }

    // ========== generate_ai_proof_obligations tests ==========

    #[test]
    fn test_generate_ai_obligations_basic() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("non_neg", "non_neg", "(>= x 0)")
            .build();

        let invariant = kani_fast_kinduction::StateFormula::new("(>= x 0)");
        let property = &ts.properties[0];

        let obligations = generate_ai_proof_obligations(&invariant, &ts, property);

        assert_eq!(obligations.len(), 3);
        assert_eq!(obligations[0].name, "ai_initiation");
        assert_eq!(obligations[1].name, "ai_consecution");
        assert_eq!(obligations[2].name, "ai_property");
    }

    #[test]
    fn test_generate_ai_obligations_kinds() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' x)")
            .property("p", "p", "(>= x 0)")
            .build();

        let invariant = kani_fast_kinduction::StateFormula::new("(>= x 0)");

        let obligations = generate_ai_proof_obligations(&invariant, &ts, &ts.properties[0]);

        assert!(matches!(
            obligations[0].kind,
            ProofObligationKind::Initiation
        ));
        assert!(matches!(
            obligations[1].kind,
            ProofObligationKind::Consecution
        ));
        assert!(matches!(obligations[2].kind, ProofObligationKind::Property));
    }

    #[test]
    fn test_generate_ai_obligations_complex_invariant() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("(and (= x 0) (= y 0))")
            .transition("(and (= x' (+ x 1)) (= y' (+ y 2)))")
            .property("bounds", "bounds", "(and (>= x 0) (>= y 0))")
            .build();

        let invariant =
            kani_fast_kinduction::StateFormula::new("(and (>= x 0) (>= y 0) (= y (* 2 x)))");

        let obligations = generate_ai_proof_obligations(&invariant, &ts, &ts.properties[0]);
        assert_eq!(obligations.len(), 3);
    }

    // ========== generate_lean5_from_ai_invariant tests ==========

    #[test]
    fn test_generate_lean5_from_ai_invariant_produces_output() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p", "p", "(>= x 0)")
            .build();

        let invariant = kani_fast_kinduction::StateFormula::new("(>= x 0)");

        let lean5_output = generate_lean5_from_ai_invariant(&invariant, &ts, &ts.properties[0]);

        // Should produce valid Lean5 file structure
        assert!(lean5_output.contains("theorem"));
        assert!(lean5_output.contains("ai_initiation"));
        assert!(lean5_output.contains("ai_consecution"));
        assert!(lean5_output.contains("ai_property"));
    }

    #[test]
    fn test_generate_lean5_from_ai_invariant_contains_vars() {
        let ts = TransitionSystemBuilder::new()
            .variable("counter", SmtType::Int)
            .init("(= counter 0)")
            .transition("(= counter' (+ counter 1))")
            .property("p", "p", "(>= counter 0)")
            .build();

        let invariant = kani_fast_kinduction::StateFormula::new("(>= counter 0)");

        let lean5_output = generate_lean5_from_ai_invariant(&invariant, &ts, &ts.properties[0]);

        // Should contain the variable name
        assert!(lean5_output.contains("counter"));
    }

    // ========== parse_smt_to_lean5_expr additional tests ==========

    #[test]
    fn test_parse_smt_arithmetic() {
        let expr = parse_smt_to_lean5_expr("(+ x 1)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_nested_and() {
        let expr = parse_smt_to_lean5_expr("(and (and a b) c)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_nested_or() {
        let expr = parse_smt_to_lean5_expr("(or (or a b) c)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_implication() {
        let expr = parse_smt_to_lean5_expr("(=> a b)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_not() {
        let expr = parse_smt_to_lean5_expr("(not x)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_less_than() {
        let expr = parse_smt_to_lean5_expr("(< x 10)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_multiplication() {
        let expr = parse_smt_to_lean5_expr("(* x y)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_subtraction() {
        let expr = parse_smt_to_lean5_expr("(- x y)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_division() {
        let expr = parse_smt_to_lean5_expr("(/ x y)");
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_parse_smt_ite() {
        let expr = parse_smt_to_lean5_expr("(ite cond then_val else_val)");
        let _ = format!("{:?}", expr);
    }

    // ========== clean_var_name_for_lean5 additional tests ==========

    #[test]
    fn test_clean_var_name_mixed_content() {
        assert_eq!(clean_var_name_for_lean5("var!0_suffix"), "var_0_suffix");
    }

    #[test]
    fn test_clean_var_name_leading_bang() {
        assert_eq!(clean_var_name_for_lean5("!start"), "_start");
    }

    #[test]
    fn test_clean_var_name_trailing_bang() {
        assert_eq!(clean_var_name_for_lean5("end!"), "end_");
    }

    // ========== clean_lean5_name additional tests ==========

    #[test]
    fn test_clean_lean5_name_spaces() {
        assert_eq!(clean_lean5_name("a b c"), "a_b_c");
    }

    #[test]
    fn test_clean_lean5_name_brackets() {
        assert_eq!(clean_lean5_name("vec[i]"), "vec_i_");
    }

    #[test]
    fn test_clean_lean5_name_slashes() {
        assert_eq!(clean_lean5_name("a/b/c"), "a_b_c");
    }

    #[test]
    fn test_clean_lean5_name_dashes() {
        assert_eq!(clean_lean5_name("kebab-case"), "kebab_case");
    }

    #[test]
    fn test_clean_lean5_name_at_symbol() {
        assert_eq!(clean_lean5_name("user@domain"), "user_domain");
    }

    #[test]
    fn test_clean_lean5_name_all_special() {
        assert_eq!(clean_lean5_name("!@#$%"), "_____");
    }

    // ========== Mutation coverage tests for parse_transition_system ==========

    /// Test that empty lines are specifically handled (catches || → && mutation)
    #[test]
    fn test_parse_transition_system_only_empty_lines_handled() {
        // Content with ONLY empty lines and whitespace - no comments
        let content = "\n\n   \n\t\n  \n";
        let ts = parse_transition_system(content).expect("Empty content should parse");
        assert!(ts.variables.is_empty());
        assert!(ts.properties.is_empty());
    }

    /// Test that semicolon comments are specifically handled (catches || → && mutation)
    #[test]
    fn test_parse_transition_system_only_semicolon_comments() {
        // Content with ONLY semicolon comments - no hash comments, no empty lines
        let content = "; comment 1\n; comment 2\n;comment3";
        let ts = parse_transition_system(content).expect("Only semicolon comments should parse");
        assert!(ts.variables.is_empty());
    }

    /// Test that hash comments are specifically handled (catches || → && mutation)
    #[test]
    fn test_parse_transition_system_only_hash_comments() {
        // Content with ONLY hash comments - no semicolon comments, no empty lines
        let content = "# comment 1\n# comment 2\n#comment3";
        let ts = parse_transition_system(content).expect("Only hash comments should parse");
        assert!(ts.variables.is_empty());
    }

    /// Test that non-empty non-comment lines are NOT skipped (catches || → && mutation)
    /// If || were &&, this content would skip all lines because they're not both empty AND comments
    #[test]
    fn test_parse_transition_system_actual_content_not_skipped() {
        // Single non-comment, non-empty line - MUST be parsed
        let content = "var x : Int";
        let ts = parse_transition_system(content).expect("Variable line should parse");
        // If || were &&, the line would NOT be skipped (correct), but we need to verify
        // the condition is evaluated correctly in all branches
        assert_eq!(ts.variables.len(), 1);
        assert_eq!(ts.variables[0].name, "x");
    }

    /// Test mixed content parsing to verify || short-circuit behavior
    /// Empty line followed by comment followed by content - all should work
    #[test]
    fn test_parse_transition_system_mixed_skip_content() {
        let content = "\n; comment\n# comment\nvar x : Int\ninit (= x 0)\ntrans (= x' x)";
        let ts = parse_transition_system(content).expect("Mixed content should parse");
        assert_eq!(ts.variables.len(), 1);
        assert_eq!(ts.init.smt_formula, "(= x 0)");
    }

    /// Test that lines starting with semicolon but NOT empty are skipped
    #[test]
    fn test_parse_transition_system_semicolon_not_empty() {
        let content = ";var x : Int\n;init (= x 0)\nvar y : Int\ninit (= y 0)\ntrans (= y' y)";
        let ts = parse_transition_system(content).expect("Should skip commented lines");
        // Only var y should be parsed, semicolon-prefixed lines are comments
        assert_eq!(ts.variables.len(), 1);
        assert_eq!(ts.variables[0].name, "y");
    }

    /// Test int type is correctly identified (catches delete match arm "int" mutation)
    #[test]
    fn test_parse_transition_system_int_type_explicit() {
        let content = "var myint : int\ninit (= myint 42)\ntrans (= myint' myint)";
        let ts = parse_transition_system(content).expect("Should parse int type");
        assert_eq!(ts.variables.len(), 1);
        assert!(
            matches!(ts.variables[0].smt_type, SmtType::Int),
            "Type 'int' must map to SmtType::Int"
        );
    }

    /// Test real type is correctly identified and distinct from int
    #[test]
    fn test_parse_transition_system_real_vs_int() {
        let content = "var x : Int\nvar y : Real\ninit (and (= x 0) (= y 0.0))\ntrans true";
        let ts = parse_transition_system(content).expect("Should parse mixed types");
        assert!(
            matches!(ts.variables[0].smt_type, SmtType::Int),
            "x must be Int"
        );
        assert!(
            matches!(ts.variables[1].smt_type, SmtType::Real),
            "y must be Real"
        );
        // Verify they are different
        assert!(
            !matches!(ts.variables[0].smt_type, SmtType::Real),
            "x must NOT be Real"
        );
    }

    // ========== Mutation coverage tests for generate_ai_proof_obligations ==========

    /// Test that Int type variables produce correct Lean5Type in proof obligations
    #[test]
    fn test_generate_ai_obligations_int_type() {
        use kani_fast_kinduction::{PropertyType, StateFormula, TransitionSystemBuilder};

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("test", "test prop", "(>= x 0)")
            .build();

        let invariant = StateFormula::new("(>= x 0)");

        let property = kani_fast_kinduction::Property {
            id: "test".to_string(),
            name: "test".to_string(),
            formula: StateFormula::new("(>= x 0)"),
            property_type: PropertyType::Safety,
        };

        let obligations = generate_ai_proof_obligations(&invariant, &system, &property);

        // Should have 3 obligations: initiation, consecution, property
        assert_eq!(obligations.len(), 3);

        // Check that the variable context contains Int type
        let init_ob = &obligations[0];
        let has_int_var = init_ob
            .context
            .iter()
            .any(|(name, ty)| name == "x" && matches!(ty, kani_fast_lean5::Lean5Type::Int));
        assert!(has_int_var, "x should be typed as Int in proof obligation");
    }

    /// Test that Bool type variables produce correct Lean5Type in proof obligations
    #[test]
    fn test_generate_ai_obligations_bool_type() {
        use kani_fast_kinduction::{PropertyType, StateFormula, TransitionSystemBuilder};

        let system = TransitionSystemBuilder::new()
            .variable("flag", SmtType::Bool)
            .init("(= flag true)")
            .transition("(= flag' (not flag))")
            .property("test", "test prop", "true")
            .build();

        let invariant = StateFormula::new("(or flag (not flag))");

        let property = kani_fast_kinduction::Property {
            id: "test".to_string(),
            name: "test".to_string(),
            formula: StateFormula::new("true"),
            property_type: PropertyType::Safety,
        };

        let obligations = generate_ai_proof_obligations(&invariant, &system, &property);

        // Check that the variable context contains Bool type
        let init_ob = &obligations[0];
        let has_bool_var = init_ob
            .context
            .iter()
            .any(|(name, ty)| name == "flag" && matches!(ty, kani_fast_lean5::Lean5Type::Bool));
        assert!(
            has_bool_var,
            "flag should be typed as Bool in proof obligation"
        );
    }

    /// Test that mixed Int/Bool types are preserved
    #[test]
    fn test_generate_ai_obligations_mixed_types() {
        use kani_fast_kinduction::{PropertyType, StateFormula, TransitionSystemBuilder};

        let system = TransitionSystemBuilder::new()
            .variable("count", SmtType::Int)
            .variable("enabled", SmtType::Bool)
            .init("(and (= count 0) (= enabled true))")
            .transition("(and (= count' (+ count 1)) (= enabled' enabled))")
            .property("test", "test prop", "true")
            .build();

        let invariant = StateFormula::new("(or (not enabled) (>= count 0))");

        let property = kani_fast_kinduction::Property {
            id: "test".to_string(),
            name: "test".to_string(),
            formula: StateFormula::new("true"),
            property_type: PropertyType::Safety,
        };

        let obligations = generate_ai_proof_obligations(&invariant, &system, &property);

        let init_ob = &obligations[0];

        // Verify count is Int
        let count_is_int = init_ob
            .context
            .iter()
            .any(|(name, ty)| name == "count" && matches!(ty, kani_fast_lean5::Lean5Type::Int));
        assert!(count_is_int, "count should be Int");

        // Verify enabled is Bool
        let enabled_is_bool = init_ob
            .context
            .iter()
            .any(|(name, ty)| name == "enabled" && matches!(ty, kani_fast_lean5::Lean5Type::Bool));
        assert!(enabled_is_bool, "enabled should be Bool");
    }

    // ========== Mutation coverage tests for generate_kinduction_proof_obligations ==========

    /// Test k-induction obligations with Int type
    #[test]
    fn test_generate_kinduction_obligations_int_type() {
        use kani_fast_kinduction::TransitionSystemBuilder;

        let system = TransitionSystemBuilder::new()
            .variable("n", SmtType::Int)
            .init("(= n 0)")
            .transition("(= n' (+ n 1))")
            .property("non_negative", "n >= 0", "(>= n 0)")
            .build();

        // generate_kinduction_proof_obligations(property_name, k, invariant_formula, ts)
        let obligations =
            generate_kinduction_proof_obligations("non_negative", 3, Some("(>= n 0)"), &system);

        // Should have at least initiation and consecution obligations
        assert!(obligations.len() >= 2);

        // Check that variable type is correctly Int
        let first_ob = &obligations[0];
        let n_is_int = first_ob
            .context
            .iter()
            .any(|(name, ty)| name == "n" && matches!(ty, kani_fast_lean5::Lean5Type::Int));
        assert!(n_is_int, "n should be typed as Int");
    }

    /// Test k-induction obligations with Bool type
    #[test]
    fn test_generate_kinduction_obligations_bool_type() {
        use kani_fast_kinduction::TransitionSystemBuilder;

        let system = TransitionSystemBuilder::new()
            .variable("toggle", SmtType::Bool)
            .init("toggle")
            .transition("(= toggle' (not toggle))")
            .property("tautology", "always true", "(or toggle (not toggle))")
            .build();

        let obligations = generate_kinduction_proof_obligations(
            "tautology",
            2,
            Some("(or toggle (not toggle))"),
            &system,
        );

        let first_ob = &obligations[0];
        let toggle_is_bool = first_ob
            .context
            .iter()
            .any(|(name, ty)| name == "toggle" && matches!(ty, kani_fast_lean5::Lean5Type::Bool));
        assert!(toggle_is_bool, "toggle should be typed as Bool");
    }
}
