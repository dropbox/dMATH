#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::Read;
use std::path::{Path, PathBuf};
#[cfg(feature = "prove")]
use std::time::Duration;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use tla_check::{
    liveness_trace_to_dot, resolve_spec_from_config_with_extends, trace_to_dot, AdaptiveChecker,
    CheckResult, Config, FingerprintStorage, JsonOutput, ModelChecker, ParallelChecker, Progress,
    SimulationConfig, SimulationResult, TraceFile, TraceLocationsStorage,
};
use tla_codegen::{generate_rust, CodeGenOptions};
use tla_core::diagnostic::{lower_error_diagnostic, parse_error_diagnostic};
use tla_core::loader::ModuleLoader;
use tla_core::{lower, parse, pretty_module, FileId, SyntaxNode};

#[cfg(feature = "prove")]
use tla_prove::{ModuleResult, ProofOutcome, Prover};

/// Output format for model checking results
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output (default)
    #[default]
    Human,
    /// Structured JSON output for AI agents and automation
    Json,
    /// Streaming JSON Lines format (one JSON object per line)
    JsonLines,
}

/// Format for counterexample traces
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum TraceFormat {
    /// Human-readable text format (default)
    #[default]
    Text,
    /// GraphViz DOT format for visualization
    Dot,
}

#[derive(Parser)]
#[command(name = "tla", version, about = "TLA2 CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a TLA+ source file and report syntax errors.
    Parse { file: PathBuf },
    /// Parse + lower a TLA+ source file and dump the lowered AST (Debug).
    Ast { file: PathBuf },
    /// Parse + lower a TLA+ source file and pretty-print the module to stdout.
    Fmt { file: PathBuf },
    /// Model check a TLA+ specification.
    Check {
        /// TLA+ source file to check.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of worker threads.
        /// 0 = auto (adaptive selection based on spec characteristics).
        /// 1 = sequential (no parallelism overhead).
        /// N = parallel with N workers.
        #[arg(short, long, default_value = "0")]
        workers: usize,
        /// Disable deadlock checking.
        #[arg(long)]
        no_deadlock: bool,
        /// Maximum number of states to explore (0 = unlimited).
        #[arg(long, default_value = "0")]
        max_states: usize,
        /// Maximum BFS depth to explore (0 = unlimited).
        #[arg(long, default_value = "0")]
        max_depth: usize,
        /// Show progress during model checking.
        #[arg(long)]
        progress: bool,
        /// Show per-action coverage statistics.
        ///
        /// Note: Coverage collection is only supported in sequential mode today.
        /// Use `--workers 1` or `--workers 0` (auto, which will force sequential).
        #[arg(long)]
        coverage: bool,
        /// Maximum memory efficiency: disable all trace reconstruction.
        ///
        /// By default, TLA2 stores only fingerprints with a temp trace file for
        /// counterexample reconstruction (42x less memory than full states).
        /// Use --no-trace to also disable the trace file, making counterexample
        /// traces completely unavailable when violations are found.
        ///
        /// Note: Liveness checking is disabled in this mode.
        #[arg(long)]
        no_trace: bool,
        /// Store full states in memory (legacy mode, 42x more memory).
        ///
        /// By default, TLA2 stores only fingerprints with disk-based trace
        /// reconstruction. Use --store-states to keep full states in memory,
        /// which provides faster trace reconstruction but uses ~42x more memory.
        /// This was the default behavior before v0.6.
        ///
        /// Conflicts with --no-trace.
        #[arg(long, conflicts_with = "no_trace")]
        store_states: bool,
        /// Use memory-mapped fingerprint storage with given capacity.
        ///
        /// This enables exploring state spaces larger than available RAM by using
        /// memory-mapped storage that can page to disk. The capacity specifies the
        /// maximum number of fingerprints to store (e.g., "1000000" for 1M states).
        ///
        /// Incompatible with --store-states. If not set, uses in-memory hash set.
        #[arg(long, value_name = "CAPACITY", conflicts_with = "store_states")]
        mmap_fingerprints: Option<usize>,
        /// Use disk-backed fingerprint storage with automatic eviction.
        ///
        /// This enables exploring billion-state specs by automatically evicting
        /// fingerprints from memory to disk when the primary storage fills up.
        /// The capacity specifies the in-memory primary storage size before eviction.
        ///
        /// Requires --mmap-dir. Incompatible with --store-states and --mmap-fingerprints.
        #[arg(long, value_name = "CAPACITY", conflicts_with_all = ["mmap_fingerprints", "store_states"])]
        disk_fingerprints: Option<usize>,
        /// Directory for memory-mapped or disk-backed fingerprint storage.
        ///
        /// If specified with --mmap-fingerprints, creates a file-backed mapping
        /// in this directory, allowing the OS to page fingerprints to disk.
        /// If not specified, uses anonymous memory mapping (in-memory, but with
        /// mmap semantics for potentially better OS memory management).
        ///
        /// Required for --disk-fingerprints. The evicted fingerprints are stored
        /// as sorted files in this directory.
        #[arg(long, value_name = "DIR")]
        mmap_dir: Option<PathBuf>,
        /// Path to explicit disk-based trace file for counterexample reconstruction.
        ///
        /// By default, TLA2 creates a temporary trace file automatically. Use this
        /// to specify a persistent location. The file stores (predecessor, fingerprint)
        /// pairs for trace reconstruction. Useful for debugging or keeping traces.
        ///
        /// Incompatible with --store-states. If file already exists, it will be overwritten.
        #[arg(long, value_name = "FILE", conflicts_with = "store_states")]
        trace_file: Option<PathBuf>,
        /// Use memory-mapped storage for trace file location mapping.
        ///
        /// When using --trace-file, this option enables memory-mapped storage
        /// for the fingerprint-to-offset mapping. Specify the capacity (maximum
        /// number of states). This reduces memory usage for large state spaces.
        ///
        /// Requires --trace-file. Uses the same directory as --mmap-dir if specified.
        #[arg(long, value_name = "CAPACITY")]
        mmap_trace_locations: Option<usize>,
        /// Directory for saving checkpoints during model checking.
        ///
        /// When specified, the model checker will periodically save checkpoint
        /// files to this directory. Checkpoints allow resuming interrupted model
        /// checking runs. The checkpoint interval can be set with --checkpoint-interval.
        #[arg(long, value_name = "DIR")]
        checkpoint: Option<PathBuf>,
        /// Checkpoint interval in seconds (default: 300).
        ///
        /// How often to save checkpoints during model checking. Only used when
        /// --checkpoint is specified.
        #[arg(long, default_value = "300")]
        checkpoint_interval: u64,
        /// Resume model checking from a checkpoint directory.
        ///
        /// Loads the checkpoint from the specified directory and continues
        /// model checking from where it left off.
        #[arg(long, value_name = "DIR")]
        resume: Option<PathBuf>,
        /// Output format: human (default), json, or json-lines.
        ///
        /// Use `json` for structured output suitable for AI agents and automated tooling.
        /// Use `json-lines` for streaming output with one JSON object per line.
        #[arg(long, value_enum, default_value = "human")]
        output: OutputFormat,
        /// Format for counterexample traces: text (default) or dot.
        ///
        /// Use `dot` to output traces in GraphViz DOT format for visualization.
        /// The DOT output can be rendered using: dot -Tpng trace.dot -o trace.png
        #[arg(long, value_enum, default_value = "text")]
        trace_format: TraceFormat,
    },
    /// Simulate a TLA+ specification (random trace exploration).
    ///
    /// Unlike exhaustive model checking, simulation generates random traces
    /// through the state space. This is useful for:
    /// - Quick exploration of large state spaces
    /// - Finding bugs that require deep traces
    /// - Probabilistic coverage when exhaustive checking is infeasible
    Simulate {
        /// TLA+ source file to simulate.
        file: PathBuf,
        /// Configuration file (.cfg). If not specified, looks for `<file>.cfg`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of random traces to generate.
        #[arg(short, long, default_value = "1000")]
        num_traces: usize,
        /// Maximum length of each trace (steps from initial state).
        #[arg(short = 'l', long, default_value = "100")]
        max_trace_length: usize,
        /// Random seed for reproducibility (0 = random seed).
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Disable invariant checking during simulation.
        #[arg(long)]
        no_invariants: bool,
    },
    /// Start the Language Server Protocol server.
    Lsp,
    /// Prove theorems in a TLA+ specification (requires Z3).
    #[cfg(feature = "prove")]
    Prove {
        /// TLA+ source file to prove.
        file: PathBuf,
        /// Timeout per obligation in seconds.
        #[arg(short, long, default_value = "60")]
        timeout: u64,
        /// Only check specific theorem(s) by name (comma-separated).
        #[arg(long)]
        theorem: Option<String>,
    },
    /// Generate Rust code from a TLA+ specification.
    Codegen {
        /// TLA+ source file.
        file: PathBuf,
        /// Output file (default: stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Generate Kani verification harnesses.
        #[arg(long)]
        kani: bool,
        /// Generate proptest property-based tests.
        #[arg(long)]
        proptest: bool,
    },
}

// Use a larger stack size (64MB) to handle deeply recursive TLA+ expressions
// The default 2MB stack is insufficient for specs with deeply nested recursive functions
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // Run the main logic in a thread with larger stack
    let result = std::thread::Builder::new()
        .name("tla2-main".to_string())
        .stack_size(64 * 1024 * 1024) // 64MB stack
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async_main())
        })
        .expect("Failed to spawn main thread")
        .join()
        .expect("Main thread panicked");
    result
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Parse { file } => cmd_parse(&file),
        Command::Ast { file } => cmd_ast(&file),
        Command::Fmt { file } => cmd_fmt(&file),
        Command::Check {
            file,
            config,
            workers,
            no_deadlock,
            max_states,
            max_depth,
            progress,
            coverage,
            no_trace,
            store_states,
            mmap_fingerprints,
            disk_fingerprints,
            mmap_dir,
            trace_file,
            mmap_trace_locations,
            checkpoint,
            checkpoint_interval,
            resume,
            output,
            trace_format,
        } => cmd_check(
            &file,
            config.as_deref(),
            workers,
            no_deadlock,
            max_states,
            max_depth,
            progress,
            coverage,
            no_trace,
            store_states,
            mmap_fingerprints,
            disk_fingerprints,
            mmap_dir,
            trace_file,
            mmap_trace_locations,
            checkpoint,
            checkpoint_interval,
            resume,
            output,
            trace_format,
        ),
        Command::Simulate {
            file,
            config,
            num_traces,
            max_trace_length,
            seed,
            no_invariants,
        } => cmd_simulate(
            &file,
            config.as_deref(),
            num_traces,
            max_trace_length,
            seed,
            no_invariants,
        ),
        Command::Lsp => {
            tla_lsp::run_server().await;
            Ok(())
        }
        #[cfg(feature = "prove")]
        Command::Prove {
            file,
            timeout,
            theorem,
        } => cmd_prove(&file, timeout, theorem.as_deref()),
        Command::Codegen {
            file,
            output,
            kani,
            proptest,
        } => cmd_codegen(&file, output.as_deref(), kani, proptest),
    }
}

fn read_source(file: &Path) -> Result<String> {
    if file.as_os_str() == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("read stdin")?;
        Ok(buf)
    } else {
        std::fs::read_to_string(file).with_context(|| format!("read {}", file.display()))
    }
}

fn parse_or_report(file: &Path, source: &str) -> Result<SyntaxNode> {
    let result = parse(source);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = parse_error_diagnostic(&file_path, &err.message, err.start, err.end);
            diagnostic.eprint(&file_path, source);
        }
        bail!("parse failed with {} error(s)", result.errors.len());
    }
    Ok(SyntaxNode::new_root(result.green_node))
}

fn cmd_parse(file: &Path) -> Result<()> {
    let source = read_source(file)?;
    let _ = parse_or_report(file, &source)?;
    Ok(())
}

fn cmd_ast(file: &Path) -> Result<()> {
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;

    let result = lower(FileId(0), &tree);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", result.errors.len());
    }
    let module = result.module.context("lower produced no module")?;
    println!("{module:#?}");
    Ok(())
}

fn cmd_fmt(file: &Path) -> Result<()> {
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;

    let result = lower(FileId(0), &tree);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", result.errors.len());
    }
    let module = result.module.context("lower produced no module")?;
    print!("{}", pretty_module(&module));
    Ok(())
}

// Allow many arguments: These mirror clap Command::Check fields for direct dispatch
#[allow(clippy::too_many_arguments)]
fn cmd_check(
    file: &Path,
    config_path: Option<&Path>,
    workers: usize,
    no_deadlock: bool,
    max_states: usize,
    max_depth: usize,
    show_progress: bool,
    show_coverage: bool,
    no_trace: bool,
    mut store_states: bool,
    mmap_fingerprints: Option<usize>,
    disk_fingerprints: Option<usize>,
    mmap_dir: Option<PathBuf>,
    trace_file_path: Option<PathBuf>,
    mmap_trace_locations: Option<usize>,
    checkpoint_dir: Option<PathBuf>,
    checkpoint_interval: u64,
    resume_from: Option<PathBuf>,
    output_format: OutputFormat,
    trace_format: TraceFormat,
) -> Result<()> {
    // Parse the TLA+ source file
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;

    let result = lower(FileId(0), &tree);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", result.errors.len());
    }
    let module = result.module.context("lower produced no module")?;

    // Load extended and instanced modules (non-stdlib only)
    let mut loader = ModuleLoader::new(file);

    // Seed the loader cache with inline modules from the main file.
    // This enables EXTENDS to resolve modules defined inline in the same file
    // (e.g., BufferedRandomAccessFile.tla defines Common and RandomAccessFile inline).
    loader.seed_from_syntax_tree(&tree, file);

    // First, load all modules (extends then instances) to populate the cache
    let extended_module_names = match loader.load_extends(&module) {
        Ok(names) => {
            if !names.is_empty() {
                eprintln!("Loaded extended modules: {}", names.join(", "));
            }
            names
        }
        Err(e) => {
            bail!("Failed to load extended modules: {}", e);
        }
    };

    // Load INSTANCE modules (non-stdlib only)
    // INSTANCE without WITH substitution works like EXTENDS for operator import
    let instance_module_names = match loader.load_instances(&module) {
        Ok(names) => {
            if !names.is_empty() {
                eprintln!("Loaded instanced modules: {}", names.join(", "));
            }
            names
        }
        Err(e) => {
            bail!("Failed to load instanced modules: {}", e);
        }
    };

    // Now collect references from the cache (no more mutation needed)
    let mut all_module_names: Vec<String> = extended_module_names;
    for name in instance_module_names {
        if !all_module_names.contains(&name) {
            all_module_names.push(name);
        }
    }

    let extended_modules: Vec<_> = all_module_names
        .iter()
        .filter_map(|name| loader.get(name).map(|l| &l.module))
        .collect();
    let extended_syntax_trees: Vec<_> = all_module_names
        .iter()
        .filter_map(|name| loader.get(name).map(|l| &l.syntax_tree))
        .collect();

    // Find and parse config file
    let config_path = match config_path {
        Some(p) => p.to_path_buf(),
        None => {
            // Look for <file>.cfg
            let mut cfg_path = file.to_path_buf();
            cfg_path.set_extension("cfg");
            if !cfg_path.exists() {
                bail!(
                    "No config file specified and {} does not exist.\n\
                     Use --config to specify a configuration file.",
                    cfg_path.display()
                );
            }
            cfg_path
        }
    };

    let config_source = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;

    let mut config = Config::parse(&config_source).map_err(|errors| {
        for err in &errors {
            eprintln!("{}:{}: {}", config_path.display(), err.line, err.message);
        }
        anyhow::anyhow!("config parse failed with {} error(s)", errors.len())
    })?;

    // Resolve Init/Next/Fairness from SPECIFICATION if needed
    // Also search extended modules for the SPECIFICATION operator
    let mut resolved_fairness = Vec::new();
    let mut resolved_spec: Option<tla_check::ResolvedSpec> = None;
    if (config.init.is_none() || config.next.is_none()) && config.specification.is_some() {
        match resolve_spec_from_config_with_extends(&config, &tree, &extended_syntax_trees) {
            Ok(resolved) => {
                if config.init.is_none() {
                    config.init = Some(resolved.init.clone());
                }
                if config.next.is_none() {
                    config.next = Some(resolved.next.clone());
                }
                // Capture fairness constraints for liveness checking
                resolved_fairness = resolved.fairness.clone();
                // Store full resolved spec for inline NEXT handling
                resolved_spec = Some(resolved);
            }
            Err(e) => {
                bail!("Failed to resolve SPECIFICATION: {}", e);
            }
        }
    }

    // Auto-enable store_full_states when config has liveness properties (#92)
    // Liveness checking requires full states, so we must store them.
    // Unless --no-trace was explicitly specified (user opts out of liveness).
    if !no_trace && !config.properties.is_empty() && !store_states {
        store_states = true;
    }

    // Print header (only for human output format)
    if matches!(output_format, OutputFormat::Human) {
        println!("Model checking: {}", file.display());
        println!("Config: {}", config_path.display());
        if let Some(ref spec) = config.specification {
            println!(
                "SPECIFICATION: {} (resolved to INIT: {}, NEXT: {})",
                spec,
                config.init.as_deref().unwrap_or("?"),
                config.next.as_deref().unwrap_or("?")
            );
        } else {
            if let Some(ref init) = config.init {
                println!("INIT: {}", init);
            }
            if let Some(ref next) = config.next {
                println!("NEXT: {}", next);
            }
        }
        if !config.invariants.is_empty() {
            println!("INVARIANTS: {}", config.invariants.join(", "));
        }
        if !config.properties.is_empty() {
            println!("PROPERTIES: {}", config.properties.join(", "));
        }
        if workers == 0 {
            println!("Mode: auto (adaptive strategy selection)");
        } else if workers == 1 {
            println!("Mode: sequential (1 worker)");
        } else {
            println!("Mode: parallel ({} workers)", workers);
        }
        println!();

        // Print limits if set
        if max_states > 0 {
            println!("Max states: {}", max_states);
        }
        if max_depth > 0 {
            println!("Max depth: {}", max_depth);
        }
        if store_states {
            println!("Store-states mode: full states in memory (42x more memory)");
        } else if no_trace {
            println!("No-trace mode: counterexample traces will be unavailable");
            if !config.properties.is_empty() {
                eprintln!("Warning: Liveness checking is disabled in no-trace mode");
            }
        }
        // Default mode: fingerprint-only with auto temp trace file (no message needed)
    }

    // Validate and create fingerprint storage if requested (mmap or disk)
    let fingerprint_storage = if let Some(capacity) = mmap_fingerprints {
        if store_states {
            bail!("--mmap-fingerprints is incompatible with --store-states");
        }
        if matches!(output_format, OutputFormat::Human) {
            println!(
                "Mmap fingerprint storage: capacity {} ({:.1} MB)",
                capacity,
                (capacity * 8) as f64 / (1024.0 * 1024.0)
            );
            if let Some(ref dir) = mmap_dir {
                println!("Mmap backing directory: {}", dir.display());
            }
        }
        let storage = FingerprintStorage::mmap(capacity, mmap_dir.clone())
            .with_context(|| "Failed to create mmap fingerprint storage")?;
        Some(std::sync::Arc::new(storage))
    } else if let Some(capacity) = disk_fingerprints {
        if store_states {
            bail!("--disk-fingerprints is incompatible with --store-states");
        }
        let disk_dir = mmap_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--disk-fingerprints requires --mmap-dir"))?;
        if matches!(output_format, OutputFormat::Human) {
            println!(
                "Disk fingerprint storage: primary capacity {} ({:.1} MB in-memory)",
                capacity,
                (capacity * 8) as f64 / (1024.0 * 1024.0)
            );
            println!("Disk backing directory: {}", disk_dir.display());
            println!("  (fingerprints will evict to disk when primary fills)");
        }
        let storage = FingerprintStorage::disk(capacity, disk_dir.clone())
            .with_context(|| "Failed to create disk fingerprint storage")?;
        Some(std::sync::Arc::new(storage))
    } else {
        if mmap_dir.is_some() {
            bail!("--mmap-dir requires --mmap-fingerprints or --disk-fingerprints");
        }
        None
    };

    // Validate and create trace file if requested
    let trace_file = if let Some(ref path) = trace_file_path {
        if store_states {
            bail!("--trace-file is incompatible with --store-states");
        }
        if matches!(output_format, OutputFormat::Human) {
            println!("Trace file: {}", path.display());
            println!("  (enables counterexample reconstruction from disk)");
        }
        let tf = TraceFile::create(path)
            .with_context(|| format!("Failed to create trace file: {}", path.display()))?;
        Some(tf)
    } else {
        None
    };

    // Create trace locations storage if mmap-trace-locations is specified
    let trace_locs_storage = if let Some(capacity) = mmap_trace_locations {
        if trace_file_path.is_none() {
            bail!("--mmap-trace-locations requires --trace-file");
        }
        if matches!(output_format, OutputFormat::Human) {
            println!("Mmap trace locations: {} capacity", capacity);
        }
        let storage = TraceLocationsStorage::mmap(capacity, mmap_dir.clone())
            .with_context(|| "Failed to create mmap trace locations storage")?;
        Some(storage)
    } else {
        None
    };

    // Create progress callback if requested (only for human output)
    let progress_callback = |progress: &Progress| {
        eprint!(
            "\rProgress: {} states, depth {}, {} transitions, {:.1} states/sec, {:.1}s elapsed",
            progress.states_found,
            progress.current_depth,
            progress.transitions,
            progress.states_per_sec,
            progress.elapsed_secs
        );
    };

    // Determine deadlock checking:
    // 1. If --no-deadlock flag is set, always disable
    // 2. If spec uses [A]_v form (stuttering allowed) AND CHECK_DEADLOCK not explicit, disable
    // 3. Otherwise use config value
    let check_deadlock = if no_deadlock {
        false
    } else if let Some(ref resolved) = resolved_spec {
        // If spec allows stuttering and CHECK_DEADLOCK not explicitly set, disable deadlock check
        if resolved.stuttering_allowed && !config.check_deadlock_explicit {
            false
        } else {
            config.check_deadlock
        }
    } else {
        config.check_deadlock
    };

    // Log checkpoint configuration (only for human output)
    if matches!(output_format, OutputFormat::Human) {
        if let Some(ref dir) = checkpoint_dir {
            println!("Checkpoint directory: {}", dir.display());
            println!("Checkpoint interval: {} seconds", checkpoint_interval);
        }
    }

    // Log resume information if resuming
    if let Some(ref dir) = resume_from {
        use tla_check::Checkpoint;
        if matches!(output_format, OutputFormat::Human) {
            println!("Resuming from checkpoint: {}", dir.display());
        }
        // Validate checkpoint exists and is loadable (preview info)
        let checkpoint = Checkpoint::load(dir)
            .with_context(|| format!("Failed to load checkpoint from {}", dir.display()))?;
        if matches!(output_format, OutputFormat::Human) {
            println!(
                "  Previous progress: {} states, {} frontier, {} transitions",
                checkpoint.metadata.stats.states_found,
                checkpoint.frontier.len(),
                checkpoint.metadata.stats.transitions
            );
        }
    }

    // Validate checkpoint/resume mode constraints
    if (checkpoint_dir.is_some() || resume_from.is_some()) && workers != 1 {
        bail!("--checkpoint and --resume are only supported with --workers 1 (sequential mode)");
    }

    // Run model checker
    let start = Instant::now();
    let (result, strategy_info) = if workers == 0 {
        // Auto mode: use adaptive checker
        // Trace file not supported in auto mode (would need to propagate to selected strategy)
        if trace_file.is_some() {
            bail!("--trace-file is only supported with --workers 1 (sequential mode)");
        }
        if trace_locs_storage.is_some() {
            bail!("--mmap-trace-locations is only supported with --workers 1 (sequential mode)");
        }
        let mut checker = AdaptiveChecker::new_with_extends(&module, &extended_modules, &config);
        // Register inline NEXT expression if present (e.g., Spec == Init /\ [][\E n: Next(n)]_vars)
        if let Some(ref resolved) = resolved_spec {
            checker.register_inline_next(resolved)?;
        }
        checker.set_deadlock_check(check_deadlock);
        checker.set_collect_coverage(show_coverage);
        // store_states enables legacy 42x-memory mode; default is fingerprint-only (#88)
        if store_states {
            checker.set_store_states(true);
        }
        // --no-trace disables auto trace file creation (maximum memory efficiency)
        if no_trace {
            checker.set_auto_create_trace_file(false);
        }
        // Pass mmap fingerprint storage if configured
        if let Some(ref storage) = fingerprint_storage {
            checker.set_fingerprint_storage(storage.clone());
        }
        // Pass fairness constraints for liveness checking
        if !resolved_fairness.is_empty() {
            checker.set_fairness(resolved_fairness.clone());
        }
        if max_states > 0 {
            checker.set_max_states(max_states);
        }
        if max_depth > 0 {
            checker.set_max_depth(max_depth);
        }
        // Only show progress for human output format
        if show_progress && matches!(output_format, OutputFormat::Human) {
            checker.set_progress_callback(Box::new(progress_callback));
        }
        let (result, analysis) = checker.check();
        let strategy_info = analysis.map(|a| {
            format!(
                "Strategy: {} (estimated {} states, branching factor {:.2})",
                a.strategy, a.estimated_states, a.avg_branching_factor
            )
        });
        (result, strategy_info)
    } else if workers == 1 {
        let mut checker = ModelChecker::new_with_extends(&module, &extended_modules, &config);
        // Register inline NEXT expression if present
        if let Some(ref resolved) = resolved_spec {
            checker.register_inline_next(resolved)?;
        }
        checker.set_deadlock_check(check_deadlock);
        checker.set_collect_coverage(show_coverage);
        // store_states enables legacy 42x-memory mode; default is fingerprint-only (#88)
        if store_states {
            checker.set_store_states(true);
        }
        // --no-trace disables auto trace file creation (maximum memory efficiency)
        if no_trace {
            checker.set_auto_create_trace_file(false);
        }
        // Pass mmap fingerprint storage if configured
        if let Some(ref storage) = fingerprint_storage {
            checker.set_fingerprint_storage(storage.clone());
        }
        // Set explicit trace file if configured (overrides auto temp trace file)
        if let Some(tf) = trace_file {
            checker.set_trace_file(tf);
        }
        // Set trace locations storage if mmap requested
        if let Some(storage) = trace_locs_storage {
            checker.set_trace_locations_storage(storage);
        }
        // Set fairness constraints extracted from SPECIFICATION formula
        if !resolved_fairness.is_empty() {
            checker.set_fairness(resolved_fairness.clone());
        }
        if max_states > 0 {
            checker.set_max_states(max_states);
        }
        if max_depth > 0 {
            checker.set_max_depth(max_depth);
        }
        // Only show progress for human output format
        if show_progress && matches!(output_format, OutputFormat::Human) {
            checker.set_progress_callback(Box::new(progress_callback));
        }
        // Set checkpoint configuration if enabled
        if let Some(ref dir) = checkpoint_dir {
            checker.set_checkpoint(dir.clone(), checkpoint_interval);
            checker.set_checkpoint_paths(
                Some(file.to_string_lossy().to_string()),
                Some(config_path.to_string_lossy().to_string()),
            );
        }
        // Either resume from checkpoint or start fresh
        let result = if let Some(ref resume_dir) = resume_from {
            checker.check_with_resume(resume_dir).with_context(|| {
                format!("Failed to resume from checkpoint: {}", resume_dir.display())
            })?
        } else {
            checker.check()
        };
        (result, None)
    } else {
        // Parallel mode - trace file not supported
        if trace_file.is_some() {
            bail!("--trace-file is only supported with --workers 1 (sequential mode)");
        }
        if trace_locs_storage.is_some() {
            bail!("--mmap-trace-locations is only supported with --workers 1 (sequential mode)");
        }
        if show_coverage {
            bail!("--coverage is only supported with --workers 0 or --workers 1");
        }
        let mut checker =
            ParallelChecker::new_with_extends(&module, &extended_modules, &config, workers);
        // Register inline NEXT expression if present
        if let Some(ref resolved) = resolved_spec {
            checker.register_inline_next(resolved)?;
        }
        checker.set_deadlock_check(check_deadlock);
        // store_states enables legacy 42x-memory mode; default is fingerprint-only (#88)
        if store_states {
            checker.set_store_states(true);
        }
        // --no-trace disables auto trace file creation (maximum memory efficiency)
        if no_trace {
            checker.set_auto_create_trace_file(false);
        }
        // Pass mmap fingerprint storage if configured
        if let Some(ref storage) = fingerprint_storage {
            checker.set_fingerprint_storage(storage.clone());
        }
        if max_states > 0 {
            checker.set_max_states(max_states);
        }
        if max_depth > 0 {
            checker.set_max_depth(max_depth);
        }
        // Only show progress for human output format
        if show_progress && matches!(output_format, OutputFormat::Human) {
            checker.set_progress_callback(Box::new(progress_callback));
        }
        (checker.check(), None)
    };
    let elapsed = start.elapsed();

    // Print strategy info if adaptive mode was used (only for human output)
    if let Some(ref info) = strategy_info {
        if matches!(output_format, OutputFormat::Human) {
            println!("{}", info);
            println!();
        }
    }

    // Clear progress line if we were showing progress (only for human output)
    if show_progress && matches!(output_format, OutputFormat::Human) {
        eprintln!();
    }

    // Report results based on output format
    match output_format {
        OutputFormat::Json | OutputFormat::JsonLines => {
            // Extract variable names from all modules (main + extended)
            // Extended modules may define state variables (e.g., HourClock defines `hr`)
            let mut variables: Vec<String> = Vec::new();

            // First collect from extended modules
            for ext_mod in &extended_modules {
                for unit in &ext_mod.units {
                    if let tla_core::ast::Unit::Variable(vars) = &unit.node {
                        for v in vars {
                            if !variables.contains(&v.node) {
                                variables.push(v.node.clone());
                            }
                        }
                    }
                }
            }

            // Then from main module (may shadow)
            for unit in &module.units {
                if let tla_core::ast::Unit::Variable(vars) = &unit.node {
                    for v in vars {
                        if !variables.contains(&v.node) {
                            variables.push(v.node.clone());
                        }
                    }
                }
            }

            // Sort for consistent output
            variables.sort();

            // Build JSON output
            let mut json_output =
                JsonOutput::new(file, Some(&config_path), &module.name.node, workers)
                    .with_spec_info(
                        config.init.as_deref(),
                        config.next.as_deref(),
                        config.invariants.clone(),
                        config.properties.clone(),
                        variables,
                    )
                    .with_check_result(&result, elapsed);

            // Add strategy info if available
            if let Some(info) = &strategy_info {
                json_output.add_info("I001", info);
            }

            // Output JSON
            let json_str = if matches!(output_format, OutputFormat::JsonLines) {
                json_output.to_json_compact().context("serialize JSON")?
            } else {
                json_output.to_json().context("serialize JSON")?
            };
            println!("{}", json_str);

            // For JSON output, return Ok for success/limit_reached, Err for errors
            match &result {
                CheckResult::Success(_) | CheckResult::LimitReached { .. } => Ok(()),
                CheckResult::InvariantViolation { .. } => bail!("Invariant violation detected"),
                CheckResult::PropertyViolation { .. } => bail!("Property violation detected"),
                CheckResult::LivenessViolation { .. } => bail!("Liveness violation detected"),
                CheckResult::Deadlock { .. } => bail!("Deadlock detected"),
                CheckResult::Error { error, .. } => bail!("Model checking failed: {}", error),
            }
        }
        OutputFormat::Human => {
            // Human-readable output (existing behavior)
            match result {
                CheckResult::Success(stats) => {
                    println!("Model checking complete: No errors found.");
                    println!();
                    println!("Statistics:");
                    println!("  States found: {}", stats.states_found);
                    println!("  Initial states: {}", stats.initial_states);
                    println!("  Transitions: {}", stats.transitions);
                    println!("  Max queue depth: {}", stats.max_queue_depth);
                    println!("  Time: {:.3}s", elapsed.as_secs_f64());
                    if !stats.detected_actions.is_empty() {
                        println!();
                        println!("Detected actions ({}):", stats.detected_actions.len());
                        for action in &stats.detected_actions {
                            println!("  {}", action);
                        }
                    }
                    if let Some(coverage) = stats.coverage.as_ref() {
                        println!();
                        println!("{}", coverage.format_report());
                    }
                    #[cfg(feature = "memory-stats")]
                    tla_check::value::memory_stats::print_stats();
                    Ok(())
                }
                CheckResult::InvariantViolation {
                    invariant,
                    trace,
                    stats,
                } => {
                    eprintln!("Error: Invariant '{}' violated!", invariant);
                    eprintln!();
                    if trace.is_empty() {
                        eprintln!("Counterexample trace: unavailable (--no-trace mode)");
                    } else {
                        match trace_format {
                            TraceFormat::Text => {
                                eprintln!("Counterexample trace ({} states):", trace.len());
                                eprintln!("{}", trace);
                            }
                            TraceFormat::Dot => {
                                eprintln!(
                                    "Counterexample trace ({} states) in DOT format:",
                                    trace.len()
                                );
                                eprintln!();
                                println!("{}", trace_to_dot(&trace, None));
                            }
                        }
                    }
                    eprintln!("Statistics:");
                    eprintln!("  States found: {}", stats.states_found);
                    eprintln!("  Initial states: {}", stats.initial_states);
                    eprintln!("  Transitions: {}", stats.transitions);
                    eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
                    bail!("Invariant violation detected");
                }
                CheckResult::PropertyViolation {
                    property,
                    trace,
                    stats,
                } => {
                    eprintln!("Error: Property '{}' violated!", property);
                    eprintln!();
                    if trace.is_empty() {
                        eprintln!("Counterexample trace: unavailable (--no-trace mode)");
                    } else {
                        match trace_format {
                            TraceFormat::Text => {
                                eprintln!("Counterexample trace ({} states):", trace.len());
                                eprintln!("{}", trace);
                            }
                            TraceFormat::Dot => {
                                eprintln!(
                                    "Counterexample trace ({} states) in DOT format:",
                                    trace.len()
                                );
                                eprintln!();
                                println!("{}", trace_to_dot(&trace, None));
                            }
                        }
                    }
                    eprintln!("Statistics:");
                    eprintln!("  States found: {}", stats.states_found);
                    eprintln!("  Initial states: {}", stats.initial_states);
                    eprintln!("  Transitions: {}", stats.transitions);
                    eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
                    bail!("Property violation detected");
                }
                CheckResult::LivenessViolation {
                    property,
                    prefix,
                    cycle,
                    stats,
                } => {
                    eprintln!("Error: Liveness property '{}' violated!", property);
                    eprintln!();
                    match trace_format {
                        TraceFormat::Text => {
                            eprintln!("Counterexample (lasso shape):");
                            eprintln!();
                            eprintln!("Prefix ({} states):", prefix.len());
                            eprintln!("{}", prefix);
                            eprintln!("Cycle ({} states):", cycle.len());
                            eprintln!("{}", cycle);
                        }
                        TraceFormat::Dot => {
                            eprintln!("Counterexample (lasso shape) in DOT format:");
                            eprintln!(
                                "  Prefix: {} states, Cycle: {} states",
                                prefix.len(),
                                cycle.len()
                            );
                            eprintln!();
                            println!("{}", liveness_trace_to_dot(&prefix, &cycle));
                        }
                    }
                    eprintln!("Statistics:");
                    eprintln!("  States found: {}", stats.states_found);
                    eprintln!("  Initial states: {}", stats.initial_states);
                    eprintln!("  Transitions: {}", stats.transitions);
                    eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
                    bail!("Liveness violation detected");
                }
                CheckResult::Deadlock { trace, stats } => {
                    eprintln!("Error: Deadlock detected!");
                    eprintln!();
                    if trace.is_empty() {
                        eprintln!("Trace to deadlock: unavailable (--no-trace mode)");
                    } else {
                        match trace_format {
                            TraceFormat::Text => {
                                eprintln!("Trace to deadlock ({} states):", trace.len());
                                eprintln!("{}", trace);
                            }
                            TraceFormat::Dot => {
                                eprintln!(
                                    "Trace to deadlock ({} states) in DOT format:",
                                    trace.len()
                                );
                                eprintln!();
                                println!("{}", trace_to_dot(&trace, None));
                            }
                        }
                    }
                    eprintln!("Statistics:");
                    eprintln!("  States found: {}", stats.states_found);
                    eprintln!("  Initial states: {}", stats.initial_states);
                    eprintln!("  Transitions: {}", stats.transitions);
                    eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
                    eprintln!();
                    eprintln!("Hint: Use --no-deadlock to disable deadlock checking");
                    bail!("Deadlock detected");
                }
                CheckResult::Error { error, stats } => {
                    eprintln!("Error: {}", error);
                    eprintln!();
                    eprintln!("Statistics:");
                    eprintln!("  States found: {}", stats.states_found);
                    eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
                    bail!("Model checking failed: {}", error);
                }
                CheckResult::LimitReached { limit_type, stats } => {
                    let limit_name = match limit_type {
                        tla_check::LimitType::States => "state",
                        tla_check::LimitType::Depth => "depth",
                    };
                    println!("Model checking stopped: {} limit reached.", limit_name);
                    println!();
                    println!("Statistics:");
                    println!("  States found: {}", stats.states_found);
                    println!("  Initial states: {}", stats.initial_states);
                    println!("  Transitions: {}", stats.transitions);
                    println!("  Max depth: {}", stats.max_depth);
                    println!("  Time: {:.3}s", elapsed.as_secs_f64());
                    println!();
                    println!("Hint: Use --max-states or --max-depth to adjust limits");
                    Ok(())
                }
            }
        }
    }
}

fn cmd_simulate(
    file: &Path,
    config_path: Option<&Path>,
    num_traces: usize,
    max_trace_length: usize,
    seed: u64,
    no_invariants: bool,
) -> Result<()> {
    // Parse the TLA+ source file
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;

    let result = lower(FileId(0), &tree);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", result.errors.len());
    }
    let module = result.module.context("lower produced no module")?;

    // Load extended modules
    let mut loader = ModuleLoader::new(file);

    // Seed the loader cache with inline modules from the main file.
    loader.seed_from_syntax_tree(&tree, file);

    let extended_module_names = match loader.load_extends(&module) {
        Ok(names) => {
            if !names.is_empty() {
                eprintln!("Loaded extended modules: {}", names.join(", "));
            }
            names
        }
        Err(e) => {
            bail!("Failed to load extended modules: {}", e);
        }
    };

    let extended_modules: Vec<_> = extended_module_names
        .iter()
        .filter_map(|name| loader.get(name).map(|l| &l.module))
        .collect();

    // Find and parse config file
    let config_path = match config_path {
        Some(p) => p.to_path_buf(),
        None => {
            let mut cfg_path = file.to_path_buf();
            cfg_path.set_extension("cfg");
            if !cfg_path.exists() {
                bail!(
                    "No config file specified and {} does not exist.\n\
                     Use --config to specify a configuration file.",
                    cfg_path.display()
                );
            }
            cfg_path
        }
    };

    let config_source = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;

    let mut config = Config::parse(&config_source).map_err(|errors| {
        for err in &errors {
            eprintln!("{}:{}: {}", config_path.display(), err.line, err.message);
        }
        anyhow::anyhow!("config parse failed with {} error(s)", errors.len())
    })?;

    // Resolve Init/Next from SPECIFICATION if needed
    let mut sim_resolved_spec: Option<tla_check::ResolvedSpec> = None;
    if (config.init.is_none() || config.next.is_none()) && config.specification.is_some() {
        let extended_syntax_trees: Vec<_> = extended_module_names
            .iter()
            .filter_map(|name| loader.get(name).map(|l| &l.syntax_tree))
            .collect();

        match resolve_spec_from_config_with_extends(&config, &tree, &extended_syntax_trees) {
            Ok(resolved) => {
                if config.init.is_none() {
                    config.init = Some(resolved.init.clone());
                }
                if config.next.is_none() {
                    config.next = Some(resolved.next.clone());
                }
                sim_resolved_spec = Some(resolved);
            }
            Err(e) => {
                bail!("Failed to resolve SPECIFICATION: {}", e);
            }
        }
    }

    // Print header
    println!("Simulating: {}", file.display());
    println!("Config: {}", config_path.display());
    if let Some(ref init) = config.init {
        println!("INIT: {}", init);
    }
    if let Some(ref next) = config.next {
        println!("NEXT: {}", next);
    }
    if !config.invariants.is_empty() && !no_invariants {
        println!("INVARIANTS: {}", config.invariants.join(", "));
    }
    println!("Traces: {}", num_traces);
    println!("Max trace length: {}", max_trace_length);
    if seed != 0 {
        println!("Seed: {}", seed);
    }
    println!();

    // Configure simulation
    let sim_config = SimulationConfig {
        num_traces,
        max_trace_length,
        seed: if seed == 0 { None } else { Some(seed) },
        check_invariants: !no_invariants,
        action_constraints: config.action_constraints.clone(),
    };

    // Run simulation
    let start = Instant::now();
    let mut checker = ModelChecker::new_with_extends(&module, &extended_modules, &config);
    // Register inline NEXT expression if present
    if let Some(ref resolved) = sim_resolved_spec {
        checker.register_inline_next(resolved)?;
    }
    let result = checker.simulate(&sim_config);
    let elapsed = start.elapsed();

    match result {
        SimulationResult::Success(stats) => {
            println!("Simulation complete: No errors found.");
            println!();
            println!("Statistics:");
            println!("  Traces generated: {}", stats.traces_generated);
            println!("  States visited: {}", stats.states_visited);
            println!("  Distinct states: {}", stats.distinct_states);
            println!("  Transitions: {}", stats.transitions);
            println!("  Max trace length: {}", stats.max_trace_length);
            println!("  Avg trace length: {:.1}", stats.avg_trace_length);
            println!("  Deadlocked traces: {}", stats.deadlocked_traces);
            println!("  Truncated traces: {}", stats.truncated_traces);
            println!("  Time: {:.3}s", elapsed.as_secs_f64());
            let states_per_sec = if elapsed.as_secs_f64() > 0.0 {
                stats.states_visited as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            println!("  States/sec: {:.0}", states_per_sec);
            Ok(())
        }
        SimulationResult::InvariantViolation {
            invariant,
            trace,
            stats,
        } => {
            eprintln!(
                "Error: Invariant '{}' violated during simulation!",
                invariant
            );
            eprintln!();
            eprintln!("Counterexample trace ({} states):", trace.len());
            eprintln!("{}", trace);
            eprintln!("Statistics:");
            eprintln!("  Traces generated: {}", stats.traces_generated);
            eprintln!("  States visited: {}", stats.states_visited);
            eprintln!("  Distinct states: {}", stats.distinct_states);
            eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
            bail!("Invariant violation detected during simulation");
        }
        SimulationResult::Error { error, stats } => {
            eprintln!("Error: {}", error);
            eprintln!();
            eprintln!("Statistics:");
            eprintln!("  Traces generated: {}", stats.traces_generated);
            eprintln!("  States visited: {}", stats.states_visited);
            eprintln!("  Time: {:.3}s", elapsed.as_secs_f64());
            bail!("Simulation failed: {}", error);
        }
    }
}

#[cfg(feature = "prove")]
fn cmd_prove(file: &Path, timeout_secs: u64, theorem_filter: Option<&str>) -> Result<()> {
    // Parse the TLA+ source file
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;

    let result = lower(FileId(0), &tree);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", result.errors.len());
    }
    let module = result.module.context("lower produced no module")?;

    // Parse theorem filter if provided
    let filter_names: Option<Vec<&str>> = theorem_filter.map(|f| f.split(',').collect());

    // Set up prover
    let mut prover = Prover::new();
    prover.set_timeout(Duration::from_secs(timeout_secs));

    // Print header
    println!("Proving theorems in: {}", file.display());
    println!("Timeout per obligation: {}s", timeout_secs);
    if let Some(ref names) = filter_names {
        println!("Theorem filter: {}", names.join(", "));
    }
    println!();

    // Run prover
    let start = Instant::now();
    let result = prover
        .check_module(&module)
        .map_err(|e| anyhow::anyhow!("Proof error: {}", e))?;
    let elapsed = start.elapsed();

    // Filter results if needed
    let theorems: Vec<_> = if let Some(ref names) = filter_names {
        result
            .theorems
            .iter()
            .filter(|t| names.iter().any(|n| *n == t.name))
            .collect()
    } else {
        result.theorems.iter().collect()
    };

    // Report results
    report_proof_results(&result, &theorems, elapsed)
}

#[cfg(feature = "prove")]
fn report_proof_results(
    result: &ModuleResult,
    theorems: &[&tla_prove::TheoremResult],
    elapsed: Duration,
) -> Result<()> {
    let mut all_proved = true;

    for thm in theorems {
        let status = if thm.is_proved() {
            "PROVED"
        } else if thm.failed_count() > 0 {
            all_proved = false;
            "FAILED"
        } else {
            all_proved = false;
            "UNKNOWN"
        };

        println!(
            "THEOREM {}: {} ({} obligations, {:.3}s)",
            thm.name,
            status,
            thm.obligations.len(),
            thm.duration.as_secs_f64()
        );

        // Show obligation details for non-proved theorems
        if !thm.is_proved() {
            for (i, obl) in thm.obligations.iter().enumerate() {
                let obl_status = match &obl.outcome {
                    ProofOutcome::Proved => "proved",
                    ProofOutcome::Failed { .. } => "FAILED",
                    ProofOutcome::Unknown { .. } => "unknown",
                };
                println!(
                    "  Obligation {}: {} (backend: {}, {:.3}s)",
                    i + 1,
                    obl_status,
                    obl.backend,
                    obl.duration.as_secs_f64()
                );

                // Show counterexample if available
                if let ProofOutcome::Failed {
                    message,
                    counterexample,
                } = &obl.outcome
                {
                    println!("    {}", message);
                    if let Some(ce) = counterexample {
                        for (var, val) in ce {
                            println!("    {} = {}", var, val);
                        }
                    }
                }

                // Show reason for unknown
                if let ProofOutcome::Unknown { reason } = &obl.outcome {
                    println!("    Reason: {}", reason);
                }
            }
        }
    }

    println!();
    println!("Summary:");
    println!("  Module: {}", result.name);
    println!(
        "  Theorems: {} proved, {} failed, {} unknown",
        theorems.iter().filter(|t| t.is_proved()).count(),
        theorems.iter().filter(|t| t.failed_count() > 0).count(),
        theorems
            .iter()
            .filter(|t| !t.is_proved() && t.failed_count() == 0)
            .count()
    );
    println!(
        "  Obligations: {} total, {} proved, {} failed, {} unknown",
        result.total_obligations(),
        result.proved_count(),
        result.failed_count(),
        result.unknown_count()
    );
    println!("  Cache hits: {}", prover_cache_note());
    println!("  Time: {:.3}s", elapsed.as_secs_f64());

    if all_proved && !theorems.is_empty() {
        println!();
        println!("All theorems proved successfully!");
        Ok(())
    } else if theorems.is_empty() {
        println!();
        println!("No theorems found in module.");
        Ok(())
    } else {
        println!();
        bail!("Some theorems could not be proved");
    }
}

#[cfg(feature = "prove")]
fn prover_cache_note() -> &'static str {
    "(check prover.cache_size() for actual count)"
}

fn cmd_codegen(file: &Path, output: Option<&Path>, kani: bool, proptest: bool) -> Result<()> {
    // Parse the TLA+ source file
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;

    let result = lower(FileId(0), &tree);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", result.errors.len());
    }
    let module = result.module.context("lower produced no module")?;

    // Configure code generation
    let options = CodeGenOptions {
        module_name: None, // Use module name from TLA+ spec
        generate_kani: kani,
        generate_proptest: proptest,
    };

    // Generate Rust code
    let rust_code = generate_rust(&module, &options).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Write output
    match output {
        Some(path) => {
            std::fs::write(path, &rust_code)
                .with_context(|| format!("write {}", path.display()))?;
            eprintln!("Generated Rust code written to: {}", path.display());
        }
        None => {
            print!("{}", rust_code);
        }
    }

    Ok(())
}
