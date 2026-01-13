// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // CLI commands don't need must_use
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::use_self)] // Self vs TypeName - style preference
#![allow(clippy::unused_self)] // Some methods keep self for API compatibility
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::missing_errors_doc)] // CLI errors are obvious
#![allow(clippy::uninlined_format_args)] // Named args are clearer
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::similar_names)] // e.g., args/arg, path/paths
#![allow(clippy::redundant_closure_for_method_calls)] // Minor style issue
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
#![allow(clippy::needless_pass_by_value)] // Ownership semantics may be intentional
#![allow(clippy::too_many_lines)] // CLI handlers may be long
#![allow(clippy::wildcard_imports)] // Re-exports use wildcard pattern
#![allow(clippy::trivially_copy_pass_by_ref)] // &BackendId is API consistency
#![allow(clippy::needless_raw_string_hashes)] // Raw strings for templates
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::cast_precision_loss)] // usize to f64 is intentional
#![allow(clippy::single_match_else)] // match is clearer for some patterns
#![allow(clippy::format_push_string)] // Common pattern in CLI output
#![allow(clippy::struct_excessive_bools)] // CLI flags may be bools
#![allow(clippy::module_name_repetitions)] // cli::CliArgs is clear
#![allow(clippy::match_wildcard_for_single_variants)] // Explicit wildcard is clearer
#![allow(clippy::or_fun_call)] // Style preference, ok_or pattern is clear
#![allow(clippy::semicolon_if_nothing_returned)] // Style preference
#![allow(clippy::manual_let_else)] // if let is clearer in some contexts

//! DashProve CLI library
//!
//! This module exposes the CLI argument types for testing and benchmarking.

/// CLI argument definitions
pub mod cli {
    use clap::{Parser, Subcommand};

    #[derive(Parser)]
    #[command(name = "dashprove")]
    #[command(about = "Unified AI-native verification platform")]
    #[command(version = env!("CARGO_PKG_VERSION"))]
    #[command(after_help = "\
EXAMPLES:
    Verify a specification:
      dashprove verify spec.usl
      dashprove verify spec.usl --backends lean,tla+
      dashprove verify spec.usl --learn --suggest

    Verify Rust code against contracts:
      dashprove verify-code --code src/lib.rs --spec contracts.usl

    Export to backend format:
      dashprove export spec.usl --target lean -o spec.lean

    Analyze counterexamples:
      dashprove explain counterexample.json
      dashprove visualize counterexample.json --format html
      dashprove analyze counterexample.json suggest
      dashprove cluster cx1.json cx2.json cx3.json

    Proof corpus operations:
      dashprove corpus stats
      dashprove search \"termination for recursive functions\"
      dashprove prove spec.usl --hints

    See 'dashprove help <command>' for more information on a specific command.")]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Commands,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        /// Verify a specification against formal verification backends
        #[command(after_help = "\
EXAMPLES:
    Basic verification:
      dashprove verify spec.usl

    Use specific backends:
      dashprove verify spec.usl --backends lean,tla+
      dashprove verify spec.usl --backends kani

    With learning and suggestions:
      dashprove verify spec.usl --learn --suggest

    Incremental verification:
      dashprove verify spec.usl --incremental --since HEAD~1

    ML-based backend selection:
      dashprove verify spec.usl --ml --ml-model ~/.dashprove/strategy_model.json --ml-confidence 0.6

    Verbose output (shows what's happening):
      dashprove verify spec.usl --verbose")]
        Verify {
            /// Path to USL specification file
            path: String,
            /// Backends to use (comma-separated: lean,tla+,kani,alloy)
            #[arg(long)]
            backends: Option<String>,
            /// Timeout in seconds for each verification
            #[arg(long, default_value = "120")]
            timeout: u64,
            /// Skip health checks for faster startup
            #[arg(long)]
            skip_health_check: bool,
            /// Record verification results to learning corpus
            #[arg(long)]
            learn: bool,
            /// Directory for learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Show tactic suggestions before verification
            #[arg(long)]
            suggest: bool,
            /// Enable incremental verification (only re-verify changed properties)
            #[arg(long)]
            incremental: bool,
            /// Git ref to compare against for incremental verification (e.g., HEAD~1, main)
            #[arg(long, requires = "incremental")]
            since: Option<String>,
            /// Enable ML-based backend selection (requires model or uses default)
            #[arg(long)]
            ml: bool,
            /// Path to ML strategy model (JSON saved via dashprove-ai)
            #[arg(long)]
            ml_model: Option<String>,
            /// Minimum confidence threshold (0.0-1.0) for ML predictions
            #[arg(long, default_value = "0.5")]
            ml_confidence: f64,
            /// Show verbose output explaining each step
            #[arg(short, long)]
            verbose: bool,
        },
        /// Export specification to backend format
        #[command(after_help = "\
EXAMPLES:
    Export to LEAN 4:
      dashprove export spec.usl --target lean
      dashprove export spec.usl --target lean -o spec.lean

    Export to TLA+:
      dashprove export spec.usl --target tla+ -o spec.tla

    Export to Coq:
      dashprove export spec.usl --target coq -o spec.v

    Export to Isabelle/HOL:
      dashprove export spec.usl --target isabelle -o spec.thy

    Export to Dafny:
      dashprove export spec.usl --target dafny -o spec.dfy

    Export to SMT-LIB2:
      dashprove export spec.usl --target smtlib -o spec.smt2
      dashprove export spec.usl --target smtlib:QF_LIA -o spec.smt2

SUPPORTED TARGETS:
    lean, tla+, kani, alloy, coq, isabelle, dafny, smtlib
    For SMT-LIB with specific logic: smtlib:QF_LIA, smtlib:QF_BV, etc.")]
        Export {
            /// Path to USL specification file
            path: String,
            /// Target backend (lean, tla+, kani, alloy, coq, isabelle, dafny, smtlib)
            #[arg(long)]
            target: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
        },
        /// Check available backends and their health status
        #[command(after_help = "\
EXAMPLES:
    Check all backends:
      dashprove backends

OUTPUT:
    Shows each backend's availability, version, and health status.
    A healthy backend can execute verification tasks.")]
        Backends,
        /// Check installation status of all verification tools (200+ backends)
        #[command(
            name = "check-tools",
            after_help = "\
EXAMPLES:
    Check all tools:
      dashprove check-tools

    Check only missing tools:
      dashprove check-tools --missing

    Check specific category:
      dashprove check-tools --category rust
      dashprove check-tools --category fuzzer
      dashprove check-tools --category llm-eval

    Show install hints:
      dashprove check-tools --verbose

    Output as JSON:
      dashprove check-tools --format json

CATEGORIES:
    theorem, model, neural, prob, security, rust, smt, sat, sanitizer,
    concurrency, fuzz, pbt, static, ai-opt, compress, data,
    fairness, interpret, guardrails, eval, hallucination

OUTPUT:
    Shows installation status of all 113 verification tools across categories.
    Use --verbose to see install commands for missing tools."
        )]
        CheckTools {
            /// Show verbose output with install hints
            #[arg(short, long)]
            verbose: bool,
            /// Filter to specific category
            #[arg(short, long)]
            category: Option<String>,
            /// Show only missing tools
            #[arg(short, long)]
            missing: bool,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
        /// Explain a counterexample from a verification result
        #[command(after_help = "\
EXAMPLES:
    Explain from JSON file (backend auto-detected):
      dashprove explain counterexample.json

    Explain plain text output:
      dashprove explain trace.txt --backend tla+

    The explain command parses counterexample output and provides:
    - Variable bindings that led to the violation
    - Execution trace showing how the violation occurred
    - Suggestions for fixing the specification or code")]
        Explain {
            /// Path to counterexample JSON file (from a previous verification)
            path: String,
            /// Backend that produced the counterexample (lean, tla+, kani, alloy)
            #[arg(long)]
            backend: Option<String>,
        },
        /// Proof corpus operations (stats, search, history)
        #[command(after_help = "\
EXAMPLES:
    View corpus statistics:
      dashprove corpus stats

    Search for similar proofs:
      dashprove corpus search spec.usl
      dashprove corpus cx-search counterexample.json

    View history and compare periods:
      dashprove corpus history --corpus counterexamples
      dashprove corpus compare --baseline-from 2024-01-01 --baseline-to 2024-01-31 \\
        --compare-from 2024-02-01 --compare-to 2024-02-28")]
        Corpus {
            #[command(subcommand)]
            action: CorpusAction,
        },
        /// Search proof corpus for similar proofs (text query)
        #[command(after_help = "\
EXAMPLES:
    Search by concept:
      dashprove search \"termination for recursive functions\"
      dashprove search \"mutex invariant\" -n 5

    The search uses semantic similarity to find relevant proofs
    from the learning corpus that may help with your specification.")]
        Search {
            /// Text query (e.g., "termination for recursive functions")
            query: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Number of results to return
            #[arg(short = 'n', long, default_value = "10")]
            limit: usize,
        },
        /// Interactive proof mode with step-by-step guidance
        #[command(after_help = "\
EXAMPLES:
    Start interactive proof session:
      dashprove prove spec.usl
      dashprove prove spec.usl --hints

INTERACTIVE COMMANDS:
    verify <property>  - Verify a specific property
    hint               - Get a hint for the current goal
    tactics            - Show available tactics
    quit               - Exit interactive mode")]
        Prove {
            /// Path to USL specification file
            path: String,
            /// Show hints and suggestions during proof
            #[arg(long)]
            hints: bool,
        },
        /// Generate runtime monitors from specifications
        #[command(after_help = "\
EXAMPLES:
    Generate Rust monitor with assertions:
      dashprove monitor spec.usl --target rust --assertions

    Generate TypeScript monitor with logging:
      dashprove monitor spec.usl --target typescript --logging -o monitor.ts

    Generate Python monitor with metrics:
      dashprove monitor spec.usl --target python --metrics -o monitor.py

    Combine options:
      dashprove monitor spec.usl --assertions --logging --metrics")]
        Monitor {
            /// Path to USL specification file
            path: String,
            /// Target language (rust, typescript, python)
            #[arg(long, default_value = "rust")]
            target: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Generate assertions (panics on violation)
            #[arg(long)]
            assertions: bool,
            /// Generate logging on property checks
            #[arg(long)]
            logging: bool,
            /// Generate metrics/counters for property checks
            #[arg(long)]
            metrics: bool,
        },
        /// Visualize a counterexample as Mermaid, DOT, or HTML
        #[command(after_help = "\
EXAMPLES:
    Generate interactive HTML visualization:
      dashprove visualize counterexample.json --format html -o trace.html

    Generate Mermaid diagram (for markdown):
      dashprove visualize counterexample.json --format mermaid

    Generate DOT graph (for Graphviz):
      dashprove visualize counterexample.json --format dot -o trace.dot
      dot -Tpng trace.dot -o trace.png")]
        Visualize {
            /// Path to counterexample JSON file
            path: String,
            /// Output format (mermaid, dot, html)
            #[arg(long, default_value = "html")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Title for HTML output
            #[arg(long)]
            title: Option<String>,
        },
        /// Analyze a counterexample trace for patterns, compression, or minimization
        #[command(after_help = "\
EXAMPLES:
    Get suggestions for understanding the counterexample:
      dashprove analyze counterexample.json suggest

    Compress repeating patterns:
      dashprove analyze counterexample.json compress --format html

    Detect actor interleavings:
      dashprove analyze counterexample.json interleavings

    Minimize the trace:
      dashprove analyze counterexample.json minimize --max-states 10

    Compare two counterexamples:
      dashprove analyze cx1.json diff cx2.json --format html")]
        Analyze {
            /// Path to counterexample JSON file
            path: String,
            /// Analysis action to perform
            #[command(subcommand)]
            action: AnalyzeAction,
        },
        /// Cluster multiple counterexamples to identify common failure patterns
        #[command(after_help = "\
EXAMPLES:
    Cluster counterexamples with default threshold:
      dashprove cluster cx1.json cx2.json cx3.json

    Cluster with stricter similarity:
      dashprove cluster cx1.json cx2.json --threshold 0.9

    Generate HTML clustering report:
      dashprove cluster *.json --format html -o clusters.html

    The clustering identifies common patterns across counterexamples
    to help understand systematic issues in specifications.")]
        Cluster {
            /// Paths to counterexample JSON files (at least 2 required)
            #[arg(required = true, num_args = 2..)]
            paths: Vec<String>,
            /// Similarity threshold for clustering (0.0 to 1.0, higher = more similar)
            #[arg(long, default_value = "0.7")]
            threshold: f64,
            /// Output format (text, mermaid, html, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Title for HTML output
            #[arg(long)]
            title: Option<String>,
        },
        /// Show detailed help on a topic (usl, backends, counterexamples, learning)
        #[command(after_help = "\
TOPICS:
    usl             - Unified Specification Language syntax and semantics
    backends        - Available verification backends and their capabilities
    counterexamples - Understanding and analyzing counterexamples
    learning        - The proof learning system and corpus
    properties      - Types of properties (theorems, invariants, contracts, etc.)

EXAMPLES:
    dashprove topics
    dashprove topics usl
    dashprove topics backends")]
        Topics {
            /// Topic to get help on (usl, backends, counterexamples, learning, properties)
            topic: Option<String>,
        },
        /// Verify Rust code against USL contracts using Kani
        #[command(
            name = "verify-code",
            after_help = "\
EXAMPLES:
    Verify a Rust file against a USL spec:
      dashprove verify-code --code src/lib.rs --spec contracts.usl

    Verify code from stdin:
      cat src/lib.rs | dashprove verify-code --spec contracts.usl

    With verbose output:
      dashprove verify-code --code src/lib.rs --spec contracts.usl --verbose

    The verify-code command uses Kani to verify that Rust code satisfies
    the contracts defined in a USL specification. The USL spec must contain
    contract properties that can be compiled to Kani proof harnesses.

NOTE:
    This command requires Kani to be installed. Install with:
      cargo install --locked kani-verifier
      cargo kani setup"
        )]
        VerifyCode {
            /// Path to Rust code file (reads from stdin if not specified)
            #[arg(long)]
            code: Option<String>,
            /// Path to USL specification file containing contracts
            #[arg(long)]
            spec: String,
            /// Timeout in seconds for Kani verification
            #[arg(long, default_value = "300")]
            timeout: u64,
            /// Show verbose output explaining each step
            #[arg(short, long)]
            verbose: bool,
        },
        /// Train an ML model for strategy prediction from the proof corpus
        #[command(after_help = "\
EXAMPLES:
    Train with default settings:
      dashprove train

    Train with custom parameters:
      dashprove train --learning-rate 0.05 --epochs 50

    Train with early stopping (recommended for larger datasets):
      dashprove train --early-stopping --patience 5 --epochs 100

    Train with checkpointing (save model snapshots during training):
      dashprove train --checkpoint --checkpoint-dir ~/.dashprove/checkpoints
      dashprove train --checkpoint --checkpoint-interval 10 --keep-best 3

    Resume training from a checkpoint:
      dashprove train --resume ~/.dashprove/checkpoints/checkpoint_epoch_50.json

    Train and save to custom path:
      dashprove train --output ~/models/my_strategy.json

    Train with verbose output:
      dashprove train --verbose

PREREQUISITES:
    Before training, build a proof corpus by running verifications with --learn:
      dashprove verify spec.usl --learn

    The larger and more diverse the corpus, the better the trained model.

EARLY STOPPING:
    Use --early-stopping to automatically stop training when validation loss
    stops improving. This prevents overfitting and saves training time.
    The best model weights are automatically restored when stopped early.

CHECKPOINTING:
    Use --checkpoint to save model snapshots during training. This allows:
    - Recovery from interrupted training sessions
    - Keeping the best N models by validation loss
    - Resuming training from any saved checkpoint
    Checkpoints are saved on validation improvement and optionally at intervals.

LR SCHEDULING:
    Use --lr-scheduler to adjust learning rate during training:
      constant  - Fixed learning rate (default)
      step      - Reduce by gamma every lr-step-size epochs
      exp       - Exponential decay: lr = initial * gamma^epoch
      cosine    - Cosine annealing between initial and lr-min
      plateau   - Reduce when validation loss plateaus
      warmup    - Linear warmup followed by decay

USAGE:
    After training, use the model for verification:
      dashprove verify spec.usl --ml --ml-model ~/.dashprove/strategy_model.json")]
        Train {
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output path for the trained model (default: DATA_DIR/strategy_model.json)
            #[arg(short, long)]
            output: Option<String>,
            /// Learning rate for training (0.001 to 0.5)
            #[arg(long, default_value = "0.01")]
            learning_rate: f64,
            /// Number of training epochs (more epochs = longer training but potentially better results)
            #[arg(long, default_value = "20")]
            epochs: usize,
            /// Show verbose output with training details
            #[arg(short, long)]
            verbose: bool,
            /// Enable early stopping to prevent overfitting
            #[arg(long)]
            early_stopping: bool,
            /// Patience for early stopping: epochs without improvement before stopping
            #[arg(long, default_value = "5")]
            patience: usize,
            /// Minimum improvement in validation loss to reset patience
            #[arg(long, default_value = "0.001")]
            min_delta: f64,
            /// Fraction of data to use for validation (0.1 to 0.5)
            #[arg(long, default_value = "0.2")]
            validation_split: f64,
            /// Learning rate scheduler (constant, step, exp, cosine, plateau, warmup)
            #[arg(long, default_value = "constant")]
            lr_scheduler: String,
            /// Step size for step scheduler (epochs between reductions)
            #[arg(long, default_value = "10")]
            lr_step_size: usize,
            /// Decay factor (gamma) for step/exponential schedulers (0.1 to 0.99)
            #[arg(long, default_value = "0.5")]
            lr_gamma: f64,
            /// Minimum learning rate for cosine/plateau schedulers
            #[arg(long, default_value = "0.0001")]
            lr_min: f64,
            /// Warmup epochs for warmup_decay scheduler
            #[arg(long, default_value = "5")]
            lr_warmup_epochs: usize,
            /// Enable model checkpointing during training
            #[arg(long)]
            checkpoint: bool,
            /// Directory to save checkpoints (default: DATA_DIR/checkpoints)
            #[arg(long)]
            checkpoint_dir: Option<String>,
            /// Save checkpoint every N epochs (0 = only save on improvement)
            #[arg(long, default_value = "0")]
            checkpoint_interval: usize,
            /// Keep only the N best checkpoints by validation loss (0 = keep all)
            #[arg(long, default_value = "3")]
            keep_best: usize,
            /// Resume training from a checkpoint file
            #[arg(long)]
            resume: Option<String>,
        },
        /// Automatically tune hyperparameters for ML model training
        #[command(
            name = "tune",
            after_help = "\
EXAMPLES:
    Grid search with default settings:
      dashprove tune --method grid

    Random search with 20 iterations:
      dashprove tune --method random --iterations 20

    Bayesian optimization (most efficient):
      dashprove tune --method bayesian --iterations 25
      dashprove tune --method bayesian --initial-samples 5 --kappa 2.5

    Grid search with custom learning rates:
      dashprove tune --method grid --lr-values 0.001,0.01,0.05

    Random search with custom ranges:
      dashprove tune --method random --lr-min 0.001 --lr-max 0.1 --epochs-min 20 --epochs-max 100

    Use cross-validation for more robust results:
      dashprove tune --method bayesian --cv-folds 5

    Save and use the best model:
      dashprove tune --method bayesian --output tuned_model.json
      dashprove verify spec.usl --ml --ml-model tuned_model.json

SEARCH METHODS:
    grid     - Evaluates all combinations of specified hyperparameter values.
               Best for small search spaces with known good ranges.

    random   - Samples random configurations from specified ranges.
               More efficient for large search spaces; often finds good
               configurations faster than grid search.

    bayesian - Uses Gaussian Process to model the objective function.
               Most sample-efficient method; balances exploration and
               exploitation using Upper Confidence Bound acquisition.

OUTPUT:
    Displays all evaluated configurations ranked by validation loss.
    The best configuration is automatically used to train and save the model.
    Use --verbose to see detailed progress during search."
        )]
        Tune {
            /// Search method (grid, random, bayesian)
            #[arg(long, default_value = "bayesian")]
            method: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output path for the tuned model (default: DATA_DIR/strategy_model.json)
            #[arg(short, long)]
            output: Option<String>,
            /// Number of iterations to try (for random/bayesian search)
            #[arg(long, default_value = "25")]
            iterations: usize,
            /// Random seed for reproducibility
            #[arg(long, default_value = "42")]
            seed: u64,
            /// Learning rates to try (comma-separated, for grid search)
            #[arg(long, default_value = "0.001,0.01,0.05,0.1")]
            lr_values: String,
            /// Epoch counts to try (comma-separated, for grid search)
            #[arg(long, default_value = "50,100")]
            epoch_values: String,
            /// Minimum learning rate (for random/bayesian search)
            #[arg(long, default_value = "0.0001")]
            lr_min: f64,
            /// Maximum learning rate (for random/bayesian search)
            #[arg(long, default_value = "0.5")]
            lr_max: f64,
            /// Minimum epochs (for random/bayesian search)
            #[arg(long, default_value = "20")]
            epochs_min: usize,
            /// Maximum epochs (for random/bayesian search)
            #[arg(long, default_value = "200")]
            epochs_max: usize,
            /// Number of initial random samples before using GP (for bayesian search)
            #[arg(long, default_value = "5")]
            initial_samples: usize,
            /// Exploration-exploitation trade-off parameter (for bayesian search, higher = more exploration)
            #[arg(long, default_value = "2.576")]
            kappa: f64,
            /// Number of cross-validation folds (0 = no CV, use simple validation split)
            #[arg(long, default_value = "0")]
            cv_folds: usize,
            /// Show verbose output with progress
            #[arg(short, long)]
            verbose: bool,
        },
        /// Combine multiple trained models into an ensemble
        #[command(
            name = "ensemble",
            after_help = "\
EXAMPLES:
  Combine two tuned models with equal weight:
    dashprove ensemble --models tuned_a.json,tuned_b.json

  Weighted ensemble favoring the first model:
    dashprove ensemble --models best.json,robust.json --weights 0.7,0.3 --method weighted

  Save ensemble to a custom location:
    dashprove ensemble --models m1.json,m2.json --output ~/.dashprove/ensemble.json"
        )]
        Ensemble {
            /// Paths to model files (comma-separated or repeated)
            #[arg(long, value_delimiter = ',', required = true)]
            models: Vec<String>,
            /// Optional comma-separated weights matching model order
            #[arg(long)]
            weights: Option<String>,
            /// Aggregation method (soft|weighted)
            #[arg(long, default_value = "soft")]
            method: String,
            /// Output path for the ensemble model
            #[arg(short, long)]
            output: Option<String>,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Show verbose output
            #[arg(long)]
            verbose: bool,
        },
        /// Run bisimulation equivalence check between oracle and subject
        #[command(
            name = "bisim",
            after_help = "\
EXAMPLES:
    Basic bisimulation check:
      dashprove bisim --oracle ./oracle --subject ./subject --inputs tests.json

    Using recorded traces as oracle:
      dashprove bisim --oracle traces/ --recorded-traces --subject ./subject --inputs tests.json

    With custom threshold and ignore timing:
      dashprove bisim --oracle ./oracle --subject ./subject --inputs tests.json \\
        --threshold 0.9 --ignore-timing

    Handle nondeterministic behavior:
      dashprove bisim --oracle ./oracle --subject ./subject --inputs tests.json \\
        --nondeterminism retry

    Verbose output:
      dashprove bisim --oracle ./oracle --subject ./subject --inputs tests.json --verbose

NONDETERMINISM STRATEGIES:
    strict     - Require exact match (default)
    retry      - Retry tests up to 3 times on mismatch
    statistical - Run multiple samples and compute confidence interval
    ignore     - Ignore nondeterministic differences

INPUT FILE FORMAT:
    The inputs file should be a JSON array of test cases:
    [
      {\"id\": \"test1\", \"data\": {\"input\": \"hello\"}},
      {\"id\": \"test2\", \"data\": {\"input\": \"world\"}}
    ]"
        )]
        Bisim {
            /// Path to oracle binary or trace directory
            #[arg(long)]
            oracle: String,
            /// Path to subject binary to test
            #[arg(long)]
            subject: String,
            /// Path to JSON file containing test inputs
            #[arg(long)]
            inputs: Option<String>,
            /// Use recorded traces as oracle (oracle path is a directory)
            #[arg(long)]
            recorded_traces: bool,
            /// Timeout in seconds for each test
            #[arg(long, default_value = "60")]
            timeout: u64,
            /// Similarity threshold (0.0 to 1.0)
            #[arg(long, default_value = "0.95")]
            threshold: f64,
            /// Ignore timing differences in outputs
            #[arg(long)]
            ignore_timing: bool,
            /// Nondeterminism handling strategy (strict, retry, statistical, ignore)
            #[arg(long, default_value = "strict")]
            nondeterminism: String,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
        /// Verify an execution trace against a TLA+ spec and/or invariants
        #[command(
            name = "verify-trace",
            after_help = "\
EXAMPLES:
    Verify a trace against a TLA+ specification:
      dashprove verify-trace trace.json --spec agent.tla

    Check invariants on a trace:
      dashprove verify-trace trace.json --invariants value_positive,counter_monotonic

    Check liveness properties:
      dashprove verify-trace trace.json --liveness terminates,progress

    Full verification with all options:
      dashprove verify-trace trace.json --spec agent.tla \\
        --invariants positive,bounded --liveness terminates --verbose

    Output as JSON:
      dashprove verify-trace trace.json --spec agent.tla --format json

TRACE FILE FORMAT:
    The trace file should be a JSON file with ExecutionTrace format:
    {
      \"initial_state\": {\"counter\": 0},
      \"transitions\": [
        {\"from_state\": {...}, \"event\": \"increment\", \"to_state\": {...}},
        ...
      ],
      \"final_state\": {\"counter\": 10}
    }

BUILT-IN INVARIANTS:
    positive, non_negative  - Check field 'value' >= 0
    bounded                 - Check field 'value' in [0, 1000]
    monotonic              - Check field 'value' monotonically increasing
    <field>_positive       - Check specific field is positive
    <field>_monotonic      - Check specific field is monotonic

BUILT-IN LIVENESS:
    terminates, done       - Check 'done'/'terminated'/'complete' becomes true
    progress               - Check that state changes occur
    <field>_eventually     - Check specific boolean field becomes true"
        )]
        VerifyTrace {
            /// Path to execution trace JSON file
            trace: String,
            /// Path to TLA+ specification file (optional)
            #[arg(long)]
            spec: Option<String>,
            /// Invariants to check (comma-separated)
            #[arg(long, value_delimiter = ',')]
            invariants: Vec<String>,
            /// Liveness properties to check (comma-separated)
            #[arg(long, value_delimiter = ',')]
            liveness: Vec<String>,
            /// Timeout in seconds for TLA+ verification
            #[arg(long, default_value = "60")]
            timeout: u64,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
        /// Generate model-based tests from a TLA+ specification
        #[command(
            name = "mbt",
            after_help = "\
EXAMPLES:
    Generate tests with state coverage:
      dashprove mbt spec.tla --coverage state

    Generate tests with transition coverage:
      dashprove mbt spec.tla --coverage transition

    Generate boundary value tests:
      dashprove mbt spec.tla --coverage boundary

    Combined coverage (default):
      dashprove mbt spec.tla

    Generate Rust test code:
      dashprove mbt spec.tla --format rust -o tests.rs

    Generate Python test code:
      dashprove mbt spec.tla --format python -o test_spec.py

    Limit exploration:
      dashprove mbt spec.tla --max-states 1000 --max-depth 20

COVERAGE STRATEGIES:
    state       - Generate tests to visit all reachable states
    transition  - Generate tests to exercise all transitions
    boundary    - Generate tests at variable domain boundaries
    combined    - Apply all strategies (default)
    random      - Generate random walk tests

OUTPUT FORMATS:
    json     - JSON test case definitions (default)
    rust     - Rust test code with assertions
    python   - Python/pytest test code
    text     - Plain text report
    markdown - Markdown documentation"
        )]
        Mbt {
            /// Path to TLA+ specification file
            spec: String,
            /// Coverage strategy (state, transition, boundary, combined, random)
            #[arg(long, default_value = "combined")]
            coverage: String,
            /// Output format (json, rust, python, text, markdown)
            #[arg(long, default_value = "json")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Maximum states to explore
            #[arg(long, default_value = "10000")]
            max_states: usize,
            /// Maximum exploration depth
            #[arg(long, default_value = "100")]
            max_depth: usize,
            /// Maximum test length
            #[arg(long, default_value = "20")]
            max_test_length: usize,
            /// Maximum number of tests to generate
            #[arg(long, default_value = "100")]
            max_tests: usize,
            /// Random seed for reproducibility
            #[arg(long)]
            seed: Option<u64>,
            /// Exploration timeout in seconds
            #[arg(long, default_value = "60")]
            timeout: u64,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
        /// Run MIRI to detect undefined behavior in Rust code
        #[command(
            name = "miri",
            after_help = "\
EXAMPLES:
    Run MIRI on a project:
      dashprove miri .
      dashprove miri /path/to/project

    Run MIRI with test filter:
      dashprove miri . --test-filter test_foo

    Run with strict checking:
      dashprove miri . --track-raw-pointers

    Disable isolation (for tests that need file I/O):
      dashprove miri . --disable-isolation

    Generate harness for a specific function:
      dashprove miri src/lib.rs --generate-harness foo -o tests/miri_test.rs

    Setup MIRI sysroot:
      dashprove miri --setup-only

    Output as JSON:
      dashprove miri . --format json -o results.json

CHECKING OPTIONS:
    --disable-isolation     Allow file I/O and other system calls
    --skip-stacked-borrows  Skip stacked borrows checking (faster)
    --skip-data-races       Skip data race detection
    --track-raw-pointers    Enable raw pointer tracking (thorough)
    --seed N                Use deterministic execution with seed N

NOTE:
    Requires MIRI to be installed. Install with:
      rustup +nightly component add miri
      cargo +nightly miri setup"
        )]
        Miri {
            /// Path to project directory or Cargo.toml
            path: String,
            /// Test filter pattern (passed to cargo test)
            #[arg(long)]
            test_filter: Option<String>,
            /// Timeout in seconds for MIRI execution
            #[arg(long, default_value = "300")]
            timeout: u64,
            /// Disable MIRI isolation (allows file I/O)
            #[arg(long)]
            disable_isolation: bool,
            /// Skip stacked borrows checking
            #[arg(long)]
            skip_stacked_borrows: bool,
            /// Skip data race detection
            #[arg(long)]
            skip_data_races: bool,
            /// Track raw pointers for more thorough checking
            #[arg(long)]
            track_raw_pointers: bool,
            /// Seed for deterministic execution
            #[arg(long)]
            seed: Option<u64>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Generate harness for a specific function
            #[arg(long)]
            generate_harness: Option<String>,
            /// Only run MIRI setup (don't run tests)
            #[arg(long)]
            setup_only: bool,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
        /// Get expert recommendations from the knowledge base
        #[command(
            name = "expert",
            after_help = "\
EXAMPLES:
    Get backend recommendation for a specification:
      dashprove expert backend --spec spec.usl
      dashprove expert backend --property-types safety,liveness --code-lang rust

    Explain a verification error:
      dashprove expert error --message \"type mismatch\" --backend lean4
      dashprove expert error --file error.log --backend kani

    Get tactic suggestions for a proof goal:
      dashprove expert tactic --goal \"forall n, n = n\" --backend lean4
      dashprove expert tactic --goal \"Init => Inv\" --backend tlaplus

    Get compilation guidance for a backend:
      dashprove expert compile --spec spec.usl --backend lean4
      dashprove expert compile --spec spec.usl --backend tlaplus

The expert commands use the RAG knowledge base to provide intelligent
recommendations based on documentation, research papers, and examples."
        )]
        Expert {
            #[command(subcommand)]
            action: ExpertAction,
        },
        /// View and manage backend reputation data
        #[command(
            name = "reputation",
            after_help = "\
EXAMPLES:
    View reputation statistics:
      dashprove reputation stats
      dashprove reputation stats --domains

    View reputation for specific backend:
      dashprove reputation stats --backend lean4

    Export/import reputation data:
      dashprove reputation export reputation.json
      dashprove reputation import reputation.json --merge

    Reset reputation data:
      dashprove reputation reset

Reputation tracking helps the system learn which backends perform
best for different types of properties. Enable learning with:
  dashprove verify spec.usl --learn"
        )]
        Reputation {
            #[command(subcommand)]
            action: ReputationAction,
        },
        /// Research paper and repository management
        #[command(
            name = "research",
            after_help = "\
EXAMPLES:
    Fetch papers from ArXiv:
      dashprove research arxiv fetch
      dashprove research arxiv fetch --categories cs.LO,cs.PL --since 2024-06-01
      dashprove research arxiv fetch --download-pdfs --extract-text

    Search papers:
      dashprove research arxiv search \"formal verification\"
      dashprove research arxiv search \"neural network\" --category cs.LG --limit 20

    Fetch GitHub repositories:
      dashprove research github fetch
      dashprove research github fetch --queries \"theorem prover,smt solver\" --min-stars 100

    View research corpus statistics:
      dashprove research stats

CATEGORIES (ArXiv):
    cs.LO  - Logic in Computer Science
    cs.PL  - Programming Languages
    cs.SE  - Software Engineering
    cs.AI  - Artificial Intelligence
    cs.CR  - Cryptography and Security
    cs.LG  - Machine Learning

The research command manages a local corpus of academic papers and
GitHub repositories relevant to formal verification and proof systems."
        )]
        Research {
            #[command(subcommand)]
            action: ResearchAction,
        },
        /// AI-driven iterative proof search with tactic learning and hint propagation
        #[command(
            name = "proof-search",
            after_help = "\
EXAMPLES:
    Basic proof search:
      dashprove proof-search spec.usl --backend lean4

    Search for a specific property:
      dashprove proof-search spec.usl --backend lean4 --property my_theorem

    With custom iterations and threshold:
      dashprove proof-search spec.usl --backend lean4 --max-iterations 5 --threshold 0.8

    With hints and preferred tactics:
      dashprove proof-search spec.usl --backend lean4 --hints \"try induction\" --tactics \"simp,intro\"

    Propagate hints to other backends:
      dashprove proof-search spec.usl --backend lean4 --propagate-to coq,isabelle

    JSON output:
      dashprove proof-search spec.usl --backend lean4 --format json

SEARCH PROCESS:
    1. Collects tactic candidates from LLM and structural analysis
    2. Ranks tactics using learned weights + exploration rate
    3. Generates proof attempts with hints from tactics
    4. Validates attempts and assigns rewards to update policy
    5. Propagates successful hints to other backends

The search continues until a valid proof is found or max iterations reached."
        )]
        ProofSearch {
            /// Path to USL specification file
            path: String,
            /// Target backend (lean4, tlaplus, kani, coq, alloy, isabelle, dafny)
            #[arg(long)]
            backend: String,
            /// Property name to prove (proves all if not specified)
            #[arg(long)]
            property: Option<String>,
            /// Maximum search iterations per property
            #[arg(long, default_value = "4")]
            max_iterations: u32,
            /// Validation threshold (0.0 to 1.0) for accepting a proof
            #[arg(long, default_value = "0.75")]
            threshold: f64,
            /// Additional hints to guide search (comma-separated or repeated)
            #[arg(long, value_delimiter = ',')]
            hints: Vec<String>,
            /// Preferred tactics to try first (comma-separated or repeated)
            #[arg(long, value_delimiter = ',')]
            tactics: Vec<String>,
            /// Backends to propagate hints to (comma-separated)
            #[arg(long, value_delimiter = ',')]
            propagate_to: Vec<String>,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Enable hierarchical decomposition of complex properties
            #[arg(long)]
            hierarchical: bool,
            /// Maximum decomposition depth for hierarchical search
            #[arg(long, default_value = "3")]
            max_depth: u32,
            /// Complexity threshold for decomposition (0.0 to 1.0)
            #[arg(long, default_value = "0.7")]
            complexity_threshold: f64,
            /// Induction mode (simple, strong, well-founded)
            #[arg(long, default_value = "simple")]
            induction_mode: String,
        },
        /// Verified self-improvement infrastructure management
        #[command(
            name = "selfimp",
            after_help = "\
EXAMPLES:
    View self-improvement status:
      dashprove selfimp status
      dashprove selfimp status --verbose

    View version history:
      dashprove selfimp history
      dashprove selfimp history --limit 20 --verbose

    Verify an improvement proposal (dry run):
      dashprove selfimp verify --proposal proposal.json --current-version v1.0.0 --dry-run

    Apply a verified improvement:
      dashprove selfimp verify --proposal proposal.json --current-version v1.0.0

    View rollback history:
      dashprove selfimp rollback
      dashprove selfimp rollback --target v0.9.0 --dry-run

    View verification gate details:
      dashprove selfimp gate
      dashprove selfimp gate --all-checks

PROPOSAL FILE FORMAT:
    {
      \"kind\": \"code_change\",
      \"target\": \"self\",
      \"description\": \"Improve proof search efficiency\",
      \"old_content\": \"...\",
      \"new_content\": \"...\"
    }

    Kinds: code_change, config_change, model_update, policy_change, knowledge_addition
    Targets: self, child, peer

SAFETY GUARANTEES:
    - ALL modifications MUST pass through the verification gate
    - The gate CANNOT be bypassed, disabled, or weakened
    - Automatic rollback on verification failure
    - Cryptographic proof certificates for every version"
        )]
        SelfImp {
            #[command(subcommand)]
            action: SelfImpAction,
        },
        /// Verification cache management and compaction time-series analysis
        #[command(
            name = "cache",
            after_help = "\
EXAMPLES:
    View cache statistics:
      dashprove cache stats
      dashprove cache stats --snapshot ~/.dashprove/cache.json --verbose

    View compaction time-series:
      dashprove cache time-series
      dashprove cache time-series --snapshot ~/.dashprove/cache.json --limit 50

    Clear cache:
      dashprove cache clear --dry-run
      dashprove cache clear

    JSON output:
      dashprove cache stats --format json
      dashprove cache time-series --format json

TIME WINDOWS:
    The --window parameter (in seconds) controls the analysis window for rates:
    - 3600 (default): 1 hour window
    - 86400: 24 hour window
    - 604800: 1 week window

COMPACTION TRIGGER TYPES:
    - SizeBased: Cache exceeded size limit
    - TimeBased: Entries exceeded TTL
    - HitRateBased: Hit rate dropped below threshold
    - PartitionImbalance: Partition sizes became imbalanced
    - InsertBased: Triggered by insert count
    - MemoryBased: Memory pressure triggered compaction"
        )]
        Cache {
            #[command(subcommand)]
            action: CacheAction,
        },
    }

    #[derive(Subcommand)]
    pub enum SelfImpAction {
        /// Show self-improvement infrastructure status
        Status {
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
        /// View version history
        History {
            /// Maximum number of versions to show
            #[arg(long, default_value = "10")]
            limit: usize,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output with capability details
            #[arg(short, long)]
            verbose: bool,
        },
        /// Verify an improvement proposal through the verification gate
        Verify {
            /// Path to improvement proposal JSON file
            #[arg(long)]
            proposal: String,
            /// Current version identifier
            #[arg(long)]
            current_version: String,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
            /// Dry run (check without applying)
            #[arg(long)]
            dry_run: bool,
        },
        /// View rollback history or perform rollback
        Rollback {
            /// Target version to rollback to
            #[arg(long)]
            target: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
            /// Dry run (check without actually rolling back)
            #[arg(long)]
            dry_run: bool,
        },
        /// View verification gate details and checks
        Gate {
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show all gate configuration details
            #[arg(long)]
            all_checks: bool,
        },
    }

    #[derive(Subcommand)]
    pub enum CacheAction {
        /// Show verification cache statistics and compaction summary
        Stats {
            /// Path to cache snapshot file (optional, uses empty cache if not provided)
            #[arg(long)]
            snapshot: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output with trigger breakdown
            #[arg(short, long)]
            verbose: bool,
            /// Time window for compaction rates in seconds (default: 1 hour)
            #[arg(long, default_value = "3600")]
            window: u64,
        },
        /// Show compaction time-series data
        TimeSeries {
            /// Path to cache snapshot file (optional, uses empty cache if not provided)
            #[arg(long)]
            snapshot: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output with trigger distribution
            #[arg(short, long)]
            verbose: bool,
            /// Time window for analysis in seconds (default: 1 hour)
            #[arg(long, default_value = "3600")]
            window: u64,
            /// Maximum number of entries to display
            #[arg(long, default_value = "20")]
            limit: usize,
        },
        /// Clear verification cache
        Clear {
            /// Path to cache directory (default: ~/.dashprove/cache)
            #[arg(long)]
            cache_dir: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Dry run (don't actually clear)
            #[arg(long)]
            dry_run: bool,
        },
        /// Show autosave session metrics from cache snapshot
        Autosave {
            /// Path to cache snapshot file (optional, uses empty cache if not provided)
            #[arg(long)]
            snapshot: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show verbose output with save reason breakdown
            #[arg(short, long)]
            verbose: bool,
        },
    }

    #[derive(Subcommand)]
    pub enum ReputationAction {
        /// Show reputation statistics for all backends
        Stats {
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Show domain-specific (property type) statistics
            #[arg(long)]
            domains: bool,
            /// Filter to specific backend
            #[arg(long)]
            backend: Option<String>,
        },
        /// Reset all reputation data
        Reset {
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Export reputation data to a file
        Export {
            /// Output file path
            output: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Import reputation data from a file
        Import {
            /// Input file path
            input: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Merge with existing data instead of replacing
            #[arg(long)]
            merge: bool,
        },
    }

    #[derive(Subcommand)]
    pub enum ExpertAction {
        /// Get backend recommendation based on specification or property types
        Backend {
            /// Path to USL specification file
            #[arg(long)]
            spec: Option<String>,
            /// Property types to verify (comma-separated: safety, liveness, temporal, correctness, probabilistic, neural, security, refinement, smt)
            #[arg(long)]
            property_types: Option<String>,
            /// Code language (rust, etc.)
            #[arg(long)]
            code_lang: Option<String>,
            /// Additional context tags
            #[arg(long)]
            tags: Option<String>,
            /// Directory for knowledge data (default: ~/.dashprove/knowledge)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
        /// Explain a verification error
        Error {
            /// Error message to explain
            #[arg(long, conflicts_with = "file")]
            message: Option<String>,
            /// File containing error output
            #[arg(long, conflicts_with = "message")]
            file: Option<String>,
            /// Backend that produced the error (lean4, tlaplus, kani, coq, etc.)
            #[arg(long)]
            backend: Option<String>,
            /// Directory for knowledge data (default: ~/.dashprove/knowledge)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
        /// Get tactic suggestions for a proof goal
        Tactic {
            /// Description of the proof goal
            #[arg(long)]
            goal: String,
            /// Backend for tactic suggestions (lean4, coq, isabelle, tlaplus)
            #[arg(long)]
            backend: String,
            /// Additional proof context
            #[arg(long)]
            context: Option<String>,
            /// Directory for knowledge data (default: ~/.dashprove/knowledge)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
        /// Get compilation guidance for a target backend
        Compile {
            /// Path to USL specification or code file
            #[arg(long)]
            spec: String,
            /// Target backend (lean4, tlaplus, kani, coq, isabelle, etc.)
            #[arg(long)]
            backend: String,
            /// Directory for knowledge data (default: ~/.dashprove/knowledge)
            #[arg(long)]
            data_dir: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
    }

    #[derive(Subcommand)]
    pub enum CorpusAction {
        /// Show corpus statistics
        Stats {
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Search for similar proofs given a USL file
        Search {
            /// Path to a USL specification file to search for similar proofs
            query: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Number of similar proofs to return
            #[arg(short = 'n', long, default_value = "5")]
            limit: usize,
        },
        /// Search for similar counterexamples given a counterexample JSON file
        CxSearch {
            /// Path to a counterexample JSON file to search for similar counterexamples
            path: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Number of similar counterexamples to return
            #[arg(short = 'n', long, default_value = "5")]
            limit: usize,
        },
        /// Add a counterexample to the corpus
        CxAdd {
            /// Path to a counterexample JSON file to add
            path: String,
            /// Property name associated with this counterexample
            #[arg(long)]
            property: String,
            /// Backend that generated this counterexample (tlaplus, kani, lean4, alloy)
            #[arg(long, default_value = "tlaplus")]
            backend: String,
            /// Cluster label for this counterexample (optional)
            #[arg(long)]
            cluster: Option<String>,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Classify a counterexample against stored cluster patterns
        CxClassify {
            /// Path to a counterexample JSON file to classify
            path: String,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Record cluster patterns from clustering results
        CxRecordClusters {
            /// Paths to counterexample JSON files to cluster and record
            #[arg(required = true, num_args = 2..)]
            paths: Vec<String>,
            /// Similarity threshold for clustering (0.0 to 1.0)
            #[arg(long, default_value = "0.7")]
            threshold: f64,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Show corpus history over time
        History {
            /// Corpus to visualize (proofs or counterexamples)
            #[arg(long, default_value = "counterexamples")]
            corpus: String,
            /// Time period granularity (day, week, month)
            #[arg(long, default_value = "day")]
            period: String,
            /// Output format (text, json, html)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Filter entries recorded on or after this date (YYYY-MM-DD)
            #[arg(long)]
            from: Option<String>,
            /// Filter entries recorded on or before this date (YYYY-MM-DD)
            #[arg(long)]
            to: Option<String>,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Compare two time periods in the corpus
        Compare {
            /// Corpus to compare (proofs or counterexamples)
            #[arg(long, default_value = "counterexamples")]
            corpus: String,
            /// Baseline period start date (YYYY-MM-DD)
            #[arg(long)]
            baseline_from: String,
            /// Baseline period end date (YYYY-MM-DD)
            #[arg(long)]
            baseline_to: String,
            /// Comparison period start date (YYYY-MM-DD)
            #[arg(long)]
            compare_from: String,
            /// Comparison period end date (YYYY-MM-DD)
            #[arg(long)]
            compare_to: String,
            /// Time period granularity for internal calculations (day, week, month)
            #[arg(long, default_value = "day")]
            period: String,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
        /// Suggest comparison periods based on available data
        SuggestCompare {
            /// Corpus to analyze (proofs or counterexamples)
            #[arg(long, default_value = "counterexamples")]
            corpus: String,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
            /// Directory containing learning data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
    }

    #[derive(Subcommand)]
    pub enum AnalyzeAction {
        /// Suggest patterns and filters to help understand the counterexample
        Suggest {
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
        /// Compress the trace by detecting repeating patterns
        Compress {
            /// Output format (text, mermaid, dot, html, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
        },
        /// Detect actor interleavings in the trace
        Interleavings {
            /// Output format (text, mermaid, html, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
        },
        /// Minimize the counterexample by removing irrelevant variables and states
        Minimize {
            /// Maximum number of states to keep (0 = no limit)
            #[arg(long, default_value = "0")]
            max_states: usize,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
        },
        /// Abstract consecutive similar states into groups
        Abstract {
            /// Minimum group size to form an abstraction
            #[arg(long, default_value = "2")]
            min_group_size: usize,
            /// Output format (text, mermaid, dot, html, json)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
        },
        /// Compare two counterexample traces
        Diff {
            /// Path to second counterexample JSON file
            other: String,
            /// Output format (text, mermaid, dot, html)
            #[arg(long, default_value = "text")]
            format: String,
            /// Output file (prints to stdout if not specified)
            #[arg(short, long)]
            output: Option<String>,
        },
    }

    #[derive(Subcommand)]
    pub enum ResearchAction {
        /// ArXiv paper operations
        #[command(subcommand)]
        Arxiv(ArxivAction),
        /// GitHub repository operations
        #[command(subcommand)]
        Github(GithubAction),
        /// Show research corpus statistics
        Stats {
            /// Directory containing research data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
        },
    }

    #[derive(Subcommand)]
    pub enum ArxivAction {
        /// Fetch papers from ArXiv
        Fetch {
            /// Output directory for fetched papers
            #[arg(short, long)]
            output: Option<String>,
            /// ArXiv categories to fetch (comma-separated: cs.LO,cs.PL,cs.SE,cs.AI,cs.CR,cs.LG)
            #[arg(long)]
            categories: Option<String>,
            /// Start date for papers (YYYY-MM-DD)
            #[arg(long)]
            since: Option<String>,
            /// Maximum papers per category
            #[arg(long, default_value = "100")]
            max_per_category: usize,
            /// Download PDFs for each paper
            #[arg(long)]
            download_pdfs: bool,
            /// Extract text from downloaded PDFs
            #[arg(long)]
            extract_text: bool,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
        /// Search papers in the local corpus
        Search {
            /// Search query
            query: String,
            /// Directory containing research data (default: ~/.dashprove)
            #[arg(long)]
            data_dir: Option<String>,
            /// Maximum number of results
            #[arg(short = 'n', long, default_value = "10")]
            limit: usize,
            /// Filter by ArXiv category
            #[arg(long)]
            category: Option<String>,
            /// Output format (text, json)
            #[arg(long, default_value = "text")]
            format: String,
        },
    }

    #[derive(Subcommand)]
    pub enum GithubAction {
        /// Fetch repositories from GitHub
        Fetch {
            /// Output directory for fetched repository data
            #[arg(short, long)]
            output: Option<String>,
            /// Search queries (comma-separated)
            #[arg(long)]
            queries: Option<String>,
            /// Minimum stars filter
            #[arg(long, default_value = "20")]
            min_stars: usize,
            /// Show verbose output
            #[arg(short, long)]
            verbose: bool,
        },
    }
}
