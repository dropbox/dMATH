//! γ-CROWN command-line interface.
//!
//! Provides verification capabilities for neural networks with a focus on
//! Whisper-scale transformer models.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use gamma_core::{Bound, VerificationSpec};
use gamma_gpu::{Backend, ComputeDevice};
use gamma_onnx::{load_onnx, load_whisper};
use gamma_propagate::{
    compute_model_hash, AlphaCrownConfig, BabVerificationStatus, BetaCrownConfig, BetaCrownResult,
    BetaCrownVerifier, BlockProgress, BoundPropagation, BranchingHeuristic, GELULayer,
    GradientMethod, GraphPrecomputedBounds, Layer, LayerNormLayer, LayerProgress, LinearLayer,
    MatMulLayer, Network, Optimizer, PropagationConfig, PropagationMethod, SoftmaxLayer,
    VerificationCheckpoint, Verifier,
};
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Compute backend selection for accelerated operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum BackendArg {
    /// CPU with Rayon parallelization (default, always available)
    #[default]
    Cpu,
    /// wgpu GPU compute (cross-platform: Metal on macOS, Vulkan on Linux, DX12 on Windows)
    Wgpu,
    /// MLX for Apple Silicon (macOS only, requires `mlx` feature at build time)
    Mlx,
}

impl From<BackendArg> for Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Cpu => Backend::Cpu,
            BackendArg::Wgpu => Backend::Wgpu,
            BackendArg::Mlx => Backend::Mlx,
        }
    }
}

impl std::fmt::Display for BackendArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendArg::Cpu => write!(f, "cpu"),
            BackendArg::Wgpu => write!(f, "wgpu"),
            BackendArg::Mlx => write!(f, "mlx"),
        }
    }
}

/// Log output format for structured logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum LogFormat {
    /// Human-readable text output (default)
    #[default]
    Text,
    /// JSON lines format for machine parsing
    Json,
}

#[derive(Parser)]
#[command(name = "gamma")]
#[command(author = "γ-CROWN Team")]
#[command(version = "0.1.0")]
#[command(about = "Neural network verification at Whisper scale", long_about = None)]
struct Cli {
    /// Verbosity level (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Log output format (text or json)
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify a neural network property
    Verify {
        /// Path to model (ONNX, SafeTensors, PyTorch, GGUF, NNet)
        model: PathBuf,

        /// Input perturbation epsilon (ignored if --property is specified)
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// VNN-LIB property file (.vnnlib) specifying input bounds and output constraints
        #[arg(short, long)]
        property: Option<PathBuf>,

        /// Verification method (ibp, crown, alpha, beta)
        #[arg(long, default_value = "alpha")]
        method: String,

        /// Timeout in seconds
        #[arg(short, long, default_value = "60")]
        timeout: u64,

        /// Compute backend (cpu, wgpu, mlx)
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,

        /// Use GPU acceleration (deprecated, use --backend wgpu instead)
        #[arg(long, default_value_t = false, hide = true)]
        gpu: bool,

        /// Load as native format (PyTorch/SafeTensors/GGUF) instead of ONNX
        #[arg(long, default_value_t = false)]
        native: bool,

        /// Use conservative (sound) LayerNorm bounds (disables forward-mode stabilization for IBP)
        #[arg(long, default_value_t = false)]
        conservative_layernorm: bool,

        /// Layer-by-layer verification mode: outputs bound statistics per node
        /// Useful for large models where full verification may timeout
        #[arg(long, default_value_t = false, conflicts_with = "block_wise")]
        layer_by_layer: bool,

        /// Block-wise verification mode: resets bounds at each transformer block
        /// Prevents bound explosion and enables per-block zonotope tightening
        #[arg(long, default_value_t = false, conflicts_with = "layer_by_layer")]
        block_wise: bool,

        /// Show progress during verification (useful for large models)
        /// Works with `--block-wise` and `--layer-by-layer`.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Output progress as JSON lines to stderr (for programmatic monitoring)
        /// Each line is a complete JSON object with progress information.
        /// Implies --progress. Works with `--block-wise` and `--layer-by-layer`.
        #[arg(long, default_value_t = false)]
        progress_json: bool,

        /// Maximum number of blocks to verify (0 = all blocks)
        /// Useful for partial verification of very large models
        /// Requires --block-wise or --checkpoint
        #[arg(long, default_value_t = 0)]
        max_blocks: usize,

        /// Checkpoint file for save/resume (implies --block-wise)
        /// If file exists and matches config, resume from checkpoint.
        /// Saves progress after each block for crash recovery.
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Load and inspect a model
    Inspect {
        /// Path to model (ONNX, PyTorch .pt/.pth, SafeTensors)
        model: PathBuf,

        /// Load as native format (PyTorch/SafeTensors) instead of ONNX
        #[arg(long, default_value_t = false)]
        native: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare two model implementations (for porting verification)
    Compare {
        /// Path to reference model (e.g., PyTorch ONNX export)
        reference: PathBuf,

        /// Path to target model (e.g., Metal implementation)
        target: PathBuf,

        /// Maximum allowed difference in output bounds
        #[arg(short, long, default_value = "0.001")]
        tolerance: f32,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Verification method (ibp, crown, alpha)
        #[arg(short = 'm', long, default_value = "crown")]
        method: String,

        /// Print per-element comparison details
        #[arg(long, default_value_t = false)]
        verbose: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Find where two models diverge (layer-by-layer diff for porting)
    Diff {
        /// Path to first ONNX model (e.g., PyTorch export)
        model_a: PathBuf,

        /// Path to second ONNX model (e.g., CoreML/Metal export)
        model_b: PathBuf,

        /// Path to input data file (.npy format)
        #[arg(long)]
        input: Option<PathBuf>,

        /// Maximum allowed difference before flagging divergence
        #[arg(short, long, default_value = "1e-5")]
        tolerance: f32,

        /// Continue comparing after first divergence
        #[arg(long, default_value_t = true)]
        continue_after_divergence: bool,

        /// Enable root cause diagnosis (analyzes divergence patterns)
        #[arg(long, default_value_t = false)]
        diagnose: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Analyze layer sensitivity (which layers amplify input noise)
    Sensitivity {
        /// Path to ONNX model
        model: PathBuf,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Continue analysis after overflow
        #[arg(long, default_value_t = true)]
        continue_after_overflow: bool,

        /// Show only high-sensitivity layers (sensitivity > threshold)
        #[arg(long)]
        threshold: Option<f32>,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Check if layers can be safely quantized to float16/int8
    QuantizeCheck {
        /// Path to ONNX model
        model: PathBuf,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Continue analysis after overflow
        #[arg(long, default_value_t = true)]
        continue_after_overflow: bool,

        /// Check only float16 (skip int8 analysis)
        #[arg(long, default_value_t = false)]
        float16_only: bool,

        /// Check only int8 (skip float16 analysis)
        #[arg(long, default_value_t = false)]
        int8_only: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Profile bound width growth through the network
    ProfileBounds {
        /// Path to model (ONNX, SafeTensors, PyTorch, GGUF)
        model: PathBuf,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Continue analysis after overflow
        #[arg(long, default_value_t = true)]
        continue_after_overflow: bool,

        /// Show only layers with growth ratio above threshold
        #[arg(long)]
        threshold: Option<f32>,

        /// Load as native format (PyTorch/SafeTensors/GGUF) instead of ONNX
        #[arg(long, default_value_t = false)]
        native: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Use zeros-centered input (for validation against Auto-LiRPA).
        /// Default is unit-variance input (±1 alternating) for realistic LayerNorm bounds.
        #[arg(long, default_value_t = false)]
        center_zeros: bool,
    },

    /// Verify a Whisper model component
    Whisper {
        /// Path to Whisper ONNX model
        model: PathBuf,

        /// Component to verify (encoder, decoder, attention)
        #[arg(short, long, default_value = "encoder")]
        component: String,

        /// Specific layer index (optional)
        #[arg(short, long)]
        layer: Option<usize>,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Verify multiple Whisper encoder blocks sequentially (diagnostic tool)
    WhisperSeq {
        /// Path to Whisper ONNX model
        model: PathBuf,

        /// First encoder block (0-indexed)
        #[arg(long, default_value_t = 0)]
        start_block: usize,

        /// End encoder block (exclusive). Defaults to all blocks.
        #[arg(long)]
        end_block: Option<usize>,

        /// Include encoder stem (mel -> hidden) before the first block
        #[arg(long, default_value_t = false)]
        include_stem: bool,

        /// Include final encoder LayerNorm (ln_post) after the last block
        #[arg(long, default_value_t = false)]
        include_ln_post: bool,

        /// Batch size for synthetic input (hidden states or mel)
        #[arg(long, default_value_t = 1)]
        batch: usize,

        /// Sequence length for synthetic hidden-state input (ignored if --include-stem)
        #[arg(long, default_value_t = 4)]
        seq_len: usize,

        /// Mel bins for synthetic mel input (only used with --include-stem)
        #[arg(long, default_value_t = 80)]
        n_mels: usize,

        /// Time dimension for synthetic mel input (only used with --include-stem)
        #[arg(long, default_value_t = 3000)]
        time: usize,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Compute backend (cpu, wgpu, mlx)
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,

        /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
        #[arg(long, default_value_t = false, hide = true)]
        gpu: bool,

        /// Multi-block config preset: default, strict, diagnostic
        #[arg(long, default_value = "default")]
        mode: String,

        /// Override: maximum bound width threshold before early termination
        #[arg(long)]
        max_bound_width: Option<f32>,

        /// Override: terminate on NaN/Infinity (true/false)
        #[arg(long, value_parser = clap::value_parser!(bool))]
        terminate_on_overflow: Option<bool>,

        /// Override: continue after overflow by clamping (true/false)
        #[arg(long, value_parser = clap::value_parser!(bool))]
        continue_after_overflow: Option<bool>,

        /// Override: clamp value used when continuing after overflow
        #[arg(long)]
        overflow_clamp_value: Option<f32>,

        /// Reset zonotope correlations at block boundaries to prevent bound explosion
        /// Normalizes input bounds and rescales output for deep transformers (28+ layers)
        #[arg(long, default_value_t = false)]
        reset_zonotope_blocks: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Sweep epsilon for sequential verification (find stable ranges)
    WhisperSweep {
        /// Path to Whisper ONNX model
        model: PathBuf,

        /// First encoder block (0-indexed)
        #[arg(long, default_value_t = 0)]
        start_block: usize,

        /// End encoder block (exclusive). Defaults to all blocks.
        #[arg(long)]
        end_block: Option<usize>,

        /// Include encoder stem (mel -> hidden) before the first block
        #[arg(long, default_value_t = false)]
        include_stem: bool,

        /// Include final encoder LayerNorm (ln_post) after the last block
        #[arg(long, default_value_t = false)]
        include_ln_post: bool,

        /// Batch size for synthetic input (hidden states or mel)
        #[arg(long, default_value_t = 1)]
        batch: usize,

        /// Sequence length for synthetic hidden-state input (ignored if --include-stem)
        #[arg(long, default_value_t = 4)]
        seq_len: usize,

        /// Mel bins for synthetic mel input (only used with --include-stem)
        #[arg(long, default_value_t = 80)]
        n_mels: usize,

        /// Time dimension for synthetic mel input (only used with --include-stem)
        #[arg(long, default_value_t = 3000)]
        time: usize,

        /// Minimum epsilon (inclusive)
        #[arg(long, default_value = "0.000001")]
        epsilon_min: f32,

        /// Maximum epsilon (inclusive)
        #[arg(long, default_value = "0.01")]
        epsilon_max: f32,

        /// Number of sweep points
        #[arg(long, default_value_t = 10)]
        steps: usize,

        /// Sweep linearly instead of logarithmically
        #[arg(long, default_value_t = false)]
        linear: bool,

        /// Compute backend (cpu, wgpu, mlx)
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,

        /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
        #[arg(long, default_value_t = false, hide = true)]
        gpu: bool,

        /// Multi-block config preset: default, strict, diagnostic (default: strict)
        #[arg(long, default_value = "strict")]
        mode: String,

        /// Override: maximum bound width threshold before early termination
        #[arg(long)]
        max_bound_width: Option<f32>,

        /// Reset zonotope correlations at block boundaries to prevent bound explosion
        #[arg(long, default_value_t = false)]
        reset_zonotope_blocks: bool,

        /// Print per-block width details for each epsilon point
        #[arg(long, default_value_t = false)]
        per_block: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Binary-search for maximum epsilon that completes N blocks
    WhisperEpsSearch {
        /// Path to Whisper ONNX model
        model: PathBuf,

        /// First encoder block (0-indexed)
        #[arg(long, default_value_t = 0)]
        start_block: usize,

        /// End encoder block (exclusive). Defaults to all blocks.
        #[arg(long)]
        end_block: Option<usize>,

        /// Target number of blocks to complete without overflow/early termination
        #[arg(long)]
        target_blocks: Option<usize>,

        /// Include encoder stem (mel -> hidden) before the first block
        #[arg(long, default_value_t = false)]
        include_stem: bool,

        /// Include final encoder LayerNorm (ln_post) after the last block
        #[arg(long, default_value_t = false)]
        include_ln_post: bool,

        /// Batch size for synthetic input (hidden states or mel)
        #[arg(long, default_value_t = 1)]
        batch: usize,

        /// Sequence length for synthetic hidden-state input (ignored if --include-stem)
        #[arg(long, default_value_t = 4)]
        seq_len: usize,

        /// Mel bins for synthetic mel input (only used with --include-stem)
        #[arg(long, default_value_t = 80)]
        n_mels: usize,

        /// Time dimension for synthetic mel input (only used with --include-stem)
        #[arg(long, default_value_t = 3000)]
        time: usize,

        /// Minimum epsilon (inclusive, search lower bound)
        #[arg(long, default_value = "0.000001")]
        epsilon_min: f32,

        /// Maximum epsilon (inclusive, search upper bound)
        #[arg(long, default_value = "0.01")]
        epsilon_max: f32,

        /// Number of binary search iterations (default: 20 for ~1e-6 precision ratio)
        #[arg(long, default_value_t = 20)]
        iterations: usize,

        /// Compute backend (cpu, wgpu, mlx)
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,

        /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
        #[arg(long, default_value_t = false, hide = true)]
        gpu: bool,

        /// Multi-block config preset: default, strict, diagnostic (default: strict)
        #[arg(long, default_value = "strict")]
        mode: String,

        /// Override: maximum bound width threshold before early termination
        #[arg(long)]
        max_bound_width: Option<f32>,

        /// Reset zonotope correlations at block boundaries to prevent bound explosion
        #[arg(long, default_value_t = false)]
        reset_zonotope_blocks: bool,

        /// Show progress for each iteration
        #[arg(long, default_value_t = false)]
        verbose_search: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Generate export script for PyTorch models
    Export {
        /// Model type (whisper, custom)
        #[arg(short, long, default_value = "whisper")]
        model_type: String,

        /// Model size (tiny, base, small, medium, large)
        #[arg(short, long, default_value = "tiny")]
        size: String,

        /// Output script path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run benchmarks
    Bench {
        /// Benchmark type (layer, attention, full)
        #[arg(short, long, default_value = "layer")]
        benchmark: String,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Run β-CROWN branch-and-bound complete verification
    BetaCrown {
        /// Path to ONNX model
        model: PathBuf,

        /// VNN-LIB property file (.vnnlib) specifying input bounds and output constraints
        #[arg(short, long)]
        property: Option<PathBuf>,

        /// Input perturbation epsilon (ignored if --property is specified)
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Property threshold: verify that output > threshold (ignored if --property is specified)
        #[arg(short, long, default_value = "0.0")]
        threshold: f32,

        /// Maximum number of domains to explore
        #[arg(long, default_value_t = 10000)]
        max_domains: usize,

        /// Timeout in seconds
        #[arg(long, default_value_t = 300)]
        timeout: u64,

        /// Maximum search tree depth (number of splits)
        #[arg(long, default_value_t = 100)]
        max_depth: usize,

        /// Branching heuristic: width, impact, sequential, input
        #[arg(long, default_value = "width")]
        branching: String,

        /// Number of candidate neurons to evaluate for FSB branching.
        #[arg(long, default_value_t = 8)]
        fsb_candidates: usize,

        /// Disable α-CROWN optimization (use CROWN-IBP only). Faster but looser bounds.
        #[arg(long, default_value_t = false)]
        no_alpha: bool,

        /// Number of α-CROWN optimization iterations (higher = tighter bounds, slower)
        /// For DAG models, each iteration = 1 + 2*spsa_samples CROWN passes.
        /// Default: 20 iterations (matches α,β-CROWN's init_iteration default).
        #[arg(long, default_value_t = 20)]
        alpha_iterations: usize,

        /// Disable adaptive α-CROWN skipping for deep networks.
        /// By default, α-CROWN is automatically skipped for very deep networks (>25 ReLU layers)
        /// where optimization doesn't help because bounds are fundamentally loose.
        /// Use this flag to always run α-CROWN regardless of network depth.
        #[arg(long, default_value_t = false)]
        no_adaptive_alpha_skip: bool,

        /// Depth threshold for adaptive α-CROWN skipping (number of ReLU layers).
        /// Networks with more than this many ReLU layers will skip α-CROWN if adaptive skip is enabled.
        /// Default: 8 (ResNet-2b has 6 and benefits, ResNet-4b has 10 and doesn't benefit)
        #[arg(long, default_value_t = 8)]
        alpha_skip_depth: usize,

        /// Use CROWN-IBP bounds for intermediate nodes (O(N²) but tighter).
        /// By default, uses IBP bounds (O(N), faster but looser - 3000x+ expansion).
        /// This matches α,β-CROWN's fix_interm_bounds=False setting.
        /// WARNING: Can be slow for deep networks (ResNet-4b: ~80s init vs <5s with IBP).
        #[arg(long, default_value_t = false)]
        crown_ibp_intermediates: bool,

        /// Number of SPSA samples per α-CROWN iteration (default: 1).
        /// Higher values reduce gradient variance at the cost of more CROWN passes.
        /// Each sample requires 2 CROWN passes (plus/minus perturbation).
        /// Formula: total_passes = iterations * (1 + 2 * samples)
        #[arg(long, default_value_t = 1)]
        alpha_spsa_samples: usize,

        /// Learning rate for α-CROWN optimization (default: 0.5).
        /// Higher values converge faster but may overshoot. Lower values are more stable.
        /// Default 0.1 matches α,β-CROWN for Adam optimizer.
        #[arg(long, default_value_t = 0.1)]
        alpha_lr: f32,

        /// Gradient method for α-CROWN optimization (default: spsa).
        /// - spsa: SPSA zero-order optimization (O(1) passes per iteration, robust)
        /// - fd: Finite differences (O(n) passes, accurate but slow)
        /// - analytic: Experimental - local gradients from CROWN backward (incomplete)
        /// - analytic-chain: True chain-rule gradients (NOT YET IMPLEMENTED - Issue #8)
        #[arg(long, default_value = "spsa", value_parser = ["spsa", "fd", "analytic", "analytic-chain"])]
        alpha_gradient_method: String,

        /// Optimizer for α-CROWN parameter updates.
        /// - adam: Adam optimizer (adaptive moment estimation, default - matches α,β-CROWN)
        /// - sgd: SGD with momentum
        #[arg(long, default_value = "adam", value_parser = ["adam", "sgd"])]
        alpha_optimizer: String,

        /// Number of β-CROWN optimization iterations per domain
        /// Per-domain optimization is expensive; default 0 for throughput.
        /// Use --beta-iterations 5 for single-objective verification where bound quality matters.
        #[arg(long, default_value_t = 0)]
        beta_iterations: usize,

        /// Maximum depth for per-domain β optimization (default: 3)
        /// Only applies when beta_iterations > 0. Domains deeper than this
        /// skip optimization and rely on inherited β values from warmup.
        #[arg(long, default_value_t = 3)]
        beta_max_depth: usize,

        /// Learning rate for β optimization (default: 0.05)
        /// α,β-CROWN default is 0.05.
        #[arg(long, default_value_t = 0.05)]
        lr_beta: f32,

        /// Use CROWN-IBP for tighter intermediate bounds (~66% tighter than IBP). Enabled by default.
        #[arg(long, default_value_t = true)]
        crown_ibp: bool,

        /// Batch size for parallel domain processing (1 = sequential, 64 = GPU-optimized default)
        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        /// Disable parallel child domain creation (default: enabled)
        #[arg(long, default_value_t = false)]
        sequential_children: bool,

        /// Enable GCP-CROWN cutting planes (improves verification even without generating cuts)
        #[arg(long, default_value_t = true)]
        enable_cuts: bool,

        /// Disable GCP-CROWN cutting planes (for comparison/debugging)
        #[arg(long, default_value_t = false)]
        no_cuts: bool,

        /// Maximum number of cutting planes to retain
        #[arg(long, default_value_t = 1000)]
        max_cuts: usize,

        /// Minimum depth for cut generation (deeper domains produce more specific cuts)
        #[arg(long, default_value_t = 2)]
        min_cut_depth: usize,

        /// Enable near-miss cut generation (experimental)
        /// Generates cuts from domains close to verification but not quite verified.
        #[arg(long, default_value_t = false)]
        enable_near_miss_cuts: bool,

        /// Margin for near-miss cut generation (fraction of threshold or absolute if threshold=0)
        #[arg(long, default_value_t = 0.1)]
        near_miss_margin: f32,

        /// Enable proactive cut generation (BICCOS-lite).
        /// Generates cuts for unstable ReLUs BEFORE BaB starts.
        /// Helps on hard instances where initial bounds are loose.
        #[arg(long, default_value_t = false)]
        proactive_cuts: bool,

        /// Maximum number of proactive cuts to generate.
        #[arg(long, default_value_t = 100)]
        max_proactive_cuts: usize,

        /// Enable PGD attack to find counterexamples when verification fails
        #[arg(long, default_value_t = false)]
        pgd_attack: bool,

        /// Number of PGD attack restarts
        #[arg(long, default_value_t = 100)]
        pgd_restarts: usize,

        /// Number of PGD gradient steps per restart
        #[arg(long, default_value_t = 50)]
        pgd_steps: usize,

        /// Compute backend (cpu, wgpu, mlx).
        /// Note: GPU is ~30% slower for BaB due to per-domain kernel overhead.
        /// Use cpu (default) for best throughput until Issue #12 implements tensor batching.
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,

        /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
        #[arg(long, default_value_t = false, hide = true)]
        gpu: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Inspect and compare model weights (SafeTensors, ONNX)
    Weights {
        #[command(subcommand)]
        action: WeightsAction,
    },
}

#[derive(Subcommand)]
enum WeightsAction {
    /// Show information about weights in a file
    Info {
        /// Path to weights file (ONNX, SafeTensors, PyTorch, CoreML, or GGUF)
        #[arg(short, long)]
        file: PathBuf,

        /// Show detailed per-tensor info
        #[arg(long, default_value_t = false)]
        detailed: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare weights between two files
    Diff {
        /// Path to first weights file
        #[arg(long)]
        file_a: PathBuf,

        /// Path to second weights file
        #[arg(long)]
        file_b: PathBuf,

        /// Maximum allowed absolute difference
        #[arg(short, long, default_value = "1e-5")]
        tolerance: f32,

        /// Show all differing tensors (not just first)
        #[arg(long, default_value_t = false)]
        show_all: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compute per-block weight norms (for sensitivity correlation analysis)
    Norms {
        /// Path to weights file (GGUF, SafeTensors, ONNX, etc.)
        #[arg(short, long)]
        file: PathBuf,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

fn make_synthetic_input(
    whisper: &gamma_onnx::WhisperModel,
    include_stem: bool,
    batch: usize,
    seq_len: usize,
    n_mels: usize,
    time: usize,
    epsilon: f32,
) -> BoundedTensor {
    if include_stem {
        let data = ArrayD::from_elem(IxDyn(&[batch, n_mels, time]), 0.0f32);
        BoundedTensor::from_epsilon(data, epsilon)
    } else {
        let hidden_dim = whisper.hidden_dim;
        let data = ArrayD::from_elem(IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
        BoundedTensor::from_epsilon(data, epsilon)
    }
}

fn make_multiblock_config(
    mode: &str,
    max_bound_width: Option<f32>,
    terminate_on_overflow: Option<bool>,
    continue_after_overflow: Option<bool>,
    overflow_clamp_value: Option<f32>,
    reset_zonotope_between_blocks: bool,
) -> Result<gamma_onnx::MultiBlockConfig> {
    let mut config = match mode {
        "default" => gamma_onnx::MultiBlockConfig::default(),
        "strict" => gamma_onnx::MultiBlockConfig::strict(),
        "diagnostic" => gamma_onnx::MultiBlockConfig::diagnostic(),
        _ => anyhow::bail!("Unknown mode: {}. Use default, strict, or diagnostic", mode),
    };

    if let Some(max_width) = max_bound_width {
        config.max_bound_width = max_width;
    }
    if let Some(v) = terminate_on_overflow {
        config.terminate_on_overflow = v;
    }
    if let Some(v) = continue_after_overflow {
        config.continue_after_overflow = v;
    }
    if let Some(v) = overflow_clamp_value {
        config.overflow_clamp_value = v;
    }
    if reset_zonotope_between_blocks {
        config.reset_zonotope_between_blocks = true;
    }

    Ok(config)
}

fn eps_sweep(epsilon_min: f32, epsilon_max: f32, steps: usize, linear: bool) -> Result<Vec<f32>> {
    if steps == 0 {
        anyhow::bail!("steps must be >= 1");
    }
    if steps == 1 {
        return Ok(vec![epsilon_min]);
    }
    if epsilon_min.is_nan() || epsilon_max.is_nan() || epsilon_min <= 0.0 || epsilon_max <= 0.0 {
        anyhow::bail!(
            "epsilon_min and epsilon_max must be > 0 (got {}..{})",
            epsilon_min,
            epsilon_max
        );
    }
    if epsilon_min > epsilon_max {
        anyhow::bail!(
            "epsilon_min must be <= epsilon_max (got {}..{})",
            epsilon_min,
            epsilon_max
        );
    }

    let mut eps = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let v = if linear {
            epsilon_min + t * (epsilon_max - epsilon_min)
        } else {
            let ratio = epsilon_max / epsilon_min;
            epsilon_min * ratio.powf(t)
        };
        eps.push(v);
    }
    Ok(eps)
}

/// Resolve the effective backend from --backend and --gpu flags.
///
/// --backend takes precedence. If --backend is not specified (default Cpu),
/// but --gpu is true, use wgpu for backward compatibility.
fn resolve_backend(backend: BackendArg, gpu: bool) -> BackendArg {
    if backend != BackendArg::Cpu {
        // --backend was explicitly specified
        backend
    } else if gpu {
        // Legacy --gpu flag, use wgpu for backward compat
        BackendArg::Wgpu
    } else {
        BackendArg::Cpu
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let level = match cli.verbose {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    // Configure subscriber based on format
    match cli.log_format {
        LogFormat::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_target(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .context("Failed to set tracing subscriber")?;
        }
        LogFormat::Json => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .json()
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .context("Failed to set tracing subscriber")?;
        }
    }

    match cli.command {
        Commands::Verify {
            model,
            epsilon,
            property,
            method,
            timeout,
            backend,
            gpu,
            native,
            conservative_layernorm,
            layer_by_layer,
            block_wise,
            progress,
            progress_json,
            max_blocks,
            checkpoint,
            json,
        } => {
            info!("Verifying model: {}", model.display());

            let method = match method.as_str() {
                "ibp" => PropagationMethod::Ibp,
                "crown" => PropagationMethod::Crown,
                "alpha" => PropagationMethod::AlphaCrown,
                "sdp" | "sdp-crown" => PropagationMethod::SdpCrown,
                "beta" => PropagationMethod::BetaCrown,
                _ => {
                    anyhow::bail!(
                        "Unknown method: {}. Use ibp, crown, alpha, sdp-crown, or beta",
                        method
                    );
                }
            };

            // Resolve effective backend (--gpu for backward compat, --backend takes precedence)
            let effective_backend = resolve_backend(backend, gpu);
            let use_gpu = effective_backend != BackendArg::Cpu;

            let config = PropagationConfig {
                method,
                max_iterations: 100,
                tolerance: 1e-4,
                use_gpu,
            };

            // Initialize compute device if not CPU
            let compute_device = if effective_backend != BackendArg::Cpu {
                match ComputeDevice::new(effective_backend.into()) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to create {} device: {}. Falling back to CPU.",
                            effective_backend, e
                        );
                        None
                    }
                }
            } else {
                None
            };
            if !json {
                info!("Backend: {}", effective_backend);
            }

            // Check if NNet format
            let is_nnet = model.extension().and_then(|e| e.to_str()) == Some("nnet");

            // Auto-detect native format based on extension/type
            let use_native = !is_nnet
                && (native || model.is_dir() || {
                    let ext = model.extension().and_then(|e| e.to_str()).unwrap_or("");
                    matches!(
                        ext,
                        "pt" | "pth" | "bin" | "safetensors" | "gguf" | "mlmodel" | "mlpackage"
                    )
                });

            // Enum to hold either a sequential Network or a DAG-based GraphNetwork
            enum VerifiableNetwork {
                Sequential(gamma_propagate::Network),
                Graph(gamma_propagate::GraphNetwork),
            }

            // Load model (NNet, native, or ONNX)
            let (network_choice, mut input_shape, output_dim) = if is_nnet {
                // Load NNet format (VNN-COMP / ACAS-Xu)
                use gamma_onnx::nnet::load_nnet;

                let nnet = load_nnet(&model)?;
                info!(
                    "Loaded NNet: {} layers, {} inputs, {} outputs, {} params",
                    nnet.num_layers,
                    nnet.input_size,
                    nnet.output_size,
                    nnet.param_count()
                );

                let prop_net = nnet.to_prop_network()?;
                let input_shape = vec![1, nnet.input_size];
                let output_dim = nnet.output_size;

                info!(
                    "Converted to propagate network: {} layers",
                    prop_net.layers.len()
                );
                (
                    VerifiableNetwork::Sequential(prop_net),
                    input_shape,
                    output_dim,
                )
            } else if use_native {
                use gamma_onnx::native::NativeModel;

                let native_model = NativeModel::load(&model)?;
                let network = &native_model.network;
                info!(
                    "Loaded native model: {} ({:?}, {} params)",
                    network.name, native_model.config.architecture, network.param_count
                );

                // Use GraphNetwork for native models (supports binary ops like attention MatMul)
                let mut graph_net = native_model.to_graph_network()?;
                if !conservative_layernorm {
                    let num_modified = graph_net.set_layernorm_forward_mode(true);
                    if num_modified > 0 && method == PropagationMethod::Ibp && !json {
                        eprintln!(
                            "Warning: enabling LayerNorm forward-mode for IBP ({} LayerNorm nodes) to reduce bound explosion; use --conservative-layernorm to disable.",
                            num_modified
                        );
                    }
                }

                // Get input/output shapes from network spec
                // For verification, strip the batch dimension (first dim if dynamic)
                // Propagation layers expect unbatched input like [channels, length]
                let input_shape: Vec<usize> = network
                    .inputs
                    .first()
                    .map(|i| {
                        let full_shape: Vec<usize> = i
                            .shape
                            .iter()
                            .map(|&d| if d < 0 { 16 } else { d as usize })
                            .collect();
                        // If first dimension is batch (dynamic in spec), skip it
                        if i.shape.first() == Some(&-1) && full_shape.len() > 2 {
                            full_shape[1..].to_vec()
                        } else {
                            full_shape
                        }
                    })
                    .unwrap_or_else(|| vec![native_model.config.hidden_dim]);

                let output_dim = network
                    .outputs
                    .first()
                    .map(|o| {
                        o.shape
                            .iter()
                            .map(|&d| if d < 0 { 16 } else { d })
                            .product::<i64>() as usize
                    })
                    .unwrap_or(native_model.config.hidden_dim);

                info!(
                    "Converted to graph network: {} nodes",
                    graph_net.num_nodes()
                );
                (VerifiableNetwork::Graph(graph_net), input_shape, output_dim)
            } else {
                let onnx_model = load_onnx(&model)?;
                let onnx_network = &onnx_model.network;
                info!(
                    "Loaded network: {} ({} layers)",
                    onnx_network.name,
                    onnx_network.layers.len()
                );

                // Convert ONNX network to graph network (supports DAG structures like ViT)
                let graph_net = onnx_model.to_graph_network()?;

                // Use batch size 1 for dynamic dimensions (VNNLIB properties are single-instance)
                // Note: Both 0 and -1 indicate dynamic dimensions in ONNX
                // TensorFlow-exported models use 0 for dynamic batch, PyTorch uses -1
                let input_shape: Vec<usize> = onnx_network
                    .inputs
                    .first()
                    .map(|i| {
                        i.shape
                            .iter()
                            .map(|&d| if d <= 0 { 1 } else { d as usize })
                            .collect()
                    })
                    .unwrap_or_else(|| vec![100]);

                let output_dim = onnx_network
                    .outputs
                    .first()
                    .map(|o| {
                        o.shape
                            .iter()
                            .map(|&d| if d <= 0 { 1 } else { d })
                            .product::<i64>() as usize
                    })
                    .unwrap_or(10);

                info!(
                    "Converted to graph network: {} nodes",
                    graph_net.num_nodes()
                );
                (VerifiableNetwork::Graph(graph_net), input_shape, output_dim)
            };

            let verifier = Verifier::new(config);

            // Keep original input shape from model (for VNNLIB)
            let input_shape_original = input_shape.clone();
            let input_dim_original: usize = input_shape.iter().product();

            // Squeeze out leading batch dimension of 1 for epsilon-ball verification
            // Conv1d/Conv2d layers expect unbatched inputs
            // Note: Do NOT squeeze when using VNNLIB - constants in model need original shape
            if input_shape.len() >= 2 && input_shape[0] == 1 {
                input_shape.remove(0);
                debug!("Squeezed batch dimension, input shape: {:?}", input_shape);
            }

            let input_dim: usize = input_shape.iter().product();

            // Load property file if provided, otherwise use epsilon bounds
            let (spec, vnnlib_spec) = if let Some(prop_path) = &property {
                use gamma_onnx::vnnlib::load_vnnlib;

                let vnnlib = load_vnnlib(prop_path)?;
                if !json {
                    println!(
                        "Loaded VNN-LIB property: {} inputs, {} outputs, {} constraints",
                        vnnlib.num_inputs,
                        vnnlib.num_outputs,
                        vnnlib.output_constraints.len()
                    );
                }

                // Validate dimensions (use original dim, not squeezed)
                if vnnlib.num_inputs != input_dim_original {
                    anyhow::bail!(
                        "Property file specifies {} inputs but model expects {} (shape {:?})",
                        vnnlib.num_inputs,
                        input_dim_original,
                        input_shape_original
                    );
                }

                // Convert VNN-LIB bounds to VerificationSpec
                // Squeeze leading batch dimension of 1 for conv layer compatibility
                // VNNLIB specifies per-sample bounds, so unbatched shape is correct
                let (lower, upper) = vnnlib.get_input_bounds_f32();
                let input_bounds: Vec<Bound> = lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&l, &u)| Bound::new(l, u))
                    .collect();

                // Use squeezed shape (remove batch dim of 1) for propagation
                let mut vnnlib_shape = input_shape_original.clone();
                if vnnlib_shape.len() >= 2 && vnnlib_shape[0] == 1 {
                    vnnlib_shape.remove(0);
                    debug!(
                        "Squeezed batch dimension for VNNLIB, shape: {:?}",
                        vnnlib_shape
                    );
                }

                let spec = VerificationSpec {
                    input_bounds,
                    output_bounds: vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); output_dim],
                    timeout_ms: Some(timeout * 1000),
                    input_shape: Some(vnnlib_shape), // Use unbatched shape for conv compatibility
                };

                (spec, Some(vnnlib))
            } else {
                // For epsilon-ball, use squeezed shape (for conv compatibility)
                let spec = VerificationSpec {
                    input_bounds: vec![Bound::new(-epsilon, epsilon); input_dim],
                    output_bounds: vec![Bound::new(f32::NEG_INFINITY, f32::INFINITY); output_dim],
                    timeout_ms: Some(timeout * 1000),
                    input_shape: Some(input_shape),
                };
                (spec, None::<gamma_onnx::vnnlib::VnnLibSpec>)
            };

            // Layer-by-layer mode: stream per-node bounds instead of full verification
            if layer_by_layer {
                // Create input tensor using shape from spec
                let effective_shape = spec.input_shape.as_ref().unwrap();
                let input_data = ArrayD::zeros(IxDyn(effective_shape));
                let input_bounded = BoundedTensor::from_epsilon(input_data, epsilon);

                match &network_choice {
                    VerifiableNetwork::Sequential(_net) => {
                        if !json {
                            println!("Layer-by-layer mode is only supported for native models (GraphNetwork)");
                            println!(
                                "Hint: use --native flag with PyTorch/SafeTensors/GGUF models"
                            );
                        } else {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "error": "layer-by-layer mode requires native model format"
                                })
                            );
                        }
                        return Ok(());
                    }
                    VerifiableNetwork::Graph(graph) => {
                        let start = Instant::now();
                        let show_progress = progress || progress_json;
                        let result = if show_progress {
                            let progress_callback = |p: LayerProgress| {
                                let pct = ((p.node_index + 1) as f32 / p.total_nodes as f32 * 100.0)
                                    as u32;
                                let eta = p.estimated_remaining();
                                if progress_json {
                                    // Output JSON line to stderr
                                    eprintln!(
                                        "{}",
                                        serde_json::json!({
                                            "type": "layer_progress",
                                            "node_index": p.node_index,
                                            "total_nodes": p.total_nodes,
                                            "node_name": p.node_name,
                                            "layer_type": p.layer_type,
                                            "percent": pct,
                                            "current_max_sensitivity": p.current_max_sensitivity,
                                            "degraded_so_far": p.degraded_so_far,
                                            "elapsed_ms": p.elapsed.as_millis(),
                                            "eta_ms": eta.as_millis()
                                        })
                                    );
                                } else {
                                    // Human-readable progress
                                    eprint!(
                                        "\r[{:3}%] Node {}/{}: {} ({}) | max_sens: {:.2e} | elapsed: {:.1?} | ETA: {:.1?}  ",
                                        pct,
                                        p.node_index + 1,
                                        p.total_nodes,
                                        p.node_name,
                                        p.layer_type,
                                        p.current_max_sensitivity,
                                        p.elapsed,
                                        eta
                                    );
                                }
                            };
                            let result = graph.propagate_ibp_detailed_with_progress(
                                &input_bounded,
                                epsilon,
                                Some(progress_callback),
                            )?;
                            if !progress_json {
                                eprintln!(); // Clear line for human-readable
                            }
                            result
                        } else {
                            graph.propagate_ibp_detailed(&input_bounded, epsilon)?
                        };
                        let elapsed = start.elapsed();

                        if json {
                            let nodes_json: Vec<_> = result
                                .nodes
                                .iter()
                                .map(|n| {
                                    serde_json::json!({
                                        "name": n.name,
                                        "layer_type": n.layer_type,
                                        "input_width": n.input_width,
                                        "output_width": n.output_width,
                                        "sensitivity": n.sensitivity,
                                        "output_shape": n.output_shape,
                                        "min_bound": n.min_bound,
                                        "max_bound": n.max_bound,
                                        "saturated": n.saturated,
                                        "has_nan": n.has_nan,
                                        "has_infinite": n.has_infinite,
                                        "status": n.status()
                                    })
                                })
                                .collect();

                            println!(
                                "{}",
                                serde_json::json!({
                                    "mode": "layer_by_layer",
                                    "method": "ibp",
                                    "input_epsilon": result.input_epsilon,
                                    "final_width": result.final_width,
                                    "total_nodes": result.total_nodes,
                                    "degraded_at_node": result.degraded_at_node,
                                    "elapsed_ms": elapsed.as_millis(),
                                    "nodes": nodes_json
                                })
                            );
                        } else {
                            println!("{}", result.summary());
                            println!("\nElapsed: {:.2?}", elapsed);
                        }
                        return Ok(());
                    }
                }
            }

            // Block-wise mode: resets bounds at each transformer block for zonotope tightening
            // Checkpoint implies block_wise mode
            let use_block_wise = block_wise || checkpoint.is_some();

            if use_block_wise {
                // Create input tensor using shape from spec
                let effective_shape = spec.input_shape.as_ref().unwrap();
                let input_data = ArrayD::zeros(IxDyn(effective_shape));
                let input_bounded = BoundedTensor::from_epsilon(input_data, epsilon);

                match &network_choice {
                    VerifiableNetwork::Sequential(_net) => {
                        if !json {
                            println!("Block-wise mode is only supported for native models (GraphNetwork)");
                            println!(
                                "Hint: use --native flag with PyTorch/SafeTensors/GGUF models"
                            );
                        } else {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "error": "block-wise mode requires native model format"
                                })
                            );
                        }
                        return Ok(());
                    }
                    VerifiableNetwork::Graph(graph) => {
                        let start = Instant::now();
                        let show_progress = progress || progress_json;

                        // Handle checkpoint: load existing or create new
                        let method_str = format!("{:?}", method).to_lowercase();
                        let backend_str = format!("{}", effective_backend);
                        let model_hash = compute_model_hash(&model)?;

                        let (existing_checkpoint, checkpoint_path) = if let Some(ref ckpt_path) =
                            checkpoint
                        {
                            if ckpt_path.exists() {
                                // Load and validate existing checkpoint
                                let ckpt = VerificationCheckpoint::load(ckpt_path)?;
                                ckpt.validate(
                                    &model,
                                    &model_hash,
                                    epsilon,
                                    &method_str,
                                    &backend_str,
                                )?;

                                if ckpt.is_complete() {
                                    // Checkpoint is already complete, just return results
                                    if !json {
                                        println!("Checkpoint is already complete. Showing previous results:");
                                    }
                                    let result = ckpt.into_result();
                                    if json {
                                        // Output JSON (simplified for resume case)
                                        println!(
                                            "{}",
                                            serde_json::json!({
                                                "mode": "block_wise",
                                                "method": "ibp+zonotope",
                                                "resumed_from_checkpoint": true,
                                                "total_blocks": result.total_blocks,
                                                "max_sensitivity": result.max_sensitivity,
                                                "degraded_blocks": result.degraded_blocks
                                            })
                                        );
                                    } else {
                                        println!("{}", result.summary());
                                    }
                                    return Ok(());
                                }

                                if !json {
                                    eprintln!(
                                        "Resuming from checkpoint: {}/{} blocks complete",
                                        ckpt.next_block_index, ckpt.total_blocks
                                    );
                                }
                                (Some(ckpt), Some(ckpt_path.clone()))
                            } else {
                                // Will create new checkpoint after first block
                                (None, Some(ckpt_path.clone()))
                            }
                        } else {
                            (None, None)
                        };

                        // Set up checkpoint state for callback (use RefCell for interior mutability)
                        use std::cell::RefCell;
                        use std::rc::Rc;

                        let checkpoint_state: Rc<RefCell<Option<VerificationCheckpoint>>> =
                            Rc::new(RefCell::new(existing_checkpoint.clone()));
                        let checkpoint_path_rc = Rc::new(checkpoint_path.clone());

                        let result = if show_progress || max_blocks > 0 || checkpoint.is_some() {
                            // Use progress callback to report to stderr
                            let progress_callback = if show_progress {
                                Some(|p: BlockProgress| {
                                    let pct = ((p.block_index + 1) as f32 / p.total_blocks as f32
                                        * 100.0)
                                        as u32;
                                    let eta = p.estimated_remaining();
                                    if progress_json {
                                        // Output JSON line to stderr
                                        eprintln!(
                                            "{}",
                                            serde_json::json!({
                                                "type": "block_progress",
                                                "block_index": p.block_index,
                                                "total_blocks": p.total_blocks,
                                                "block_name": p.block_name,
                                                "percent": pct,
                                                "current_max_sensitivity": p.current_max_sensitivity,
                                                "degraded_so_far": p.degraded_so_far,
                                                "elapsed_ms": p.elapsed.as_millis(),
                                                "eta_ms": eta.as_millis()
                                            })
                                        );
                                    } else {
                                        // Human-readable progress
                                        eprint!(
                                            "\r[{:3}%] Block {}/{}: {} | max_sens: {:.2e} | elapsed: {:.1?} | ETA: {:.1?}  ",
                                            pct,
                                            p.block_index + 1,
                                            p.total_blocks,
                                            p.block_name,
                                            p.current_max_sensitivity,
                                            p.elapsed,
                                            eta
                                        );
                                    }
                                })
                            } else {
                                None
                            };

                            // Checkpoint callback
                            let ckpt_state_clone = checkpoint_state.clone();
                            let ckpt_path_clone = checkpoint_path_rc.clone();
                            let model_path = model.clone();
                            let model_hash_clone = model_hash.clone();
                            let method_str_clone = method_str.clone();
                            let backend_str_clone = backend_str.clone();

                            let checkpoint_callback = if checkpoint.is_some() {
                                Some(
                                    move |block: &gamma_propagate::BlockBoundsInfo,
                                          elapsed_ms: u64,
                                          total_blocks: usize| {
                                        let mut state = ckpt_state_clone.borrow_mut();

                                        // Initialize checkpoint on first block if not resuming
                                        if state.is_none() {
                                            *state = Some(VerificationCheckpoint::new(
                                                model_path.clone(),
                                                model_hash_clone.clone(),
                                                epsilon,
                                                &method_str_clone,
                                                &backend_str_clone,
                                                total_blocks,
                                            ));
                                        }

                                        if let Some(ref mut ckpt) = *state {
                                            // Update total_blocks if needed
                                            ckpt.total_blocks = total_blocks;
                                            ckpt.update(block.clone(), elapsed_ms);

                                            // Save checkpoint to file
                                            if let Some(ref path) = *ckpt_path_clone {
                                                if let Err(e) = ckpt.save(path) {
                                                    eprintln!(
                                                        "\nWarning: Failed to save checkpoint: {}",
                                                        e
                                                    );
                                                }
                                            }
                                        }
                                    },
                                )
                            } else {
                                None
                            };

                            let result = graph.propagate_ibp_block_wise_with_checkpoint(
                                &input_bounded,
                                epsilon,
                                progress_callback,
                                checkpoint_callback,
                                max_blocks,
                                existing_checkpoint.as_ref(),
                            )?;
                            if show_progress && !progress_json {
                                eprintln!(); // newline after human-readable progress
                            }
                            result
                        } else {
                            graph.propagate_ibp_block_wise(&input_bounded, epsilon)?
                        };
                        let elapsed = start.elapsed();

                        if json {
                            let blocks_json: Vec<_> = result
                                .blocks
                                .iter()
                                .map(|b| {
                                    // Include per-node breakdown for detailed analysis
                                    let nodes_json: Vec<_> = b
                                        .nodes
                                        .iter()
                                        .map(|n| {
                                            serde_json::json!({
                                                "name": n.name,
                                                "layer_type": n.layer_type,
                                                "input_width": n.input_width,
                                                "output_width": n.output_width,
                                                "sensitivity": n.sensitivity,
                                                "saturated": n.saturated,
                                                "has_nan": n.has_nan,
                                                "has_infinite": n.has_infinite
                                            })
                                        })
                                        .collect();
                                    serde_json::json!({
                                        "block_index": b.block_index,
                                        "block_name": b.block_name,
                                        "input_width": b.input_width,
                                        "output_width": b.output_width,
                                        "sensitivity": b.sensitivity,
                                        "qk_matmul_width": b.qk_matmul_width,
                                        "swiglu_width": b.swiglu_width,
                                        "degraded": b.degraded,
                                        "status": b.status(),
                                        "num_nodes": b.nodes.len(),
                                        "nodes": nodes_json
                                    })
                                })
                                .collect();

                            // Build worst-k summary for regression tracking
                            let worst_5: Vec<_> = result
                                .worst_k_blocks(5)
                                .iter()
                                .map(|(idx, name, sens, out_width)| {
                                    serde_json::json!({
                                        "block_index": idx,
                                        "block_name": name,
                                        "sensitivity": sens,
                                        "output_width": out_width
                                    })
                                })
                                .collect();

                            println!(
                                "{}",
                                serde_json::json!({
                                    "mode": "block_wise",
                                    "method": "ibp+zonotope",
                                    "block_epsilon": result.block_epsilon,
                                    "total_blocks": result.total_blocks,
                                    "summary": {
                                        "max_sensitivity": result.max_sensitivity,
                                        "min_sensitivity": result.min_sensitivity(),
                                        "median_sensitivity": result.median_sensitivity(),
                                        "sensitivity_range": result.sensitivity_range(),
                                        "degraded_blocks": result.degraded_blocks,
                                        "worst_5_blocks": worst_5
                                    },
                                    "elapsed_ms": elapsed.as_millis(),
                                    "blocks": blocks_json
                                })
                            );
                        } else {
                            println!("{}", result.summary());
                            println!("\nElapsed: {:.2?}", elapsed);
                        }
                        return Ok(());
                    }
                }
            }

            let engine = compute_device
                .as_ref()
                .map(|d| d as &dyn gamma_core::GemmEngine);

            let result = match &network_choice {
                VerifiableNetwork::Sequential(net) => {
                    verifier.verify_with_engine(net, &spec, engine)?
                }
                VerifiableNetwork::Graph(graph) => {
                    verifier.verify_graph_with_engine(graph, &spec, engine)?
                }
            };

            // Check property constraints if VNN-LIB spec is provided
            let property_status = if let Some(ref vnnlib) = vnnlib_spec {
                // Check if the output bounds satisfy the property constraints
                // VNN-LIB specifies UNSAFE region, so if we can prove the constraints
                // CANNOT all be satisfied, the property is VERIFIED (safe)
                use gamma_onnx::vnnlib::OutputConstraint;

                let check_constraint_satisfiable = |bounds: &[Bound],
                                                    constraint: &OutputConstraint|
                 -> bool {
                    // Returns true if the constraint COULD be satisfied given the bounds
                    match constraint {
                        OutputConstraint::LessEq(i, j) => {
                            // Y_i <= Y_j could be satisfied if lower[i] <= upper[j]
                            bounds[*i].lower <= bounds[*j].upper
                        }
                        OutputConstraint::GreaterEq(i, j) => {
                            // Y_i >= Y_j could be satisfied if upper[i] >= lower[j]
                            bounds[*i].upper >= bounds[*j].lower
                        }
                        OutputConstraint::LessThan(i, j) => bounds[*i].lower < bounds[*j].upper,
                        OutputConstraint::GreaterThan(i, j) => bounds[*i].upper > bounds[*j].lower,
                        OutputConstraint::LessEqConst(i, c) => bounds[*i].lower <= *c as f32,
                        OutputConstraint::GreaterEqConst(i, c) => bounds[*i].upper >= *c as f32,
                        OutputConstraint::LessThanConst(i, c) => bounds[*i].lower < *c as f32,
                        OutputConstraint::GreaterThanConst(i, c) => bounds[*i].upper > *c as f32,
                    }
                };

                match &result {
                    gamma_core::VerificationResult::Verified { output_bounds, .. }
                    | gamma_core::VerificationResult::Unknown {
                        bounds: output_bounds,
                        ..
                    } => {
                        // VNN-LIB constraints are in disjunctive form (OR semantics):
                        // The unsafe region is satisfied if ANY constraint could be true.
                        // To prove SAFE, we need NONE of the constraints to be satisfiable.
                        let any_satisfiable = vnnlib
                            .output_constraints
                            .iter()
                            .any(|c| check_constraint_satisfiable(output_bounds, c));

                        if any_satisfiable {
                            Some("unknown") // At least one constraint might be satisfied
                        } else {
                            Some("safe") // No constraint can be satisfied - property verified
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            if json {
                use serde_json::json;
                let method_str = match method {
                    PropagationMethod::Ibp => "ibp",
                    PropagationMethod::Crown => "crown",
                    PropagationMethod::AlphaCrown => "alpha",
                    PropagationMethod::SdpCrown => "sdp-crown",
                    PropagationMethod::BetaCrown => "beta",
                };
                let mut output = match &result {
                    gamma_core::VerificationResult::Verified {
                        output_bounds,
                        proof,
                    } => {
                        let mut result = json!({
                            "status": "verified",
                            "output_bounds": output_bounds.iter().map(|b| {
                                json!({"lower": b.lower, "upper": b.upper})
                            }).collect::<Vec<_>>(),
                            "epsilon": epsilon,
                            "method": method_str,
                            "backend": effective_backend.to_string()
                        });
                        // Include proof info if available
                        if let Some(proof) = proof {
                            result.as_object_mut().unwrap().insert(
                                "proof".to_string(),
                                json!({
                                    "format": format!("{:?}", proof.format),
                                    "num_steps": proof.num_steps,
                                    "size_bytes": proof.data.len()
                                }),
                            );
                        }
                        result
                    }
                    gamma_core::VerificationResult::Violated {
                        counterexample,
                        output,
                        details,
                    } => {
                        let mut json_val = json!({
                            "status": "violated",
                            "counterexample": counterexample,
                            "output": output,
                            "epsilon": epsilon,
                            "method": method_str,
                            "backend": effective_backend.to_string()
                        });
                        // Add informative details if available
                        if let Some(ref details) = details {
                            if let Some(ref vc) = details.violated_constraint {
                                json_val["violation"] = json!({
                                    "output_idx": vc.output_idx,
                                    "actual_value": vc.actual_value,
                                    "required_lower": vc.required_bound.lower,
                                    "required_upper": vc.required_bound.upper,
                                    "violation_amount": vc.violation_amount,
                                    "explanation": vc.explain()
                                });
                            }
                            json_val["explanation"] = json!(details.explanation);
                        }
                        json_val
                    }
                    gamma_core::VerificationResult::Unknown { reason, bounds } => {
                        json!({
                            "status": "unknown",
                            "reason": reason,
                            "output_bounds": bounds.iter().map(|b| {
                                json!({"lower": b.lower, "upper": b.upper})
                            }).collect::<Vec<_>>(),
                            "epsilon": epsilon,
                            "method": method_str,
                            "backend": effective_backend.to_string()
                        })
                    }
                    gamma_core::VerificationResult::Timeout { partial_bounds } => {
                        json!({
                            "status": "timeout",
                            "partial_bounds": partial_bounds.as_ref().map(|bs| {
                                bs.iter().map(|b| {
                                    json!({"lower": b.lower, "upper": b.upper})
                                }).collect::<Vec<_>>()
                            }),
                            "epsilon": epsilon,
                            "method": method_str,
                            "backend": effective_backend.to_string()
                        })
                    }
                };

                // Add property status if vnnlib was used
                if let Some(status) = property_status {
                    output["property_status"] = json!(status);
                }
                if let Some(p) = &property {
                    output["property_file"] = json!(p.display().to_string());
                }

                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("\nVerification Result:");
                println!("{:?}", result);

                // Print property status if vnnlib was used
                if let Some(status) = property_status {
                    println!("\nProperty Status: {}", status.to_uppercase());
                    if status == "safe" {
                        println!("  The output bounds prove the property CANNOT be violated.");
                    } else {
                        println!(
                            "  The output bounds do not prove safety. Property may be violated."
                        );
                    }
                }
            }
        }

        Commands::Inspect {
            model,
            native,
            json,
        } => {
            info!("Inspecting model: {}", model.display());

            // Auto-detect native format based on extension/type if --native not specified
            let use_native = native || model.is_dir() || {
                let ext = model.extension().and_then(|e| e.to_str()).unwrap_or("");
                matches!(
                    ext,
                    "pt" | "pth" | "bin" | "safetensors" | "gguf" | "mlmodel" | "mlpackage"
                )
            };

            if use_native {
                use gamma_onnx::native::NativeModel;

                let native_model = NativeModel::load(&model)?;
                let network = &native_model.network;
                let config = &native_model.config;

                if json {
                    use serde_json::json;
                    let output = json!({
                        "name": network.name,
                        "architecture": format!("{:?}", config.architecture),
                        "hidden_dim": config.hidden_dim,
                        "num_heads": config.num_heads,
                        "num_layers": config.num_layers,
                        "parameters": network.param_count,
                        "weights": native_model.weights.len(),
                        "inputs": network.inputs.iter().map(|i| {
                            json!({
                                "name": i.name,
                                "shape": i.shape,
                                "dtype": format!("{:?}", i.dtype)
                            })
                        }).collect::<Vec<_>>(),
                        "outputs": network.outputs.iter().map(|o| {
                            json!({
                                "name": o.name,
                                "shape": o.shape,
                                "dtype": format!("{:?}", o.dtype)
                            })
                        }).collect::<Vec<_>>(),
                        "layer_count": network.layers.len()
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    println!("Network: {}", network.name);
                    println!("Architecture: {:?}", config.architecture);
                    println!("Hidden dim: {}", config.hidden_dim);
                    if let Some(heads) = config.num_heads {
                        println!("Attention heads: {}", heads);
                    }
                    if let Some(layers) = config.num_layers {
                        println!("Layers: {}", layers);
                    }
                    println!("Parameters: {}", network.param_count);
                    println!("Weight tensors: {}", native_model.weights.len());
                    println!("\nInputs:");
                    for input in &network.inputs {
                        println!("  {}: {:?} ({:?})", input.name, input.shape, input.dtype);
                    }
                    println!("\nOutputs:");
                    for output in &network.outputs {
                        println!("  {}: {:?} ({:?})", output.name, output.shape, output.dtype);
                    }
                    println!("\nNetwork layers: {}", network.layers.len());

                    // Show first few weight names
                    println!("\nFirst 10 weight tensors:");
                    for (name, weight) in native_model.weights.iter().take(10) {
                        println!("  {}: shape {:?}", name, weight.shape());
                    }
                    if native_model.weights.len() > 10 {
                        println!("  ... and {} more", native_model.weights.len() - 10);
                    }
                }
            } else {
                let onnx_model = load_onnx(&model)?;
                let network = &onnx_model.network;

                if json {
                    use serde_json::json;
                    let output = json!({
                        "name": network.name,
                        "parameters": network.param_count,
                        "inputs": network.inputs.iter().map(|i| {
                            json!({
                                "name": i.name,
                                "shape": i.shape,
                                "dtype": format!("{:?}", i.dtype)
                            })
                        }).collect::<Vec<_>>(),
                        "outputs": network.outputs.iter().map(|o| {
                            json!({
                                "name": o.name,
                                "shape": o.shape,
                                "dtype": format!("{:?}", o.dtype)
                            })
                        }).collect::<Vec<_>>(),
                        "layer_count": network.layers.len()
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    println!("Network: {}", network.name);
                    println!("Parameters: {}", network.param_count);
                    println!("\nInputs:");
                    for input in &network.inputs {
                        println!("  {}: {:?} ({:?})", input.name, input.shape, input.dtype);
                    }
                    println!("\nOutputs:");
                    for output in &network.outputs {
                        println!("  {}: {:?} ({:?})", output.name, output.shape, output.dtype);
                    }
                    println!("\nLayers: {}", network.layers.len());
                }
            }
        }

        Commands::Compare {
            reference,
            target,
            tolerance,
            epsilon,
            method,
            verbose,
            json: _,
        } => {
            info!("Comparing models:");
            info!("  Reference: {}", reference.display());
            info!("  Target: {}", target.display());
            info!("  Tolerance: {}", tolerance);
            info!("  Epsilon: {}", epsilon);
            info!("  Method: {}", method);

            // Load both models
            let ref_model = load_onnx(&reference).with_context(|| {
                format!("Failed to load reference model: {}", reference.display())
            })?;
            let target_model = load_onnx(&target)
                .with_context(|| format!("Failed to load target model: {}", target.display()))?;

            // Convert to propagation networks
            let ref_network = ref_model
                .to_propagate_network()
                .context("Failed to convert reference model to propagation network")?;
            let target_network = target_model
                .to_propagate_network()
                .context("Failed to convert target model to propagation network")?;

            // Get input shapes (use reference model's input shape)
            let ref_onnx_net = &ref_model.network;
            let target_onnx_net = &target_model.network;

            println!("Reference model: {}", ref_onnx_net.name);
            println!(
                "  Inputs: {:?}",
                ref_onnx_net
                    .inputs
                    .iter()
                    .map(|i| (&i.name, &i.shape))
                    .collect::<Vec<_>>()
            );
            println!(
                "  Outputs: {:?}",
                ref_onnx_net
                    .outputs
                    .iter()
                    .map(|o| (&o.name, &o.shape))
                    .collect::<Vec<_>>()
            );
            println!("  Layers: {}", ref_network.layers.len());

            println!("Target model: {}", target_onnx_net.name);
            println!(
                "  Inputs: {:?}",
                target_onnx_net
                    .inputs
                    .iter()
                    .map(|i| (&i.name, &i.shape))
                    .collect::<Vec<_>>()
            );
            println!(
                "  Outputs: {:?}",
                target_onnx_net
                    .outputs
                    .iter()
                    .map(|o| (&o.name, &o.shape))
                    .collect::<Vec<_>>()
            );
            println!("  Layers: {}", target_network.layers.len());

            // Verify input shapes match
            let ref_input_shape: Vec<usize> = ref_onnx_net
                .inputs
                .first()
                .map(|i| i.shape.iter().map(|&d| d.max(1) as usize).collect())
                .unwrap_or_else(|| vec![1]);

            let target_input_shape: Vec<usize> = target_onnx_net
                .inputs
                .first()
                .map(|i| i.shape.iter().map(|&d| d.max(1) as usize).collect())
                .unwrap_or_else(|| vec![1]);

            if ref_input_shape != target_input_shape {
                anyhow::bail!(
                    "Input shapes don't match: reference {:?} vs target {:?}",
                    ref_input_shape,
                    target_input_shape
                );
            }

            // Create bounded input
            let input_data = ArrayD::from_elem(IxDyn(&ref_input_shape), 0.0f32);
            let input = BoundedTensor::from_epsilon(input_data, epsilon);

            println!("\nInput: shape {:?}, epsilon {}", ref_input_shape, epsilon);
            println!("Method: {}", method);

            // Run bound propagation on both models
            let start = std::time::Instant::now();
            let ref_output = match method.as_str() {
                "ibp" => ref_network.propagate_ibp(&input)?,
                "crown" => ref_network.propagate_crown(&input)?,
                "alpha" => ref_network.propagate_alpha_crown(&input)?,
                _ => anyhow::bail!("Unknown method: {}. Use ibp, crown, or alpha", method),
            };
            let ref_time = start.elapsed();

            let start = std::time::Instant::now();
            let target_output = match method.as_str() {
                "ibp" => target_network.propagate_ibp(&input)?,
                "crown" => target_network.propagate_crown(&input)?,
                "alpha" => target_network.propagate_alpha_crown(&input)?,
                _ => unreachable!(),
            };
            let target_time = start.elapsed();

            // Compare outputs
            println!("\n--- Propagation Results ---");
            println!(
                "Reference: {:?} in {:.2}ms",
                ref_output.shape(),
                ref_time.as_secs_f64() * 1000.0
            );
            println!(
                "  Lower: min={:.6}, max={:.6}",
                ref_output
                    .lower
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min),
                ref_output
                    .lower
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            );
            println!(
                "  Upper: min={:.6}, max={:.6}",
                ref_output
                    .upper
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min),
                ref_output
                    .upper
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            );
            println!("  Max width: {:.6e}", ref_output.max_width());

            println!(
                "Target: {:?} in {:.2}ms",
                target_output.shape(),
                target_time.as_secs_f64() * 1000.0
            );
            println!(
                "  Lower: min={:.6}, max={:.6}",
                target_output
                    .lower
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min),
                target_output
                    .lower
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            );
            println!(
                "  Upper: min={:.6}, max={:.6}",
                target_output
                    .upper
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min),
                target_output
                    .upper
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            );
            println!("  Max width: {:.6e}", target_output.max_width());

            // Verify shapes match
            if ref_output.shape() != target_output.shape() {
                println!("\n--- FAIL: Output shapes don't match ---");
                anyhow::bail!(
                    "Output shapes don't match: reference {:?} vs target {:?}",
                    ref_output.shape(),
                    target_output.shape()
                );
            }

            // Compare element-wise: check if bounds are equivalent within tolerance
            // Two models are equivalent if for all elements:
            //   |ref_lower - target_lower| <= tolerance AND |ref_upper - target_upper| <= tolerance
            let ref_lower = &ref_output.lower;
            let ref_upper = &ref_output.upper;
            let target_lower = &target_output.lower;
            let target_upper = &target_output.upper;

            let mut max_lower_diff: f32 = 0.0;
            let mut max_upper_diff: f32 = 0.0;
            let mut violations = Vec::new();

            for (idx, (((&rl, &ru), &tl), &tu)) in ref_lower
                .iter()
                .zip(ref_upper.iter())
                .zip(target_lower.iter())
                .zip(target_upper.iter())
                .enumerate()
            {
                let lower_diff = (rl - tl).abs();
                let upper_diff = (ru - tu).abs();

                max_lower_diff = max_lower_diff.max(lower_diff);
                max_upper_diff = max_upper_diff.max(upper_diff);

                if (lower_diff > tolerance || upper_diff > tolerance)
                    && (violations.len() < 10 || verbose)
                {
                    violations.push((idx, rl, ru, tl, tu, lower_diff, upper_diff));
                }
            }

            // Compute overlap metric: what percentage of outputs have overlapping bounds?
            let mut overlap_count = 0usize;
            let total = ref_lower.len();
            for (((&rl, &ru), &tl), &tu) in ref_lower
                .iter()
                .zip(ref_upper.iter())
                .zip(target_lower.iter())
                .zip(target_upper.iter())
            {
                // Bounds overlap if max(lower) <= min(upper)
                let overlap = rl.max(tl) <= ru.min(tu);
                if overlap {
                    overlap_count += 1;
                }
            }
            let overlap_pct = 100.0 * overlap_count as f64 / total as f64;

            println!("\n--- Comparison Results ---");
            println!("Max lower bound diff: {:.6e}", max_lower_diff);
            println!("Max upper bound diff: {:.6e}", max_upper_diff);
            println!("Tolerance: {:.6e}", tolerance);
            println!(
                "Bound overlap: {}/{} ({:.2}%)",
                overlap_count, total, overlap_pct
            );

            let equivalent = max_lower_diff <= tolerance && max_upper_diff <= tolerance;

            if equivalent {
                println!("\n✓ EQUIVALENT: Models produce matching bounds within tolerance");
            } else {
                println!("\n✗ NOT EQUIVALENT: Models differ beyond tolerance");
                println!(
                    "\nViolations (first {}{}): ",
                    violations.len(),
                    if violations.len() < 10 && !verbose {
                        ""
                    } else {
                        ", use --verbose for all"
                    }
                );
                for (idx, rl, ru, tl, tu, ld, ud) in &violations {
                    println!(
                        "  [{}] ref=[{:.6}, {:.6}] target=[{:.6}, {:.6}] diff=({:.3e}, {:.3e})",
                        idx, rl, ru, tl, tu, ld, ud
                    );
                }
            }

            // Structured JSON output for AI workflows
            println!("\n--- JSON Summary ---");
            println!(
                r#"{{"equivalent": {}, "max_lower_diff": {:.6e}, "max_upper_diff": {:.6e}, "tolerance": {:.6e}, "overlap_pct": {:.2}, "ref_max_width": {:.6e}, "target_max_width": {:.6e}}}"#,
                equivalent,
                max_lower_diff,
                max_upper_diff,
                tolerance,
                overlap_pct,
                ref_output.max_width(),
                target_output.max_width()
            );
        }

        Commands::Diff {
            model_a,
            model_b,
            input,
            tolerance,
            continue_after_divergence,
            diagnose,
            json,
        } => {
            use gamma_onnx::diff::{diff_models, load_npy, DiffConfig, DiffStatus};

            if !json {
                info!("Comparing models layer-by-layer:");
                info!("  Model A: {}", model_a.display());
                info!("  Model B: {}", model_b.display());
                if diagnose {
                    info!("  Diagnosis: enabled");
                }
            }

            // Load input if provided
            let input_array = if let Some(input_path) = &input {
                if !json {
                    info!("  Input: {}", input_path.display());
                }
                Some(load_npy(input_path).context("Failed to load input .npy file")?)
            } else {
                if !json {
                    info!("  Input: synthetic zeros");
                }
                None
            };

            let config = DiffConfig {
                tolerance,
                continue_after_divergence,
                input: input_array,
                layer_mapping: std::collections::HashMap::new(),
                diagnose,
            };

            let result = diff_models(&model_a, &model_b, &config)
                .map_err(|e| anyhow::anyhow!("Diff failed: {}", e))?;

            if json {
                // JSON output for programmatic use
                let statuses = result.statuses();
                let layers_json: Vec<_> = result
                    .layers
                    .iter()
                    .zip(statuses.iter())
                    .map(|(layer, status)| {
                        let status_str = match status {
                            DiffStatus::Ok => "ok",
                            DiffStatus::DriftStarts => "drift_starts",
                            DiffStatus::ExceedsTolerance => "exceeds_tolerance",
                            DiffStatus::ShapeMismatch => "shape_mismatch",
                        };
                        serde_json::json!({
                            "name": layer.name,
                            "name_b": layer.name_b,
                            "max_diff": layer.max_diff,
                            "mean_diff": layer.mean_diff,
                            "exceeds_tolerance": layer.exceeds_tolerance,
                            "shape_a": layer.shape_a,
                            "shape_b": layer.shape_b,
                            "status": status_str
                        })
                    })
                    .collect();

                // Build diagnosis JSON if available
                let diagnosis_json = result.diagnosis.as_ref().map(|d| {
                    serde_json::json!({
                        "divergence_layer": d.divergence_layer,
                        "layer_type": format!("{:?}", d.layer_type),
                        "pattern": format!("{}", d.pattern),
                        "explanation": d.explanation,
                        "suggestion": d.suggestion,
                        "confidence": d.confidence,
                        "evidence": d.evidence
                    })
                });

                println!(
                    "{}",
                    serde_json::json!({
                        "equivalent": result.is_equivalent(),
                        "max_divergence": result.max_divergence,
                        "tolerance": result.tolerance,
                        "first_bad_layer": result.first_bad_layer,
                        "first_bad_layer_name": result.first_bad_layer_name(),
                        "drift_start_layer": result.drift_start_layer,
                        "suggestion": result.suggestion,
                        "diagnosis": diagnosis_json,
                        "model_a": model_a.display().to_string(),
                        "model_b": model_b.display().to_string(),
                        "input": input.as_ref().map(|p| p.display().to_string()),
                        "layers": layers_json
                    })
                );
            } else {
                // Human-readable output

                // Display results in table format
                println!("\nLayer-by-Layer Comparison");
                println!("==========================");
                println!("{:<40} | {:<12} | Status", "Layer", "Max Diff");
                println!("{:-<40}-+-{:-<12}-+--------", "", "");

                let statuses = result.statuses();
                for (layer, status) in result.layers.iter().zip(statuses.iter()) {
                    let status_str = match status {
                        DiffStatus::Ok => "OK",
                        DiffStatus::DriftStarts => "DRIFT STARTS HERE",
                        DiffStatus::ExceedsTolerance => "EXCEEDS TOLERANCE",
                        DiffStatus::ShapeMismatch => "SHAPE MISMATCH",
                    };
                    println!(
                        "{:<40} | {:>12.3e} | {}",
                        layer.name, layer.max_diff, status_str
                    );
                }

                // Print summary
                println!();
                if result.is_equivalent() {
                    println!(
                        "EQUIVALENT: Models produce matching outputs within tolerance {:.2e}",
                        tolerance
                    );
                } else {
                    println!(
                        "DIVERGENT: Models differ beyond tolerance {:.2e}",
                        tolerance
                    );

                    // Display detailed diagnosis if available
                    if let Some(ref diagnosis) = result.diagnosis {
                        println!("\nRoot Cause Analysis:");
                        println!("--------------------");
                        print!("{}", diagnosis.format_report());
                    } else {
                        // Fall back to simple root cause display
                        if let Some(name) = result.first_bad_layer_name() {
                            println!("\nFirst divergence at: {}", name);
                        }
                        if let Some(suggestion) = &result.suggestion {
                            println!("Suggestion: {}", suggestion);
                        }
                    }
                }
            }
        }

        Commands::Sensitivity {
            model,
            epsilon,
            continue_after_overflow,
            threshold,
            json,
        } => {
            use gamma_onnx::sensitivity::{analyze_sensitivity, SensitivityConfig};

            info!("Analyzing sensitivity: {}", model.display());

            let config = SensitivityConfig {
                epsilon,
                continue_after_overflow,
                input: None,
            };

            let result = analyze_sensitivity(&model, &config)
                .map_err(|e| anyhow::anyhow!("Sensitivity analysis failed: {}", e))?;

            if json {
                // JSON output for programmatic use
                let layers_json: Vec<_> = result
                    .layers
                    .iter()
                    .map(|l| {
                        serde_json::json!({
                            "name": l.name,
                            "layer_type": l.layer_type,
                            "input_width": l.input_width,
                            "output_width": l.output_width,
                            "sensitivity": l.sensitivity,
                            "has_overflow": l.has_overflow
                        })
                    })
                    .collect();

                println!(
                    "{}",
                    serde_json::json!({
                        "layers": layers_json,
                        "total_sensitivity": result.total_sensitivity,
                        "max_sensitivity": result.max_sensitivity,
                        "max_sensitivity_layer": result.max_sensitivity_layer
                            .and_then(|i| result.layers.get(i))
                            .map(|l| l.name.as_str()),
                        "input_epsilon": result.input_epsilon,
                        "final_width": result.final_width,
                        "overflow_at_layer": result.overflow_at_layer
                            .and_then(|i| result.layers.get(i))
                            .map(|l| l.name.as_str())
                    })
                );
            } else {
                // Human-readable output
                if let Some(thresh) = threshold {
                    // Filter to high-sensitivity layers only
                    let hot_spots = result.hot_spots(thresh);
                    if hot_spots.is_empty() {
                        println!("No layers with sensitivity > {:.2} found.", thresh);
                    } else {
                        println!("High-Sensitivity Layers (sensitivity > {:.2}):", thresh);
                        println!("{:-<60}", "");
                        for layer in hot_spots {
                            println!("  {:<40} sensitivity={:.2}", layer.name, layer.sensitivity);
                        }
                    }
                } else {
                    // Full summary
                    println!("{}", result.summary());
                }
            }
        }

        Commands::QuantizeCheck {
            model,
            epsilon,
            continue_after_overflow,
            float16_only,
            int8_only,
            json,
        } => {
            use gamma_onnx::quantize::{analyze_quantization, QuantFormat, QuantizeConfig};

            info!("Analyzing quantization safety: {}", model.display());

            let formats = if float16_only {
                vec![QuantFormat::Float16]
            } else if int8_only {
                vec![QuantFormat::Int8]
            } else {
                vec![QuantFormat::Float16, QuantFormat::Int8]
            };

            let config = QuantizeConfig {
                epsilon,
                continue_after_overflow,
                formats,
                input: None,
            };

            let result = analyze_quantization(&model, &config)
                .map_err(|e| anyhow::anyhow!("Quantization analysis failed: {}", e))?;

            if json {
                // JSON output for programmatic use
                let layers_json: Vec<_> = result
                    .layers
                    .iter()
                    .map(|l| {
                        serde_json::json!({
                            "name": l.name,
                            "layer_type": l.layer_type,
                            "min_bound": l.min_bound,
                            "max_bound": l.max_bound,
                            "max_abs": l.max_abs,
                            "float16_safety": format!("{}", l.float16_safety),
                            "int8_safety": format!("{}", l.int8_safety),
                            "int8_scale": l.int8_scale,
                            "has_overflow": l.has_overflow
                        })
                    })
                    .collect();

                println!(
                    "{}",
                    serde_json::json!({
                        "layers": layers_json,
                        "float16_safe": result.float16_safe,
                        "int8_safe": result.int8_safe,
                        "float16_overflow_count": result.float16_overflow_count,
                        "int8_overflow_count": result.int8_overflow_count,
                        "denormal_count": result.denormal_count,
                        "input_epsilon": result.input_epsilon
                    })
                );
            } else {
                // Human-readable output
                println!("{}", result.summary());

                // Print suggestions
                if !result.float16_safe {
                    println!("\nFloat16 Unsafe Layers:");
                    for layer in result.float16_unsafe_layers() {
                        println!(
                            "  {}: bounds [{:.3e}, {:.3e}]",
                            layer.name, layer.min_bound, layer.max_bound
                        );
                    }
                }

                if !result.int8_safe {
                    println!("\nInt8 Unsafe Layers:");
                    for layer in result.int8_unsafe_layers() {
                        println!(
                            "  {}: bounds [{:.3e}, {:.3e}]",
                            layer.name, layer.min_bound, layer.max_bound
                        );
                    }
                }

                if result.denormal_count > 0 {
                    println!("\nDenormal Warning Layers:");
                    for layer in result.denormal_layers() {
                        println!("  {}: values may be in float16 denormal range", layer.name);
                    }
                }
            }
        }

        Commands::ProfileBounds {
            model,
            epsilon,
            continue_after_overflow,
            threshold,
            native,
            json,
            center_zeros,
        } => {
            use gamma_onnx::profile::{profile_bounds_graph, ProfileConfig};

            info!("Profiling bounds: {}", model.display());

            // Note: input will be set below based on model input shape
            let mut config = ProfileConfig {
                epsilon,
                continue_after_overflow,
                input: None,
            };

            // If center_zeros, we need to create zeros-centered input after getting input shape
            let use_center_zeros = center_zeros;

            // Auto-detect native format based on extension if --native not specified
            let use_native = native || model.is_dir() || {
                let ext = model.extension().and_then(|e| e.to_str()).unwrap_or("");
                matches!(
                    ext,
                    "pt" | "pth" | "bin" | "safetensors" | "gguf" | "mlmodel" | "mlpackage"
                )
            };

            let result = if use_native {
                use gamma_onnx::native::NativeModel;

                let native_model = NativeModel::load(&model)?;
                let network = &native_model.network;

                info!(
                    "Loaded native model: {} ({:?}, {} params)",
                    network.name, native_model.config.architecture, network.param_count
                );

                // Convert to GraphNetwork for profiling
                let mut graph_net = native_model.to_graph_network()?;
                // Enable forward-mode LayerNorm for tighter bounds
                let num_modified = graph_net.set_layernorm_forward_mode(true);
                if num_modified > 0 && !json {
                    eprintln!(
                        "Note: enabled LayerNorm forward-mode for {} LayerNorm nodes",
                        num_modified
                    );
                }

                // Get input shape from network spec
                let input_shape: Vec<usize> = network
                    .inputs
                    .first()
                    .map(|i| {
                        i.shape
                            .iter()
                            .map(|&d| if d < 0 { 16 } else { d as usize })
                            .collect()
                    })
                    .unwrap_or_else(|| vec![native_model.config.hidden_dim]);

                // If center_zeros, create zeros-centered input for validation
                if use_center_zeros {
                    use gamma_tensor::BoundedTensor;
                    use ndarray::{ArrayD, IxDyn};
                    let zeros = ArrayD::zeros(IxDyn(&input_shape));
                    config.input = Some(BoundedTensor::from_epsilon(zeros, epsilon));
                }

                profile_bounds_graph(&graph_net, &config, &input_shape)
                    .map_err(|e| anyhow::anyhow!("Bound profiling failed: {}", e))?
            } else {
                use gamma_onnx::{load_onnx, profile::profile_bounds_model};

                // Load ONNX model first to get input shape
                let onnx_model = load_onnx(&model)?;

                // If center_zeros, create zeros-centered input for validation
                if use_center_zeros {
                    use gamma_tensor::BoundedTensor;
                    use ndarray::{ArrayD, IxDyn};

                    let input_spec =
                        onnx_model.network.inputs.first().ok_or_else(|| {
                            anyhow::anyhow!("No input specification in ONNX model")
                        })?;
                    let shape: Vec<usize> = input_spec
                        .shape
                        .iter()
                        .map(|&d| if d > 0 { d as usize } else { 1 })
                        .collect();
                    let zeros = ArrayD::zeros(IxDyn(&shape));
                    config.input = Some(BoundedTensor::from_epsilon(zeros, epsilon));
                }

                profile_bounds_model(&onnx_model, &config)
                    .map_err(|e| anyhow::anyhow!("Bound profiling failed: {}", e))?
            };

            if json {
                // JSON output for programmatic use
                let layers_json: Vec<_> = result
                    .layers
                    .iter()
                    .map(|l| {
                        serde_json::json!({
                            "name": l.name,
                            "layer_type": l.layer_type,
                            "input_width": l.input_width,
                            "output_width": l.output_width,
                            "mean_output_width": l.mean_output_width,
                            "median_output_width": l.median_output_width,
                            "growth_ratio": l.growth_ratio,
                            "cumulative_expansion": l.cumulative_expansion,
                            "status": format!("{}", l.status)
                        })
                    })
                    .collect();

                println!(
                    "{}",
                    serde_json::json!({
                        "layers": layers_json,
                        "input_epsilon": result.input_epsilon,
                        "initial_width": result.initial_width,
                        "final_width": result.final_width,
                        "total_expansion": result.total_expansion,
                        "max_growth_ratio": result.max_growth_ratio,
                        "max_growth_layer": result.max_growth_layer
                            .and_then(|i| result.layers.get(i))
                            .map(|l| l.name.as_str()),
                        "overflow_at_layer": result.overflow_at_layer
                            .and_then(|i| result.layers.get(i))
                            .map(|l| l.name.as_str()),
                        "difficulty_score": result.difficulty_score
                    })
                );
            } else {
                // Human-readable output
                if let Some(thresh) = threshold {
                    // Filter to high-growth layers only
                    let choke_points = result.choke_points(thresh);
                    if choke_points.is_empty() {
                        println!("No layers with growth > {:.2}x found.", thresh);
                    } else {
                        println!("Choke Points (growth > {:.2}x):", thresh);
                        println!("{:-<60}", "");
                        for layer in choke_points {
                            println!(
                                "  {:<40} growth={:.2}x status={}",
                                layer.name, layer.growth_ratio, layer.status
                            );
                        }
                    }
                } else {
                    // Full summary
                    println!("{}", result.summary());
                }

                // Print problematic layers
                let problems = result.problematic_layers();
                if !problems.is_empty() {
                    println!("\nProblematic Layers (WIDE or worse):");
                    for layer in problems {
                        println!(
                            "  {}: width={:.3e}, growth={:.2}x, status={}",
                            layer.name, layer.output_width, layer.growth_ratio, layer.status
                        );
                    }
                }
            }
        }

        Commands::Whisper {
            model,
            component,
            layer,
            epsilon,
            json,
        } => {
            if !json {
                info!("Verifying Whisper component: {}", component);
            }

            let whisper = load_whisper(&model)?;
            if !json {
                info!(
                    "Loaded Whisper model: {} encoder layers, {} decoder layers",
                    whisper.encoder_layers, whisper.decoder_layers
                );
            }

            let network = match component.as_str() {
                "encoder" => {
                    if let Some(idx) = layer {
                        whisper.encoder_layer(idx)?
                    } else {
                        whisper.encoder()?
                    }
                }
                "attention" => {
                    let layer_idx = layer.unwrap_or(0);
                    whisper.attention_head(layer_idx, 0)?
                }
                _ => {
                    anyhow::bail!("Unknown component: {}. Use encoder or attention", component);
                }
            };

            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "model": model.display().to_string(),
                        "component": component,
                        "layer": layer,
                        "epsilon": epsilon,
                        "num_layers": network.num_layers(),
                        "encoder_layers": whisper.encoder_layers,
                        "decoder_layers": whisper.decoder_layers,
                        "hidden_dim": whisper.hidden_dim,
                        "status": "not_implemented",
                        "message": "Verification not yet fully implemented for Whisper"
                    })
                );
            } else {
                println!(
                    "Component to verify: {} (layers: {})",
                    component,
                    network.num_layers()
                );
                println!("Perturbation epsilon: {}", epsilon);
                println!("\nVerification not yet fully implemented for Whisper");
            }
        }

        Commands::WhisperSeq {
            model,
            start_block,
            end_block,
            include_stem,
            include_ln_post,
            batch,
            seq_len,
            n_mels,
            time,
            epsilon,
            backend,
            gpu,
            mode,
            max_bound_width,
            terminate_on_overflow,
            continue_after_overflow,
            overflow_clamp_value,
            reset_zonotope_blocks,
            json,
        } => {
            let whisper = load_whisper(&model)?;
            let end_block = end_block.unwrap_or(whisper.encoder_layers);

            let config = make_multiblock_config(
                &mode,
                max_bound_width,
                terminate_on_overflow,
                continue_after_overflow,
                overflow_clamp_value,
                reset_zonotope_blocks,
            )?;

            let input = make_synthetic_input(
                &whisper,
                include_stem,
                batch,
                seq_len,
                n_mels,
                time,
                epsilon,
            );

            // Resolve backend and create device
            let effective_backend = resolve_backend(backend, gpu);
            let gpu_device = match effective_backend {
                BackendArg::Cpu => None,
                BackendArg::Wgpu => Some(ComputeDevice::new(Backend::Wgpu)?),
                BackendArg::Mlx => match ComputeDevice::new(Backend::Mlx) {
                    Ok(dev) => Some(dev),
                    Err(e) => {
                        if !json {
                            eprintln!("MLX backend not available: {}. Using CPU.", e);
                        }
                        None
                    }
                },
            };
            let gpu_ref = gpu_device.as_ref();

            if !json {
                println!("Model: {}", model.display());
                println!(
                    "Blocks: {}..{} ({} total in model)",
                    start_block, end_block, whisper.encoder_layers
                );
                println!(
                    "Input: {} (epsilon {})",
                    if include_stem {
                        format!("[batch={}, n_mels={}, time={}]", batch, n_mels, time)
                    } else {
                        format!(
                            "[batch={}, seq_len={}, hidden={}]",
                            batch, seq_len, whisper.hidden_dim
                        )
                    },
                    epsilon
                );
                println!(
                    "Config: mode={}, max_bound_width={:.2e}, terminate_on_overflow={}, continue_after_overflow={}, overflow_clamp_value={:.2e}",
                    mode,
                    config.max_bound_width,
                    config.terminate_on_overflow,
                    config.continue_after_overflow,
                    config.overflow_clamp_value
                );
                println!(
                    "Backend: {} (GPU: {})",
                    effective_backend,
                    if gpu_ref.is_some() {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }

            let (_out, details) = whisper.verify_encoder_sequential_with_config(
                &input,
                start_block,
                end_block,
                include_stem,
                include_ln_post,
                gpu_ref,
                &config,
            )?;

            if json {
                let blocks_json: Vec<_> = details
                    .block_details
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        serde_json::json!({
                            "block": start_block + i,
                            "attention_delta_width": b.attention_delta_width,
                            "x_attn_width": b.x_attn_width,
                            "mlp_delta_width": b.mlp_delta_width,
                            "output_width": b.output_width,
                            "used_gpu_attention": b.used_gpu_attention,
                            "seq_len": b.seq_len
                        })
                    })
                    .collect();

                println!(
                    "{}",
                    serde_json::json!({
                        "model": model.display().to_string(),
                        "start_block": start_block,
                        "end_block": end_block,
                        "include_stem": include_stem,
                        "include_ln_post": include_ln_post,
                        "epsilon": epsilon,
                        "backend": effective_backend.to_string(),
                        "gpu_enabled": gpu_ref.is_some(),
                        "config": {
                            "mode": mode,
                            "max_bound_width": config.max_bound_width,
                            "terminate_on_overflow": config.terminate_on_overflow,
                            "continue_after_overflow": config.continue_after_overflow,
                            "overflow_clamp_value": config.overflow_clamp_value
                        },
                        "result": {
                            "blocks_completed": details.blocks_completed,
                            "num_blocks": details.num_blocks,
                            "early_terminated": details.early_terminated,
                            "overflow_at_block": details.overflow_at_block,
                            "termination_reason": details.termination_reason,
                            "final_output_width": details.final_output_width,
                            "total_time_ms": details.total_time_ms
                        },
                        "blocks": blocks_json
                    })
                );
            } else {
                println!("\nResult:");
                println!(
                    "blocks_completed={} / {}, early_terminated={}, overflow_at_block={:?}",
                    details.blocks_completed,
                    details.num_blocks,
                    details.early_terminated,
                    details.overflow_at_block
                );
                if let Some(reason) = &details.termination_reason {
                    println!("termination_reason={}", reason);
                }
                println!("final_output_width={:.6e}", details.final_output_width);
                println!("total_time_ms={}", details.total_time_ms);

                if !details.block_details.is_empty() {
                    println!("\nPer-block:");
                    println!(
                        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>6} {:>6}",
                        "block", "attn", "x+attn", "mlp", "out", "gpu", "seq"
                    );
                    for (i, b) in details.block_details.iter().enumerate() {
                        println!(
                            "{:>6} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>6} {:>6}",
                            start_block + i,
                            b.attention_delta_width,
                            b.x_attn_width,
                            b.mlp_delta_width,
                            b.output_width,
                            if b.used_gpu_attention { "yes" } else { "no" },
                            b.seq_len
                        );
                    }
                }
            }
        }

        Commands::WhisperSweep {
            model,
            start_block,
            end_block,
            include_stem,
            include_ln_post,
            batch,
            seq_len,
            n_mels,
            time,
            epsilon_min,
            epsilon_max,
            steps,
            linear,
            backend,
            gpu,
            mode,
            max_bound_width,
            reset_zonotope_blocks,
            per_block,
            json,
        } => {
            let whisper = load_whisper(&model)?;
            let end_block = end_block.unwrap_or(whisper.encoder_layers);

            let mut config = make_multiblock_config(
                &mode,
                max_bound_width,
                None,
                None,
                None,
                reset_zonotope_blocks,
            )?;

            // If the caller left strict defaults but removed the threshold, enforce a reasonable default
            // so the sweep terminates before f32 overflow.
            if mode == "strict" && config.max_bound_width == f32::MAX {
                config.max_bound_width = 1e20;
            }

            // Resolve backend and create device
            let effective_backend = resolve_backend(backend, gpu);
            let gpu_device = match effective_backend {
                BackendArg::Cpu => None,
                BackendArg::Wgpu => Some(ComputeDevice::new(Backend::Wgpu)?),
                BackendArg::Mlx => match ComputeDevice::new(Backend::Mlx) {
                    Ok(dev) => Some(dev),
                    Err(e) => {
                        if !json {
                            eprintln!("MLX backend not available: {}. Using CPU.", e);
                        }
                        None
                    }
                },
            };
            let gpu_ref = gpu_device.as_ref();

            let eps_list = eps_sweep(epsilon_min, epsilon_max, steps, linear)?;

            if !json {
                println!("Model: {}", model.display());
                println!(
                    "Blocks: {}..{} ({} blocks requested)",
                    start_block,
                    end_block,
                    end_block.saturating_sub(start_block)
                );
                println!(
                    "Input: {}",
                    if include_stem {
                        format!("[batch={}, n_mels={}, time={}]", batch, n_mels, time)
                    } else {
                        format!(
                            "[batch={}, seq_len={}, hidden={}]",
                            batch, seq_len, whisper.hidden_dim
                        )
                    }
                );
                println!(
                    "Sweep: {} points, {} space, eps in [{:.2e}, {:.2e}]",
                    steps,
                    if linear { "linear" } else { "log" },
                    epsilon_min,
                    epsilon_max
                );
                println!(
                    "Config: mode={}, max_bound_width={:.2e}, terminate_on_overflow={}, continue_after_overflow={}",
                    mode, config.max_bound_width, config.terminate_on_overflow, config.continue_after_overflow
                );
                println!(
                    "Backend: {} (GPU: {})",
                    effective_backend,
                    if gpu_ref.is_some() {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );

                println!(
                    "\n{:>12} {:>8} {:>8} {:>12} {:>12} {:>10}",
                    "epsilon", "done", "early", "overflow", "final_w", "time_ms"
                );
            }

            // Collect results for JSON output
            let mut sweep_results = Vec::new();

            for eps in &eps_list {
                let input = make_synthetic_input(
                    &whisper,
                    include_stem,
                    batch,
                    seq_len,
                    n_mels,
                    time,
                    *eps,
                );

                let res = whisper.verify_encoder_sequential_with_config(
                    &input,
                    start_block,
                    end_block,
                    include_stem,
                    include_ln_post,
                    gpu_ref,
                    &config,
                );

                match res {
                    Ok((_out, details)) => {
                        if json {
                            let blocks_json: Vec<_> = if per_block {
                                details
                                    .block_details
                                    .iter()
                                    .enumerate()
                                    .map(|(i, b)| {
                                        serde_json::json!({
                                            "block": start_block + i,
                                            "attention_delta_width": b.attention_delta_width,
                                            "x_attn_width": b.x_attn_width,
                                            "mlp_delta_width": b.mlp_delta_width,
                                            "output_width": b.output_width
                                        })
                                    })
                                    .collect()
                            } else {
                                Vec::new()
                            };

                            sweep_results.push(serde_json::json!({
                                "epsilon": eps,
                                "blocks_completed": details.blocks_completed,
                                "num_blocks": details.num_blocks,
                                "early_terminated": details.early_terminated,
                                "overflow_at_block": details.overflow_at_block,
                                "final_output_width": details.final_output_width,
                                "total_time_ms": details.total_time_ms,
                                "blocks": blocks_json,
                                "error": serde_json::Value::Null
                            }));
                        } else {
                            println!(
                                "{:>12.3e} {:>8} {:>8} {:>12} {:>12.3e} {:>10}",
                                eps,
                                format!("{}/{}", details.blocks_completed, details.num_blocks),
                                if details.early_terminated {
                                    "yes"
                                } else {
                                    "no"
                                },
                                details
                                    .overflow_at_block
                                    .map(|b| b.to_string())
                                    .unwrap_or_else(|| "-".to_string()),
                                details.final_output_width,
                                details.total_time_ms
                            );

                            // Print per-block widths if requested
                            if per_block && !details.block_details.is_empty() {
                                for (i, b) in details.block_details.iter().enumerate() {
                                    println!(
                                        "      block[{}]: attn={:.2e} x+attn={:.2e} mlp={:.2e} out={:.2e}",
                                        start_block + i,
                                        b.attention_delta_width,
                                        b.x_attn_width,
                                        b.mlp_delta_width,
                                        b.output_width
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if json {
                            sweep_results.push(serde_json::json!({
                                "epsilon": eps,
                                "error": e.to_string()
                            }));
                        } else {
                            println!(
                                "{:>12.3e} {:>8} {:>8} {:>12} {:>12} {:>10}",
                                eps, "err", "-", "-", "-", "-"
                            );
                            info!("epsilon {} failed: {:?}", eps, e);
                        }
                    }
                }
            }

            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "model": model.display().to_string(),
                        "start_block": start_block,
                        "end_block": end_block,
                        "include_stem": include_stem,
                        "include_ln_post": include_ln_post,
                        "backend": effective_backend.to_string(),
                        "gpu_enabled": gpu_ref.is_some(),
                        "sweep": {
                            "epsilon_min": epsilon_min,
                            "epsilon_max": epsilon_max,
                            "steps": steps,
                            "linear": linear
                        },
                        "config": {
                            "mode": mode,
                            "max_bound_width": config.max_bound_width,
                            "terminate_on_overflow": config.terminate_on_overflow,
                            "continue_after_overflow": config.continue_after_overflow
                        },
                        "results": sweep_results
                    })
                );
            }
        }

        Commands::WhisperEpsSearch {
            model,
            start_block,
            end_block,
            target_blocks,
            include_stem,
            include_ln_post,
            batch,
            seq_len,
            n_mels,
            time,
            epsilon_min,
            epsilon_max,
            iterations,
            backend,
            gpu,
            mode,
            max_bound_width,
            reset_zonotope_blocks,
            verbose_search,
            json,
        } => {
            let whisper = load_whisper(&model)?;
            let end_block = end_block.unwrap_or(whisper.encoder_layers);
            let num_blocks = end_block.saturating_sub(start_block);
            let target_blocks = target_blocks.unwrap_or(num_blocks);

            if target_blocks == 0 || target_blocks > num_blocks {
                anyhow::bail!(
                    "target_blocks must be in [1, {}] (got {})",
                    num_blocks,
                    target_blocks
                );
            }

            let mut config = make_multiblock_config(
                &mode,
                max_bound_width,
                None,
                None,
                None,
                reset_zonotope_blocks,
            )?;
            if mode == "strict" && config.max_bound_width == f32::MAX {
                config.max_bound_width = 1e20;
            }

            // Resolve backend and create device
            let effective_backend = resolve_backend(backend, gpu);
            let gpu_device = match effective_backend {
                BackendArg::Cpu => None,
                BackendArg::Wgpu => Some(ComputeDevice::new(Backend::Wgpu)?),
                BackendArg::Mlx => match ComputeDevice::new(Backend::Mlx) {
                    Ok(dev) => Some(dev),
                    Err(e) => {
                        if !json {
                            eprintln!("MLX backend not available: {}. Using CPU.", e);
                        }
                        None
                    }
                },
            };
            let gpu_ref = gpu_device.as_ref();

            if !json {
                println!("Model: {}", model.display());
                println!(
                    "Blocks: {}..{} ({} blocks), target={} to complete",
                    start_block, end_block, num_blocks, target_blocks
                );
                println!(
                    "Input: {}",
                    if include_stem {
                        format!("[batch={}, n_mels={}, time={}]", batch, n_mels, time)
                    } else {
                        format!(
                            "[batch={}, seq_len={}, hidden={}]",
                            batch, seq_len, whisper.hidden_dim
                        )
                    }
                );
                println!(
                    "Search: {} iterations in [{:.2e}, {:.2e}]",
                    iterations, epsilon_min, epsilon_max
                );
                println!(
                    "Config: mode={}, max_bound_width={:.2e}, terminate_on_overflow={}",
                    mode, config.max_bound_width, config.terminate_on_overflow
                );
                println!(
                    "Backend: {} (GPU: {})",
                    effective_backend,
                    if gpu_ref.is_some() {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }

            // Binary search in log space for numerical stability
            let mut log_low = epsilon_min.ln();
            let mut log_high = epsilon_max.ln();
            let mut best_eps: Option<f32> = None;
            let mut best_details: Option<gamma_onnx::MultiBlockDetails> = None;

            // Collect search history for JSON output
            let mut search_history: Vec<serde_json::Value> = Vec::new();

            // Helper to test an epsilon value
            let test_eps = |eps: f32| -> Result<(bool, gamma_onnx::MultiBlockDetails)> {
                let input =
                    make_synthetic_input(&whisper, include_stem, batch, seq_len, n_mels, time, eps);
                let (_out, details) = whisper.verify_encoder_sequential_with_config(
                    &input,
                    start_block,
                    end_block,
                    include_stem,
                    include_ln_post,
                    gpu_ref,
                    &config,
                )?;
                let success =
                    details.blocks_completed >= target_blocks && !details.early_terminated;
                Ok((success, details))
            };

            if verbose_search && !json {
                println!(
                    "\n{:>5} {:>12} {:>8} {:>8} {:>12} {:>10}",
                    "iter", "epsilon", "done", "success", "final_w", "time_ms"
                );
            }

            for iter in 0..iterations {
                let log_mid = (log_low + log_high) / 2.0;
                let eps_mid = log_mid.exp();

                match test_eps(eps_mid) {
                    Ok((success, details)) => {
                        if json && verbose_search {
                            search_history.push(serde_json::json!({
                                "iteration": iter,
                                "epsilon": eps_mid,
                                "blocks_completed": details.blocks_completed,
                                "num_blocks": details.num_blocks,
                                "success": success,
                                "final_output_width": details.final_output_width,
                                "total_time_ms": details.total_time_ms,
                                "error": serde_json::Value::Null
                            }));
                        } else if verbose_search {
                            println!(
                                "{:>5} {:>12.3e} {:>8} {:>8} {:>12.3e} {:>10}",
                                iter,
                                eps_mid,
                                format!("{}/{}", details.blocks_completed, details.num_blocks),
                                if success { "yes" } else { "no" },
                                details.final_output_width,
                                details.total_time_ms
                            );
                        }

                        if success {
                            // Can go higher
                            best_eps = Some(eps_mid);
                            best_details = Some(details);
                            log_low = log_mid;
                        } else {
                            // Need to go lower
                            log_high = log_mid;
                        }
                    }
                    Err(e) => {
                        if json && verbose_search {
                            search_history.push(serde_json::json!({
                                "iteration": iter,
                                "epsilon": eps_mid,
                                "success": false,
                                "error": e.to_string()
                            }));
                        } else if verbose_search {
                            println!(
                                "{:>5} {:>12.3e} {:>8} {:>8} {:>12} {:>10}",
                                iter, eps_mid, "err", "no", "-", "-"
                            );
                        }
                        if !json {
                            info!("epsilon {} failed: {:?}", eps_mid, e);
                        }
                        // Treat errors as needing to go lower
                        log_high = log_mid;
                    }
                }
            }

            // Check if epsilon_max also succeeds
            let epsilon_max_succeeds = test_eps(epsilon_max).map(|(s, _)| s).unwrap_or(false);

            if json {
                let result_json = match (best_eps, &best_details) {
                    (Some(eps), Some(details)) => {
                        serde_json::json!({
                            "found": true,
                            "max_epsilon": eps,
                            "blocks_completed": details.blocks_completed,
                            "num_blocks": details.num_blocks,
                            "target_blocks": target_blocks,
                            "final_output_width": details.final_output_width,
                            "total_time_ms": details.total_time_ms,
                            "epsilon_max_succeeds": epsilon_max_succeeds
                        })
                    }
                    _ => {
                        serde_json::json!({
                            "found": false,
                            "target_blocks": target_blocks,
                            "suggestion": "Try lowering epsilon_min or increasing target_blocks tolerance"
                        })
                    }
                };

                println!(
                    "{}",
                    serde_json::json!({
                        "model": model.display().to_string(),
                        "start_block": start_block,
                        "end_block": end_block,
                        "num_blocks": num_blocks,
                        "target_blocks": target_blocks,
                        "include_stem": include_stem,
                        "include_ln_post": include_ln_post,
                        "backend": effective_backend.to_string(),
                        "gpu_enabled": gpu_ref.is_some(),
                        "search": {
                            "epsilon_min": epsilon_min,
                            "epsilon_max": epsilon_max,
                            "iterations": iterations
                        },
                        "config": {
                            "mode": mode,
                            "max_bound_width": config.max_bound_width,
                            "terminate_on_overflow": config.terminate_on_overflow
                        },
                        "history": if verbose_search { search_history } else { Vec::new() },
                        "result": result_json
                    })
                );
            } else {
                println!("\n--- Search Result ---");
                match (best_eps, best_details) {
                    (Some(eps), Some(details)) => {
                        println!("max_epsilon={:.6e}", eps);
                        println!(
                            "blocks_completed={}/{}, target={}",
                            details.blocks_completed, details.num_blocks, target_blocks
                        );
                        println!("final_output_width={:.6e}", details.final_output_width);
                        println!("total_time_ms={}", details.total_time_ms);

                        // Check if we should also test bounds
                        if epsilon_max_succeeds {
                            println!(
                                "\nNote: epsilon_max ({:.2e}) also succeeds. Max may be higher.",
                                epsilon_max
                            );
                        }
                    }
                    _ => {
                        println!(
                            "No epsilon found in [{:.2e}, {:.2e}] that completes {} blocks.",
                            epsilon_min, epsilon_max, target_blocks
                        );
                        println!("Try lowering epsilon_min or increasing target_blocks tolerance.");
                    }
                }
            }
        }

        Commands::Export {
            model_type,
            size,
            output,
        } => {
            let script = match model_type.as_str() {
                "whisper" => gamma_onnx::generate_whisper_export_script(&size),
                _ => {
                    anyhow::bail!("Unknown model type: {}", model_type);
                }
            };

            if let Some(path) = output {
                std::fs::write(&path, &script)?;
                println!("Export script written to: {}", path.display());
            } else {
                println!("{}", script);
            }
        }

        Commands::Bench { benchmark, json } => {
            if !json {
                info!("Running benchmark: {}", benchmark);
            }
            run_benchmarks(&benchmark, json)?;
        }

        Commands::BetaCrown {
            model,
            property,
            epsilon,
            threshold,
            max_domains,
            timeout,
            max_depth,
            branching,
            fsb_candidates,
            no_alpha,
            alpha_iterations,
            no_adaptive_alpha_skip,
            alpha_skip_depth,
            crown_ibp_intermediates,
            alpha_spsa_samples,
            alpha_lr,
            alpha_gradient_method,
            alpha_optimizer,
            beta_iterations,
            beta_max_depth,
            lr_beta,
            crown_ibp,
            batch_size,
            sequential_children,
            enable_cuts,
            no_cuts,
            max_cuts,
            min_cut_depth,
            enable_near_miss_cuts,
            near_miss_margin,
            proactive_cuts,
            max_proactive_cuts,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            backend,
            gpu,
            json,
        } => {
            let use_alpha = !no_alpha;

            // Resolve effective backend (--gpu for backward compat, --backend takes precedence)
            let effective_backend = resolve_backend(backend, gpu);
            let use_gpu_backend = effective_backend != BackendArg::Cpu;
            let enable_cuts = enable_cuts && !no_cuts;

            // GPU batching is now automatic for DAG models (Issue #12 implemented)
            // No warning needed - tensor-level batching reduces kernel launch overhead

            info!("Running β-CROWN verification on: {}", model.display());

            enum BetaCrownModel {
                Sequential(gamma_propagate::Network),
                Graph(gamma_propagate::GraphNetwork),
            }

            // Parse branching heuristic
            // Note: "relu" is a special value that triggers ReLU-splitting for DAG models
            let (branching_heuristic, use_relu_split) = match branching.as_str() {
                "width" => (BranchingHeuristic::LargestBoundWidth, false),
                "impact" => (BranchingHeuristic::BoundImpact, false),
                "fsb" => (BranchingHeuristic::FilteredSmartBranching, false),
                "sequential" => (BranchingHeuristic::Sequential, false),
                "input" => (BranchingHeuristic::InputSplit, false),
                "relu" => (BranchingHeuristic::LargestBoundWidth, true), // ReLU split for DAGs
                _ => {
                    anyhow::bail!(
                        "Unknown branching heuristic: {}. Use width, impact, fsb, sequential, input, or relu",
                        branching
                    );
                }
            };

            // Parallel children is enabled by default, --sequential-children disables it
            let parallel_children = !sequential_children;

            // Check if NNet format
            let is_nnet = model.extension().and_then(|e| e.to_str()) == Some("nnet");

            // Load model (NNet or ONNX)
            let (model_net, input_dim, output_dim, input_shape_from_model, is_graph_model) =
                if is_nnet {
                    use gamma_onnx::nnet::load_nnet;

                    let nnet = load_nnet(&model)?;
                    info!(
                        "Loaded NNet: {} layers, {} inputs, {} outputs, {} params",
                        nnet.num_layers,
                        nnet.input_size,
                        nnet.output_size,
                        nnet.param_count()
                    );

                    let network = nnet.to_prop_network()?;
                    // NNet format has flat inputs [1, num_inputs], and is always sequential
                    (
                        BetaCrownModel::Sequential(network),
                        nnet.input_size,
                        nnet.output_size,
                        vec![1, nnet.input_size],
                        false, // NNet models are always sequential, not graphs
                    )
                } else {
                    let onnx_model = load_onnx(&model)?;
                    let onnx_network = &onnx_model.network;
                    info!(
                        "Loaded network: {} ({} layers, {} params)",
                        onnx_network.name,
                        onnx_network.layers.len(),
                        onnx_network.param_count
                    );

                    // Convert to GraphNetwork and route DAG models (residual connections, attention, etc.)
                    let graph = onnx_model.to_graph_network()?;
                    let needs_graph = graph.node_names().iter().any(|name| {
                        graph
                            .get_node(name)
                            .is_some_and(|node| node.layer.is_binary())
                    });
                    if needs_graph {
                        info!(
                            "Detected non-sequential graph (binary ops); using GraphNetwork path"
                        );
                        if !matches!(branching_heuristic, BranchingHeuristic::InputSplit)
                            && !use_relu_split
                        {
                            anyhow::bail!(
                            "Model is a DAG (e.g., residual/attention). β-CROWN supports DAGs with --branching input (input splitting) or --branching relu (ReLU splitting)"
                        );
                        }
                    }

                    // Get actual input shape from ONNX model (preserves all dimensions)
                    // Note: Both 0 and -1 indicate dynamic dimensions in ONNX
                    let input_shape_vec: Vec<usize> = onnx_network
                        .inputs
                        .first()
                        .map(|i| {
                            i.shape
                                .iter()
                                .map(|&d| if d <= 0 { 1 } else { d as usize })
                                .collect()
                        })
                        .unwrap_or_else(|| vec![1]);

                    let input_dim: usize = input_shape_vec.iter().product();

                    let output_dim: usize = onnx_network
                        .outputs
                        .first()
                        .map(|o| {
                            o.shape
                                .iter()
                                .map(|&d| if d <= 0 { 1 } else { d as usize })
                                .product()
                        })
                        .unwrap_or(1);

                    let model_net = if needs_graph {
                        BetaCrownModel::Graph(graph)
                    } else {
                        BetaCrownModel::Sequential(onnx_model.to_propagate_network()?)
                    };

                    (
                        model_net,
                        input_dim,
                        output_dim,
                        input_shape_vec,
                        needs_graph,
                    )
                };

            match &model_net {
                BetaCrownModel::Sequential(network) => {
                    if network.layers.is_empty() {
                        anyhow::bail!("No layers in network - β-CROWN requires at least one layer");
                    }
                }
                BetaCrownModel::Graph(graph) => {
                    if graph.num_nodes() == 0 {
                        anyhow::bail!(
                            "No nodes in graph - β-CROWN requires at least one operation"
                        );
                    }
                }
            }

            // Initialize compute device if not CPU
            let compute_device = if effective_backend != BackendArg::Cpu {
                match ComputeDevice::new(effective_backend.into()) {
                    Ok(d) => {
                        info!(
                            "Using {} backend for GPU-accelerated CROWN",
                            effective_backend
                        );
                        Some(d)
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to create {} device: {}. Falling back to CPU.",
                            effective_backend, e
                        );
                        None
                    }
                }
            } else {
                None
            };
            if use_gpu_backend && compute_device.is_none() {
                warn!("GPU backend requested but unavailable - using CPU");
            }

            // Create GemmEngine reference for GPU-accelerated CROWN operations
            let gemm_engine = compute_device
                .as_ref()
                .map(|d| d as &dyn gamma_core::GemmEngine);

            // Determine whether we need to squeeze batch dimension for Conv inputs.
            // This handles both:
            // 1. Direct Conv2d first layer (NCHW format)
            // 2. Transpose -> Conv2d (NHWC format converted to NCHW)
            let needs_squeeze = match &model_net {
                BetaCrownModel::Sequential(network) => {
                    // Check if first layer is Conv, or first is Transpose and second is Conv
                    let first_is_conv = network.layers.first().is_some_and(|l| {
                        matches!(
                            l,
                            gamma_propagate::Layer::Conv2d(_) | gamma_propagate::Layer::Conv1d(_)
                        )
                    });
                    let transpose_then_conv = network.layers.len() >= 2
                        && matches!(network.layers[0], gamma_propagate::Layer::Transpose(_))
                        && matches!(
                            network.layers[1],
                            gamma_propagate::Layer::Conv2d(_) | gamma_propagate::Layer::Conv1d(_)
                        );
                    first_is_conv || transpose_then_conv
                }
                BetaCrownModel::Graph(graph) => {
                    let exec_order = graph.topological_sort()?;

                    // Find nodes that directly take "_input"
                    let first_nodes: Vec<_> = exec_order
                        .iter()
                        .filter_map(|name| {
                            let node = graph.get_node(name)?;
                            if node.inputs.iter().any(|i| i == "_input") {
                                Some((name.as_str(), &node.layer))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Check if first layer is Conv, or if first is Transpose and second is Conv
                    // The latter handles NHWC -> NCHW conversion before Conv
                    let squeeze = first_nodes.iter().any(|(name, layer)| {
                        if matches!(
                            layer,
                            gamma_propagate::Layer::Conv2d(_) | gamma_propagate::Layer::Conv1d(_)
                        ) {
                            return true;
                        }
                        // If first layer is Transpose, check if its output feeds a Conv
                        if matches!(layer, gamma_propagate::Layer::Transpose(_)) {
                            // Find nodes that take this node's output
                            for next_name in &exec_order {
                                if let Some(next_node) = graph.get_node(next_name) {
                                    if next_node.inputs.iter().any(|i| i == *name)
                                        && matches!(
                                            next_node.layer,
                                            gamma_propagate::Layer::Conv2d(_)
                                                | gamma_propagate::Layer::Conv1d(_)
                                        )
                                    {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    });
                    squeeze
                }
            };

            // Create input bounds (VNNLIB or epsilon-ball)
            let (
                input,
                effective_threshold,
                vnnlib_spec,
                verify_upper,
                has_relational,
                const_output_idx,
            ) = if let Some(prop_path) = &property {
                use gamma_onnx::vnnlib::{load_vnnlib, OutputConstraint};

                let vnnlib = load_vnnlib(prop_path)?;
                info!(
                    "Loaded VNNLIB: {} inputs, {} outputs, {} constraints",
                    vnnlib.num_inputs,
                    vnnlib.num_outputs,
                    vnnlib.output_constraints.len()
                );

                // Create input bounds from VNNLIB using model's expected input shape
                let (lower_bounds, upper_bounds) = vnnlib.get_input_bounds_f32();

                // Validate dimensions match
                if vnnlib.num_inputs != input_dim {
                    anyhow::bail!(
                        "VNNLIB specifies {} inputs but model expects {} (shape {:?})",
                        vnnlib.num_inputs,
                        input_dim,
                        input_shape_from_model
                    );
                }

                // Use model's actual input shape (may be [1,1,1,5] for ACAS-Xu ONNX)
                // For Conv2d first layer, squeeze batch dimension of 1 since Conv2d expects (C,H,W)
                let mut effective_shape = input_shape_from_model.clone();
                if needs_squeeze && effective_shape.len() >= 2 && effective_shape[0] == 1 {
                    effective_shape.remove(0);
                    info!(
                        "Squeezed batch dimension for Conv layer, shape: {:?}",
                        effective_shape
                    );
                }
                let lower =
                    ArrayD::from_shape_vec(IxDyn(&effective_shape), lower_bounds).map_err(|e| {
                        anyhow::anyhow!(
                            "Failed to create lower bounds with shape {:?}: {}",
                            effective_shape,
                            e
                        )
                    })?;
                let upper =
                    ArrayD::from_shape_vec(IxDyn(&effective_shape), upper_bounds).map_err(|e| {
                        anyhow::anyhow!(
                            "Failed to create upper bounds with shape {:?}: {}",
                            effective_shape,
                            e
                        )
                    })?;
                let input = BoundedTensor::new(lower, upper)?;

                // Classify constraints: constant vs relational
                let has_relational = vnnlib.output_constraints.iter().any(|c| {
                    matches!(
                        c,
                        OutputConstraint::LessEq(_, _)
                            | OutputConstraint::GreaterEq(_, _)
                            | OutputConstraint::LessThan(_, _)
                            | OutputConstraint::GreaterThan(_, _)
                    )
                });

                // Extract threshold from VNNLIB (for simple GreaterEqConst/LessEqConst properties)
                // VNNLIB specifies UNSAFE region:
                // - Y_i >= c (GreaterEqConst) means unsafe if output >= c, so prove upper < c
                // - Y_i <= c (LessEqConst) means unsafe if output <= c, so prove lower > c
                let (effective_threshold, verify_upper, const_output_idx) = if has_relational {
                    // For relational constraints, threshold is 0 (prove difference > 0)
                    (0.0f32, false, None)
                } else {
                    vnnlib
                        .output_constraints
                        .iter()
                        .find_map(|c| match c {
                            // GreaterEq: unsafe if Y >= c, safe if Y < c (verify upper bound)
                            OutputConstraint::GreaterEqConst(i, val) => {
                                Some((*val as f32, true, Some(*i)))
                            }
                            OutputConstraint::GreaterThanConst(i, val) => {
                                Some((*val as f32, true, Some(*i)))
                            }
                            // LessEq: unsafe if Y <= c, safe if Y > c (verify lower bound)
                            OutputConstraint::LessEqConst(i, val) => {
                                Some((*val as f32, false, Some(*i)))
                            }
                            OutputConstraint::LessThanConst(i, val) => {
                                Some((*val as f32, false, Some(*i)))
                            }
                            _ => None,
                        })
                        .unwrap_or((threshold, false, None))
                };

                (
                    input,
                    effective_threshold,
                    Some(vnnlib),
                    verify_upper,
                    has_relational,
                    const_output_idx,
                )
            } else {
                // Use epsilon-ball around zero with model's expected shape
                // For Conv2d first layer, squeeze batch dimension of 1 since Conv2d expects (C,H,W)
                let mut effective_shape = input_shape_from_model.clone();
                if needs_squeeze && effective_shape.len() >= 2 && effective_shape[0] == 1 {
                    effective_shape.remove(0);
                    info!(
                        "Squeezed batch dimension for Conv layer (epsilon-ball), shape: {:?}",
                        effective_shape
                    );
                }
                let center = ArrayD::from_elem(IxDyn(&effective_shape), 0.0f32);
                let input = BoundedTensor::from_epsilon(center, epsilon);
                (input, threshold, None, false, false, None) // Default: verify lower > threshold, no relational
            };

            // For graph models, automatically disable features not yet supported
            // ReLU splitting now supports cuts (GCP-CROWN for DAGs)
            // Input splitting still requires disabling cuts
            let (use_alpha_effective, crown_ibp_effective, enable_cuts_effective) =
                if is_graph_model && matches!(branching_heuristic, BranchingHeuristic::InputSplit) {
                    // Input splitting: disable all advanced features
                    if (use_alpha || crown_ibp || enable_cuts) && !json {
                        eprintln!(
                        "Note: Graph model with input splitting - disabling unsupported features (alpha={}, crown_ibp={}, cuts={})",
                        use_alpha, crown_ibp, enable_cuts
                    );
                    }
                    (false, false, false)
                } else if is_graph_model && use_relu_split {
                    // ReLU splitting: α-CROWN and cuts supported; crown_ibp still not supported
                    if crown_ibp && !json {
                        eprintln!(
                        "Note: Graph model with ReLU splitting - disabling CROWN-IBP (crown_ibp={})",
                        crown_ibp
                    );
                    }
                    (use_alpha, false, enable_cuts) // α-CROWN and cuts supported for graph ReLU splitting
                } else {
                    (use_alpha, crown_ibp, enable_cuts)
                };

            if !json {
                println!("Model: {}", model.display());
                if let Some(prop_path) = &property {
                    println!("Property: {}", prop_path.display());
                    if let Some(ref spec) = vnnlib_spec {
                        println!("Input region: {} dimensions", spec.num_inputs);
                        for (i, (l, u)) in spec.input_bounds.iter().enumerate() {
                            println!("  X_{}: [{:.6}, {:.6}] (width: {:.6})", i, l, u, u - l);
                        }
                    }
                } else {
                    println!("Input shape: {:?}, epsilon: {}", input.shape(), epsilon);
                }
                let verify_msg = if verify_upper {
                    format!(
                        "output < {} (unsafe if output >= {})",
                        effective_threshold, effective_threshold
                    )
                } else {
                    format!(
                        "output > {} (unsafe if output <= {})",
                        effective_threshold, effective_threshold
                    )
                };
                println!(
                    "Threshold: {} (verifying {})",
                    effective_threshold, verify_msg
                );
                println!(
                    "Config: max_domains={}, timeout={}s, max_depth={}, branching={}, fsb_candidates={}, use_alpha={}, alpha_iter={}, alpha_grad={}, alpha_opt={}, alpha_lr={}, crown_ibp_interm={}, beta_iter={}, lr_beta={}, crown_ibp={}, batch_size={}, parallel={}, enable_cuts={}, max_cuts={}, min_cut_depth={}, pgd_attack={}",
                    max_domains, timeout, max_depth, branching, fsb_candidates, use_alpha_effective, alpha_iterations, alpha_gradient_method, alpha_optimizer, alpha_lr, crown_ibp_intermediates, beta_iterations, lr_beta, crown_ibp_effective, batch_size, parallel_children, enable_cuts_effective, max_cuts, min_cut_depth, pgd_attack
                );
            }

            // Configure β-CROWN
            let config = BetaCrownConfig {
                max_domains,
                timeout: std::time::Duration::from_secs(timeout),
                max_depth,
                use_alpha_crown: use_alpha_effective,
                use_crown_ibp: crown_ibp_effective,
                alpha_config: AlphaCrownConfig {
                    iterations: alpha_iterations,
                    adaptive_skip: !no_adaptive_alpha_skip,
                    adaptive_skip_depth_threshold: alpha_skip_depth,
                    // fix_interm_bounds=true means use IBP (fast but loose)
                    // --crown-ibp-intermediates inverts this to use CROWN-IBP (slow but tight)
                    fix_interm_bounds: !crown_ibp_intermediates,
                    // SPSA configuration (used when gradient_method is Spsa)
                    spsa_samples: alpha_spsa_samples,
                    learning_rate: alpha_lr,
                    gradient_method: match alpha_gradient_method.as_str() {
                        "analytic" => GradientMethod::Analytic,
                        "analytic-chain" => GradientMethod::AnalyticChain,
                        "spsa" => GradientMethod::Spsa,
                        "fd" => GradientMethod::FiniteDifferences,
                        _ => GradientMethod::Spsa, // Default to SPSA (matches config default)
                    },
                    optimizer: match alpha_optimizer.as_str() {
                        "adam" => Optimizer::Adam,
                        "sgd" => Optimizer::Sgd,
                        _ => Optimizer::Adam, // Default to Adam (matches α,β-CROWN)
                    },
                    ..Default::default()
                },
                branching_heuristic,
                fsb_candidates,
                batch_size,
                parallel_children,
                enable_cuts: enable_cuts_effective,
                max_cuts,
                min_cut_depth,
                enable_near_miss_cuts,
                near_miss_margin,
                enable_proactive_cuts: proactive_cuts,
                max_proactive_cuts,
                verify_upper_bound: verify_upper,
                enable_pgd_attack: pgd_attack,
                pgd_restarts,
                pgd_steps,
                beta_lr: lr_beta,
                beta_iterations,
                beta_max_depth,
                ..Default::default()
            };

            let verifier = BetaCrownVerifier::new(config.clone());

            // Handle relational constraints specially
            let result = if let (true, Some(vnnlib)) = (has_relational, vnnlib_spec.as_ref()) {
                use gamma_onnx::vnnlib::OutputConstraint;
                use gamma_propagate::layers::LinearLayer;
                use gamma_propagate::Layer as PropLayer;
                use ndarray::Array2;

                if let BetaCrownModel::Graph(graph) = &model_net {
                    // GraphNetwork β-CROWN currently supports only input-space splitting, so we
                    // reuse the relational-constraint loop but verify each linear objective via
                    // interval-linear evaluation over GraphNetwork CROWN bounds.

                    // Count relational constraints for timeout budgeting
                    let relational_count = vnnlib
                        .output_constraints
                        .iter()
                        .filter(|c| {
                            matches!(
                                c,
                                OutputConstraint::LessEq(_, _)
                                    | OutputConstraint::GreaterEq(_, _)
                                    | OutputConstraint::LessThan(_, _)
                                    | OutputConstraint::GreaterThan(_, _)
                            )
                        })
                        .count();

                    let is_disjunction = vnnlib.is_disjunction;

                    // For disjunctive properties: need ALL constraints verified, early exit on failure
                    // Give more time per constraint since we exit immediately on any failure
                    // For conjunctive properties: need ANY constraint verified, divide timeout fairly
                    let per_constraint_timeout = if is_disjunction {
                        // Disjunction: use 1/3 of total timeout per constraint (early exit limits total)
                        // This gives each constraint 3x more time than dividing equally
                        std::time::Duration::from_secs(timeout / 3)
                    } else if relational_count > 0 {
                        std::time::Duration::from_secs(timeout) / (relational_count as u32 + 1)
                    } else {
                        std::time::Duration::from_secs(timeout)
                    };

                    // For disjunctive properties with ReLU splitting, use multi-objective BaB
                    // This verifies ALL constraints simultaneously, sharing computation across objectives
                    if is_disjunction && use_relu_split && relational_count >= 2 {
                        let num_outputs = vnnlib.num_outputs;

                        // Collect all objectives into Vec<Vec<f32>>
                        let objectives: Vec<Vec<f32>> = vnnlib
                            .output_constraints
                            .iter()
                            .filter_map(|c| match c {
                                OutputConstraint::LessEq(i, j)
                                | OutputConstraint::LessThan(i, j) => {
                                    let mut obj = vec![0.0f32; num_outputs];
                                    obj[*i] = 1.0;
                                    obj[*j] = -1.0;
                                    Some(obj)
                                }
                                OutputConstraint::GreaterEq(i, j)
                                | OutputConstraint::GreaterThan(i, j) => {
                                    let mut obj = vec![0.0f32; num_outputs];
                                    obj[*j] = 1.0;
                                    obj[*i] = -1.0;
                                    Some(obj)
                                }
                                _ => None,
                            })
                            .collect();

                        let thresholds = vec![0.0f32; objectives.len()];

                        if !json {
                            println!("\nRunning multi-objective Graph β-CROWN (ReLU splitting) for disjunction...");
                            println!(
                                "Verifying {} constraints simultaneously (shared computation)",
                                objectives.len()
                            );
                            println!("SAFE requires: ALL constraints provably violated");
                        }

                        let multi_verifier = BetaCrownVerifier::new(config.clone());
                        let result = multi_verifier
                            .verify_graph_relu_split_multi_objective_with_engine(
                                graph,
                                &input,
                                &objectives,
                                &thresholds,
                                gemm_engine,
                            )?;

                        if !json {
                            println!("\n  Multi-objective result: {:?}", result.result);
                            println!("    Domains explored: {}", result.domains_explored);
                            println!("    Domains verified: {}", result.domains_verified);
                            println!("    Max depth: {}", result.max_depth_reached);
                            println!("    Time: {:.2}s", result.time_elapsed.as_secs_f64());
                        }

                        result
                    } else {
                        // Per-constraint verification (non-disjunction or input splitting)
                        if !json {
                            let split_type = if use_relu_split {
                                "ReLU splitting"
                            } else {
                                "input splitting"
                            };
                            println!(
                                "\nRunning Graph β-CROWN ({}) with relational constraints...",
                                split_type
                            );
                            if is_disjunction {
                                println!("Disjunctive property: SAFE if ALL constraints are provably violated.");
                                println!("Early-exit strategy: stop immediately when any constraint fails to verify.");
                            } else {
                                println!("Conjunctive property: SAFE if ANY constraint is provably violated.");
                            }
                            println!(
                                "Timeout budget: {:.1}s per constraint ({} constraints)",
                                per_constraint_timeout.as_secs_f64(),
                                relational_count
                            );
                        }

                        let num_outputs = vnnlib.num_outputs;
                        let mut verified_count = 0usize; // Count of constraints proved violated
                        let mut total_domains = 0;
                        let mut max_depth = 0usize;
                        let mut total_time = std::time::Duration::ZERO;
                        let mut constraint_results = Vec::new();

                        let overall_start = std::time::Instant::now();
                        let overall_timeout = std::time::Duration::from_secs(timeout);

                        // Pre-compute α-CROWN bounds once for all constraints (major optimization)
                        // This avoids re-computing bounds for each constraint, providing ~Nx speedup
                        let precomputed_bounds = if use_relu_split && relational_count > 1 {
                            if !json {
                                println!("\n  Pre-computing α-CROWN bounds (shared across {} constraints)...", relational_count);
                            }
                            let bounds_start = std::time::Instant::now();
                            let bounds_verifier = BetaCrownVerifier::new(config.clone());
                            let (node_bounds, output_bounds) =
                                bounds_verifier.compute_initial_graph_bounds(graph, &input)?;
                            if !json {
                                println!(
                                    "    Bounds computed in {:.2}s",
                                    bounds_start.elapsed().as_secs_f64()
                                );
                            }
                            Some((node_bounds, output_bounds))
                        } else {
                            None
                        };

                        for (idx, constraint) in vnnlib.output_constraints.iter().enumerate() {
                            let spec_vec = match constraint {
                                OutputConstraint::LessEq(i, j)
                                | OutputConstraint::LessThan(i, j) => {
                                    let mut c = vec![0.0f32; num_outputs];
                                    c[*i] = 1.0;
                                    c[*j] = -1.0;
                                    Some((
                                        c,
                                        format!("Y_{} <= Y_{}", i, j),
                                        format!("Y_{} - Y_{}", i, j),
                                    ))
                                }
                                OutputConstraint::GreaterEq(i, j)
                                | OutputConstraint::GreaterThan(i, j) => {
                                    let mut c = vec![0.0f32; num_outputs];
                                    c[*j] = 1.0;
                                    c[*i] = -1.0;
                                    Some((
                                        c,
                                        format!("Y_{} >= Y_{}", i, j),
                                        format!("Y_{} - Y_{}", j, i),
                                    ))
                                }
                                _ => None,
                            };

                            if let Some((spec_coeffs, constraint_desc, diff_desc)) = spec_vec {
                                if overall_start.elapsed() >= overall_timeout {
                                    if !json {
                                        println!("\n  Overall timeout reached, stopping constraint iteration");
                                    }
                                    break;
                                }

                                let remaining =
                                    overall_timeout.saturating_sub(overall_start.elapsed());
                                let this_timeout = per_constraint_timeout.min(remaining);
                                if this_timeout.as_millis() < 100 {
                                    break;
                                }

                                if !json {
                                    println!(
                                        "\n  Constraint {}: {} (verify {} > 0, timeout: {:.1}s)",
                                        idx + 1,
                                        constraint_desc,
                                        diff_desc,
                                        this_timeout.as_secs_f64()
                                    );
                                }

                                let constraint_config = BetaCrownConfig {
                                    timeout: this_timeout,
                                    ..config.clone()
                                };
                                let constraint_verifier = BetaCrownVerifier::new(constraint_config);

                                let constraint_result = if use_relu_split {
                                    if let Some((ref node_bounds, ref output_bounds)) =
                                        precomputed_bounds
                                    {
                                        // Use pre-computed bounds (fast path)
                                        let bounds =
                                            GraphPrecomputedBounds::new(node_bounds, output_bounds);
                                        constraint_verifier
                                            .verify_graph_relu_split_with_bounds_with_engine(
                                                graph,
                                                &input,
                                                &spec_coeffs,
                                                0.0,
                                                &bounds,
                                                gemm_engine,
                                            )?
                                    } else {
                                        // Fall back to computing bounds each time
                                        constraint_verifier
                                            .verify_graph_relu_split_with_engine_gpu(
                                                graph,
                                                &input,
                                                &spec_coeffs,
                                                0.0,
                                                gemm_engine,
                                            )?
                                    }
                                } else {
                                    constraint_verifier.verify_graph_input_split_with_engine(
                                        graph,
                                        &input,
                                        &spec_coeffs,
                                        0.0,
                                        gemm_engine,
                                    )?
                                };
                                total_domains += constraint_result.domains_explored;
                                max_depth = max_depth.max(constraint_result.max_depth_reached);
                                total_time += constraint_result.time_elapsed;

                                let status_str = match &constraint_result.result {
                                    BabVerificationStatus::Verified => "VIOLATED (safe)",
                                    BabVerificationStatus::Violated { .. } => {
                                        "VIOLATED (counterexample found)"
                                    }
                                    BabVerificationStatus::PotentialViolation => "MAY HOLD",
                                    BabVerificationStatus::Unknown { .. } => "UNKNOWN",
                                };

                                if !json {
                                    println!(
                                        "    Result: {} ({} domains, {:.2}s)",
                                        status_str,
                                        constraint_result.domains_explored,
                                        constraint_result.time_elapsed.as_secs_f64()
                                    );
                                }

                                constraint_results
                                    .push((constraint_desc.clone(), constraint_result.clone()));

                                let constraint_verified = matches!(
                                    constraint_result.result,
                                    BabVerificationStatus::Verified
                                        | BabVerificationStatus::Violated { .. }
                                );
                                if constraint_verified {
                                    verified_count += 1;
                                    // For conjunction: exit early when ANY constraint is proved violated
                                    // For disjunction: must prove ALL constraints violated, so continue
                                    if !is_disjunction {
                                        if !json {
                                            println!("\n  Early exit: constraint {} violated (conjunctive property satisfied)", constraint_desc);
                                        }
                                        break;
                                    }
                                } else if is_disjunction {
                                    // For disjunction: if ANY constraint fails to be proved, property is unknown
                                    if !json {
                                        println!("\n  Early exit: constraint {} not violated (disjunctive property requires ALL)", constraint_desc);
                                    }
                                    break;
                                }
                            }
                        }

                        // Determine final status based on property type
                        let final_status = if is_disjunction {
                            // For disjunction: need ALL constraints violated
                            if verified_count == relational_count && relational_count > 0 {
                                BabVerificationStatus::Verified
                            } else {
                                BabVerificationStatus::Unknown {
                                reason: format!(
                                    "Only {}/{} constraints provably violated (disjunction requires all)",
                                    verified_count, relational_count
                                ),
                            }
                            }
                        } else {
                            // For conjunction: need ANY constraint violated
                            if verified_count > 0 {
                                BabVerificationStatus::Verified
                            } else {
                                BabVerificationStatus::Unknown {
                                    reason: format!(
                                    "No constraint was provably violated ({} constraints checked)",
                                    constraint_results.len()
                                ),
                                }
                            }
                        };

                        BetaCrownResult {
                            result: final_status,
                            domains_explored: total_domains,
                            domains_verified: verified_count,
                            cuts_generated: 0,
                            max_depth_reached: max_depth,
                            time_elapsed: total_time,
                            output_bounds: None,
                        }
                    } // Close the multi-objective else block
                } else {
                    let network = match &model_net {
                        BetaCrownModel::Sequential(network) => network,
                        BetaCrownModel::Graph(_) => unreachable!(),
                    };

                    // For conjunctive constraints (AND), property is SAFE if ANY constraint is VIOLATED for all inputs
                    // For each relational constraint Y_i <= Y_j (unsafe), prove Y_i > Y_j (i.e., Y_i - Y_j > 0)

                    // Fast path for common targeted-robustness properties:
                    //
                    // VNNLIB output constraints encode an UNSAFE region as a conjunction, e.g.
                    //   (Y_target >= Y_0) ∧ ... ∧ (Y_target >= Y_8)
                    //
                    // This conjunction is satisfiable iff max_j(Y_j - Y_target) <= 0.
                    // Therefore, it is SAFE (UNSAT) iff max_j(Y_j - Y_target) > 0 for all inputs.
                    //
                    // Reduce the whole conjunction to a single scalar objective:
                    //   maxdiff = max_j(signed_diff_j)
                    // and run β-CROWN once.
                    let reduced_result = (|| -> anyhow::Result<Option<BetaCrownResult>> {
                        use gamma_propagate::layers::{MaxPool2dLayer, ReshapeLayer};

                        let mut rel_family: Option<&'static str> = None; // "ge" or "le"
                        let mut lhs_idx: Option<usize> = None;
                        let mut rhs_indices: Vec<usize> = Vec::new();

                        for c in &vnnlib.output_constraints {
                            match c {
                                OutputConstraint::GreaterEq(i, j)
                                | OutputConstraint::GreaterThan(i, j) => {
                                    if rel_family.is_none() {
                                        rel_family = Some("ge");
                                    }
                                    if rel_family != Some("ge") {
                                        return Ok(None);
                                    }
                                    if lhs_idx.is_none() {
                                        lhs_idx = Some(*i);
                                    }
                                    if lhs_idx != Some(*i) {
                                        return Ok(None);
                                    }
                                    rhs_indices.push(*j);
                                }
                                OutputConstraint::LessEq(i, j)
                                | OutputConstraint::LessThan(i, j) => {
                                    if rel_family.is_none() {
                                        rel_family = Some("le");
                                    }
                                    if rel_family != Some("le") {
                                        return Ok(None);
                                    }
                                    if lhs_idx.is_none() {
                                        lhs_idx = Some(*i);
                                    }
                                    if lhs_idx != Some(*i) {
                                        return Ok(None);
                                    }
                                    rhs_indices.push(*j);
                                }
                                _ => return Ok(None),
                            }
                        }

                        rhs_indices.sort_unstable();
                        rhs_indices.dedup();
                        let (Some(family), Some(lhs)) = (rel_family, lhs_idx) else {
                            return Ok(None);
                        };
                        if rhs_indices.is_empty() {
                            return Ok(None);
                        }

                        let num_outputs = vnnlib.num_outputs;
                        let k = rhs_indices.len();

                        if !json {
                            let constraint_desc = match family {
                                "ge" => format!("Y_{} >= Y_j for j in {:?}", lhs, rhs_indices),
                                "le" => format!("Y_{} <= Y_j for j in {:?}", lhs, rhs_indices),
                                _ => "unknown".to_string(),
                            };
                            println!(
                                "\nRelational constraint reduction detected: {}",
                                constraint_desc
                            );
                            println!(
                            "Reducing to single objective: maxdiff = max_j(signed_diff_j), verify maxdiff > 0"
                        );
                        }

                        // Build Linear layer computing signed differences, one per RHS:
                        // - For unsafe (Y_lhs >= Y_rhs): signed_diff = Y_rhs - Y_lhs
                        // - For unsafe (Y_lhs <= Y_rhs): signed_diff = Y_lhs - Y_rhs
                        let mut weights = vec![0.0f32; k * num_outputs];
                        for (row, &rhs) in rhs_indices.iter().enumerate() {
                            let row_start = row * num_outputs;
                            match family {
                                "ge" => {
                                    weights[row_start + rhs] = 1.0;
                                    weights[row_start + lhs] = -1.0;
                                }
                                "le" => {
                                    weights[row_start + lhs] = 1.0;
                                    weights[row_start + rhs] = -1.0;
                                }
                                _ => unreachable!(),
                            }
                        }

                        let spec_weight =
                            Array2::from_shape_vec((k, num_outputs), weights).unwrap();
                        let mut augmented = (*network).clone();
                        augmented
                            .add_layer(PropLayer::Linear(LinearLayer::new(spec_weight, None)?));
                        augmented
                            .add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1, 1, k as i64])));
                        augmented.add_layer(PropLayer::MaxPool2d(MaxPool2dLayer::new(
                            (1, k),
                            (1, k),
                            (0, 0),
                        )));
                        augmented.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1])));

                        Ok(Some(verifier.verify_with_engine(
                            &augmented,
                            &input,
                            0.0,
                            gemm_engine,
                        )?))
                    })()?;

                    if let Some(reduced_result) = reduced_result {
                        reduced_result
                    } else {
                        // Count relational constraints for timeout budgeting
                        let relational_count = vnnlib
                            .output_constraints
                            .iter()
                            .filter(|c| {
                                matches!(
                                    c,
                                    OutputConstraint::LessEq(_, _)
                                        | OutputConstraint::GreaterEq(_, _)
                                        | OutputConstraint::LessThan(_, _)
                                        | OutputConstraint::GreaterThan(_, _)
                                )
                            })
                            .count();

                        // Budget timeout across constraints: give each constraint timeout/(n+1) to leave buffer
                        // This prevents one hard constraint from consuming all time
                        let per_constraint_timeout = if relational_count > 0 {
                            std::time::Duration::from_secs(timeout) / (relational_count as u32 + 1)
                        } else {
                            std::time::Duration::from_secs(timeout)
                        };

                        if !json {
                            println!("\nRunning β-CROWN with relational constraints...");
                            println!("Conjunctive property: SAFE if ANY constraint is provably violated.");
                            println!(
                                "Timeout budget: {:.1}s per constraint ({} constraints)",
                                per_constraint_timeout.as_secs_f64(),
                                relational_count
                            );
                        }

                        let num_outputs = vnnlib.num_outputs;
                        let mut any_verified = false;
                        let mut total_domains = 0;
                        let mut max_depth = 0usize;
                        let mut total_time = std::time::Duration::ZERO;
                        let mut constraint_results = Vec::new();

                        // Track overall start time for global timeout
                        let overall_start = std::time::Instant::now();
                        let overall_timeout = std::time::Duration::from_secs(timeout);

                        // Try conjunctive PGD attack first to find counterexample satisfying ALL constraints
                        // This handles the common pattern: Y_target <= Y_j for all j (e.g., ACAS-Xu prop_3/prop_4)
                        let conjunctive_counterexample = if pgd_attack {
                            use gamma_propagate::pgd_attack::{PgdAttacker, PgdConfig};

                            // Check if all constraints are LessEq with same LHS (target index)
                            let mut target_idx: Option<usize> = None;
                            let mut comparison_indices = Vec::new();
                            let mut all_less_eq = true;

                            for constraint in &vnnlib.output_constraints {
                                match constraint {
                                    OutputConstraint::LessEq(i, j)
                                    | OutputConstraint::LessThan(i, j) => {
                                        if target_idx.is_none() {
                                            target_idx = Some(*i);
                                        }
                                        if target_idx == Some(*i) {
                                            comparison_indices.push(*j);
                                        } else {
                                            all_less_eq = false;
                                            break;
                                        }
                                    }
                                    _ => {
                                        all_less_eq = false;
                                        break;
                                    }
                                }
                            }

                            if let (true, Some(target), false) =
                                (all_less_eq, target_idx, comparison_indices.is_empty())
                            {
                                if !json {
                                    println!("\n  Running conjunctive PGD attack: find x where Y_{} <= Y_j for all j in {:?}",
                                target, comparison_indices);
                                }

                                let pgd_config = PgdConfig {
                                    num_restarts: pgd_restarts,
                                    num_steps: pgd_steps,
                                    step_size: 0.01,
                                    spsa_delta: 0.001,
                                    seed: 42,
                                    parallel: true,
                                };
                                let attacker = PgdAttacker::new(pgd_config);

                                match attacker.attack_conjunctive_less_eq(
                                    network,
                                    &input,
                                    target,
                                    &comparison_indices,
                                ) {
                                    Ok(result) if result.found_counterexample => {
                                        if !json {
                                            println!("  Conjunctive PGD found counterexample! max(Y_{} - Y_j) = {}",
                                        target, result.best_output_value);
                                        }
                                        Some((
                                            result.counterexample.unwrap(),
                                            result.output.unwrap(),
                                        ))
                                    }
                                    Ok(result) => {
                                        if !json {
                                            println!("  Conjunctive PGD: no counterexample found. Best max diff: {}",
                                        result.best_output_value);
                                        }
                                        None
                                    }
                                    Err(e) => {
                                        if !json {
                                            println!("  Conjunctive PGD error: {}", e);
                                        }
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        // If conjunctive attack found counterexample, return violated
                        if let Some((counterexample, output)) = conjunctive_counterexample {
                            BetaCrownResult {
                                result: BabVerificationStatus::Violated {
                                    counterexample: counterexample.iter().copied().collect(),
                                    output: output.iter().copied().collect(),
                                },
                                domains_explored: 0,
                                domains_verified: 0,
                                cuts_generated: 0,
                                max_depth_reached: 0,
                                time_elapsed: overall_start.elapsed(),
                                output_bounds: None,
                            }
                        } else {
                            for (idx, constraint) in vnnlib.output_constraints.iter().enumerate() {
                                // Build specification vector: for Y_i <= Y_j, we want to prove Y_i - Y_j > 0
                                // Create c such that c @ Y = Y_i - Y_j (c[i] = 1, c[j] = -1)
                                let spec_vec = match constraint {
                                    OutputConstraint::LessEq(i, j)
                                    | OutputConstraint::LessThan(i, j) => {
                                        // Y_i <= Y_j is UNSAFE. To prove safe, show Y_i > Y_j, i.e., Y_i - Y_j > 0
                                        let mut c = vec![0.0f32; num_outputs];
                                        c[*i] = 1.0; // +Y_i
                                        c[*j] = -1.0; // -Y_j
                                        Some((
                                            c,
                                            format!("Y_{} <= Y_{}", i, j),
                                            format!("Y_{} - Y_{}", i, j),
                                        ))
                                    }
                                    OutputConstraint::GreaterEq(i, j)
                                    | OutputConstraint::GreaterThan(i, j) => {
                                        // Y_i >= Y_j is UNSAFE. To prove safe, show Y_i < Y_j, i.e., Y_j - Y_i > 0
                                        let mut c = vec![0.0f32; num_outputs];
                                        c[*j] = 1.0; // +Y_j
                                        c[*i] = -1.0; // -Y_i
                                        Some((
                                            c,
                                            format!("Y_{} >= Y_{}", i, j),
                                            format!("Y_{} - Y_{}", j, i),
                                        ))
                                    }
                                    _ => None, // Skip constant constraints
                                };

                                if let Some((spec_coeffs, constraint_desc, diff_desc)) = spec_vec {
                                    // Check if we've exceeded overall timeout
                                    if overall_start.elapsed() >= overall_timeout {
                                        if !json {
                                            println!("\n  Overall timeout reached, stopping constraint iteration");
                                        }
                                        break;
                                    }

                                    // Calculate remaining time and use min(per_constraint, remaining)
                                    let remaining =
                                        overall_timeout.saturating_sub(overall_start.elapsed());
                                    let this_timeout = per_constraint_timeout.min(remaining);

                                    if this_timeout.as_millis() < 100 {
                                        // Not enough time remaining
                                        break;
                                    }

                                    if !json {
                                        println!("\n  Constraint {}: {} (verify {} > 0, timeout: {:.1}s)",
                                idx + 1, constraint_desc, diff_desc, this_timeout.as_secs_f64());
                                    }

                                    // Create augmented network: append linear layer to compute c @ Y
                                    let mut augmented = (*network).clone();
                                    let spec_weight =
                                        Array2::from_shape_vec((1, num_outputs), spec_coeffs)
                                            .unwrap();
                                    let spec_layer = LinearLayer::new(spec_weight, None)?;
                                    augmented.add_layer(PropLayer::Linear(spec_layer));

                                    // Create verifier with per-constraint timeout
                                    let constraint_config = BetaCrownConfig {
                                        timeout: this_timeout,
                                        ..config.clone()
                                    };
                                    let constraint_verifier =
                                        BetaCrownVerifier::new(constraint_config);

                                    // Run β-CROWN on augmented network, verify output > 0
                                    let constraint_result = constraint_verifier
                                        .verify_with_engine(&augmented, &input, 0.0, gemm_engine)?;
                                    total_domains += constraint_result.domains_explored;
                                    max_depth = max_depth.max(constraint_result.max_depth_reached);
                                    total_time += constraint_result.time_elapsed;

                                    let status_str = match &constraint_result.result {
                                        BabVerificationStatus::Verified => "VIOLATED (safe)",
                                        BabVerificationStatus::Violated { .. } => {
                                            "VIOLATED (counterexample found)"
                                        }
                                        BabVerificationStatus::PotentialViolation => "MAY HOLD",
                                        BabVerificationStatus::Unknown { .. } => "UNKNOWN",
                                    };

                                    if !json {
                                        println!(
                                            "    Result: {} ({} domains, {:.2}s)",
                                            status_str,
                                            constraint_result.domains_explored,
                                            constraint_result.time_elapsed.as_secs_f64()
                                        );
                                    }

                                    constraint_results
                                        .push((constraint_desc, constraint_result.clone()));

                                    if matches!(
                                        constraint_result.result,
                                        BabVerificationStatus::Verified
                                            | BabVerificationStatus::Violated { .. }
                                    ) {
                                        any_verified = true;
                                        // For conjunctive property, one violated constraint is enough
                                        break;
                                    }
                                }
                            }

                            // Build final result
                            let final_status = if any_verified {
                                BabVerificationStatus::Verified
                            } else {
                                BabVerificationStatus::Unknown {
                        reason: format!("No constraint was provably violated ({} constraints checked)", constraint_results.len()),
                    }
                            };

                            BetaCrownResult {
                                result: final_status,
                                domains_explored: total_domains,
                                domains_verified: if any_verified { 1 } else { 0 },
                                cuts_generated: 0,
                                max_depth_reached: max_depth,
                                time_elapsed: total_time,
                                output_bounds: None,
                            }
                        } // end else (no reduction)
                    } // end else (no conjunctive counterexample)
                } // end Sequential Network branch
            } else {
                // No relational constraints - use standard verification
                if !json {
                    println!("\nRunning β-CROWN...");
                }
                match &model_net {
                    BetaCrownModel::Sequential(network) => {
                        if let Some(output_idx) = const_output_idx {
                            use gamma_propagate::layers::LinearLayer;
                            use gamma_propagate::Layer as PropLayer;
                            use ndarray::Array2;

                            let mut coeffs = vec![0.0f32; output_dim];
                            if output_idx >= coeffs.len() {
                                anyhow::bail!(
                                    "VNNLIB output index {} out of range (model outputs={})",
                                    output_idx,
                                    coeffs.len()
                                );
                            }
                            coeffs[output_idx] = 1.0;

                            let spec_weight =
                                Array2::from_shape_vec((1, coeffs.len()), coeffs).unwrap();
                            let spec_layer = LinearLayer::new(spec_weight, None)?;
                            let mut augmented = (*network).clone();
                            augmented.add_layer(PropLayer::Linear(spec_layer));
                            verifier.verify_with_engine(
                                &augmented,
                                &input,
                                effective_threshold,
                                gemm_engine,
                            )?
                        } else {
                            verifier.verify_with_engine(
                                network,
                                &input,
                                effective_threshold,
                                gemm_engine,
                            )?
                        }
                    }
                    BetaCrownModel::Graph(graph) => {
                        let output_idx = const_output_idx.ok_or_else(|| {
                            anyhow::anyhow!(
                                "GraphNetwork β-CROWN requires a property with an explicit output index"
                            )
                        })?;
                        if output_idx >= output_dim {
                            anyhow::bail!(
                                "VNNLIB output index {} out of range (model outputs={})",
                                output_idx,
                                output_dim
                            );
                        }
                        let mut objective = vec![0.0f32; output_dim];
                        objective[output_idx] = 1.0;
                        if use_relu_split {
                            verifier.verify_graph_relu_split_with_engine_gpu(
                                graph,
                                &input,
                                &objective,
                                effective_threshold,
                                gemm_engine,
                            )?
                        } else {
                            verifier.verify_graph_input_split_with_engine(
                                graph,
                                &input,
                                &objective,
                                effective_threshold,
                                gemm_engine,
                            )?
                        }
                    }
                }
            };

            if json {
                use serde_json::json;
                let (status, reason, counterexample) = match &result.result {
                    BabVerificationStatus::Verified => ("verified", None, None),
                    BabVerificationStatus::Violated {
                        counterexample,
                        output,
                    } => (
                        "violated",
                        None,
                        Some((counterexample.clone(), output.clone())),
                    ),
                    BabVerificationStatus::PotentialViolation => {
                        ("potential_violation", None, None)
                    }
                    BabVerificationStatus::Unknown { reason } => {
                        ("unknown", Some(reason.clone()), None)
                    }
                };
                let output_bound_width = result
                    .output_bounds
                    .as_ref()
                    .map(|b| b.width().iter().cloned().fold(0.0f32, f32::max));
                let json_output = json!({
                    "status": status,
                    "reason": reason,
                    "counterexample": counterexample.as_ref().map(|(cx, out)| json!({
                        "input": cx,
                        "output": out
                    })),
                    "property_file": property.as_ref().map(|p| p.display().to_string()),
                    "epsilon": if property.is_none() { Some(epsilon) } else { None },
                    "threshold": effective_threshold,
                    "domains_explored": result.domains_explored,
                    "domains_verified": result.domains_verified,
                    "cuts_generated": result.cuts_generated,
                    "max_depth_reached": result.max_depth_reached,
                    "time_elapsed_s": result.time_elapsed.as_secs_f64(),
                    "output_bound_width": output_bound_width
                });
                println!("{}", serde_json::to_string_pretty(&json_output)?);
            } else {
                println!("\n--- Result ---");
                match &result.result {
                    BabVerificationStatus::Verified => {
                        println!("Status: VERIFIED");
                        println!("All inputs produce output > {}", effective_threshold);
                    }
                    BabVerificationStatus::Violated {
                        counterexample,
                        output,
                    } => {
                        println!("Status: VIOLATED");
                        println!(
                            "Found counterexample where output <= {}",
                            effective_threshold
                        );
                        println!("Counterexample input: {:?}", counterexample);
                        println!("Counterexample output: {:?}", output);
                    }
                    BabVerificationStatus::PotentialViolation => {
                        println!("Status: POTENTIAL VIOLATION");
                        println!("Found region where output may be < {}", effective_threshold);
                    }
                    BabVerificationStatus::Unknown { reason } => {
                        println!("Status: UNKNOWN");
                        println!("Reason: {}", reason);
                    }
                }
                println!("Domains explored: {}", result.domains_explored);
                println!("Domains verified: {}", result.domains_verified);
                if result.cuts_generated > 0 {
                    println!("Cuts generated: {}", result.cuts_generated);
                }
                println!("Max depth reached: {}", result.max_depth_reached);
                println!("Time elapsed: {:.2}s", result.time_elapsed.as_secs_f64());

                if let Some(bounds) = &result.output_bounds {
                    let width = bounds.width();
                    let max_width = width.iter().cloned().fold(0.0f32, f32::max);
                    println!("Output bound width: {:.6e}", max_width);
                }
            }
        }

        Commands::Weights { action } => {
            handle_weights_command(action)?;
        }
    }

    Ok(())
}

/// Handle the Weights subcommand
fn handle_weights_command(action: WeightsAction) -> Result<()> {
    use gamma_onnx::safetensors::safetensors_info;
    use serde_json::json;

    match action {
        WeightsAction::Info {
            file,
            detailed,
            json,
        } => {
            let ext = file
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if ext == "safetensors" {
                // SafeTensors file
                let info = safetensors_info(&file).with_context(|| {
                    format!("Failed to read SafeTensors file: {}", file.display())
                })?;

                if json {
                    let output = json!({
                        "format": "safetensors",
                        "file": file.to_string_lossy(),
                        "tensor_count": info.tensor_count,
                        "param_count": info.param_count,
                        "tensors": if detailed {
                            info.tensors.iter().map(|(name, shape, dtype)| {
                                json!({
                                    "name": name,
                                    "shape": shape,
                                    "dtype": dtype,
                                    "elements": shape.iter().product::<usize>()
                                })
                            }).collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    println!("Format: SafeTensors");
                    println!("File: {}", file.display());
                    println!("Tensors: {}", info.tensor_count);
                    println!(
                        "Parameters: {} ({:.2}M)",
                        info.param_count,
                        info.param_count as f64 / 1e6
                    );

                    if detailed {
                        println!("\nTensor Details:");
                        for (name, shape, dtype) in &info.tensors {
                            let elements: usize = shape.iter().product();
                            println!(
                                "  {}: {:?} ({}) - {} elements",
                                name, shape, dtype, elements
                            );
                        }
                    }
                }
            } else if ext == "onnx" {
                // ONNX file - extract weight info
                let model = load_onnx(&file)
                    .with_context(|| format!("Failed to load ONNX file: {}", file.display()))?;

                let weight_count = model.weights.len();
                let param_count: usize = model.weights.iter().map(|(_, w)| w.len()).sum();

                if json {
                    let output = json!({
                        "format": "onnx",
                        "file": file.to_string_lossy(),
                        "tensor_count": weight_count,
                        "param_count": param_count,
                        "tensors": if detailed {
                            model.weights.iter().map(|(name, w)| {
                                json!({
                                    "name": name,
                                    "shape": w.shape(),
                                    "dtype": "F32",
                                    "elements": w.len()
                                })
                            }).collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    println!("Format: ONNX");
                    println!("File: {}", file.display());
                    println!("Tensors: {}", weight_count);
                    println!(
                        "Parameters: {} ({:.2}M)",
                        param_count,
                        param_count as f64 / 1e6
                    );

                    if detailed {
                        println!("\nTensor Details:");
                        for (name, w) in model.weights.iter() {
                            println!("  {}: {:?} - {} elements", name, w.shape(), w.len());
                        }
                    }
                }
            } else if ext == "pt" || ext == "pth" || ext == "bin" {
                #[cfg(feature = "pytorch")]
                {
                    use gamma_onnx::pytorch::pytorch_info;
                    // PyTorch file
                    let info = pytorch_info(&file).with_context(|| {
                        format!("Failed to read PyTorch file: {}", file.display())
                    })?;

                    if json {
                        let output = json!({
                            "format": "pytorch",
                            "file": file.to_string_lossy(),
                            "tensor_count": info.tensor_count,
                            "param_count": info.param_count,
                            "tensors": if detailed {
                                info.tensors.iter().map(|(name, shape, dtype)| {
                                    json!({
                                        "name": name,
                                        "shape": shape,
                                        "dtype": dtype,
                                        "elements": shape.iter().product::<usize>()
                                    })
                                }).collect::<Vec<_>>()
                            } else {
                                vec![]
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("Format: PyTorch");
                        println!("File: {}", file.display());
                        println!("Tensors: {}", info.tensor_count);
                        println!(
                            "Parameters: {} ({:.2}M)",
                            info.param_count,
                            info.param_count as f64 / 1e6
                        );

                        if detailed {
                            println!("\nTensor Details:");
                            for (name, shape, dtype) in &info.tensors {
                                let elements: usize = shape.iter().product();
                                println!(
                                    "  {}: {:?} ({}) - {} elements",
                                    name, shape, dtype, elements
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "pytorch"))]
                {
                    anyhow::bail!("PyTorch support not enabled. Rebuild with --features pytorch");
                }
            } else if ext == "mlmodel"
                || ext == "mlpackage"
                || file.to_string_lossy().ends_with(".mlpackage")
            {
                #[cfg(feature = "coreml")]
                {
                    use gamma_onnx::coreml::coreml_info;
                    // CoreML file
                    let info = coreml_info(&file).with_context(|| {
                        format!("Failed to read CoreML file: {}", file.display())
                    })?;

                    if json {
                        let output = json!({
                            "format": "coreml",
                            "file": file.to_string_lossy(),
                            "spec_version": info.spec_version,
                            "tensor_count": info.tensor_count,
                            "param_count": info.param_count,
                            "tensors": if detailed {
                                info.tensors.iter().map(|(name, shape, dtype)| {
                                    json!({
                                        "name": name,
                                        "shape": shape,
                                        "dtype": dtype,
                                        "elements": shape.iter().product::<usize>()
                                    })
                                }).collect::<Vec<_>>()
                            } else {
                                vec![]
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("Format: CoreML");
                        println!("File: {}", file.display());
                        println!("Spec Version: {}", info.spec_version);
                        println!("Tensors: {}", info.tensor_count);
                        println!(
                            "Parameters: {} ({:.2}M)",
                            info.param_count,
                            info.param_count as f64 / 1e6
                        );

                        if detailed {
                            println!("\nTensor Details:");
                            for (name, shape, dtype) in &info.tensors {
                                let elements: usize = shape.iter().product();
                                println!(
                                    "  {}: {:?} ({}) - {} elements",
                                    name, shape, dtype, elements
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "coreml"))]
                {
                    anyhow::bail!("CoreML support not enabled. Rebuild with --features coreml");
                }
            } else if ext == "gguf" {
                #[cfg(feature = "gguf")]
                {
                    use gamma_onnx::gguf::gguf_info;
                    // GGUF file (llama.cpp format)
                    let info = gguf_info(&file)
                        .with_context(|| format!("Failed to read GGUF file: {}", file.display()))?;

                    if json {
                        let output = json!({
                            "format": "gguf",
                            "file": file.to_string_lossy(),
                            "version": info.version,
                            "architecture": info.architecture,
                            "model_name": info.model_name,
                            "tensor_count": info.tensor_count,
                            "param_count": info.param_count,
                            "metadata": info.metadata.iter().map(|(k, v)| json!({"key": k, "value": v})).collect::<Vec<_>>(),
                            "tensors": if detailed {
                                info.tensors.iter().map(|(name, shape, dtype, is_quantized)| {
                                    json!({
                                        "name": name,
                                        "shape": shape,
                                        "dtype": dtype,
                                        "quantized": is_quantized,
                                        "elements": shape.iter().product::<u64>()
                                    })
                                }).collect::<Vec<_>>()
                            } else {
                                vec![]
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("Format: GGUF (llama.cpp)");
                        println!("File: {}", file.display());
                        println!("Version: {}", info.version);
                        if let Some(arch) = &info.architecture {
                            println!("Architecture: {}", arch);
                        }
                        if let Some(name) = &info.model_name {
                            println!("Model Name: {}", name);
                        }
                        println!("Tensors: {}", info.tensor_count);
                        println!(
                            "Parameters: {} ({:.2}M)",
                            info.param_count,
                            info.param_count as f64 / 1e6
                        );

                        // Count quantized vs non-quantized
                        let quantized_count = info.tensors.iter().filter(|(_, _, _, q)| *q).count();
                        if quantized_count > 0 {
                            println!(
                                "Quantized tensors: {} (not loadable for diff)",
                                quantized_count
                            );
                        }

                        if !info.metadata.is_empty() {
                            println!("\nMetadata:");
                            for (key, value) in &info.metadata {
                                println!("  {}: {}", key, value);
                            }
                        }

                        if detailed {
                            println!("\nTensor Details:");
                            for (name, shape, dtype, is_quantized) in &info.tensors {
                                let elements: u64 = shape.iter().product();
                                let quant_marker = if *is_quantized { " [Q]" } else { "" };
                                println!(
                                    "  {}: {:?} ({}{}) - {} elements",
                                    name, shape, dtype, quant_marker, elements
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "gguf"))]
                {
                    anyhow::bail!("GGUF support not enabled. Rebuild with --features gguf");
                }
            } else {
                anyhow::bail!("Unsupported file format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .mlmodel, .mlpackage, or .gguf", ext);
            }
        }

        WeightsAction::Diff {
            file_a,
            file_b,
            tolerance,
            show_all,
            json,
        } => {
            // Load weights from both files
            let weights_a = load_weights_from_file(&file_a)?;
            let weights_b = load_weights_from_file(&file_b)?;

            // Compare
            let mut comparisons = Vec::new();
            let mut max_diff = 0.0f32;
            let mut differing_count = 0;

            // Find common tensors
            for (name, tensor_a) in weights_a.iter() {
                if let Some(tensor_b) = weights_b.get(name) {
                    // Compare shapes
                    if tensor_a.shape() != tensor_b.shape() {
                        comparisons.push(json!({
                            "name": name,
                            "status": "shape_mismatch",
                            "shape_a": tensor_a.shape(),
                            "shape_b": tensor_b.shape()
                        }));
                        differing_count += 1;
                        continue;
                    }

                    // Compare values
                    let diff = tensor_a
                        .iter()
                        .zip(tensor_b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);

                    max_diff = max_diff.max(diff);

                    if diff > tolerance {
                        differing_count += 1;
                        comparisons.push(json!({
                            "name": name,
                            "status": "differs",
                            "max_diff": diff,
                            "shape": tensor_a.shape()
                        }));
                    } else if show_all {
                        comparisons.push(json!({
                            "name": name,
                            "status": "match",
                            "max_diff": diff,
                            "shape": tensor_a.shape()
                        }));
                    }
                } else {
                    comparisons.push(json!({
                        "name": name,
                        "status": "missing_in_b"
                    }));
                    differing_count += 1;
                }
            }

            // Check for tensors only in B
            for name in weights_b.keys() {
                if weights_a.get(name).is_none() {
                    comparisons.push(json!({
                        "name": name,
                        "status": "missing_in_a"
                    }));
                    differing_count += 1;
                }
            }

            let is_match = differing_count == 0;

            if json {
                let output = json!({
                    "file_a": file_a.to_string_lossy(),
                    "file_b": file_b.to_string_lossy(),
                    "tolerance": tolerance,
                    "result": if is_match { "match" } else { "differs" },
                    "max_difference": max_diff,
                    "differing_tensors": differing_count,
                    "total_tensors_a": weights_a.len(),
                    "total_tensors_b": weights_b.len(),
                    "comparisons": comparisons
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("File A: {}", file_a.display());
                println!("File B: {}", file_b.display());
                println!("Tolerance: {}", tolerance);
                println!();

                if is_match {
                    println!("Result: MATCH");
                    println!("Max difference: {:.6e}", max_diff);
                } else {
                    println!("Result: DIFFERS");
                    println!("Max difference: {:.6e}", max_diff);
                    println!("Differing tensors: {}", differing_count);
                    println!();

                    for comp in &comparisons {
                        let status = comp["status"].as_str().unwrap_or("");
                        let name = comp["name"].as_str().unwrap_or("");
                        match status {
                            "differs" => {
                                let diff = comp["max_diff"].as_f64().unwrap_or(0.0);
                                println!("  {} - max diff: {:.6e}", name, diff);
                            }
                            "shape_mismatch" => {
                                println!(
                                    "  {} - SHAPE MISMATCH: {:?} vs {:?}",
                                    name, comp["shape_a"], comp["shape_b"]
                                );
                            }
                            "missing_in_a" => {
                                println!("  {} - only in file B", name);
                            }
                            "missing_in_b" => {
                                println!("  {} - only in file A", name);
                            }
                            _ => {}
                        }
                        if !show_all && differing_count > 10 && comparisons.len() > 10 {
                            println!("  ... and {} more", differing_count - 10);
                            break;
                        }
                    }
                }
            }
        }

        WeightsAction::Norms { file, json } => {
            handle_weights_norms(&file, json)?;
        }
    }

    Ok(())
}

/// Load weights from a file (SafeTensors, ONNX, PyTorch, or CoreML)
fn load_weights_from_file(path: &std::path::Path) -> Result<gamma_onnx::WeightStore> {
    use gamma_onnx::safetensors::load_safetensors;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check for .mlpackage directory (CoreML)
    if ext == "mlpackage" || (path.is_dir() && path.to_string_lossy().ends_with(".mlpackage")) {
        #[cfg(feature = "coreml")]
        {
            use gamma_onnx::coreml::load_coreml;
            return load_coreml(path)
                .map_err(|e| anyhow::anyhow!("Failed to load CoreML package: {}", e));
        }
        #[cfg(not(feature = "coreml"))]
        {
            anyhow::bail!("CoreML support not enabled. Rebuild with --features coreml");
        }
    }

    match ext.as_str() {
        "safetensors" => {
            load_safetensors(path)
                .map_err(|e| anyhow::anyhow!("Failed to load SafeTensors: {}", e))
        }
        "onnx" => {
            let model = load_onnx(path)?;
            Ok(model.weights)
        }
        #[cfg(feature = "pytorch")]
        "pt" | "pth" | "bin" => {
            use gamma_onnx::pytorch::load_pytorch;
            load_pytorch(path)
                .map_err(|e| anyhow::anyhow!("Failed to load PyTorch file: {}", e))
        }
        #[cfg(not(feature = "pytorch"))]
        "pt" | "pth" | "bin" => {
            anyhow::bail!("PyTorch support not enabled. Rebuild with --features pytorch")
        }
        #[cfg(feature = "coreml")]
        "mlmodel" => {
            use gamma_onnx::coreml::load_coreml;
            load_coreml(path)
                .map_err(|e| anyhow::anyhow!("Failed to load CoreML model: {}", e))
        }
        #[cfg(not(feature = "coreml"))]
        "mlmodel" => {
            anyhow::bail!("CoreML support not enabled. Rebuild with --features coreml")
        }
        #[cfg(feature = "gguf")]
        "gguf" => {
            use gamma_onnx::gguf::load_gguf;
            load_gguf(path)
                .map_err(|e| anyhow::anyhow!("Failed to load GGUF file: {}", e))
        }
        #[cfg(not(feature = "gguf"))]
        "gguf" => {
            anyhow::bail!("GGUF support not enabled. Rebuild with --features gguf")
        }
        _ => anyhow::bail!("Unsupported file format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .mlmodel, .mlpackage, or .gguf", ext),
    }
}

/// Handle the Weights Norms subcommand - compute per-block weight norms
fn handle_weights_norms(file: &std::path::Path, json_output: bool) -> Result<()> {
    use serde_json::json;

    let weights = load_weights_from_file(file)?;

    // Detect block naming pattern and extract per-block norms
    let block_norms = compute_block_norms(&weights);

    if json_output {
        let blocks_json: Vec<_> = block_norms
            .iter()
            .map(|b| {
                json!({
                    "block": b.block_index,
                    "attn_q_frobenius": b.attn_q_frobenius,
                    "attn_q_spectral": b.attn_q_spectral,
                    "attn_k_frobenius": b.attn_k_frobenius,
                    "attn_k_spectral": b.attn_k_spectral,
                    "attn_v_frobenius": b.attn_v_frobenius,
                    "attn_v_spectral": b.attn_v_spectral,
                    "attn_output_frobenius": b.attn_output_frobenius,
                    "attn_output_spectral": b.attn_output_spectral,
                    "ffn_up_frobenius": b.ffn_up_frobenius,
                    "ffn_up_spectral": b.ffn_up_spectral,
                    "ffn_down_frobenius": b.ffn_down_frobenius,
                    "ffn_down_spectral": b.ffn_down_spectral,
                    "ffn_gate_frobenius": b.ffn_gate_frobenius,
                    "ffn_gate_spectral": b.ffn_gate_spectral,
                    "total_frobenius": b.total_frobenius,
                    "max_spectral": b.max_spectral,
                })
            })
            .collect();

        // Compute summary stats
        let total_frobenius_values: Vec<f64> =
            block_norms.iter().map(|b| b.total_frobenius).collect();
        let max_spectral_values: Vec<f64> = block_norms.iter().map(|b| b.max_spectral).collect();

        let max_total_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_total_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let frob_range = if min_total_frob > 0.0 {
            max_total_frob / min_total_frob
        } else {
            f64::INFINITY
        };
        let spec_range = if min_spec > 0.0 {
            max_spec / min_spec
        } else {
            f64::INFINITY
        };

        let output = json!({
            "file": file.to_string_lossy(),
            "block_count": block_norms.len(),
            "blocks": blocks_json,
            "summary": {
                "max_total_frobenius": max_total_frob,
                "min_total_frobenius": min_total_frob,
                "max_spectral": max_spec,
                "min_spectral": min_spec,
                "frobenius_range": frob_range,
                "spectral_range": spec_range,
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Weight Norms Analysis");
        println!("File: {}", file.display());
        println!("Blocks: {}\n", block_norms.len());

        println!(
            "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Block", "Attn_out_F", "FFN_down_F", "FFN_up_F", "Max_Spec", "Total_F"
        );
        println!("{}", "-".repeat(72));

        for b in &block_norms {
            println!(
                "{:>6} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}",
                b.block_index,
                b.attn_output_frobenius,
                b.ffn_down_frobenius,
                b.ffn_up_frobenius,
                b.max_spectral,
                b.total_frobenius
            );
        }

        // Summary
        let total_frobenius_values: Vec<f64> =
            block_norms.iter().map(|b| b.total_frobenius).collect();
        let max_spectral_values: Vec<f64> = block_norms.iter().map(|b| b.max_spectral).collect();

        let max_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_frob = total_frobenius_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_spec = max_spectral_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        println!("\nSummary:");
        println!(
            "  Total Frobenius range: {:.4e} - {:.4e} ({:.2}x)",
            min_frob,
            max_frob,
            max_frob / min_frob
        );
        println!(
            "  Max Spectral range: {:.4e} - {:.4e} ({:.2}x)",
            min_spec,
            max_spec,
            max_spec / min_spec
        );
    }

    Ok(())
}

/// Per-block weight norm statistics
struct BlockNorms {
    block_index: usize,
    attn_q_frobenius: f64,
    attn_q_spectral: f64,
    attn_k_frobenius: f64,
    attn_k_spectral: f64,
    attn_v_frobenius: f64,
    attn_v_spectral: f64,
    attn_output_frobenius: f64,
    attn_output_spectral: f64,
    ffn_up_frobenius: f64,
    ffn_up_spectral: f64,
    ffn_down_frobenius: f64,
    ffn_down_spectral: f64,
    ffn_gate_frobenius: f64,
    ffn_gate_spectral: f64,
    total_frobenius: f64,
    max_spectral: f64,
}

/// Compute Frobenius norm of a tensor: ||A||_F = sqrt(sum(a_ij^2))
fn frobenius_norm(tensor: &ndarray::ArrayD<f32>) -> f64 {
    tensor
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Approximate spectral norm via power iteration (largest singular value)
/// For matrix A, spectral norm = max singular value = sqrt(max eigenvalue of A^T A)
fn spectral_norm_approx(tensor: &ndarray::ArrayD<f32>, iterations: usize) -> f64 {
    // Only works for 2D matrices
    if tensor.ndim() != 2 {
        return frobenius_norm(tensor); // Fallback for non-2D
    }

    let shape = tensor.shape();
    let (m, n) = (shape[0], shape[1]);

    // Reshape to 2D for matrix operations
    let matrix = tensor.view().into_shape_with_order((m, n)).unwrap();

    // Initialize random vector
    let mut v: Vec<f64> = (0..n)
        .map(|i| ((i * 31337) % 1000) as f64 / 1000.0)
        .collect();

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    // Power iteration: v = A^T A v / ||A^T A v||
    for _ in 0..iterations {
        // u = A v
        let mut u = vec![0.0f64; m];
        for i in 0..m {
            for j in 0..n {
                u[i] += matrix[[i, j]] as f64 * v[j];
            }
        }

        // v = A^T u
        let mut v_new = vec![0.0f64; n];
        for j in 0..n {
            for i in 0..m {
                v_new[j] += matrix[[i, j]] as f64 * u[i];
            }
        }

        // Normalize
        let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return 0.0;
        }
        for x in &mut v_new {
            *x /= norm;
        }
        v = v_new;
    }

    // Compute ||Av|| as estimate of largest singular value
    let mut av = vec![0.0f64; m];
    for i in 0..m {
        for j in 0..n {
            av[i] += matrix[[i, j]] as f64 * v[j];
        }
    }
    av.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Extract block number from GGUF-style name (e.g., "blk.5.attn_q.weight" -> 5)
fn extract_gguf_block_number(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        num_str.parse().ok()
    } else {
        None
    }
}

/// Extract block number from Whisper/HF-style name (e.g., "encoder.layers.5.self_attn.q_proj.weight" -> 5)
fn extract_hf_block_number(name: &str) -> Option<usize> {
    // Try patterns like "encoder.layers.N", "decoder.layers.N", "model.layers.N", "layers.N"
    let patterns = [
        "encoder.layers.",
        "decoder.layers.",
        "model.layers.",
        "layers.",
    ];
    for pattern in patterns {
        if let Some(rest) = name.find(pattern) {
            let after_pattern = &name[rest + pattern.len()..];
            let num_str: String = after_pattern
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = num_str.parse() {
                return Some(n);
            }
        }
    }
    None
}

/// Compute per-block weight norms for a model
fn compute_block_norms(weights: &gamma_onnx::WeightStore) -> Vec<BlockNorms> {
    use std::collections::BTreeMap;

    // Determine naming convention by checking first weight name
    let names: Vec<&String> = weights.keys().collect();
    let is_gguf_style = names.iter().any(|n| n.starts_with("blk."));

    // Group weights by block
    let mut block_weights: BTreeMap<usize, Vec<(&String, &ndarray::ArrayD<f32>)>> = BTreeMap::new();

    for (name, tensor) in weights.iter() {
        let block_num = if is_gguf_style {
            extract_gguf_block_number(name)
        } else {
            extract_hf_block_number(name)
        };

        if let Some(n) = block_num {
            block_weights.entry(n).or_default().push((name, tensor));
        }
    }

    // For each block, compute norms
    let mut results = Vec::new();

    for (&block_idx, tensors) in &block_weights {
        let mut norms = BlockNorms {
            block_index: block_idx,
            attn_q_frobenius: 0.0,
            attn_q_spectral: 0.0,
            attn_k_frobenius: 0.0,
            attn_k_spectral: 0.0,
            attn_v_frobenius: 0.0,
            attn_v_spectral: 0.0,
            attn_output_frobenius: 0.0,
            attn_output_spectral: 0.0,
            ffn_up_frobenius: 0.0,
            ffn_up_spectral: 0.0,
            ffn_down_frobenius: 0.0,
            ffn_down_spectral: 0.0,
            ffn_gate_frobenius: 0.0,
            ffn_gate_spectral: 0.0,
            total_frobenius: 0.0,
            max_spectral: 0.0,
        };

        for (name, tensor) in tensors {
            let frob = frobenius_norm(tensor);
            let spec = if tensor.ndim() == 2 {
                spectral_norm_approx(tensor, 20) // 20 iterations
            } else {
                frob // For non-2D, just use Frobenius
            };

            // Categorize by weight name pattern
            let name_lower = name.to_lowercase();

            if name_lower.contains("attn_q") || name_lower.contains("q_proj") {
                norms.attn_q_frobenius = frob;
                norms.attn_q_spectral = spec;
            } else if name_lower.contains("attn_k") || name_lower.contains("k_proj") {
                norms.attn_k_frobenius = frob;
                norms.attn_k_spectral = spec;
            } else if name_lower.contains("attn_v") || name_lower.contains("v_proj") {
                norms.attn_v_frobenius = frob;
                norms.attn_v_spectral = spec;
            } else if name_lower.contains("attn_output")
                || name_lower.contains("out_proj")
                || name_lower.contains("o_proj")
            {
                norms.attn_output_frobenius = frob;
                norms.attn_output_spectral = spec;
            } else if name_lower.contains("ffn_up")
                || name_lower.contains("up_proj")
                || name_lower.contains("fc1")
            {
                norms.ffn_up_frobenius = frob;
                norms.ffn_up_spectral = spec;
            } else if name_lower.contains("ffn_down")
                || name_lower.contains("down_proj")
                || name_lower.contains("fc2")
            {
                norms.ffn_down_frobenius = frob;
                norms.ffn_down_spectral = spec;
            } else if name_lower.contains("ffn_gate") || name_lower.contains("gate_proj") {
                norms.ffn_gate_frobenius = frob;
                norms.ffn_gate_spectral = spec;
            }

            norms.total_frobenius += frob * frob; // Sum squares for total
            if spec > norms.max_spectral {
                norms.max_spectral = spec;
            }
        }

        norms.total_frobenius = norms.total_frobenius.sqrt(); // sqrt of sum of squares
        results.push(norms);
    }

    results
}

/// Helper to run a benchmark function with warmup and timing
/// Benchmark result for JSON output
struct BenchResult {
    name: String,
    iterations: usize,
    per_iter_ns: u64,
    total_ns: u64,
}

fn bench_collect<F: FnMut()>(name: &str, iterations: usize, mut f: F) -> BenchResult {
    // Warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;

    BenchResult {
        name: name.to_string(),
        iterations,
        per_iter_ns: per_iter.as_nanos() as u64,
        total_ns: elapsed.as_nanos() as u64,
    }
}

fn bench<F: FnMut()>(
    name: &str,
    iterations: usize,
    f: F,
    json: bool,
    results: &mut Vec<BenchResult>,
) {
    let result = bench_collect(name, iterations, f);
    if !json {
        println!(
            "{}: {:?} per iteration ({} iterations)",
            name,
            std::time::Duration::from_nanos(result.per_iter_ns),
            iterations
        );
    }
    results.push(result);
}

/// Create a BoundedTensor with specified shape from center and epsilon
fn make_bench_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon)
}

/// Run benchmarks based on the benchmark type
fn run_benchmarks(benchmark: &str, json: bool) -> Result<()> {
    if !json {
        println!("γ-CROWN Benchmark Suite");
        println!("========================\n");
    }

    // Whisper-tiny dimensions
    let batch = 1;
    let seq_len = 16;
    let hidden_dim = 384;
    let intermediate_dim = 1536;
    let num_heads = 6;
    let head_dim = 64;
    let epsilon = 0.01_f32;

    if !json {
        println!("Dimensions:");
        println!(
            "  batch={}, seq={}, hidden={}, intermediate={}",
            batch, seq_len, hidden_dim, intermediate_dim
        );
        println!("  heads={}, head_dim={}\n", num_heads, head_dim);
    }

    // Create common layers
    let linear_weight = Array2::from_shape_fn((intermediate_dim, hidden_dim), |_| 0.01_f32);
    let linear_bias = Some(Array1::zeros(intermediate_dim));
    let linear1 = LinearLayer::new(linear_weight.clone(), linear_bias.clone())?;

    let linear_weight2 = Array2::from_shape_fn((hidden_dim, intermediate_dim), |_| 0.01_f32);
    let linear_bias2 = Some(Array1::zeros(hidden_dim));
    let linear2 = LinearLayer::new(linear_weight2, linear_bias2)?;

    let gelu = GELULayer::default();
    let layernorm = LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5);

    // Collect all benchmark results
    let mut results: Vec<BenchResult> = Vec::new();
    let mut unknown_type = false;

    match benchmark {
        "layer" => {
            if !json {
                println!("=== Layer Benchmarks (IBP) ===\n");
            }

            let input = make_bench_input(&[batch, seq_len, hidden_dim], 0.5, epsilon);

            // Linear layer
            let mut linear_output = input.clone();
            bench(
                "Linear IBP [384->1536]",
                100,
                || {
                    linear_output = linear1.propagate_ibp(&input).unwrap();
                },
                json,
                &mut results,
            );

            // GELU
            bench(
                "GELU IBP [1536]",
                100,
                || {
                    let _ = gelu.propagate_ibp(&linear_output);
                },
                json,
                &mut results,
            );
            let gelu_output = gelu.propagate_ibp(&linear_output)?;

            // Linear back
            bench(
                "Linear IBP [1536->384]",
                100,
                || {
                    let _ = linear2.propagate_ibp(&gelu_output);
                },
                json,
                &mut results,
            );
            let final_output = linear2.propagate_ibp(&gelu_output)?;

            // LayerNorm
            bench(
                "LayerNorm IBP [384]",
                100,
                || {
                    let _ = layernorm.propagate_ibp(&final_output);
                },
                json,
                &mut results,
            );

            if !json {
                println!("\n=== Full MLP Path IBP ===\n");
            }

            let mut mlp = Network::new();
            mlp.add_layer(Layer::Linear(linear1.clone()));
            mlp.add_layer(Layer::GELU(gelu.clone()));
            mlp.add_layer(Layer::Linear(linear2.clone()));

            bench(
                "Full MLP IBP [384->1536->384]",
                100,
                || {
                    let _ = mlp.propagate_ibp(&input);
                },
                json,
                &mut results,
            );
        }

        "attention" => {
            if !json {
                println!("=== Attention Component Benchmarks (IBP) ===\n");
            }

            // MatMul: Q @ K^T
            let q_input = make_bench_input(&[batch, num_heads, seq_len, head_dim], 0.5, 0.1);
            let k_input = make_bench_input(&[batch, num_heads, head_dim, seq_len], 0.5, 0.1);

            let matmul = MatMulLayer::new(false, None);

            bench(
                &format!(
                    "MatMul IBP [{},{},{},{}] @ [{},{},{},{}]",
                    batch, num_heads, seq_len, head_dim, batch, num_heads, head_dim, seq_len
                ),
                100,
                || {
                    let _ = matmul.propagate_ibp_binary(&q_input, &k_input);
                },
                json,
                &mut results,
            );

            // Softmax
            let attn_input = make_bench_input(&[batch, num_heads, seq_len, seq_len], 0.0, 1.0);
            let softmax = SoftmaxLayer::new(-1);

            bench(
                &format!(
                    "Softmax IBP [{},{},{},{}]",
                    batch, num_heads, seq_len, seq_len
                ),
                100,
                || {
                    let _ = softmax.propagate_ibp(&attn_input);
                },
                json,
                &mut results,
            );

            if !json {
                println!("\n=== MatMul Scaling ===\n");
            }

            for seq in [4, 16, 64] {
                let q = make_bench_input(&[batch, num_heads, seq, head_dim], 0.5, 0.1);
                let k = make_bench_input(&[batch, num_heads, head_dim, seq], 0.5, 0.1);
                let iterations = if seq <= 16 { 100 } else { 20 };

                bench(
                    &format!("MatMul IBP seq={}", seq),
                    iterations,
                    || {
                        let _ = matmul.propagate_ibp_binary(&q, &k);
                    },
                    json,
                    &mut results,
                );
            }
        }

        "full" => {
            if !json {
                println!("=== Full Pipeline Benchmarks ===\n");
            }

            let mut mlp = Network::new();
            mlp.add_layer(Layer::Linear(linear1.clone()));
            mlp.add_layer(Layer::GELU(gelu.clone()));
            mlp.add_layer(Layer::Linear(linear2.clone()));

            if !json {
                println!("=== IBP Scaling ===\n");
            }

            for seq in [4, 16, 64, 128] {
                let input = make_bench_input(&[batch, seq, hidden_dim], 0.5, epsilon);
                let iterations = if seq <= 16 {
                    100
                } else if seq <= 64 {
                    20
                } else {
                    5
                };

                bench(
                    &format!("MLP IBP seq={}", seq),
                    iterations,
                    || {
                        let _ = mlp.propagate_ibp(&input);
                    },
                    json,
                    &mut results,
                );
            }

            if !json {
                println!("\n=== CROWN (1-D) ===\n");
            }

            let input_1d = make_bench_input(&[hidden_dim], 0.5, epsilon);
            bench(
                "Full MLP CROWN 1-D [384]",
                100,
                || {
                    let _ = mlp.propagate_crown(&input_1d);
                },
                json,
                &mut results,
            );
        }

        _ => {
            unknown_type = true;
            if !json {
                println!(
                    "Unknown benchmark type: {}. Available: layer, attention, full",
                    benchmark
                );
            }
        }
    }

    if json {
        let results_json: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "name": r.name,
                    "iterations": r.iterations,
                    "per_iter_ns": r.per_iter_ns,
                    "per_iter_us": r.per_iter_ns as f64 / 1000.0,
                    "per_iter_ms": r.per_iter_ns as f64 / 1_000_000.0,
                    "total_ns": r.total_ns,
                    "total_ms": r.total_ns as f64 / 1_000_000.0
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::json!({
                "benchmark_type": benchmark,
                "valid_type": !unknown_type,
                "dimensions": {
                    "batch": batch,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "intermediate_dim": intermediate_dim,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "epsilon": epsilon
                },
                "results": results_json
            })
        );
    } else {
        println!("\n=== Summary ===");
        println!("Benchmark complete. Use --benchmark <type> to run specific benchmarks.");
        println!("  layer     - Individual layer IBP performance");
        println!("  attention - Attention component (MatMul, Softmax) performance");
        println!("  full      - Full pipeline scaling tests");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // eps_sweep tests
    // ============================================================

    #[test]
    fn test_eps_sweep_linear_basic() {
        let result = eps_sweep(0.1, 1.0, 5, true).unwrap();
        assert_eq!(result.len(), 5);
        // Linear: 0.1, 0.325, 0.55, 0.775, 1.0
        assert!((result[0] - 0.1).abs() < 1e-6);
        assert!((result[4] - 1.0).abs() < 1e-6);
        // Check linear spacing
        let diff1 = result[1] - result[0];
        let diff2 = result[2] - result[1];
        assert!((diff1 - diff2).abs() < 1e-6);
    }

    #[test]
    fn test_eps_sweep_log_basic() {
        let result = eps_sweep(0.001, 1.0, 4, false).unwrap();
        assert_eq!(result.len(), 4);
        // Log scale: 0.001, 0.01, 0.1, 1.0 (ratio = 1000, each step = 10x)
        assert!((result[0] - 0.001).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
        // Check geometric spacing (ratio between consecutive should be constant)
        let ratio1 = result[1] / result[0];
        let ratio2 = result[2] / result[1];
        assert!((ratio1 - ratio2).abs() < 0.01);
    }

    #[test]
    fn test_eps_sweep_single_step() {
        let result = eps_sweep(0.5, 2.0, 1, true).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_eps_sweep_two_steps() {
        let result = eps_sweep(0.1, 0.9, 2, true).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.1).abs() < 1e-6);
        assert!((result[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_eps_sweep_zero_steps_error() {
        let result = eps_sweep(0.1, 1.0, 0, true);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("steps must be >= 1"));
    }

    #[test]
    fn test_eps_sweep_negative_epsilon_error() {
        let result = eps_sweep(-0.1, 1.0, 5, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));
    }

    #[test]
    fn test_eps_sweep_zero_epsilon_error() {
        let result = eps_sweep(0.0, 1.0, 5, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));
    }

    #[test]
    fn test_eps_sweep_nan_error() {
        let result = eps_sweep(f32::NAN, 1.0, 5, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));
    }

    #[test]
    fn test_eps_sweep_min_greater_than_max_error() {
        let result = eps_sweep(2.0, 1.0, 5, true);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("epsilon_min must be <= epsilon_max"));
    }

    #[test]
    fn test_eps_sweep_equal_min_max() {
        let result = eps_sweep(0.5, 0.5, 3, true).unwrap();
        assert_eq!(result.len(), 3);
        for v in &result {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    // ============================================================
    // resolve_backend tests
    // ============================================================

    #[test]
    fn test_resolve_backend_explicit_wgpu() {
        let result = resolve_backend(BackendArg::Wgpu, false);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_explicit_wgpu_with_gpu_flag() {
        // --backend takes precedence over --gpu
        let result = resolve_backend(BackendArg::Wgpu, true);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_explicit_mlx() {
        let result = resolve_backend(BackendArg::Mlx, false);
        assert_eq!(result, BackendArg::Mlx);
    }

    #[test]
    fn test_resolve_backend_legacy_gpu_flag() {
        // When --backend is default (Cpu) and --gpu is true, use wgpu
        let result = resolve_backend(BackendArg::Cpu, true);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_default_cpu() {
        let result = resolve_backend(BackendArg::Cpu, false);
        assert_eq!(result, BackendArg::Cpu);
    }

    // ============================================================
    // frobenius_norm tests
    // ============================================================

    #[test]
    fn test_frobenius_norm_1d() {
        // [3, 4] -> sqrt(9 + 16) = 5
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![3.0f32, 4.0]).unwrap();
        let norm = frobenius_norm(&tensor);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_frobenius_norm_2d() {
        // [[1, 2], [3, 4]] -> sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0])
                .unwrap();
        let norm = frobenius_norm(&tensor);
        assert!((norm - 30.0f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_frobenius_norm_zeros() {
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0f32; 3]).unwrap();
        let norm = frobenius_norm(&tensor);
        assert!(norm.abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm_negative() {
        // [-3, 4] -> sqrt(9 + 16) = 5 (squares make negatives positive)
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-3.0f32, 4.0]).unwrap();
        let norm = frobenius_norm(&tensor);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    // ============================================================
    // spectral_norm_approx tests
    // ============================================================

    #[test]
    fn test_spectral_norm_identity_2x2() {
        // Identity matrix has spectral norm = 1
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 0.0, 0.0, 1.0])
                .unwrap();
        let norm = spectral_norm_approx(&tensor, 20);
        assert!((norm - 1.0).abs() < 0.1, "spectral norm of I = {}", norm);
    }

    #[test]
    fn test_spectral_norm_scaled_identity() {
        // 3*I has spectral norm = 3
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![3.0f32, 0.0, 0.0, 3.0])
                .unwrap();
        let norm = spectral_norm_approx(&tensor, 20);
        assert!((norm - 3.0).abs() < 0.3, "spectral norm of 3I = {}", norm);
    }

    #[test]
    fn test_spectral_norm_non_square() {
        // 2x3 matrix - check it doesn't crash and returns reasonable value
        let tensor = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let norm = spectral_norm_approx(&tensor, 20);
        // Known: singular values are approximately 9.508 and 0.773, largest is ~9.5
        assert!(norm > 8.0 && norm < 11.0, "spectral norm = {}", norm);
    }

    #[test]
    fn test_spectral_norm_fallback_1d() {
        // 1D tensor should fallback to Frobenius norm
        let tensor =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![3.0f32, 4.0, 0.0, 0.0])
                .unwrap();
        let norm = spectral_norm_approx(&tensor, 20);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    // ============================================================
    // extract_gguf_block_number tests
    // ============================================================

    #[test]
    fn test_extract_gguf_block_basic() {
        assert_eq!(extract_gguf_block_number("blk.5.attn_q.weight"), Some(5));
        assert_eq!(extract_gguf_block_number("blk.0.ffn_gate.weight"), Some(0));
        assert_eq!(extract_gguf_block_number("blk.31.attn_v.weight"), Some(31));
    }

    #[test]
    fn test_extract_gguf_block_large_number() {
        assert_eq!(
            extract_gguf_block_number("blk.127.attn_k.weight"),
            Some(127)
        );
    }

    #[test]
    fn test_extract_gguf_block_not_a_block() {
        assert_eq!(extract_gguf_block_number("token_embd.weight"), None);
        assert_eq!(extract_gguf_block_number("output.weight"), None);
        assert_eq!(extract_gguf_block_number("output_norm.weight"), None);
    }

    #[test]
    fn test_extract_gguf_block_malformed() {
        assert_eq!(extract_gguf_block_number("blk."), None);
        assert_eq!(extract_gguf_block_number("blk"), None);
        assert_eq!(extract_gguf_block_number("block.5.weight"), None);
    }

    // ============================================================
    // extract_hf_block_number tests
    // ============================================================

    #[test]
    fn test_extract_hf_block_encoder_layers() {
        assert_eq!(
            extract_hf_block_number("encoder.layers.5.self_attn.q_proj.weight"),
            Some(5)
        );
        assert_eq!(
            extract_hf_block_number("encoder.layers.11.mlp.fc1.weight"),
            Some(11)
        );
    }

    #[test]
    fn test_extract_hf_block_decoder_layers() {
        assert_eq!(
            extract_hf_block_number("decoder.layers.3.self_attn.k_proj.weight"),
            Some(3)
        );
        assert_eq!(
            extract_hf_block_number("decoder.layers.0.mlp.fc2.weight"),
            Some(0)
        );
    }

    #[test]
    fn test_extract_hf_block_model_layers() {
        assert_eq!(
            extract_hf_block_number("model.layers.15.self_attn.q_proj.weight"),
            Some(15)
        );
    }

    #[test]
    fn test_extract_hf_block_layers_only() {
        assert_eq!(
            extract_hf_block_number("layers.7.attention.weight"),
            Some(7)
        );
    }

    #[test]
    fn test_extract_hf_block_not_a_block() {
        assert_eq!(extract_hf_block_number("embed_tokens.weight"), None);
        assert_eq!(extract_hf_block_number("lm_head.weight"), None);
        assert_eq!(extract_hf_block_number("norm.weight"), None);
    }

    #[test]
    fn test_extract_hf_block_malformed() {
        assert_eq!(extract_hf_block_number("encoder.layers."), None);
        assert_eq!(extract_hf_block_number("layers"), None);
    }
}
