//! Configuration types for train command
//!
//! Contains TrainConfig, SchedulerType, and related helpers.

use std::path::PathBuf;

/// Learning rate scheduler type for CLI configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerType {
    /// Constant learning rate (no scheduling)
    #[default]
    Constant,
    /// Step decay: multiply by gamma every step_size epochs
    Step,
    /// Exponential decay: lr = initial_lr * gamma^epoch
    Exponential,
    /// Cosine annealing: oscillates between max and min lr
    Cosine,
    /// Reduce on plateau: reduce lr when validation loss stalls
    ReduceOnPlateau,
    /// Linear warmup followed by decay
    WarmupDecay,
}

impl std::str::FromStr for SchedulerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "constant" | "none" => Ok(SchedulerType::Constant),
            "step" => Ok(SchedulerType::Step),
            "exponential" | "exp" => Ok(SchedulerType::Exponential),
            "cosine" | "cos" => Ok(SchedulerType::Cosine),
            "plateau" | "reduce_on_plateau" | "rop" => Ok(SchedulerType::ReduceOnPlateau),
            "warmup" | "warmup_decay" => Ok(SchedulerType::WarmupDecay),
            _ => Err(format!(
                "Unknown scheduler type: '{}'. Valid options: constant, step, exponential, cosine, plateau, warmup",
                s
            )),
        }
    }
}

/// Configuration for the train command
pub struct TrainConfig<'a> {
    /// Directory containing learning data (default: ~/.dashprove)
    pub data_dir: Option<&'a str>,
    /// Output path for the trained model
    pub output: Option<&'a str>,
    /// Learning rate for training
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Show verbose output
    pub verbose: bool,
    /// Enable early stopping to prevent overfitting
    pub early_stopping: bool,
    /// Patience for early stopping (epochs without improvement before stopping)
    pub patience: usize,
    /// Minimum delta for early stopping (minimum improvement to reset patience)
    pub min_delta: f64,
    /// Validation split ratio for early stopping (0.0-0.5)
    pub validation_split: f64,
    /// Learning rate scheduler type
    pub lr_scheduler: SchedulerType,
    /// Step size for step scheduler (epochs between lr reductions)
    pub lr_step_size: usize,
    /// Gamma (decay factor) for step/exponential schedulers
    pub lr_gamma: f64,
    /// Minimum learning rate for cosine/plateau schedulers
    pub lr_min: f64,
    /// Warmup epochs for warmup_decay scheduler
    pub lr_warmup_epochs: usize,
    /// Enable model checkpointing during training
    pub checkpoint: bool,
    /// Directory to save checkpoints
    pub checkpoint_dir: Option<&'a str>,
    /// Save checkpoint every N epochs (0 = only save on improvement)
    pub checkpoint_interval: usize,
    /// Keep only the N best checkpoints by validation loss (0 = keep all)
    pub keep_best: usize,
    /// Resume training from a checkpoint file
    pub resume: Option<&'a str>,
}

/// Get the default data directory (~/.dashprove)
pub fn default_data_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".dashprove")
}
