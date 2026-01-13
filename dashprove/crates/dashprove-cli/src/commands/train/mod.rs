//! Train ML strategy prediction models from proof corpus
//!
//! This command loads successful proofs from the corpus and trains the ML-based
//! strategy predictor, which can then be used for intelligent backend selection.

pub mod checkpoint;
pub mod config;
pub mod core;
pub mod ensemble;
pub mod search_results;
pub mod summary;
mod tests;
pub mod tune;

// Re-export primary public API
pub use config::{SchedulerType, TrainConfig};
pub use core::run_train;
pub use ensemble::{run_ensemble, EnsembleConfig};
pub use tune::{run_tune, TuneConfig};
