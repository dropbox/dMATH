//! Counterexample types and analysis
//!
//! This module provides structured representations of counterexamples
//! from verification backends, along with analysis tools for:
//!
//! - **Comparison**: Diff two counterexamples to find divergences
//! - **Filtering**: Filter trace states by variable patterns
//! - **Minimization**: Remove redundant states from traces
//! - **Abstraction**: Abstract trace states to high-level patterns
//! - **Interleaving**: Analyze multi-actor traces as swimlane diagrams
//! - **Compression**: Detect and compress repeated patterns
//! - **Alignment**: Align multiple traces to find common prefixes
//! - **Suggestions**: Generate fix suggestions from counterexamples
//! - **Clustering**: Group similar counterexamples together

mod abstraction;
mod alignment;
mod cluster;
mod compression;
mod data_quality;
mod diff;
mod fairness;
mod filter;
mod formal_verification;
mod guardrails;
mod interleaving;
mod interpretability;
mod llm_eval;
mod minimization;
mod model_optimization;
mod nn;
mod robustness;
mod suggestion;
mod types;

// Re-export all types at module level for backward compatibility
pub use abstraction::*;
pub use alignment::*;
pub use cluster::*;
pub use compression::*;
pub use data_quality::*;
pub use diff::*;
pub use fairness::*;
pub use filter::*;
pub use formal_verification::*;
pub use guardrails::*;
pub use interleaving::*;
pub use interpretability::*;
pub use llm_eval::*;
pub use model_optimization::*;
pub use nn::*;
pub use robustness::*;
pub use suggestion::*;
pub use types::*;

#[cfg(test)]
mod tests;
