//! Counterexample corpus management
//!
//! This module provides types and functions for storing, searching, and analyzing
//! counterexamples from verification runs.
//!
//! # Key Types
//!
//! - [`CounterexampleCorpus`] - Main storage container for counterexamples
//! - [`CounterexampleEntry`] - A single stored counterexample with metadata
//! - [`CounterexampleFeatures`] - Extracted features for similarity search
//! - [`CorpusHistory`] - Historical statistics over time periods
//! - [`HistoryComparison`] - Comparison between two time periods
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::counterexamples::{CounterexampleCorpus, TimePeriod};
//! use dashprove_backends::traits::BackendId;
//!
//! let mut corpus = CounterexampleCorpus::new();
//! // Insert counterexamples...
//!
//! // Get history statistics
//! let history = corpus.history(TimePeriod::Day);
//! println!("{}", history.summary());
//! ```

mod comparison;
mod corpus;
mod history;
pub(crate) mod similarity;
mod suggestions;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use comparison::{GrowthProjections, HistoryComparison};
pub use corpus::CounterexampleCorpus;
pub use history::{CorpusHistory, PeriodStats, TimePeriod};
pub use suggestions::{
    format_suggestions, suggest_comparison_periods, PeriodSuggestion, PeriodSuggestionCliArgs,
    SuggestionType,
};
pub use types::{
    ClusterPattern, CounterexampleEntry, CounterexampleFeatures, CounterexampleId,
    SimilarCounterexample,
};
