//! Beautiful, actionable counterexample generation for Kani Fast
//!
//! This crate provides structured counterexample types and parsing for
//! Kani verification output. Counterexamples are parsed into rich structures
//! that support:
//! - Witness values with type information
//! - Failed check details with source locations
//! - Concrete playback test extraction
//! - State traces for visualization
//! - Counterexample minimization via delta debugging
//! - Natural language explanations
//! - Repair suggestions

pub mod explanation;
pub mod minimization;
pub mod parsing;
pub mod repair;
pub mod types;

pub use explanation::*;
pub use minimization::*;
pub use parsing::*;
pub use repair::*;
pub use types::*;
