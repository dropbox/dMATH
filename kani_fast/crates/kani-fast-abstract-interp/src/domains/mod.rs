//! Abstract domains for abstract interpretation.
//!
//! Each domain represents a particular abstraction of program values:
//! - Interval: numeric bounds [lo, hi]
//! - Nullability: Option/Result state (Some/None/Maybe)

pub mod interval;
pub mod nullability;

pub use interval::Interval;
pub use nullability::{Nullability, ResultState};
