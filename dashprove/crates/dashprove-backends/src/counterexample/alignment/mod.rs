//! Multi-trace alignment
//!
//! This module provides tools for aligning multiple traces:
//! - `MultiTraceAlignment`: Aligned view of multiple traces
//! - `MultiTraceAlignmentRow`: A single row in the alignment
//! - `DivergencePoint`: Point where traces diverge

mod align;
mod export;
mod table;
mod types;

pub use align::align_multiple_traces;
pub use types::{DivergencePoint, MultiTraceAlignment, MultiTraceAlignmentRow};
