//! Trace pattern compression
//!
//! This module provides tools for detecting and compressing repeated patterns:
//! - `CompressedTrace`: A trace with repeated patterns compressed
//! - `TraceSegment`: Individual or repeated segments in a compressed trace

mod compress;
mod export;
mod trace_export;
mod types;

pub use types::{CompressedTrace, TraceSegment};
