//! Trace abstraction functionality
//!
//! This module provides tools for abstracting counterexample traces:
//! - `AbstractedState`: A state with abstracted variable values
//! - `AbstractedValue`: Abstracted value representation (high/low/unchanged/etc)
//! - `AbstractedTrace`: A trace with abstracted segments

mod export;
mod types;

pub use types::*;
