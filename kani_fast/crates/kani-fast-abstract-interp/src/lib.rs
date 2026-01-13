//! Abstract Interpretation Engine for Kani Fast
//!
//! This crate provides abstract interpretation capabilities for computing
//! over-approximations of program behavior. Unlike SMT-based verification
//! which asks "is there a counterexample?", abstract interpretation asks
//! "what are all possible values?".
//!
//! # Domains
//!
//! - [`Interval`]: Numeric bounds [lo, hi] for proving array bounds, division safety
//! - [`Nullability`]: Option state (Some/None/Maybe) for proving .unwrap() safety
//! - [`ResultState`]: Result state (Ok/Err/Maybe) for proving .unwrap() safety
//!
//! # Example
//!
//! ```
//! use kani_fast_abstract_interp::domains::{Interval, Nullability};
//! use kani_fast_abstract_interp::lattice::Lattice;
//!
//! // Interval analysis: x in [0, 10]
//! let x = Interval::new(0, 10);
//! assert!(x.all_less_than(100)); // Proves x < 100
//! assert!(x.is_non_negative()); // Proves x >= 0
//!
//! // Nullability analysis
//! let opt = Nullability::Some;
//! assert!(opt.is_safe_to_unwrap()); // Proves .unwrap() won't panic
//!
//! // Merging at control flow join point
//! let branch1 = Nullability::Some;
//! let branch2 = Nullability::None;
//! let merged = branch1.join(&branch2);
//! assert!(!merged.is_safe_to_unwrap()); // merged is Maybe
//! ```
//!
//! # Integration with Verification
//!
//! Abstract interpretation can be used to:
//! 1. Discharge trivial checks without SMT (x in `[0,10]` → x < 100 trivially true)
//! 2. Tighten bounds for BMC (don't explore x > 10 if interval is `[0,10]`)
//! 3. Provide invariant candidates for k-induction
//! 4. Detect definite bugs (unwrap on None)

pub mod domains;
pub mod fixpoint;
pub mod lattice;

pub use domains::{Interval, Nullability, ResultState};
pub use fixpoint::{
    forward_analysis, ControlFlowGraph, FixpointConfig, FixpointResult, TransferFunction,
};
pub use lattice::{FlatLattice, Lattice, ProductLattice};
