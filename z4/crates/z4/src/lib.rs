//! Z4 - A high-performance SMT solver in Rust
//!
//! This is the main library crate that re-exports all components.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub use z4_core as core;
pub use z4_dpll as dpll;
pub use z4_frontend as frontend;
pub use z4_proof as proof;
pub use z4_sat as sat;

/// Theory solvers
pub mod theories {
    pub use z4_arrays as arrays;
    pub use z4_bv as bv;
    pub use z4_dt as dt;
    pub use z4_euf as euf;
    pub use z4_fp as fp;
    pub use z4_lia as lia;
    pub use z4_lra as lra;
    pub use z4_strings as strings;
}
