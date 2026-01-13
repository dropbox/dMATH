//! Kani Fast Compiler - Rustc Driver for CHC-based Verification
//!
//! This crate provides a rustc driver that captures MIR (Mid-level Intermediate
//! Representation) from the Rust compiler and converts it to CHC (Constrained
//! Horn Clauses) for verification.
//!
//! # Architecture
//!
//! ```text
//! Rust Source → rustc → MIR → kani-fast-compiler → CHC → Z4 PDR
//!                              ↓
//!                         MirProgram (kani-fast-chc)
//! ```
//!
//! # Stability
//!
//! This crate uses unstable rustc internals (`rustc_private`). It requires:
//! - Nightly Rust toolchain (nightly-2025-11-20)
//! - rustc-dev component installed (`rustup component add rustc-dev`)
//! - rust-src component installed (`rustup component add rust-src`)
//!
//! The APIs may change between Rust versions. We pin to a specific nightly
//! version in rust-toolchain.toml for reproducibility.

// Enable unstable features required for rustc internals
#![feature(rustc_private)]
#![feature(box_patterns)]
#![feature(extern_types)]
#![feature(more_qualified_paths)]
#![feature(iter_intersperse)]
#![feature(f128)]
#![feature(f16)]
#![feature(non_exhaustive_omitted_patterns_lint)]
#![feature(cfg_version)]
#![feature(mpmc_channel)]
#![cfg_attr(not(version("1.86")), feature(float_next_up_down))]
#![feature(try_blocks)]

// Compiler crate imports - these are provided by rustc itself
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_hir_pretty;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_mir_dataflow;
extern crate rustc_public;
extern crate rustc_public_bridge;
extern crate rustc_session;
extern crate rustc_span;

// Compatibility shims for cbmc types we don't want to depend on
pub mod compat;

// Kani modules (copied from kani-compiler)
pub mod args;
pub mod intrinsics;
pub mod kani_middle;
pub mod kani_queries;

// Kani Fast modules
pub mod callbacks;
pub mod codegen_chc;
pub mod mir_collector;
pub mod mir_to_chc;

pub use codegen_chc::{ChcCodegenCtx, CodegenError};

use rustc_driver::run_compiler;

pub use callbacks::{HarnessResult, KaniFastCallbacks, VerificationResults};

/// Entry point for the rustc driver
///
/// This function is called when rustc loads our dynamic library.
/// It sets up our callbacks to intercept MIR after it's been constructed.
///
/// Returns the verification results after compilation completes.
pub fn run_kani_fast_compiler(args: Vec<String>) -> VerificationResults {
    let mut callbacks = KaniFastCallbacks::new();

    // Run rustc with our callbacks
    run_compiler(&args, &mut callbacks);

    callbacks.results
}

/// Run the compiler with custom callbacks
pub fn run_kani_fast_compiler_with_callbacks(
    args: Vec<String>,
    callbacks: &mut KaniFastCallbacks,
) -> VerificationResults {
    run_compiler(&args, callbacks);
    callbacks.results.clone()
}

/// Run the compiler as a command-line tool
///
/// This is the main entry point that reads args from command line and runs verification.
/// Returns exit code: 0 for success, 1 for verification failure.
pub fn run_as_cli() -> i32 {
    // Get command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Run the compiler
    let results = run_kani_fast_compiler(args);

    // Return appropriate exit code
    if results.all_passed() { 0 } else { 1 }
}
