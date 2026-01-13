//! Kani Fast Driver - Standalone rustc wrapper for CHC verification
//!
//! Usage: `kani-fast-driver [rustc-args] <source-file>`
//!
//! This binary wraps rustc and intercepts MIR to perform CHC-based verification.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_session;

use rustc_session::EarlyDiagCtxt;

fn main() {
    // Initialize diagnostics context
    let early_dcx = EarlyDiagCtxt::new(rustc_session::config::ErrorOutputType::default());

    // Initialize rustc error handling
    rustc_driver::init_rustc_env_logger(&early_dcx);
    rustc_driver::install_ice_hook("kani-fast-driver", |_| ());

    // Run kani-fast compiler
    let exit_code = kani_fast_compiler::run_as_cli();
    std::process::exit(exit_code);
}
