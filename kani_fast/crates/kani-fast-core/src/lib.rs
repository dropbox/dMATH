//! Core verification pipeline for Kani Fast
//!
//! This crate provides the core verification orchestration including:
//! - KaniWrapper: Wraps `cargo kani` for baseline bounded model checking
//! - VerificationResult: Structured verification outcomes
//! - Pipeline: Configurable verification pipeline
//! - PortfolioVerifier: Run multiple solvers in parallel

mod config;
mod detection;
mod portfolio;
mod result;
mod wrapper;

// Re-use find_executable from kani-fast-portfolio to avoid duplication
pub(crate) use kani_fast_portfolio::find_executable;

pub use config::*;
pub use detection::*;
pub use portfolio::*;
pub use result::*;
pub use wrapper::*;

// Note: find_executable tests are in kani-fast-portfolio where the function is defined
