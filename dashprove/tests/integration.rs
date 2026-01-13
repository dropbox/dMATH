//! Integration tests for DashProve
//!
//! Run with: cargo test --test integration
//!
//! These tests verify end-to-end functionality across all crates.
//!
//! For real backend tests (requires actual tools installed):
//! cargo test --test integration real_backends -- --ignored

#[path = "integration/pipeline.rs"]
mod pipeline;

#[path = "integration/cli.rs"]
mod cli;

#[path = "integration/remote_client.rs"]
mod remote_client;

#[path = "integration/real_backends.rs"]
mod real_backends;
