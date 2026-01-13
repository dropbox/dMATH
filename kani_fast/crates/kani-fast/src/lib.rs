//! Kani Fast: Next-Generation Rust Verification
//!
//! Kani Fast dramatically improves on Kani with:
//! - **10-100x faster** incremental verification via clause learning and caching
//! - **Unbounded verification** via k-induction and CHC solving
//! - **Portfolio solving** with parallel SAT/SMT solvers
//! - **AI-assisted invariant synthesis** for automatic loop invariant discovery
//! - **Beautiful counterexamples** with natural language explanations
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use kani_fast::{KaniWrapper, KaniConfig};
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let wrapper = KaniWrapper::with_defaults()?;
//!     let result = wrapper.verify(Path::new("my-project")).await?;
//!
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```
//!
//! # Verification Modes
//!
//! Kani Fast supports multiple verification strategies:
//!
//! - **Bounded**: Standard Kani bounded model checking
//! - **K-Induction**: Unbounded verification for loops
//! - **CHC**: Constrained Horn Clause solving via Spacer (Z3/Z4)
//! - **Portfolio**: Run multiple SAT solvers in parallel
//! - **Auto**: Automatically select the best strategy
//!
//! # tRust Integration
//!
//! For integration with the tRust compiler, use the engine module:
//!
//! ```rust,ignore
//! use kani_fast::engine::{VerificationEngine, MirInput, VerificationConfig};
//!
//! let config = VerificationConfig::default();
//! let engine = VerificationEngine::new(&config);
//!
//! let input = MirInput::from_mir_program(program);
//! let result = engine.verify(input).await;
//! ```

pub mod engine;

pub use kani_fast_chc as chc;
pub use kani_fast_core::*;
pub use kani_fast_counterexample as counterexample;
pub use kani_fast_incremental as incremental;
pub use kani_fast_kinduction as kinduction;
pub use kani_fast_lean5 as lean5;
pub use kani_fast_portfolio as portfolio;
pub use kani_fast_proof as proof;

/// Current version of Kani Fast
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if Kani is available and properly configured
pub fn check_installation() -> Result<KaniInfo, DetectionError> {
    detect_kani()
}

/// Verify a project using default settings
///
/// This is a convenience function for simple verification cases.
///
/// # Arguments
///
/// * `project_path` - Path to the Rust project to verify
///
/// # Example
///
/// ```rust,ignore
/// let result = kani_fast::verify("my-project").await?;
/// println!("{}", result);
/// ```
pub async fn verify(
    project_path: impl AsRef<std::path::Path>,
) -> Result<VerificationResult, KaniError> {
    let wrapper = KaniWrapper::with_defaults()?;
    wrapper.verify(project_path.as_ref()).await
}

/// Verify a specific harness in a project
///
/// # Arguments
///
/// * `project_path` - Path to the Rust project
/// * `harness` - Name of the specific proof harness to verify
pub async fn verify_harness(
    project_path: impl AsRef<std::path::Path>,
    harness: &str,
) -> Result<VerificationResult, KaniError> {
    let wrapper = KaniWrapper::with_defaults()?;
    wrapper
        .verify_with_harness(project_path.as_ref(), Some(harness))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_check_installation() {
        // Just verify it doesn't panic
        let _ = check_installation();
    }
}
