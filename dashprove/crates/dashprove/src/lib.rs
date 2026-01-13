// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // API methods don't need must_use
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::use_self)] // Self vs TypeName - style preference
#![allow(clippy::unused_self)] // Some methods keep self for API compatibility
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::missing_errors_doc)] // Error docs are implementation details
#![allow(clippy::uninlined_format_args)] // Named args are clearer
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::similar_names)] // e.g., spec/specs, result/results
#![allow(clippy::redundant_closure_for_method_calls)] // Minor style issue
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
#![allow(clippy::needless_pass_by_value)] // Ownership semantics may be intentional
#![allow(clippy::too_many_lines)] // Complex methods may be inherently long
#![allow(clippy::cast_precision_loss)] // usize to f64 is intentional
#![allow(clippy::wildcard_imports)] // Re-exports use wildcard pattern
#![allow(clippy::trivially_copy_pass_by_ref)] // &BackendId is API consistency
#![allow(clippy::needless_raw_string_hashes)] // Raw strings for templates
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::return_self_not_must_use)] // Builder methods don't need must_use

//! DashProve: Unified AI-Native Verification Platform
//!
//! This is the main library crate that provides a high-level API for integration
//! with DashFlow, Dasher, and other consumers.
//!
//! # Example
//!
//! ```rust,no_run
//! use dashprove::{DashProve, DashProveConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a DashProve client
//! let mut client = DashProve::new(DashProveConfig::default());
//!
//! // Verify a specification string
//! let result = client.verify("theorem test { forall x: Bool . x or not x }").await?;
//!
//! println!("Proven: {}", result.is_proven());
//! # Ok(())
//! # }
//! ```
//!
//! # Runtime monitors
//!
//! ```rust
//! use dashprove::{MonitorConfig, RuntimeMonitor};
//! use dashprove_usl::{parse, typecheck};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let spec = parse("theorem safe { true }")?;
//! let typed = typecheck(spec)?;
//! let monitor = RuntimeMonitor::from_spec(&typed, &MonitorConfig::default());
//! assert!(monitor.code.contains("SafeMonitor"));
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod monitor;

pub mod backend_ids;
#[cfg(feature = "remote")]
pub mod remote;

// Re-export sub-crates for direct access when needed
pub use dashprove_ai as ai;
pub use dashprove_backends as backends;
pub use dashprove_dispatcher as dispatcher;
pub use dashprove_learning as learning;
pub use dashprove_usl as usl;

// Re-export key types at crate root for convenience
pub use backend_ids::{backend_metric_label, default_backends, BackendIdParam, SUPPORTED_BACKENDS};
pub use client::{DashProve, DashProveConfig, DashProveError, VerificationResult};
pub use dashprove_backends::{BackendId, VerificationStatus};
pub use dashprove_usl::ast::Spec;
pub use dashprove_usl::typecheck::TypedSpec;
pub use monitor::{MonitorConfig, RuntimeMonitor};

// Remote client exports (when feature is enabled)
#[cfg(feature = "remote")]
pub use remote::{
    // Utility functions
    create_feedback,
    extract_property_features,
    // DashFlow ML API client
    CodeContext,
    // DashProve server client
    DashFlowClient,
    DashFlowMlClient,
    DashFlowMlConfig,
    // Feedback queue for offline collection
    FeedbackQueue,
    FeedbackQueueStats,
    FeedbackResponse,
    FeedbackVerificationStatus,
    ModelStatusResponse,
    PredictResponse,
    PropertyFeatures,
    RemoteClient,
    RemoteConfig,
    RemoteError,
    VerificationFeedback,
};
