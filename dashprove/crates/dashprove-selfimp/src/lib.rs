//! Verified Self-Improvement Infrastructure for DashProve
//!
//! This crate provides the core infrastructure for safe, verified self-improvement
//! of AI systems like Dasher. The key principle is that ALL self-modifications
//! MUST pass through a verification gate that CANNOT be bypassed.
//!
//! # Key Components
//!
//! - [`VerificationGate`]: The immutable, hardcoded verification checkpoint
//! - [`RollbackManager`]: Automatic rollback on verification failure
//! - [`ImprovementProposer`]: Proposes improvements based on analysis
//! - [`ImprovementVerifier`]: Formally verifies proposed improvements
//! - [`VersionHistory`]: Maintains history of all versions with proof certificates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     SELF-IMPROVEMENT LOOP                           │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  1. Propose    ┌──────────────────┐                                │
//! │  ───────────→  │ ImprovementProposer │                             │
//! │                └────────┬─────────┘                                │
//! │                         │                                          │
//! │  2. Verify     ┌────────▼─────────┐                                │
//! │  ───────────→  │ VerificationGate │  ← CANNOT BE BYPASSED          │
//! │                │                  │                                │
//! │                │ • Soundness      │                                │
//! │                │ • Capability     │                                │
//! │                │ • Safety         │                                │
//! │                └────────┬─────────┘                                │
//! │                         │                                          │
//! │            ┌────────────┴────────────┐                             │
//! │            │                         │                             │
//! │         PASS                       FAIL                            │
//! │            │                         │                             │
//! │  3a. Apply │           3b. Rollback  │                             │
//! │            ▼                         ▼                             │
//! │  ┌─────────────────┐      ┌─────────────────┐                     │
//! │  │ VersionHistory  │      │ RollbackManager │                     │
//! │  │ + ProofCert     │      │ → Previous Ver  │                     │
//! │  └─────────────────┘      └─────────────────┘                     │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Safety Guarantees
//!
//! 1. **Immutable Verification Gate**: The verification gate is hardcoded and
//!    cannot be disabled, bypassed, or weakened at runtime.
//!
//! 2. **Soundness Preservation**: Every accepted improvement preserves the
//!    soundness property - the system never claims false proofs.
//!
//! 3. **Capability Preservation**: Capabilities can only improve or stay the
//!    same; they can never regress.
//!
//! 4. **Rollback Guarantee**: If verification fails, the system automatically
//!    rolls back to the last verified state.
//!
//! 5. **Proof Certificates**: Every version in history has a cryptographically
//!    signed proof certificate.
//!
//! # Example
//!
//! ```ignore
//! use dashprove_selfimp::{
//!     VerificationGate, Improvement, ImprovementResult,
//!     VersionHistory, Version,
//! };
//!
//! // Create a version history
//! let mut history = VersionHistory::new();
//!
//! // Register the current version
//! let current = Version::new("v1.0.0", /* ... */);
//! history.register(current)?;
//!
//! // Propose an improvement
//! let improvement = Improvement::new(/* ... */);
//!
//! // Apply through the verification gate (CANNOT BE BYPASSED)
//! let result = VerificationGate::apply_improvement(
//!     history.current(),
//!     &improvement,
//! )?;
//!
//! match result {
//!     ImprovementResult::Accepted(new_version) => {
//!         // New version has proof certificate
//!         history.register(new_version)?;
//!     }
//!     ImprovementResult::Rejected(reason) => {
//!         // Improvement was rejected, system unchanged
//!         println!("Rejected: {:?}", reason);
//!     }
//! }
//! ```

// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::use_self)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::similar_names)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::single_match_else)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::or_fun_call)]

pub mod certificate;
pub mod error;
pub mod gate;
pub mod history;
pub mod improvement;
pub mod rollback;
pub mod verifier;
pub mod version;

pub use certificate::{CertificateChain, CertificateVerification, ProofCertificate};
pub use error::{SelfImpError, SelfImpResult};
pub use gate::{
    AsyncVerificationGate, GateCheck, GateConfig, GateResult, IncrementalGateResult,
    VerificationGate,
};
pub use history::{HistoryEntry, HistoryQuery, VersionHistory};
pub use improvement::{Improvement, ImprovementKind, ImprovementResult, ImprovementTarget};
pub use rollback::{RollbackAction, RollbackConfig, RollbackManager, RollbackTrigger};
pub use verifier::{
    ActivitySample, ActivityStatistics, AdaptiveIntervalConfig, AsyncImprovementVerifier,
    AutosaveCoalesceEvent, AutosaveErrorEvent, AutosaveMetrics, AutosaveReasonCounts,
    AutosaveSaveEvent, AutosaveSaveReason, AutosaveSkipEvent, BackoffConfig,
    CacheAutosaveCallbacks, CacheAutosaveConfig, CacheAutosaveHandle, CacheAutosaveStatus,
    CacheAutosaveSummary, CacheKey, CachePartition, CacheSnapshot, CacheStats,
    CachedPropertyResult, CoalesceConfig, CompactionConfig, CompactionEvent, CompactionPolicy,
    CompactionResult, CompactionTimeSeries, CompactionTimeSeriesEntry, CompactionTimeSeriesSummary,
    CompactionTriggerCounts, CompactionTriggerType, HistoricalActivityTracker, ImprovementVerifier,
    IncrementalVerificationResult, LearningThresholdConfig, MetricSample, MetricsAggregator,
    MetricsAggregatorConfig, MetricsReport, MetricsWindow, PartitionConfig,
    PartitionOperationResult, PartitionStats, PersistedAutosaveMetrics, SnapshotCompressionLevel,
    ThresholdUpdateEvent, VerificationCache, VerificationConfig, VerificationResult,
    VerificationRetryPolicy, VerifiedProperty, WarmingConfig, WarmingEvent, WarmingResult,
    WarmingStrategy, WindowedMetricStats, WindowedMetricsReport,
};
pub use version::{Capability, CapabilitySet, Version, VersionId, VersionMetadata};

#[cfg(test)]
mod tests;
