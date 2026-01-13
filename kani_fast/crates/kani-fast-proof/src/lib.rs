//! Universal Proof Format for Kani Fast
//!
//! This crate provides a shared proof representation that all verification backends
//! (Z4, Lean5, TLA2, Kani Fast) can emit and consume. The format is designed for:
//!
//! - **Fast checking**: O(proof size) verification time
//! - **Content-addressable**: BLAKE3 hash of VC + proof + backend version
//! - **Composable**: Combine proofs without re-running provers via dependency DAG
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Universal Proof                          │
//! │                                                             │
//! │  id: ContentHash (BLAKE3)                                   │
//! │  vc_hash: ContentHash                                       │
//! │  backend: BackendId                                         │
//! │  format: ProofFormat                                        │
//! │  steps: Vec<ProofStep>                                      │
//! │  dependencies: Vec<ContentHash>                             │
//! │  metadata: ProofMetadata                                    │
//! └─────────────────────────────────────────────────────────────┘
//!          │
//!          ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    ProofStep                                │
//! │                                                             │
//! │  SAT:   DRAT clauses (additions/deletions)                  │
//! │  SMT:   Alethe-style inference steps                        │
//! │  CHC:   Invariant predicates with formulas                  │
//! │  Lean:  Proof terms (lambda calculus expressions)           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_proof::{UniversalProof, ProofFormat, BackendId, ProofStep};
//!
//! // Create a CHC proof
//! let proof = UniversalProof::builder()
//!     .backend(BackendId::KaniFast)
//!     .format(ProofFormat::Chc)
//!     .vc("(assert (=> (= x 0) (>= x 0)))")
//!     .step(ProofStep::chc_invariant("inv", vec!["x"], "(>= x 0)"))
//!     .build();
//!
//! // Content-addressable ID
//! println!("Proof ID: {}", proof.id);
//!
//! // Save to storage
//! storage.store(&proof)?;
//!
//! // Later: retrieve by ID (fast cache lookup)
//! let cached = storage.get(&proof.id)?;
//! ```

pub mod checker;
pub mod format;
pub mod hash;
pub mod step;
pub mod storage;

pub use checker::{CheckerConfig, CheckerError, ProofChecker, StepResult, VerificationResult};
pub use format::{BackendId, ProofFormat, ProofMetadata, UniversalProof, UniversalProofBuilder};
pub use hash::ContentHash;
pub use step::{ChcStep, DratStep, LeanStep, ProofStep, SmtStep};
pub use storage::{ProofStorage, ProofStorageError};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
