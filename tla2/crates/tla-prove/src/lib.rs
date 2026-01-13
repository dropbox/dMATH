//! tla-prove - Proof manager for TLA+
//!
//! This crate provides theorem proving support for TLA+ specifications.
//! It extracts proof obligations from theorems and uses backends (SMT, Zenon, etc.)
//! to discharge them.
//!
//! # Architecture
//!
//! The proof manager follows TLAPM's approach:
//!
//! 1. **Obligation Extraction**: Structured proofs are analyzed to extract
//!    individual proof obligations that must be discharged.
//!
//! 2. **Context Management**: Facts, definitions, and assumptions are tracked
//!    in a proof context that's available to the backends.
//!
//! 3. **Backend Orchestration**: Multiple proof backends (SMT, Zenon, Isabelle)
//!    can be used to attempt proving obligations.
//!
//! 4. **Caching**: Proof results are cached using content-addressed fingerprints
//!    to avoid re-proving unchanged obligations.
//!
//! # Example
//!
//! ```rust,ignore
//! use tla_prove::{Prover, ProofContext};
//! use tla_core::ast::Module;
//!
//! let module: Module = /* parse module */;
//! let mut prover = Prover::new();
//! let result = prover.check_module(&module)?;
//!
//! if result.is_proved() {
//!     println!("All theorems proved!");
//! } else {
//!     println!("Failed: {} of {} obligations",
//!         result.failed_count(), result.total_obligations());
//! }
//! ```
//!
//! # Proof Syntax
//!
//! TLA+ supports several proof constructs:
//!
//! - `OBVIOUS` - Goal should be obvious to the prover
//! - `OMITTED` - Skip this proof (placeholder)
//! - `BY facts` - Prove using specified facts
//! - Structured proofs with steps like `<1>a`, `SUFFICES`, `HAVE`, `TAKE`, etc.

mod backend;
mod context;
mod error;
mod obligation;
mod prover;
mod zenon_backend;

pub use backend::{ProofBackend, ProofOutcome, SmtBackend};
pub use context::{Definition, Fact, ProofContext};
pub use error::{ProofError, ProofResult};
pub use obligation::{Obligation, ObligationExtractor, ObligationId};
pub use prover::{ModuleResult, ObligationResult, ProofCache, Prover, TheoremResult};
pub use zenon_backend::ZenonBackend;
