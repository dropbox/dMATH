//! Incremental BMC with clause learning and caching for Kani Fast
//!
//! This crate provides incremental bounded model checking capabilities that
//! dramatically speed up re-verification by:
//!
//! 1. **Clause persistence** - Store learned clauses in SQLite database
//! 2. **Diff analysis** - Detect what changed between verification runs
//! 3. **Clause invalidation** - Determine which clauses remain valid
//! 4. **Watch mode** - Continuous verification with automatic cache reuse
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     Incremental BMC Pipeline                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  Source Code ──► Content Hash ──► Check Cache ──► Decision         │
//! │       │              │                │              │              │
//! │       │              │                │              ├─► Cache Hit  │
//! │       │              │                │              │    └─► Load  │
//! │       │              │                │              │        Clauses│
//! │       │              │                │              │              │
//! │       │              │                │              └─► Cache Miss │
//! │       │              │                │                   └─► Full │
//! │       │              │                │                       BMC   │
//! │       │              │                │                             │
//! │       ▼              ▼                ▼                             │
//! │  ┌─────────┐   ┌───────────┐   ┌───────────────┐                   │
//! │  │  Diff   │   │  Content  │   │    Clause     │                   │
//! │  │Analysis │   │  Address  │   │   Database    │                   │
//! │  └─────────┘   │  Storage  │   │   (SQLite)    │                   │
//! │                └───────────┘   └───────────────┘                   │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_incremental::{IncrementalBmc, ClauseDatabase};
//!
//! // Open or create the clause database
//! let db = ClauseDatabase::open("./kani_cache.db")?;
//!
//! // Create incremental BMC engine
//! let bmc = IncrementalBmc::new(db);
//!
//! // Verify with caching
//! let result = bmc.verify_file("src/lib.rs").await?;
//!
//! // On subsequent runs, clauses are reused automatically
//! ```

pub mod chc_cache;
pub mod clause_db;
pub mod config;
pub mod content_hash;
pub mod diff;
pub mod engine;
pub mod result;
pub mod watch;

pub use chc_cache::{
    CacheStats as ChcCacheStats, CachedChcResult, CachedOutcome, ChcCacheError,
    ChcVerificationCache, FunctionDependency,
};
pub use clause_db::{ClauseDatabase, ClauseEntry, ClauseId, ClauseValidity};
pub use config::{IncrementalConfig, IncrementalConfigBuilder};
pub use content_hash::{ContentHash, HashKind};
pub use diff::{ChangeKind, DiffAnalyzer, DiffResult, FileChange};
pub use engine::{DimacsResult, IncrementalBmc};
pub use result::{CacheStats, IncrementalResult};
pub use watch::{WatchError, WatchEvent, WatchMode, WatchModeBuilder, WatchResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports() {
        // Verify all public types are accessible
        let _ = IncrementalConfig::default();
    }
}
