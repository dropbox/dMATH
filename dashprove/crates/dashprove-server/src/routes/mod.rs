//! REST API routes for DashProve server
//!
//! This module is split into:
//! - `types` - Request/response types for the API
//! - `cache` - Cache route handlers (stats, save, load, clear)
//! - `corpus` - Corpus route handlers (search, stats, history, compare, counterexamples)
//! - `expert` - Expert API handlers (backend, error, tactic, compile)
//! - `status` - Health, version, metrics, and backend listing handlers
//! - `verification` - Verification handlers (full and incremental)
//! - `assistant` - Tactics suggestions, sketch elaboration, and counterexample explanations
//! - `mod.rs` - Route state management
//!
//! Endpoints per docs/DESIGN.md:
//! - GET /health - Health check with shutdown state
//! - GET /version - API version and metadata
//! - GET /metrics - Prometheus-compatible metrics endpoint
//! - POST /verify - Verify a specification
//! - POST /verify/incremental - Incremental verification after changes
//! - POST /sketch/elaborate - Elaborate a proof sketch
//! - GET /corpus/search - Search proof corpus
//! - GET /corpus/stats - Get corpus statistics
//! - GET /corpus/history - Get corpus history over time
//! - GET /corpus/compare - Compare two time periods in the corpus
//! - GET /corpus/suggest - Suggest comparison periods based on available data
//! - GET /corpus/counterexamples - List all counterexamples with pagination
//! - GET /corpus/counterexamples/:id - Get a single counterexample by ID
//! - POST /corpus/counterexamples/search - Search similar counterexamples (feature-based)
//! - GET /corpus/counterexamples/text-search - Search counterexamples by text keywords
//! - POST /corpus/counterexamples - Add counterexample to corpus
//! - POST /corpus/counterexamples/classify - Classify counterexample against cluster patterns
//! - POST /corpus/counterexamples/clusters - Record cluster patterns from clustering results
//! - POST /tactics/suggest - Get tactic suggestions
//! - POST /explain - Explain a counterexample in human-readable form
//! - GET /backends - List available backends with health status
//! - GET /cache/stats - Get proof cache statistics
//! - POST /cache/save - Save cache to disk
//! - POST /cache/load - Load cache from disk
//! - DELETE /cache/clear - Clear all cache entries
//! - POST /expert/backend - Get backend recommendation
//! - POST /expert/error - Get error explanation
//! - POST /expert/tactic - Get tactic suggestions
//! - POST /expert/compile - Get compilation guidance
//! - POST /proof-search - AI-driven iterative proof search with tactic learning

mod assistant;
mod cache;
mod corpus;
mod expert;
mod proof_search;
mod status;
mod types;
mod verification;

// Re-export types for external use (incrementally being adopted)
#[allow(unused_imports)]
pub use types::*;

// Re-export assistant handlers
pub use assistant::{explain, sketch_elaborate, tactics_suggest};

// Re-export cache handlers
pub use cache::{cache_clear, cache_load, cache_save, cache_stats};

// Re-export corpus handlers
pub use corpus::{
    corpus_compare, corpus_history, corpus_search, corpus_stats, corpus_suggest,
    counterexample_add, counterexample_classify, counterexample_clusters, counterexample_get,
    counterexample_list, counterexample_search, counterexample_text_search,
};

// Re-export expert handlers
pub use expert::{expert_backend, expert_compile, expert_error, expert_tactic};

// Re-export proof search handlers
pub use proof_search::proof_search;

// Re-export status handlers
pub use status::{health, list_backends, prometheus_metrics, version};

// Re-export verification handlers
pub use verification::{verify, verify_incremental};

// Re-export helpers and types needed for tests
#[cfg(test)]
pub(crate) use dashprove_knowledge::PropertyType as ExpertPropertyType;
#[cfg(test)]
pub(crate) use expert::{get_knowledge_dir, parse_expert_property_types};

use dashprove_ai::{ProofAssistant, StrategyModel};
use dashprove_learning::ProofLearningSystem;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::cache::ProofCache;
use crate::metrics::Metrics;
use crate::ws::SessionManager;

/// Shutdown state for graceful shutdown coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShutdownState {
    /// Server is running normally
    #[default]
    Running,
    /// Server is draining - accepting no new requests, waiting for in-flight to complete
    Draining,
    /// Server is shutting down
    ShuttingDown,
}

/// Shared application state
pub struct AppState {
    /// Learning system (optional)
    pub learning: Option<RwLock<ProofLearningSystem>>,
    /// AI proof assistant
    pub assistant: RwLock<ProofAssistant>,
    /// WebSocket session manager
    pub session_manager: SessionManager,
    /// Proof result cache for incremental verification
    pub proof_cache: RwLock<ProofCache>,
    /// Path for cache persistence (optional)
    pub cache_path: Option<std::path::PathBuf>,
    /// Counter for in-flight requests (for graceful shutdown)
    pub in_flight_requests: std::sync::atomic::AtomicUsize,
    /// Current shutdown state (for health endpoint reporting)
    pub shutdown_state: std::sync::atomic::AtomicU8,
    /// Prometheus-compatible metrics collector
    pub metrics: Metrics,
    /// ML strategy predictor (optional)
    pub ml_predictor: Option<Arc<StrategyModel>>,
}

impl AppState {
    /// Create new app state without learning data
    pub fn new() -> Self {
        Self {
            learning: None,
            assistant: RwLock::new(ProofAssistant::new()),
            session_manager: SessionManager::new(),
            proof_cache: RwLock::new(ProofCache::new()),
            cache_path: None,
            in_flight_requests: std::sync::atomic::AtomicUsize::new(0),
            shutdown_state: std::sync::atomic::AtomicU8::new(ShutdownState::Running as u8),
            metrics: Metrics::new(),
            ml_predictor: None,
        }
    }

    /// Create app state with learning system
    pub fn with_learning(learning: ProofLearningSystem) -> Self {
        // ProofAssistant needs its own learning system instance
        // For now, we don't share it - the assistant starts fresh
        Self {
            learning: Some(RwLock::new(learning)),
            assistant: RwLock::new(ProofAssistant::new()),
            session_manager: SessionManager::new(),
            proof_cache: RwLock::new(ProofCache::new()),
            cache_path: None,
            in_flight_requests: std::sync::atomic::AtomicUsize::new(0),
            shutdown_state: std::sync::atomic::AtomicU8::new(ShutdownState::Running as u8),
            metrics: Metrics::new(),
            ml_predictor: None,
        }
    }

    /// Set the ML strategy predictor
    pub fn with_ml_predictor(mut self, predictor: Arc<StrategyModel>) -> Self {
        self.ml_predictor = Some(predictor);
        self
    }

    /// Get current number of in-flight requests
    pub fn active_requests(&self) -> usize {
        self.in_flight_requests
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Increment in-flight request counter (called when request starts)
    pub fn request_started(&self) {
        self.in_flight_requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Decrement in-flight request counter (called when request completes)
    pub fn request_completed(&self) {
        self.in_flight_requests
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get the current shutdown state
    pub fn get_shutdown_state(&self) -> ShutdownState {
        let val = self
            .shutdown_state
            .load(std::sync::atomic::Ordering::Relaxed);
        match val {
            0 => ShutdownState::Running,
            1 => ShutdownState::Draining,
            _ => ShutdownState::ShuttingDown,
        }
    }

    /// Set the shutdown state
    pub fn set_shutdown_state(&self, state: ShutdownState) {
        self.shutdown_state
            .store(state as u8, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if server is currently draining or shutting down
    pub fn is_draining(&self) -> bool {
        self.get_shutdown_state() != ShutdownState::Running
    }

    /// Create app state with cache persistence path
    pub fn with_cache_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        let path = path.into();
        // Try to load existing cache from disk
        if path.exists() {
            if let Ok(cache) = ProofCache::load_from_file(&path) {
                self.proof_cache = RwLock::new(cache);
            }
        }
        self.cache_path = Some(path);
        self
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
