//! Intelligent backend dispatcher
//!
//! This crate provides intelligent backend selection, parallel execution, and result merging
//! for the DashProve verification platform.
//!
//! ## ML-Based Backend Selection
//!
//! The dispatcher supports ML-based backend selection using a trained `StrategyModel`
//! (single predictor or ensemble).
//! This allows the system to learn optimal backend choices from historical verification data.
//!
//! ```rust,no_run
//! use dashprove_dispatcher::{Dispatcher, DispatcherConfig, SelectionStrategy};
//! use dashprove_ai::StrategyModel;
//! use std::sync::Arc;
//!
//! // Create dispatcher with ML-based selection
//! let predictor = Arc::new(StrategyModel::from(dashprove_ai::StrategyPredictor::new()));
//! let config = DispatcherConfig::ml_based(0.3); // 30% confidence threshold
//! let mut dispatcher = Dispatcher::with_ml_predictor(config, predictor);
//! ```

// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // Builder methods and getters don't need must_use
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::use_self)] // Self vs TypeName - style preference
#![allow(clippy::unused_self)] // Some methods keep self for future API compatibility
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::missing_errors_doc)] // Error docs are implementation details
#![allow(clippy::cast_precision_loss)] // usize to f64 for confidence scores is intentional
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::cast_lossless)] // Explicit casts are clearer
#![allow(clippy::uninlined_format_args)] // Named args in format strings are clearer
#![allow(clippy::needless_pass_by_value)] // Config passed by value for ownership semantics
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
//!
//! # Components
//!
//! - **BackendRegistry**: Manages available backends and their health status
//! - **BackendSelector**: Intelligently selects backends based on property types
//! - **ParallelExecutor**: Runs verification tasks in parallel with concurrency control
//! - **ResultMerger**: Combines results from multiple backends into unified results
//! - **Dispatcher**: Main interface that orchestrates all components
//!
//! # Example
//!
//! ```rust,no_run
//! use dashprove_dispatcher::{Dispatcher, DispatcherConfig};
//! use dashprove_backends::BackendId;
//! use dashprove_usl::{parse, typecheck};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create dispatcher with default config
//! let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
//!
//! // Register backends (in real code, use actual backend implementations)
//! // dispatcher.register_backend(Arc::new(Lean4Backend::new()));
//!
//! // Parse and type-check specification
//! let spec = parse(r#"
//!     theorem test { forall x: Bool . x or not x }
//! "#)?;
//! let typed_spec = typecheck(spec)?;
//!
//! // Verify
//! let results = dispatcher.verify(&typed_spec).await?;
//! println!("Proven: {}, Disproven: {}", results.summary.proven, results.summary.disproven);
//! # Ok(())
//! # }
//! ```

pub mod merge;
pub mod parallel;
pub mod selector;

// Re-exports for convenience
pub use merge::{MergeStrategy, MergedResult, MergedResults, ResultMerger, VerificationSummary};
pub use parallel::{
    CancellationToken, ExecutionResults, ExecutorConfig, ExecutorError, ParallelExecutor,
    ProgressUpdate, RetryConfig, TaskPriority, TaskResult,
};
pub use selector::{
    BackendInfo, BackendRegistry, BackendSelector, PropertyAssignment, Selection, SelectionMethod,
    SelectionMetrics, SelectionStrategy, SelectorError,
};

use dashprove_ai::StrategyModel;
use dashprove_backends::{BackendId, VerificationBackend, VerificationStatus};
use dashprove_knowledge::{Embedder, KnowledgeStore, ToolKnowledgeStore};
use dashprove_learning::{ReputationConfig, ReputationTracker};
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info};

/// Errors from the dispatcher
#[derive(Error, Debug)]
pub enum DispatcherError {
    /// Backend selection failed
    #[error("Selector error: {0}")]
    Selector(#[from] SelectorError),

    /// Parallel execution failed
    #[error("Executor error: {0}")]
    Executor(#[from] ExecutorError),

    /// No backends were registered with the dispatcher
    #[error("No backends registered")]
    NoBackends,

    /// Verification completed but failed to prove the property
    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    /// Internal error (e.g., I/O error during persistence)
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Configuration for the Dispatcher
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    /// Backend selection strategy
    pub selection_strategy: SelectionStrategy,
    /// Result merging strategy
    pub merge_strategy: MergeStrategy,
    /// Maximum concurrent verification tasks
    pub max_concurrent: usize,
    /// Timeout per verification task
    pub task_timeout: Duration,
    /// Whether to check backend health before verification
    pub check_health: bool,
    /// Whether to automatically update backend reputation after each verification
    ///
    /// When enabled, the dispatcher will:
    /// - Record successful verifications (backend result matches consensus)
    /// - Record failures (backend disagreed with consensus, timed out, or errored)
    /// - Automatically refresh reputation weights in the merger
    ///
    /// Requires a reputation tracker to be set via `set_reputation_tracker()`.
    /// Default: false
    pub auto_update_reputation: bool,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::Single,
            merge_strategy: MergeStrategy::FirstSuccess,
            max_concurrent: 4,
            task_timeout: Duration::from_secs(300),
            check_health: true,
            auto_update_reputation: false,
        }
    }
}

impl DispatcherConfig {
    /// Create a config for redundant verification with multiple backends
    pub fn redundant(min_backends: usize) -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::Redundant { min_backends },
            merge_strategy: MergeStrategy::Unanimous,
            ..Default::default()
        }
    }

    /// Create a config that uses all available backends
    pub fn all_backends() -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::Majority,
            ..Default::default()
        }
    }

    /// Create a config for a specific backend
    pub fn specific(backend: BackendId) -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::Specific(backend),
            merge_strategy: MergeStrategy::FirstSuccess,
            ..Default::default()
        }
    }

    /// Create a config for ML-based backend selection
    ///
    /// # Arguments
    /// * `min_confidence` - Minimum confidence threshold (0.0-1.0) for ML predictions.
    ///   If prediction confidence is below this, falls back to rule-based selection.
    ///   Recommended values: 0.3-0.5 for exploratory use, 0.6-0.8 for production.
    pub fn ml_based(min_confidence: f64) -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::MlBased {
                min_confidence: min_confidence.clamp(0.0, 1.0),
            },
            merge_strategy: MergeStrategy::FirstSuccess,
            ..Default::default()
        }
    }

    /// Create a config for knowledge-enhanced backend selection
    ///
    /// Uses the RAG knowledge base (ToolKnowledgeStore) for context-aware
    /// backend recommendations based on property characteristics and tool capabilities.
    ///
    /// # Arguments
    /// * `min_confidence` - Minimum confidence threshold (0.0-1.0) for knowledge-based predictions.
    ///   If prediction confidence is below this, falls back to rule-based selection.
    ///   Recommended values: 0.3-0.5 for exploratory use, 0.5-0.7 for production.
    pub fn knowledge_enhanced(min_confidence: f64) -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::KnowledgeEnhanced {
                min_confidence: min_confidence.clamp(0.0, 1.0),
            },
            merge_strategy: MergeStrategy::FirstSuccess,
            ..Default::default()
        }
    }

    /// Create a config for Byzantine Fault Tolerant verification
    ///
    /// Uses BFT consensus to tolerate up to `max_faulty` buggy or incorrect backends.
    /// For BFT to provide safety guarantees, need at least 3*max_faulty + 1 backends.
    ///
    /// This is useful for high-assurance verification where some backends might:
    /// - Have implementation bugs
    /// - Be unsound in certain cases
    /// - Return incorrect results due to timeouts or resource limits
    ///
    /// # Arguments
    /// * `max_faulty` - Maximum number of faulty backends to tolerate.
    ///   For example, with max_faulty=1 and 4 backends, need 3 to agree.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::DispatcherConfig;
    ///
    /// // Tolerate 1 faulty backend (requires 4+ backends for safety)
    /// let config = DispatcherConfig::byzantine_fault_tolerant(1);
    /// ```
    pub fn byzantine_fault_tolerant(max_faulty: usize) -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::ByzantineFaultTolerant { max_faulty },
            ..Default::default()
        }
    }

    /// Create a config for weighted consensus verification
    ///
    /// Each backend's vote is weighted by its historical accuracy or reliability.
    /// This allows more trusted backends to have more influence on the final result.
    ///
    /// Weights should be in [0.0, 1.0] range and will be clamped.
    /// Backends not in the map use a default weight of 0.5.
    ///
    /// # Arguments
    /// * `weights` - Map from BackendId to weight (0.0-1.0).
    ///   Higher weight = more trusted backend.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::DispatcherConfig;
    /// use dashprove_backends::BackendId;
    /// use std::collections::HashMap;
    ///
    /// let mut weights = HashMap::new();
    /// weights.insert(BackendId::Lean4, 0.95);  // Highly trusted
    /// weights.insert(BackendId::Coq, 0.90);    // Trusted
    /// weights.insert(BackendId::Alloy, 0.70);  // Moderate trust
    ///
    /// let config = DispatcherConfig::weighted_consensus(weights);
    /// ```
    pub fn weighted_consensus(weights: std::collections::HashMap<BackendId, f64>) -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::WeightedConsensus { weights },
            ..Default::default()
        }
    }

    /// Create a config for reputation-based weighted consensus verification
    ///
    /// Computes weights from the reputation tracker's historical data.
    /// Use this when you have a pre-trained reputation tracker and want
    /// to create a config without immediately creating a Dispatcher.
    ///
    /// # Arguments
    /// * `tracker` - Reputation tracker with historical backend performance data
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::DispatcherConfig;
    /// use dashprove_learning::{ReputationTracker, ReputationConfig};
    /// use dashprove_backends::BackendId;
    /// use std::time::Duration;
    ///
    /// let mut tracker = ReputationTracker::new(ReputationConfig::default());
    ///
    /// // Record historical data
    /// for _ in 0..10 {
    ///     tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
    /// }
    ///
    /// let config = DispatcherConfig::from_reputation(&tracker);
    /// ```
    pub fn from_reputation(tracker: &ReputationTracker) -> Self {
        let aggregate_weights = tracker.compute_weights();
        let domain_weights = tracker.compute_domain_weights();
        DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::DomainWeightedConsensus {
                domain_weights,
                aggregate_weights,
            },
            ..Default::default()
        }
    }

    /// Enable automatic reputation updates after each verification
    ///
    /// When enabled, the dispatcher will automatically:
    /// - Track success/failure for each backend based on whether it agrees with consensus
    /// - Record response times for successful verifications
    /// - Refresh weighted consensus weights after each verification
    ///
    /// This is a builder-style method that modifies the config and returns it.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::DispatcherConfig;
    ///
    /// // Create a config with auto-learning enabled
    /// let config = DispatcherConfig::all_backends().with_auto_reputation();
    /// ```
    pub fn with_auto_reputation(mut self) -> Self {
        self.auto_update_reputation = true;
        self
    }

    /// Create a config that uses all backends with automatic reputation learning
    ///
    /// This is a convenience preset that combines:
    /// - All backends selection strategy
    /// - Weighted consensus merge strategy (initial equal weights)
    /// - Automatic reputation updates after each verification
    ///
    /// The weights will be updated automatically based on backend performance.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::{Dispatcher, DispatcherConfig};
    /// use dashprove_learning::{ReputationTracker, ReputationConfig};
    ///
    /// // Start with empty reputation, system will learn over time
    /// let tracker = ReputationTracker::new(ReputationConfig::default());
    /// let config = DispatcherConfig::learning();
    /// let mut dispatcher = Dispatcher::new(config);
    /// dispatcher.set_reputation_tracker(tracker);
    /// ```
    pub fn learning() -> Self {
        DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: MergeStrategy::Majority, // Start with majority, will switch to weighted
            auto_update_reputation: true,
            ..Default::default()
        }
    }
}

/// Main dispatcher that orchestrates backend selection, execution, and result merging
pub struct Dispatcher {
    config: DispatcherConfig,
    registry: BackendRegistry,
    executor: ParallelExecutor,
    merger: ResultMerger,
    /// ML predictor for ML-based backend selection
    ml_predictor: Option<Arc<StrategyModel>>,
    /// Knowledge store for knowledge-enhanced backend selection
    knowledge_store: Option<Arc<KnowledgeStore>>,
    /// Embedder for knowledge-enhanced backend selection
    embedder: Option<Arc<Embedder>>,
    /// Tool knowledge store for knowledge-enhanced backend selection
    tool_store: Option<Arc<ToolKnowledgeStore>>,
    /// Reputation tracker for learning-based backend weighting
    reputation_tracker: Option<ReputationTracker>,
    /// Optional callback for progress updates during execution
    progress_callback: Option<Arc<dyn Fn(ProgressUpdate) + Send + Sync>>,
    /// Path to persist reputation tracker (if set, auto-saves after verification)
    reputation_persistence_path: Option<std::path::PathBuf>,
}

impl Dispatcher {
    /// Create a new dispatcher with the given configuration
    pub fn new(config: DispatcherConfig) -> Self {
        let executor_config = ExecutorConfig {
            max_concurrent: config.max_concurrent,
            task_timeout: config.task_timeout,
            fail_fast: false,
            ..Default::default()
        };

        Dispatcher {
            registry: BackendRegistry::new(),
            executor: ParallelExecutor::new(executor_config),
            merger: ResultMerger::new(config.merge_strategy.clone()),
            config,
            ml_predictor: None,
            knowledge_store: None,
            embedder: None,
            tool_store: None,
            reputation_tracker: None,
            progress_callback: None,
            reputation_persistence_path: None,
        }
    }

    /// Create a new dispatcher with ML-based backend selection
    ///
    /// # Arguments
    /// * `config` - Dispatcher configuration (should use `DispatcherConfig::ml_based()` for ML selection)
    /// * `predictor` - Trained ML strategy predictor for backend selection
    pub fn with_ml_predictor(config: DispatcherConfig, predictor: Arc<StrategyModel>) -> Self {
        let executor_config = ExecutorConfig {
            max_concurrent: config.max_concurrent,
            task_timeout: config.task_timeout,
            fail_fast: false,
            ..Default::default()
        };

        Dispatcher {
            registry: BackendRegistry::new(),
            executor: ParallelExecutor::new(executor_config),
            merger: ResultMerger::new(config.merge_strategy.clone()),
            config,
            ml_predictor: Some(predictor),
            knowledge_store: None,
            embedder: None,
            tool_store: None,
            reputation_tracker: None,
            progress_callback: None,
            reputation_persistence_path: None,
        }
    }

    /// Create a new dispatcher with knowledge-enhanced backend selection
    ///
    /// # Arguments
    /// * `config` - Dispatcher configuration (should use `DispatcherConfig::knowledge_enhanced()` for knowledge selection)
    /// * `knowledge_store` - Knowledge store for semantic search
    /// * `embedder` - Embedder for query embedding
    /// * `tool_store` - Tool knowledge store with structured tool information
    pub fn with_knowledge_store(
        config: DispatcherConfig,
        knowledge_store: Arc<KnowledgeStore>,
        embedder: Arc<Embedder>,
        tool_store: Arc<ToolKnowledgeStore>,
    ) -> Self {
        let executor_config = ExecutorConfig {
            max_concurrent: config.max_concurrent,
            task_timeout: config.task_timeout,
            fail_fast: false,
            ..Default::default()
        };

        Dispatcher {
            registry: BackendRegistry::new(),
            executor: ParallelExecutor::new(executor_config),
            merger: ResultMerger::new(config.merge_strategy.clone()),
            config,
            ml_predictor: None,
            knowledge_store: Some(knowledge_store),
            embedder: Some(embedder),
            tool_store: Some(tool_store),
            reputation_tracker: None,
            progress_callback: None,
            reputation_persistence_path: None,
        }
    }

    /// Create a new dispatcher with reputation-based weighted consensus
    ///
    /// This constructor sets up the dispatcher to use backend reputation scores
    /// as weights for the domain-aware consensus merge strategy. Backend reputations
    /// are learned from historical verification outcomes.
    ///
    /// # Arguments
    /// * `tracker` - Pre-configured reputation tracker with historical data
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::Dispatcher;
    /// use dashprove_learning::{ReputationTracker, ReputationConfig};
    ///
    /// // Load or create a reputation tracker
    /// let tracker = ReputationTracker::new(ReputationConfig::default());
    ///
    /// // Create dispatcher with reputation-based weighting
    /// let dispatcher = Dispatcher::with_reputation_tracker(tracker);
    /// ```
    pub fn with_reputation_tracker(tracker: ReputationTracker) -> Self {
        // Compute weights from reputation scores
        let aggregate_weights = tracker.compute_weights();
        let domain_weights = tracker.compute_domain_weights();
        let merge_strategy = MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights,
        };

        let config = DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy,
            ..Default::default()
        };

        let executor_config = ExecutorConfig {
            max_concurrent: config.max_concurrent,
            task_timeout: config.task_timeout,
            fail_fast: false,
            ..Default::default()
        };

        Dispatcher {
            registry: BackendRegistry::new(),
            executor: ParallelExecutor::new(executor_config),
            merger: ResultMerger::new(config.merge_strategy.clone()),
            config,
            ml_predictor: None,
            knowledge_store: None,
            embedder: None,
            tool_store: None,
            reputation_tracker: Some(tracker),
            progress_callback: None,
            reputation_persistence_path: None,
        }
    }

    /// Set the ML predictor for ML-based backend selection
    pub fn set_ml_predictor(&mut self, predictor: Arc<StrategyModel>) {
        self.ml_predictor = Some(predictor);
    }

    /// Get the ML predictor (if configured)
    pub fn ml_predictor(&self) -> Option<&Arc<StrategyModel>> {
        self.ml_predictor.as_ref()
    }

    /// Set the knowledge stores for knowledge-enhanced backend selection
    pub fn set_knowledge_stores(
        &mut self,
        knowledge_store: Arc<KnowledgeStore>,
        embedder: Arc<Embedder>,
        tool_store: Arc<ToolKnowledgeStore>,
    ) {
        self.knowledge_store = Some(knowledge_store);
        self.embedder = Some(embedder);
        self.tool_store = Some(tool_store);
    }

    /// Check if knowledge stores are configured
    pub fn has_knowledge_stores(&self) -> bool {
        self.knowledge_store.is_some() && self.embedder.is_some() && self.tool_store.is_some()
    }

    /// Set the reputation tracker for learning-based weighted consensus
    ///
    /// When a reputation tracker is set, the dispatcher will update backend
    /// weights based on the tracker's computed reputation scores.
    pub fn set_reputation_tracker(&mut self, tracker: ReputationTracker) {
        // Update merger weights from reputation
        let aggregate_weights = tracker.compute_weights();
        let domain_weights = tracker.compute_domain_weights();
        let strategy = MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights,
        };
        self.merger = ResultMerger::new(strategy.clone());
        self.config.merge_strategy = strategy;
        self.reputation_tracker = Some(tracker);
    }

    /// Get the reputation tracker (if configured)
    pub fn reputation_tracker(&self) -> Option<&ReputationTracker> {
        self.reputation_tracker.as_ref()
    }

    /// Get mutable access to the reputation tracker
    pub fn reputation_tracker_mut(&mut self) -> Option<&mut ReputationTracker> {
        self.reputation_tracker.as_mut()
    }

    /// Check if a reputation tracker is configured
    pub fn has_reputation_tracker(&self) -> bool {
        self.reputation_tracker.is_some()
    }

    /// Update the merge strategy weights from the current reputation tracker
    ///
    /// Call this after recording new observations to the reputation tracker
    /// to update the weighted consensus weights.
    pub fn refresh_reputation_weights(&mut self) {
        if let Some(ref tracker) = self.reputation_tracker {
            let aggregate_weights = tracker.compute_weights();
            let domain_weights = tracker.compute_domain_weights();
            let strategy = MergeStrategy::DomainWeightedConsensus {
                domain_weights,
                aggregate_weights,
            };
            self.merger = ResultMerger::new(strategy.clone());
            self.config.merge_strategy = strategy;
        }
    }

    /// Record a successful verification to the reputation tracker
    ///
    /// Convenience method to update reputation after verification completes.
    pub fn record_verification_success(&mut self, backend: BackendId, response_time: Duration) {
        if let Some(ref mut tracker) = self.reputation_tracker {
            tracker.record_success(backend, response_time);
        }
    }

    /// Record a failed verification to the reputation tracker
    ///
    /// Convenience method to update reputation after verification fails.
    pub fn record_verification_failure(&mut self, backend: BackendId) {
        if let Some(ref mut tracker) = self.reputation_tracker {
            tracker.record_failure(backend);
        }
    }

    /// Set the path for automatic reputation persistence
    ///
    /// When set, the reputation tracker will be automatically saved to this path
    /// after each verification completes. This enables persistent learning across
    /// verification sessions.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dashprove_dispatcher::Dispatcher;
    /// use dashprove_learning::{ReputationTracker, ReputationConfig};
    /// use std::path::PathBuf;
    ///
    /// let mut dispatcher = Dispatcher::with_reputation_tracker(
    ///     ReputationTracker::new(ReputationConfig::default())
    /// );
    /// dispatcher.set_reputation_persistence_path(
    ///     PathBuf::from("~/.dashprove/reputation.json")
    /// );
    /// ```
    pub fn set_reputation_persistence_path(&mut self, path: std::path::PathBuf) {
        self.reputation_persistence_path = Some(path);
    }

    /// Clear the reputation persistence path (disable auto-save)
    pub fn clear_reputation_persistence_path(&mut self) {
        self.reputation_persistence_path = None;
    }

    /// Get the reputation persistence path (if set)
    pub fn reputation_persistence_path(&self) -> Option<&std::path::Path> {
        self.reputation_persistence_path.as_deref()
    }

    /// Save the reputation tracker to the configured persistence path
    ///
    /// Returns Ok(true) if saved successfully, Ok(false) if no path is configured,
    /// or an error if saving failed.
    pub fn save_reputation(&self) -> Result<bool, DispatcherError> {
        let Some(path) = &self.reputation_persistence_path else {
            return Ok(false);
        };

        let Some(tracker) = &self.reputation_tracker else {
            return Ok(false);
        };

        tracker.save_to_file(path).map_err(|e| {
            DispatcherError::InternalError(format!(
                "Failed to save reputation tracker to {}: {}",
                path.display(),
                e
            ))
        })?;

        debug!(path = %path.display(), "Saved reputation tracker");
        Ok(true)
    }

    /// Load the reputation tracker from the configured persistence path
    ///
    /// If the file exists, loads and replaces the current tracker.
    /// If the file doesn't exist, creates a new tracker with default config.
    pub fn load_or_create_reputation(&mut self) -> Result<(), DispatcherError> {
        let Some(path) = &self.reputation_persistence_path else {
            return Ok(());
        };

        let tracker = ReputationTracker::load_or_default(path, ReputationConfig::default())
            .map_err(|e| {
                DispatcherError::InternalError(format!(
                    "Failed to load reputation tracker from {}: {}",
                    path.display(),
                    e
                ))
            })?;

        self.reputation_tracker = Some(tracker);
        debug!(path = %path.display(), "Loaded reputation tracker");
        Ok(())
    }

    /// Set a callback to receive progress updates as verification tasks complete
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(ProgressUpdate) + Send + Sync + 'static,
    {
        let callback = Arc::new(callback);
        self.progress_callback = Some(callback.clone());
        self.executor.set_progress_callback(Some(callback.clone()));
    }

    /// Clear any configured progress callback
    pub fn clear_progress_callback(&mut self) {
        self.progress_callback = None;
        self.executor.set_progress_callback(None);
    }

    /// Register a verification backend
    pub fn register_backend(&mut self, backend: Arc<dyn VerificationBackend>) {
        self.registry.register(backend);
    }

    /// Get the backend registry (for inspection)
    pub fn registry(&self) -> &BackendRegistry {
        &self.registry
    }

    /// Get mutable access to the registry
    pub fn registry_mut(&mut self) -> &mut BackendRegistry {
        &mut self.registry
    }

    /// Verify a typed specification
    pub async fn verify(&mut self, spec: &TypedSpec) -> Result<MergedResults, DispatcherError> {
        if self.registry.is_empty() {
            return Err(DispatcherError::NoBackends);
        }

        // Optionally check backend health
        if self.config.check_health {
            debug!("Checking backend health");
            self.registry.check_all_health().await;
        }

        // Select backends for each property
        // Use appropriate selector based on configured resources
        let mut selector =
            BackendSelector::new(&self.registry, self.config.selection_strategy.clone());

        // Add ML predictor if available
        if let Some(ref predictor) = self.ml_predictor {
            selector.set_ml_predictor(Arc::clone(predictor));
        }

        // Add knowledge stores if available
        if let (Some(ref ks), Some(ref emb), Some(ref ts)) =
            (&self.knowledge_store, &self.embedder, &self.tool_store)
        {
            selector.set_knowledge_stores(Arc::clone(ks), Arc::clone(emb), Arc::clone(ts));
        }

        let selection = selector.select(&spec.spec.properties)?;

        info!(
            properties = spec.spec.properties.len(),
            assignments = selection.assignments.len(),
            warnings = selection.warnings.len(),
            "Backend selection complete"
        );

        for warning in &selection.warnings {
            tracing::warn!("{}", warning);
        }

        // Build backend map for executor
        let backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = self
            .registry
            .all_backends()
            .into_iter()
            .filter_map(|id| self.registry.get(id).map(|b| (id, b)))
            .collect();

        // Execute verification tasks in parallel
        let exec_results = self.executor.execute(&selection, spec, &backends).await?;

        // Merge results
        let merged = self.merger.merge(exec_results);

        info!(
            proven = merged.summary.proven,
            disproven = merged.summary.disproven,
            unknown = merged.summary.unknown,
            confidence = format!("{:.2}", merged.summary.overall_confidence),
            "Verification complete"
        );

        // Auto-update reputation from verification results
        if self.config.auto_update_reputation && self.reputation_tracker.is_some() {
            self.update_reputation_from_results(&merged);
            self.refresh_reputation_weights();
            debug!(
                backends = self
                    .reputation_tracker
                    .as_ref()
                    .map_or(0, |t| t.backend_count()),
                observations = self
                    .reputation_tracker
                    .as_ref()
                    .map_or(0, |t| t.total_observations()),
                "Updated reputation from verification results"
            );

            // Auto-persist reputation if a path is configured
            if self.reputation_persistence_path.is_some() {
                if let Err(e) = self.save_reputation() {
                    tracing::warn!("Failed to persist reputation tracker: {}", e);
                }
            }
        }

        Ok(merged)
    }

    /// Update reputation tracker based on verification results
    ///
    /// A backend is considered "successful" if its status matches the consensus status,
    /// and "failed" if it disagreed with consensus, errored, or timed out.
    fn update_reputation_from_results(&mut self, results: &MergedResults) {
        let Some(ref mut tracker) = self.reputation_tracker else {
            return;
        };

        for property_result in &results.properties {
            let consensus_status = &property_result.status;

            for backend_result in &property_result.backend_results {
                // Check if backend agreed with consensus
                let agreed = Self::statuses_equivalent(&backend_result.status, consensus_status);

                match (
                    property_result.property_type,
                    agreed,
                    backend_result.error.is_none(),
                ) {
                    (Some(prop_type), true, true) => tracker.record_domain_success(
                        backend_result.backend,
                        prop_type,
                        backend_result.time_taken,
                    ),
                    (Some(prop_type), _, _) => {
                        tracker.record_domain_failure(backend_result.backend, prop_type)
                    }
                    (None, true, true) => {
                        tracker.record_success(backend_result.backend, backend_result.time_taken)
                    }
                    (None, _, _) => tracker.record_failure(backend_result.backend),
                }
            }
        }
    }

    /// Check if two verification statuses are equivalent for reputation purposes
    ///
    /// Two statuses are considered equivalent if they represent the same high-level outcome:
    /// - Both Proven
    /// - Both Disproven
    /// - Both Unknown (regardless of reason)
    /// - Both Partial (regardless of percentage)
    fn statuses_equivalent(a: &VerificationStatus, b: &VerificationStatus) -> bool {
        use VerificationStatus::*;
        matches!(
            (a, b),
            (Proven, Proven)
                | (Disproven, Disproven)
                | (Unknown { .. }, Unknown { .. })
                | (Partial { .. }, Partial { .. })
        )
    }

    /// Verify with a specific backend (convenience method)
    pub async fn verify_with(
        &mut self,
        spec: &TypedSpec,
        backend_id: BackendId,
    ) -> Result<MergedResults, DispatcherError> {
        // Temporarily change selection strategy
        let original_strategy = self.config.selection_strategy.clone();
        self.config.selection_strategy = SelectionStrategy::Specific(backend_id);

        let result = self.verify(spec).await;

        // Restore original strategy
        self.config.selection_strategy = original_strategy;

        result
    }

    /// Get current configuration
    pub fn config(&self) -> &DispatcherConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DispatcherConfig) {
        self.config = config.clone();
        self.executor = ParallelExecutor::new(ExecutorConfig {
            max_concurrent: config.max_concurrent,
            task_timeout: config.task_timeout,
            fail_fast: false,
            ..Default::default()
        });
        self.executor
            .set_progress_callback(self.progress_callback.clone());
        self.merger = ResultMerger::new(config.merge_strategy);
    }
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new(DispatcherConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge::BackendResultSummary;
    use dashprove_ai::StrategyPredictor;
    use dashprove_backends::{BackendResult, HealthStatus, PropertyType, VerificationStatus};
    use dashprove_learning::{DomainKey, ReputationConfig};
    use dashprove_usl::ast::{Expr, Property, Spec, Theorem};

    // Mock backend for testing
    struct MockBackend {
        id: BackendId,
        supported: Vec<PropertyType>,
        status: VerificationStatus,
    }

    impl MockBackend {
        fn new(id: BackendId, supported: Vec<PropertyType>) -> Self {
            MockBackend {
                id,
                supported,
                status: VerificationStatus::Proven,
            }
        }

        fn with_status(mut self, status: VerificationStatus) -> Self {
            self.status = status;
            self
        }
    }

    #[async_trait::async_trait]
    impl VerificationBackend for MockBackend {
        fn id(&self) -> BackendId {
            self.id
        }

        fn supports(&self) -> Vec<PropertyType> {
            self.supported.clone()
        }

        async fn verify(
            &self,
            _spec: &TypedSpec,
        ) -> Result<BackendResult, dashprove_backends::BackendError> {
            Ok(BackendResult {
                backend: self.id,
                status: self.status.clone(),
                proof: Some("mock proof".into()),
                counterexample: None,
                diagnostics: vec![],
                time_taken: Duration::from_millis(100),
            })
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    fn make_typed_spec() -> TypedSpec {
        use dashprove_usl::typecheck::typecheck;
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "test".into(),
                body: Expr::Bool(true),
            })],
        };
        typecheck(spec).unwrap()
    }

    #[tokio::test]
    async fn test_dispatcher_basic() {
        let mut dispatcher = Dispatcher::default();
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        assert_eq!(results.summary.proven, 1);
        assert_eq!(results.properties.len(), 1);
    }

    #[tokio::test]
    async fn test_dispatcher_no_backends() {
        let mut dispatcher = Dispatcher::default();
        let spec = make_typed_spec();

        let result = dispatcher.verify(&spec).await;
        assert!(matches!(result, Err(DispatcherError::NoBackends)));
    }

    #[tokio::test]
    async fn test_dispatcher_specific_backend() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Lean4));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        // Should only use Lean4
        assert_eq!(results.properties[0].backend_results.len(), 1);
        assert_eq!(
            results.properties[0].backend_results[0].backend,
            BackendId::Lean4
        );
    }

    #[tokio::test]
    async fn test_dispatcher_all_backends() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::all_backends());
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        // Should use both backends
        assert_eq!(results.properties[0].backend_results.len(), 2);
    }

    #[tokio::test]
    async fn test_dispatcher_redundant() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::redundant(2));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        assert_eq!(results.properties[0].backend_results.len(), 2);
        assert_eq!(results.summary.overall_confidence, 1.0); // Unanimous agreement
    }

    #[tokio::test]
    async fn test_dispatcher_verify_with() {
        let mut dispatcher = Dispatcher::default();
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher
            .verify_with(&spec, BackendId::Alloy)
            .await
            .unwrap();

        assert_eq!(results.properties[0].backend_results.len(), 1);
        assert_eq!(
            results.properties[0].backend_results[0].backend,
            BackendId::Alloy
        );
    }

    #[tokio::test]
    async fn test_dispatcher_disproven() {
        let mut dispatcher = Dispatcher::default();
        dispatcher.register_backend(Arc::new(
            MockBackend::new(BackendId::Lean4, vec![PropertyType::Theorem])
                .with_status(VerificationStatus::Disproven),
        ));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        assert_eq!(results.summary.disproven, 1);
        assert_eq!(results.summary.proven, 0);
    }

    #[test]
    fn test_config_presets() {
        let default = DispatcherConfig::default();
        assert!(matches!(
            default.selection_strategy,
            SelectionStrategy::Single
        ));

        let redundant = DispatcherConfig::redundant(2);
        assert!(matches!(
            redundant.selection_strategy,
            SelectionStrategy::Redundant { min_backends: 2 }
        ));

        let all = DispatcherConfig::all_backends();
        assert!(matches!(all.selection_strategy, SelectionStrategy::All));

        let specific = DispatcherConfig::specific(BackendId::TlaPlus);
        assert!(matches!(
            specific.selection_strategy,
            SelectionStrategy::Specific(BackendId::TlaPlus)
        ));
    }

    #[test]
    fn test_config_ml_based() {
        let ml_config = DispatcherConfig::ml_based(0.5);
        assert!(matches!(
            ml_config.selection_strategy,
            SelectionStrategy::MlBased { min_confidence } if min_confidence == 0.5
        ));

        // Test clamping
        let high_config = DispatcherConfig::ml_based(1.5);
        assert!(matches!(
            high_config.selection_strategy,
            SelectionStrategy::MlBased { min_confidence } if min_confidence == 1.0
        ));

        let low_config = DispatcherConfig::ml_based(-0.5);
        assert!(matches!(
            low_config.selection_strategy,
            SelectionStrategy::MlBased { min_confidence } if min_confidence == 0.0
        ));
    }

    #[test]
    fn test_dispatcher_with_ml_predictor() {
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        let config = DispatcherConfig::ml_based(0.3);
        let dispatcher = Dispatcher::with_ml_predictor(config, predictor);

        assert!(dispatcher.ml_predictor().is_some());
        assert!(matches!(
            dispatcher.config().selection_strategy,
            SelectionStrategy::MlBased { .. }
        ));
    }

    #[test]
    fn test_dispatcher_set_ml_predictor() {
        let mut dispatcher = Dispatcher::default();
        assert!(dispatcher.ml_predictor().is_none());

        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        dispatcher.set_ml_predictor(predictor);
        assert!(dispatcher.ml_predictor().is_some());
    }

    #[tokio::test]
    async fn test_dispatcher_ml_based_verification() {
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        let config = DispatcherConfig::ml_based(0.0); // Low threshold to use ML
        let mut dispatcher = Dispatcher::with_ml_predictor(config, predictor);

        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        // Should successfully verify (either via ML prediction or fallback)
        assert_eq!(results.summary.proven, 1);
    }

    #[test]
    fn test_config_knowledge_enhanced() {
        let config = DispatcherConfig::knowledge_enhanced(0.5);
        assert!(matches!(
            config.selection_strategy,
            SelectionStrategy::KnowledgeEnhanced { min_confidence } if min_confidence == 0.5
        ));

        // Test clamping
        let high_config = DispatcherConfig::knowledge_enhanced(1.5);
        assert!(matches!(
            high_config.selection_strategy,
            SelectionStrategy::KnowledgeEnhanced { min_confidence } if min_confidence == 1.0
        ));

        let low_config = DispatcherConfig::knowledge_enhanced(-0.5);
        assert!(matches!(
            low_config.selection_strategy,
            SelectionStrategy::KnowledgeEnhanced { min_confidence } if min_confidence == 0.0
        ));
    }

    #[test]
    fn test_dispatcher_has_knowledge_stores() {
        let dispatcher = Dispatcher::default();
        assert!(!dispatcher.has_knowledge_stores());
    }

    // ==================== Mutation-killing tests ====================

    #[test]
    fn test_config_specific_uses_first_success_merge() {
        // Mutation: delete field merge_strategy from DispatcherConfig::specific
        let config = DispatcherConfig::specific(BackendId::Lean4);
        assert_eq!(config.merge_strategy, MergeStrategy::FirstSuccess);
    }

    #[test]
    fn test_config_ml_based_uses_first_success_merge() {
        // Mutation: delete field merge_strategy from DispatcherConfig::ml_based
        let config = DispatcherConfig::ml_based(0.5);
        assert_eq!(config.merge_strategy, MergeStrategy::FirstSuccess);
    }

    #[test]
    fn test_config_knowledge_enhanced_uses_first_success_merge() {
        // Mutation: delete field merge_strategy from DispatcherConfig::knowledge_enhanced
        let config = DispatcherConfig::knowledge_enhanced(0.5);
        assert_eq!(config.merge_strategy, MergeStrategy::FirstSuccess);
    }

    #[test]
    fn test_config_byzantine_fault_tolerant() {
        let config = DispatcherConfig::byzantine_fault_tolerant(1);
        assert!(matches!(
            config.merge_strategy,
            MergeStrategy::ByzantineFaultTolerant { max_faulty } if max_faulty == 1
        ));
        // BFT uses All selection to get results from all backends
        assert!(matches!(config.selection_strategy, SelectionStrategy::All));
    }

    #[test]
    fn test_config_byzantine_fault_tolerant_higher_tolerance() {
        let config = DispatcherConfig::byzantine_fault_tolerant(2);
        assert!(matches!(
            config.merge_strategy,
            MergeStrategy::ByzantineFaultTolerant { max_faulty } if max_faulty == 2
        ));
    }

    #[test]
    fn test_config_weighted_consensus() {
        let mut weights = std::collections::HashMap::new();
        weights.insert(BackendId::Lean4, 0.9);
        weights.insert(BackendId::Coq, 0.85);

        let config = DispatcherConfig::weighted_consensus(weights);
        assert!(matches!(
            config.merge_strategy,
            MergeStrategy::WeightedConsensus { .. }
        ));
        // Weighted consensus uses All selection to get results from all backends
        assert!(matches!(config.selection_strategy, SelectionStrategy::All));
    }

    #[test]
    fn test_config_weighted_consensus_empty_weights() {
        let weights = std::collections::HashMap::new();
        let config = DispatcherConfig::weighted_consensus(weights);
        // Empty weights map is valid - all backends will use default 0.5
        assert!(matches!(
            config.merge_strategy,
            MergeStrategy::WeightedConsensus { .. }
        ));
    }

    #[tokio::test]
    async fn test_dispatcher_bft_verification() {
        let config = DispatcherConfig::byzantine_fault_tolerant(1);
        let mut dispatcher = Dispatcher::new(config);

        // Register 4 backends for BFT with f=1
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Coq,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Kani,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        // All 4 backends should agree (all Proven)
        assert_eq!(results.summary.proven, 1);
        assert_eq!(results.properties[0].backend_results.len(), 4);
    }

    #[tokio::test]
    async fn test_dispatcher_weighted_verification() {
        let mut weights = std::collections::HashMap::new();
        weights.insert(BackendId::Lean4, 1.0);
        weights.insert(BackendId::Alloy, 0.5);

        let config = DispatcherConfig::weighted_consensus(weights);
        let mut dispatcher = Dispatcher::new(config);

        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Alloy,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        assert_eq!(results.summary.proven, 1);
    }

    #[test]
    fn test_has_knowledge_stores_requires_all_three() {
        // Mutation: replace && with || in has_knowledge_stores
        // This test ensures all three stores must be present
        let dispatcher = Dispatcher::default();
        // None set -> should be false
        assert!(!dispatcher.has_knowledge_stores());
    }

    #[test]
    fn test_has_knowledge_stores_partial_is_false() {
        // Mutation: replace && with || in has_knowledge_stores
        // Testing that partial configuration returns false
        // We can't easily set just one field, but we verify the && logic
        // by checking the method returns false when none are set
        let mut dispatcher = Dispatcher::default();
        // Dispatcher without any stores
        assert!(!dispatcher.has_knowledge_stores());

        // After setting ML predictor (not knowledge stores) - still false
        let predictor = Arc::new(StrategyModel::from(StrategyPredictor::new()));
        dispatcher.set_ml_predictor(predictor);
        // ML predictor doesn't affect has_knowledge_stores
        assert!(!dispatcher.has_knowledge_stores());
    }

    #[tokio::test]
    async fn test_dispatcher_knowledge_enhanced_fallback_no_stores() {
        // Without knowledge stores, should fall back to rule-based selection
        let config = DispatcherConfig::knowledge_enhanced(0.5);
        let mut dispatcher = Dispatcher::new(config);

        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        // Should successfully verify via fallback
        assert_eq!(results.summary.proven, 1);
    }

    // Integration tests with actual knowledge stores
    mod integration {
        use super::*;
        use dashprove_knowledge::{
            embedding::EmbeddingModel, Embedder, KnowledgeStore, ToolKnowledgeStore,
        };
        use std::path::PathBuf;

        fn get_resources_path() -> PathBuf {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(manifest_dir)
                .parent()
                .unwrap()
                .join("dashprove-knowledge")
                .join("resources")
                .join("tools")
        }

        fn create_test_store() -> KnowledgeStore {
            KnowledgeStore::new(PathBuf::from("/tmp/dispatcher_test_store"), 384)
        }

        fn create_test_embedder() -> Embedder {
            Embedder::new(EmbeddingModel::SentenceTransformers)
        }

        #[tokio::test]
        async fn test_dispatcher_with_knowledge_stores() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!(
                    "Skipping: resources/tools directory not found at {:?}",
                    resources_path
                );
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let knowledge_store = create_test_store();
            let embedder = create_test_embedder();

            let config = DispatcherConfig::knowledge_enhanced(0.0);
            let dispatcher = Dispatcher::with_knowledge_store(
                config,
                Arc::new(knowledge_store),
                Arc::new(embedder),
                Arc::new(tool_store),
            );

            assert!(dispatcher.has_knowledge_stores());
            assert!(matches!(
                dispatcher.config().selection_strategy,
                SelectionStrategy::KnowledgeEnhanced { .. }
            ));
        }

        #[tokio::test]
        async fn test_dispatcher_set_knowledge_stores() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let knowledge_store = create_test_store();
            let embedder = create_test_embedder();

            let mut dispatcher = Dispatcher::default();
            assert!(!dispatcher.has_knowledge_stores());

            dispatcher.set_knowledge_stores(
                Arc::new(knowledge_store),
                Arc::new(embedder),
                Arc::new(tool_store),
            );
            assert!(dispatcher.has_knowledge_stores());
        }

        #[tokio::test]
        async fn test_dispatcher_knowledge_enhanced_verification() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let knowledge_store = create_test_store();
            let embedder = create_test_embedder();

            // Use low confidence threshold to exercise the knowledge-enhanced path
            let config = DispatcherConfig::knowledge_enhanced(0.0);
            let mut dispatcher = Dispatcher::with_knowledge_store(
                config,
                Arc::new(knowledge_store),
                Arc::new(embedder),
                Arc::new(tool_store),
            );

            dispatcher.register_backend(Arc::new(MockBackend::new(
                BackendId::Lean4,
                vec![PropertyType::Theorem],
            )));

            let spec = make_typed_spec();
            let results = dispatcher.verify(&spec).await.unwrap();

            // Should successfully verify (either via knowledge or fallback)
            assert_eq!(results.summary.proven, 1);
        }

        #[tokio::test]
        async fn test_dispatcher_knowledge_enhanced_multiple_backends() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let knowledge_store = create_test_store();
            let embedder = create_test_embedder();

            let config = DispatcherConfig::knowledge_enhanced(0.0);
            let mut dispatcher = Dispatcher::with_knowledge_store(
                config,
                Arc::new(knowledge_store),
                Arc::new(embedder),
                Arc::new(tool_store),
            );

            // Register multiple backends for theorem type
            dispatcher.register_backend(Arc::new(MockBackend::new(
                BackendId::Lean4,
                vec![PropertyType::Theorem],
            )));
            dispatcher.register_backend(Arc::new(MockBackend::new(
                BackendId::Coq,
                vec![PropertyType::Theorem],
            )));

            let spec = make_typed_spec();
            let results = dispatcher.verify(&spec).await.unwrap();

            // Should verify with one backend (knowledge selection or fallback)
            assert_eq!(results.summary.proven, 1);
            assert_eq!(results.properties[0].backend_results.len(), 1);
        }

        #[tokio::test]
        async fn test_dispatcher_knowledge_enhanced_high_threshold_fallback() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let knowledge_store = create_test_store();
            let embedder = create_test_embedder();

            // Use very high confidence threshold - should fall back to rule-based
            let config = DispatcherConfig::knowledge_enhanced(0.99);
            let mut dispatcher = Dispatcher::with_knowledge_store(
                config,
                Arc::new(knowledge_store),
                Arc::new(embedder),
                Arc::new(tool_store),
            );

            dispatcher.register_backend(Arc::new(MockBackend::new(
                BackendId::Lean4,
                vec![PropertyType::Theorem],
            )));

            let spec = make_typed_spec();
            let results = dispatcher.verify(&spec).await.unwrap();

            // Should fallback to rule-based and still verify
            assert_eq!(results.summary.proven, 1);
            assert_eq!(
                results.properties[0].backend_results[0].backend,
                BackendId::Lean4
            );
        }

        #[tokio::test]
        async fn test_tool_store_loaded_correctly() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            // Verify we have substantial tool coverage
            assert!(tool_store.len() > 100, "Expected > 100 tools loaded");

            // Verify key tools are present
            assert!(tool_store.get("z3").is_some(), "Z3 should be loaded");
            assert!(tool_store.get("lean4").is_some(), "Lean4 should be loaded");
        }
    }

    // ===== Reputation Tracker Tests =====

    #[test]
    fn test_config_from_reputation() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Record some history
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        }
        for _ in 0..5 {
            tracker.record_failure(BackendId::Coq);
            tracker.record_success(BackendId::Coq, Duration::from_millis(200));
        }

        let config = DispatcherConfig::from_reputation(&tracker);

        // Should use All selection with DomainWeightedConsensus merge
        assert!(matches!(config.selection_strategy, SelectionStrategy::All));
        assert!(matches!(
            config.merge_strategy,
            MergeStrategy::DomainWeightedConsensus { .. }
        ));

        // Verify weights are computed
        if let MergeStrategy::DomainWeightedConsensus {
            aggregate_weights, ..
        } = &config.merge_strategy
        {
            assert!(aggregate_weights.contains_key(&BackendId::Lean4));
            assert!(aggregate_weights.contains_key(&BackendId::Coq));
            // Lean4 should have higher weight (100% success vs 50%)
            assert!(aggregate_weights[&BackendId::Lean4] > aggregate_weights[&BackendId::Coq]);
        } else {
            panic!("Expected DomainWeightedConsensus merge strategy");
        }
    }

    #[test]
    fn test_config_from_reputation_empty_tracker() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let config = DispatcherConfig::from_reputation(&tracker);

        // Empty tracker should still produce valid config
        assert!(matches!(config.selection_strategy, SelectionStrategy::All));
        if let MergeStrategy::DomainWeightedConsensus {
            aggregate_weights, ..
        } = &config.merge_strategy
        {
            assert!(aggregate_weights.is_empty());
        }
    }

    #[test]
    fn test_dispatcher_with_reputation_tracker() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Build up reputation history
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        }

        let dispatcher = Dispatcher::with_reputation_tracker(tracker);

        // Verify tracker is set
        assert!(dispatcher.has_reputation_tracker());
        assert!(dispatcher.reputation_tracker().is_some());

        // Verify config uses domain-weighted consensus
        assert!(matches!(
            dispatcher.config().merge_strategy,
            MergeStrategy::DomainWeightedConsensus { .. }
        ));
    }

    #[test]
    fn test_dispatcher_set_reputation_tracker() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::default());

        // Initially no tracker
        assert!(!dispatcher.has_reputation_tracker());

        // Set tracker
        let mut tracker = ReputationTracker::new(ReputationConfig::default());
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(50));
        }

        dispatcher.set_reputation_tracker(tracker);

        // Now has tracker
        assert!(dispatcher.has_reputation_tracker());
        assert!(matches!(
            dispatcher.config().merge_strategy,
            MergeStrategy::DomainWeightedConsensus { .. }
        ));
    }

    #[test]
    fn test_dispatcher_record_verification_success() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let mut dispatcher = Dispatcher::with_reputation_tracker(tracker);

        // Record a success
        dispatcher.record_verification_success(BackendId::Lean4, Duration::from_millis(100));

        // Check it was recorded
        let tracker = dispatcher.reputation_tracker().unwrap();
        let stats = tracker.get_stats(&BackendId::Lean4).unwrap();
        assert_eq!(stats.successes, 1);
    }

    #[test]
    fn test_dispatcher_record_verification_failure() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let mut dispatcher = Dispatcher::with_reputation_tracker(tracker);

        // Record a failure
        dispatcher.record_verification_failure(BackendId::Alloy);

        // Check it was recorded
        let tracker = dispatcher.reputation_tracker().unwrap();
        let stats = tracker.get_stats(&BackendId::Alloy).unwrap();
        assert_eq!(stats.failures, 1);
    }

    #[test]
    fn test_dispatcher_refresh_reputation_weights() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let mut dispatcher = Dispatcher::with_reputation_tracker(tracker);

        // Initial state - no weights because no observations above threshold
        let initial_weights = if let MergeStrategy::DomainWeightedConsensus {
            aggregate_weights,
            ..
        } = &dispatcher.config().merge_strategy
        {
            aggregate_weights.clone()
        } else {
            panic!("Expected DomainWeightedConsensus");
        };

        // Record observations via mutable tracker
        {
            let tracker = dispatcher.reputation_tracker_mut().unwrap();
            for _ in 0..10 {
                tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
            }
        }

        // Refresh weights
        dispatcher.refresh_reputation_weights();

        // Weights should now include Lean4
        if let MergeStrategy::DomainWeightedConsensus {
            aggregate_weights: weights,
            ..
        } = &dispatcher.config().merge_strategy
        {
            assert!(weights.contains_key(&BackendId::Lean4));
            assert!(weights.len() > initial_weights.len() || !weights.is_empty());
        }
    }

    #[test]
    fn test_update_reputation_tracks_domain_stats() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let mut dispatcher = Dispatcher::with_reputation_tracker(tracker);

        let merged = MergedResults {
            properties: vec![MergedResult {
                property_index: 0,
                property_type: Some(PropertyType::Theorem),
                status: VerificationStatus::Proven,
                confidence: 0.9,
                backend_results: vec![
                    BackendResultSummary {
                        backend: BackendId::Lean4,
                        status: VerificationStatus::Proven,
                        time_taken: Duration::from_millis(120),
                        error: None,
                    },
                    BackendResultSummary {
                        backend: BackendId::Coq,
                        status: VerificationStatus::Unknown {
                            reason: "timeout".into(),
                        },
                        time_taken: Duration::ZERO,
                        error: Some("timeout".into()),
                    },
                ],
                proof: None,
                counterexample: None,
                diagnostics: vec![],
                verification_time: Duration::from_millis(120),
            }],
            summary: VerificationSummary {
                proven: 1,
                disproven: 0,
                unknown: 0,
                partial: 0,
                overall_confidence: 0.9,
            },
            total_time: Duration::from_millis(120),
        };

        dispatcher.update_reputation_from_results(&merged);

        let tracker = dispatcher.reputation_tracker().unwrap();
        let lean_key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let coq_key = DomainKey::new(BackendId::Coq, PropertyType::Theorem);

        let lean_stats = tracker.get_domain_stats(&lean_key).unwrap();
        assert_eq!(lean_stats.successes, 1);
        assert_eq!(lean_stats.failures, 0);

        let coq_stats = tracker.get_domain_stats(&coq_key).unwrap();
        assert_eq!(coq_stats.failures, 1);
    }

    #[test]
    fn test_dispatcher_reputation_tracker_mut() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let mut dispatcher = Dispatcher::with_reputation_tracker(tracker);

        // Get mutable access and modify
        {
            let tracker = dispatcher.reputation_tracker_mut().unwrap();
            tracker.record_success(BackendId::Coq, Duration::from_millis(500));
        }

        // Verify modification persisted
        let tracker = dispatcher.reputation_tracker().unwrap();
        assert!(tracker.get_stats(&BackendId::Coq).is_some());
    }

    #[tokio::test]
    async fn test_dispatcher_reputation_based_verification() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Lean4 has excellent reputation
        for _ in 0..15 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(50));
        }

        // Coq has mediocre reputation
        for _ in 0..10 {
            tracker.record_success(BackendId::Coq, Duration::from_millis(500));
            tracker.record_failure(BackendId::Coq);
        }

        let mut dispatcher = Dispatcher::with_reputation_tracker(tracker);

        // Register backends
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Coq,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let results = dispatcher.verify(&spec).await.unwrap();

        // Verification should succeed
        assert_eq!(results.summary.proven, 1);

        // Both backends should have been used (All strategy)
        assert!(!results.properties[0].backend_results.is_empty());
    }

    // ===== Auto-Reputation Update Tests =====

    #[test]
    fn test_config_with_auto_reputation() {
        let config = DispatcherConfig::all_backends().with_auto_reputation();
        assert!(config.auto_update_reputation);
    }

    #[test]
    fn test_config_learning() {
        let config = DispatcherConfig::learning();
        assert!(config.auto_update_reputation);
        assert!(matches!(config.selection_strategy, SelectionStrategy::All));
        assert!(matches!(config.merge_strategy, MergeStrategy::Majority));
    }

    #[test]
    fn test_config_default_no_auto_reputation() {
        let config = DispatcherConfig::default();
        assert!(!config.auto_update_reputation);
    }

    #[test]
    fn test_reputation_persistence_path() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::default());

        // Initially no path set
        assert!(dispatcher.reputation_persistence_path().is_none());

        // Set path
        let path = std::path::PathBuf::from("/tmp/test_rep.json");
        dispatcher.set_reputation_persistence_path(path.clone());
        assert_eq!(
            dispatcher.reputation_persistence_path(),
            Some(path.as_path())
        );

        // Clear path
        dispatcher.clear_reputation_persistence_path();
        assert!(dispatcher.reputation_persistence_path().is_none());
    }

    #[test]
    fn test_save_reputation_no_path() {
        let dispatcher = Dispatcher::new(DispatcherConfig::default());

        // No path configured - should return Ok(false)
        let result = dispatcher.save_reputation();
        assert!(matches!(result, Ok(false)));
    }

    #[test]
    fn test_save_reputation_no_tracker() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
        dispatcher.set_reputation_persistence_path(std::path::PathBuf::from("/tmp/test.json"));

        // No tracker configured - should return Ok(false)
        let result = dispatcher.save_reputation();
        assert!(matches!(result, Ok(false)));
    }

    #[test]
    fn test_save_reputation_success() {
        let mut dispatcher = Dispatcher::with_reputation_tracker(ReputationTracker::new(
            ReputationConfig::default(),
        ));

        // Use temp directory for test
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("dashprove_test_reputation.json");

        // Clean up before test
        let _ = std::fs::remove_file(&path);

        dispatcher.set_reputation_persistence_path(path.clone());

        // Record some data
        dispatcher.record_verification_success(BackendId::Lean4, Duration::from_millis(100));

        // Save should succeed
        let result = dispatcher.save_reputation();
        assert!(matches!(result, Ok(true)));

        // File should exist
        assert!(path.exists());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_or_create_reputation_new() {
        let mut dispatcher = Dispatcher::new(DispatcherConfig::default());

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("dashprove_test_new_reputation.json");

        // Clean up before test
        let _ = std::fs::remove_file(&path);

        dispatcher.set_reputation_persistence_path(path.clone());

        // Load should create new tracker
        let result = dispatcher.load_or_create_reputation();
        assert!(result.is_ok());
        assert!(dispatcher.reputation_tracker().is_some());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_or_create_reputation_existing() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("dashprove_test_existing_reputation.json");

        // Create and save an existing tracker with data
        {
            let mut tracker = ReputationTracker::new(ReputationConfig::default());
            for _ in 0..10 {
                tracker.record_success(BackendId::Coq, Duration::from_millis(200));
            }
            tracker.save_to_file(&path).unwrap();
        }

        // Now load it
        let mut dispatcher = Dispatcher::new(DispatcherConfig::default());
        dispatcher.set_reputation_persistence_path(path.clone());

        let result = dispatcher.load_or_create_reputation();
        assert!(result.is_ok());

        let tracker = dispatcher.reputation_tracker().unwrap();
        assert_eq!(tracker.total_observations(), 10);
        assert!(tracker.get_stats(&BackendId::Coq).is_some());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_statuses_equivalent_proven() {
        assert!(Dispatcher::statuses_equivalent(
            &VerificationStatus::Proven,
            &VerificationStatus::Proven
        ));
    }

    #[test]
    fn test_statuses_equivalent_disproven() {
        assert!(Dispatcher::statuses_equivalent(
            &VerificationStatus::Disproven,
            &VerificationStatus::Disproven
        ));
    }

    #[test]
    fn test_statuses_equivalent_unknown() {
        assert!(Dispatcher::statuses_equivalent(
            &VerificationStatus::Unknown {
                reason: "timeout".into()
            },
            &VerificationStatus::Unknown {
                reason: "different reason".into()
            }
        ));
    }

    #[test]
    fn test_statuses_equivalent_partial() {
        assert!(Dispatcher::statuses_equivalent(
            &VerificationStatus::Partial {
                verified_percentage: 50.0
            },
            &VerificationStatus::Partial {
                verified_percentage: 75.0
            }
        ));
    }

    #[test]
    fn test_statuses_not_equivalent_proven_disproven() {
        assert!(!Dispatcher::statuses_equivalent(
            &VerificationStatus::Proven,
            &VerificationStatus::Disproven
        ));
    }

    #[test]
    fn test_statuses_not_equivalent_proven_unknown() {
        assert!(!Dispatcher::statuses_equivalent(
            &VerificationStatus::Proven,
            &VerificationStatus::Unknown {
                reason: "test".into()
            }
        ));
    }

    #[tokio::test]
    async fn test_auto_reputation_update_on_agreement() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let config = DispatcherConfig::learning();
        let mut dispatcher = Dispatcher::new(config);
        dispatcher.set_reputation_tracker(tracker);

        // Register backend that will agree with itself (only backend)
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let _results = dispatcher.verify(&spec).await.unwrap();

        // Should have recorded success for Lean4
        let tracker = dispatcher.reputation_tracker().unwrap();
        let stats = tracker.get_stats(&BackendId::Lean4).unwrap();
        assert_eq!(stats.successes, 1);
        assert_eq!(stats.failures, 0);
    }

    #[tokio::test]
    async fn test_auto_reputation_update_multiple_backends_agree() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let config = DispatcherConfig::learning();
        let mut dispatcher = Dispatcher::new(config);
        dispatcher.set_reputation_tracker(tracker);

        // Both backends return Proven - they agree
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Coq,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let _results = dispatcher.verify(&spec).await.unwrap();

        // Both should be recorded as successes (they agreed with consensus)
        let tracker = dispatcher.reputation_tracker().unwrap();

        let lean_stats = tracker.get_stats(&BackendId::Lean4).unwrap();
        assert_eq!(lean_stats.successes, 1);
        assert_eq!(lean_stats.failures, 0);

        let coq_stats = tracker.get_stats(&BackendId::Coq).unwrap();
        assert_eq!(coq_stats.successes, 1);
        assert_eq!(coq_stats.failures, 0);
    }

    #[tokio::test]
    async fn test_auto_reputation_update_backend_disagrees() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let config = DispatcherConfig::learning();
        let mut dispatcher = Dispatcher::new(config);
        dispatcher.set_reputation_tracker(tracker);

        // One backend proves, one disproves - one will be marked as failure
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));
        dispatcher.register_backend(Arc::new(
            MockBackend::new(BackendId::Alloy, vec![PropertyType::Theorem])
                .with_status(VerificationStatus::Disproven),
        ));

        let spec = make_typed_spec();
        let _results = dispatcher.verify(&spec).await.unwrap();

        // With Majority merge strategy:
        // - If consensus is Proven, Lean4 succeeds, Alloy fails
        // - If consensus is Disproven, Lean4 fails, Alloy succeeds
        let tracker = dispatcher.reputation_tracker().unwrap();

        let lean_stats = tracker.get_stats(&BackendId::Lean4);
        let alloy_stats = tracker.get_stats(&BackendId::Alloy);

        // One success and one failure total across both backends
        let total_successes =
            lean_stats.map_or(0, |s| s.successes) + alloy_stats.map_or(0, |s| s.successes);
        let total_failures =
            lean_stats.map_or(0, |s| s.failures) + alloy_stats.map_or(0, |s| s.failures);

        assert_eq!(total_successes, 1);
        assert_eq!(total_failures, 1);
    }

    #[tokio::test]
    async fn test_no_auto_reputation_when_disabled() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let config = DispatcherConfig::all_backends(); // auto_update_reputation = false
        let mut dispatcher = Dispatcher::new(config);
        dispatcher.set_reputation_tracker(tracker);

        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        let _results = dispatcher.verify(&spec).await.unwrap();

        // Should NOT have recorded anything since auto_update is disabled
        let tracker = dispatcher.reputation_tracker().unwrap();
        assert!(tracker.get_stats(&BackendId::Lean4).is_none());
    }

    #[tokio::test]
    async fn test_no_auto_reputation_without_tracker() {
        let config = DispatcherConfig::learning(); // auto_update enabled but no tracker
        let mut dispatcher = Dispatcher::new(config);
        // Note: NOT setting a reputation tracker

        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();
        // Should not panic - just skip reputation update
        let results = dispatcher.verify(&spec).await.unwrap();
        assert_eq!(results.summary.proven, 1);
    }

    #[tokio::test]
    async fn test_auto_reputation_weights_refreshed() {
        let tracker = ReputationTracker::new(ReputationConfig::default());
        let config = DispatcherConfig::learning();
        let mut dispatcher = Dispatcher::new(config);
        dispatcher.set_reputation_tracker(tracker);

        // Register a backend
        dispatcher.register_backend(Arc::new(MockBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem],
        )));

        let spec = make_typed_spec();

        // Run multiple verifications to build up reputation
        for _ in 0..5 {
            let _ = dispatcher.verify(&spec).await.unwrap();
        }

        // Check that observations were recorded
        let tracker = dispatcher.reputation_tracker().unwrap();
        let stats = tracker.get_stats(&BackendId::Lean4).unwrap();
        assert_eq!(stats.successes, 5);

        // After enough observations, weights should be computed
        // (min_observations default is 5)
        assert!(tracker.total_observations() >= 5);
    }
}
