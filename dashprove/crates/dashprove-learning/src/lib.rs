// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // Builder methods don't need must_use
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::use_self)] // Self vs TypeName - style preference
#![allow(clippy::unused_self)] // Some methods keep self for future API compatibility
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::missing_errors_doc)] // Error docs are implementation details
#![allow(clippy::cast_precision_loss)] // usize to f64 for scores is intentional
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::uninlined_format_args)] // Named args are clearer
#![allow(clippy::format_push_string)] // Common pattern in string builders
#![allow(clippy::similar_names)] // e.g., proof/proofs, corpus/corpora
#![allow(clippy::too_many_lines)] // Complex methods may be inherently long
#![allow(clippy::needless_pass_by_value)] // Ownership semantics may be intentional
#![allow(clippy::redundant_closure_for_method_calls)] // Minor style issue
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
#![allow(clippy::cast_possible_wrap)] // i64 to i32 checked at runtime
#![allow(clippy::cast_possible_truncation)] // Bounds checked at runtime
#![allow(clippy::cast_sign_loss)] // Bounds checked at runtime
#![allow(clippy::cast_lossless)] // Explicit casts are clearer
#![allow(clippy::or_fun_call)] // Style preference
#![allow(clippy::needless_raw_string_hashes)] // Raw strings for templates
#![allow(clippy::wildcard_imports)] // Re-exports use wildcard pattern
#![allow(clippy::module_name_repetitions)] // corpus::ProofCorpus is clear
#![allow(clippy::single_char_pattern)] // "." vs '.' - minor optimization
#![allow(clippy::struct_excessive_bools)] // Structs may have multiple bool flags
#![allow(clippy::if_then_some_else_none)] // if chain style preference
#![allow(clippy::redundant_pub_crate)] // pub(crate) in private mod is intentional
#![allow(clippy::if_not_else)] // if chain style preference
#![allow(clippy::comparison_chain)] // if chain is clearer than match for comparisons
#![allow(clippy::suboptimal_flops)] // mul_add not always clearer
#![allow(clippy::implicit_hasher)] // HashMap<K,V> is clearer than generic S
#![allow(clippy::format_collect)] // Collect into String is clear enough

//! Proof learning system
//!
//! This module provides:
//! - **ProofCorpus**: Storage and retrieval of successful proofs
//! - **CounterexampleCorpus**: Storage and retrieval of counterexamples
//! - **TacticDatabase**: Statistics on tactic effectiveness per context
//! - **Similarity Search**: Find related proofs and counterexamples based on structure
//!
//! The learning system observes verification results and builds a searchable
//! corpus that improves proof discovery, strategy selection, and counterexample
//! classification over time.

pub mod binary;
pub mod corpus;
pub mod counterexamples;
pub mod distance;
pub mod embedder;
mod io;
pub mod lsh;
pub mod ordered_float;
pub mod patterns;
pub mod pipeline;
pub mod pq;
pub mod reputation;
pub mod similarity;
pub mod tactics;
pub mod templates;

pub use binary::{
    BinaryCode, BinaryConfig, BinaryCorpus, BinaryError, BinaryMemoryStats, BinaryQuantizer,
};
pub use corpus::{
    OpqLshConfig, OpqLshMemoryStats, PqLshConfig, PqLshMemoryStats, ProofCorpus, ProofCorpusLsh,
    ProofCorpusOpqLsh, ProofCorpusPqLsh, ProofEntry, ProofHistory, ProofId,
};
pub use counterexamples::{
    format_suggestions, suggest_comparison_periods, ClusterPattern, CorpusHistory,
    CounterexampleCorpus, CounterexampleEntry, CounterexampleFeatures, CounterexampleId,
    GrowthProjections, HistoryComparison, PeriodStats, PeriodSuggestion, PeriodSuggestionCliArgs,
    SimilarCounterexample, SuggestionType, TimePeriod,
};
pub use distance::{
    dot_product, dot_product_scalar, euclidean_distance, euclidean_distance_sq,
    euclidean_distance_sq_scalar, vector_norm, vector_norm_sq, vector_norm_sq_scalar,
};
pub use embedder::{
    Embedding, EmbeddingIndex, EmbeddingIndexBuilder, PropertyEmbedder, EMBEDDING_DIM,
};
pub use lsh::{BucketStats, LshConfig, LshIndex};
pub use patterns::{
    PatternDatabase, PatternExtractor, PatternStats, ProofPattern, ProofRepairer, RepairStrategy,
    RepairSuggestion, SuggestedContinuation, TacticSequence,
};
pub use pipeline::{
    FeedbackBatch, FeedbackRecord, PipelineConfig, PipelineStats, TrainingPipeline,
};
pub use pq::{
    DistanceTable, OpqConfig, OpqCorpus, OptimizedProductQuantizer, PqConfig, PqCorpus, PqError,
    ProductQuantizer,
};
pub use reputation::{
    property_type_from_string, BackendStats, BootstrapStats, DomainKey, DomainSummary,
    ReputationConfig, ReputationFromCorpus, ReputationObservation, ReputationTracker,
};
pub use similarity::{PropertyFeatures, SimilarProof};
pub use tactics::{TacticContext, TacticDatabase, TacticStats};

use chrono::{DateTime, Utc};
use dashprove_backends::traits::{
    BackendId, CounterexampleClusters, StructuredCounterexample, VerificationStatus,
};
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};
use thiserror::Error;

/// Errors from the learning system
#[derive(Error, Debug)]
pub enum LearningError {
    /// Requested proof ID does not exist in the corpus
    #[error("Proof not found: {0}")]
    ProofNotFound(ProofId),
    /// JSON serialization/deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    /// File system operation failed
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// Product quantization error
    #[error("PQ error: {0}")]
    PqError(#[from] pq::PqError),
}

/// Result from a verification that can be learned from
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableResult {
    /// The property that was verified
    pub property: Property,
    /// Which backend succeeded
    pub backend: BackendId,
    /// Verification status
    pub status: VerificationStatus,
    /// Tactics used (if tracked)
    pub tactics: Vec<String>,
    /// Time taken for verification
    pub time_taken: Duration,
    /// Raw proof output (backend-specific)
    pub proof_output: Option<String>,
}

/// The main proof learning system
#[derive(Debug, Default)]
pub struct ProofLearningSystem {
    /// Database of successful proofs
    pub corpus: ProofCorpus,
    /// Database of counterexamples
    pub counterexamples: CounterexampleCorpus,
    /// Tactic effectiveness statistics
    pub tactics: TacticDatabase,
}

impl ProofLearningSystem {
    /// Create a new learning system
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a verification result for learning
    pub fn record(&mut self, result: &LearnableResult) {
        match &result.status {
            VerificationStatus::Proven => {
                // Add to proof corpus
                let id = self.corpus.insert(result);

                // Update tactic statistics for successful proof
                let features = similarity::extract_features(&result.property);
                let context = TacticContext::from_features(&features);
                for tactic in &result.tactics {
                    self.tactics.record_success(&context, tactic);
                }

                tracing::debug!(proof_id = %id, "Recorded successful proof");
            }
            VerificationStatus::Disproven | VerificationStatus::Unknown { .. } => {
                // Record failed tactics for negative learning
                let features = similarity::extract_features(&result.property);
                let context = TacticContext::from_features(&features);
                for tactic in &result.tactics {
                    self.tactics.record_failure(&context, tactic);
                }
                tracing::debug!(property = %result.property.name(), "Recorded failed verification");
            }
            VerificationStatus::Partial { .. } => {
                // Partial results contribute partial evidence
                let features = similarity::extract_features(&result.property);
                let context = TacticContext::from_features(&features);
                for tactic in &result.tactics {
                    self.tactics.record_partial(&context, tactic);
                }
            }
        }
    }

    /// Find proofs similar to the given property
    pub fn find_similar(&self, property: &Property, k: usize) -> Vec<SimilarProof> {
        self.corpus.find_similar(property, k)
    }

    /// Search proofs by text keywords
    ///
    /// Use this when you have a text query that isn't valid USL syntax.
    /// Searches property names, function names, variable names, etc.
    pub fn search_by_keywords(&self, query: &str, k: usize) -> Vec<SimilarProof> {
        self.corpus.search_by_keywords(query, k)
    }

    /// Get suggested tactics for a property based on past success
    pub fn suggest_tactics(&self, property: &Property, n: usize) -> Vec<(String, f64)> {
        let features = similarity::extract_features(property);
        let context = TacticContext::from_features(&features);
        self.tactics.best_for_context(&context, n)
    }

    /// Get a proof by ID
    pub fn get_proof(&self, id: &ProofId) -> Option<&ProofEntry> {
        self.corpus.get(id)
    }

    /// Total number of proofs in corpus
    pub fn proof_count(&self) -> usize {
        self.corpus.len()
    }

    /// Get proof history aggregated by time period
    pub fn proof_history(&self, period: TimePeriod) -> ProofHistory {
        self.corpus.history(period)
    }

    /// Get proof history aggregated by time period within an optional date range
    pub fn proof_history_in_range(
        &self,
        period: TimePeriod,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> ProofHistory {
        self.corpus.history_in_range(period, from, to)
    }

    // ========== Counterexample Learning Methods ==========

    /// Record a counterexample from a failed verification
    pub fn record_counterexample(
        &mut self,
        property_name: &str,
        backend: BackendId,
        counterexample: StructuredCounterexample,
        cluster_label: Option<String>,
    ) -> CounterexampleId {
        let id = self
            .counterexamples
            .insert(property_name, backend, counterexample, cluster_label);
        tracing::debug!(cx_id = %id, "Recorded counterexample");
        id
    }

    /// Record cluster patterns from a clustering result
    ///
    /// This stores the cluster patterns for future classification
    /// of new counterexamples.
    pub fn record_cluster_patterns(&mut self, clusters: &CounterexampleClusters) {
        self.counterexamples.record_clusters(clusters);
        tracing::debug!(
            cluster_count = clusters.clusters.len(),
            "Recorded cluster patterns"
        );
    }

    /// Find counterexamples similar to the given one
    pub fn find_similar_counterexamples(
        &self,
        cx: &StructuredCounterexample,
        k: usize,
    ) -> Vec<SimilarCounterexample> {
        self.counterexamples.find_similar(cx, k)
    }

    /// Classify a counterexample against stored cluster patterns
    ///
    /// Returns the best matching cluster label and its similarity score,
    /// or None if no pattern matches above threshold.
    pub fn classify_counterexample(&self, cx: &StructuredCounterexample) -> Option<(String, f64)> {
        self.counterexamples.classify(cx)
    }

    /// Search counterexamples by text keywords
    pub fn search_counterexamples_by_keywords(
        &self,
        query: &str,
        k: usize,
    ) -> Vec<SimilarCounterexample> {
        self.counterexamples.search_by_keywords(query, k)
    }

    /// Get a counterexample by ID
    pub fn get_counterexample(&self, id: &CounterexampleId) -> Option<&CounterexampleEntry> {
        self.counterexamples.get(id)
    }

    /// Total number of counterexamples in corpus
    pub fn counterexample_count(&self) -> usize {
        self.counterexamples.len()
    }

    /// Number of stored cluster patterns
    pub fn cluster_pattern_count(&self) -> usize {
        self.counterexamples.pattern_count()
    }

    // ========== Persistence Methods ==========

    /// Persist corpus, counterexamples, and tactic statistics to a directory
    ///
    /// Writes `corpus.json`, `counterexamples.json`, and `tactics.json`.
    pub fn save_to_dir<P: AsRef<Path>>(&self, dir: P) -> Result<(), LearningError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let corpus_path = dir.join("corpus.json");
        let counterexamples_path = dir.join("counterexamples.json");
        let tactics_path = dir.join("tactics.json");

        self.corpus.save_to_file(corpus_path)?;
        self.counterexamples.save_to_file(counterexamples_path)?;
        self.tactics.save_to_file(tactics_path)?;

        Ok(())
    }

    /// Load corpus, counterexamples, and tactics from a directory, tolerating missing files
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, LearningError> {
        let dir = dir.as_ref();
        let corpus = ProofCorpus::load_or_default(dir.join("corpus.json"))?;
        let counterexamples =
            CounterexampleCorpus::load_or_default(dir.join("counterexamples.json"))?;
        let tactics = TacticDatabase::load_or_default(dir.join("tactics.json"))?;

        Ok(Self {
            corpus,
            counterexamples,
            tactics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Expr, Invariant};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_simple_property(name: &str) -> Property {
        Property::Invariant(Invariant {
            name: name.to_string(),
            body: Expr::Bool(true),
        })
    }

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dir.push(format!("dashprove_learning_{prefix}_{ts}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_learning_system_creation() {
        let system = ProofLearningSystem::new();
        assert_eq!(system.proof_count(), 0);
    }

    #[test]
    fn test_record_successful_proof() {
        let mut system = ProofLearningSystem::new();

        let result = LearnableResult {
            property: make_simple_property("test_inv"),
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec!["decide".to_string()],
            time_taken: Duration::from_millis(100),
            proof_output: Some("theorem test_inv : True := trivial".to_string()),
        };

        system.record(&result);
        assert_eq!(system.proof_count(), 1);
    }

    #[test]
    fn test_find_similar_proofs() {
        let mut system = ProofLearningSystem::new();

        // Record several proofs
        for i in 0..5 {
            let result = LearnableResult {
                property: make_simple_property(&format!("inv_{}", i)),
                backend: BackendId::Lean4,
                status: VerificationStatus::Proven,
                tactics: vec!["decide".to_string()],
                time_taken: Duration::from_millis(100),
                proof_output: None,
            };
            system.record(&result);
        }

        // Find similar proofs
        let query = make_simple_property("new_inv");
        let similar = system.find_similar(&query, 3);

        assert!(similar.len() <= 3);
    }

    #[test]
    fn test_tactic_suggestion() {
        let mut system = ProofLearningSystem::new();

        // Record proofs with different tactics
        for _ in 0..10 {
            let result = LearnableResult {
                property: make_simple_property("test"),
                backend: BackendId::Lean4,
                status: VerificationStatus::Proven,
                tactics: vec!["decide".to_string()],
                time_taken: Duration::from_millis(100),
                proof_output: None,
            };
            system.record(&result);
        }

        let query = make_simple_property("query");
        let suggestions = system.suggest_tactics(&query, 5);

        // "decide" should be suggested since it worked
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_persistence_roundtrip() {
        let mut system = ProofLearningSystem::new();

        let result = LearnableResult {
            property: make_simple_property("persist_inv"),
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec!["simp".to_string(), "decide".to_string()],
            time_taken: Duration::from_millis(42),
            proof_output: Some("example proof".to_string()),
        };

        system.record(&result);
        assert_eq!(system.proof_count(), 1);

        let dir = temp_dir("persist_roundtrip");
        system.save_to_dir(&dir).unwrap();

        let loaded = ProofLearningSystem::load_from_dir(&dir).unwrap();
        assert_eq!(loaded.proof_count(), 1);

        // Tactic stats should be preserved
        let suggestions = loaded.suggest_tactics(&result.property, 2);
        let tactic_names: Vec<_> = suggestions.into_iter().map(|(name, _)| name).collect();
        assert!(tactic_names.contains(&"simp".to_string()));
        assert!(tactic_names.contains(&"decide".to_string()));

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_load_missing_defaults() {
        let dir = temp_dir("missing_defaults");
        std::fs::remove_dir_all(&dir).ok(); // ensure the directory is gone

        let loaded = ProofLearningSystem::load_from_dir(&dir).unwrap();
        assert_eq!(loaded.proof_count(), 0);
        assert_eq!(loaded.tactics.total_observations(), 0);
    }

    // ========== Counterexample Learning Tests ==========

    fn make_counterexample(
        witness_vars: &[(&str, i128)],
        check_desc: &str,
    ) -> StructuredCounterexample {
        use dashprove_backends::traits::{CounterexampleValue, FailedCheck, TraceState};

        let mut cx = StructuredCounterexample::new();

        for (name, value) in witness_vars {
            cx.witness.insert(
                name.to_string(),
                CounterexampleValue::Int {
                    value: *value,
                    type_hint: None,
                },
            );
        }

        if !check_desc.is_empty() {
            cx.failed_checks.push(FailedCheck {
                check_id: "test_check".to_string(),
                description: check_desc.to_string(),
                location: None,
                function: None,
            });
        }

        // Add a simple trace
        let mut state = TraceState::new(1);
        state.action = Some("Init".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        cx.trace.push(state);

        cx
    }

    #[test]
    fn test_record_counterexample() {
        let mut system = ProofLearningSystem::new();

        let cx = make_counterexample(&[("n", 5)], "division by zero");
        let id = system.record_counterexample("test_prop", BackendId::TlaPlus, cx, None);

        assert_eq!(system.counterexample_count(), 1);
        assert!(system.get_counterexample(&id).is_some());
    }

    #[test]
    fn test_find_similar_counterexamples() {
        let mut system = ProofLearningSystem::new();

        // Record several counterexamples
        for i in 0..5 {
            let cx = make_counterexample(&[("n", i)], "division by zero");
            system.record_counterexample(&format!("prop_{}", i), BackendId::TlaPlus, cx, None);
        }

        // Find similar to a new counterexample
        let query = make_counterexample(&[("n", 10)], "division by zero");
        let similar = system.find_similar_counterexamples(&query, 3);

        assert!(similar.len() <= 3);
        assert!(similar.iter().all(|s| s.similarity > 0.0));
    }

    #[test]
    fn test_classify_counterexample() {
        let mut system = ProofLearningSystem::new();

        // Create and record cluster patterns
        let cx1 = make_counterexample(&[("x", 1)], "overflow");
        let cx2 = make_counterexample(&[("x", 2)], "overflow");
        let clusters = CounterexampleClusters::from_counterexamples(vec![cx1, cx2], 0.5);

        system.record_cluster_patterns(&clusters);
        assert_eq!(system.cluster_pattern_count(), 1);

        // Classify a similar counterexample
        let query = make_counterexample(&[("x", 3)], "overflow");
        let classification = system.classify_counterexample(&query);

        assert!(classification.is_some());
    }

    #[test]
    fn test_counterexample_persistence() {
        let mut system = ProofLearningSystem::new();

        let cx = make_counterexample(&[("n", 42)], "test_failure");
        system.record_counterexample(
            "persist_test",
            BackendId::TlaPlus,
            cx,
            Some("cluster_1".to_string()),
        );

        let dir = temp_dir("cx_persist");
        system.save_to_dir(&dir).unwrap();

        let loaded = ProofLearningSystem::load_from_dir(&dir).unwrap();
        assert_eq!(loaded.counterexample_count(), 1);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_search_counterexamples_by_keywords() {
        let mut system = ProofLearningSystem::new();

        let cx1 = make_counterexample(&[("counter", 0)], "buffer overflow");
        system.record_counterexample("prop1", BackendId::Kani, cx1, None);

        let cx2 = make_counterexample(&[("index", 0)], "null pointer");
        system.record_counterexample("prop2", BackendId::Kani, cx2, None);

        let results = system.search_counterexamples_by_keywords("overflow", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].property_name, "prop1");
    }
}
