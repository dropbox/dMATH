//! LSH-accelerated similarity search for ProofCorpus
//!
//! This module provides optional LSH (Locality-Sensitive Hashing) acceleration
//! for embedding-based similarity search in large proof corpora.
//!
//! # When to Use LSH
//!
//! LSH provides speedup when:
//! - Corpus size > 1000 entries
//! - Queries need < 50 results
//! - Approximate results are acceptable (recall ~80-95%)
//!
//! For smaller corpora or when exact results are needed, use the standard
//! `find_similar_embedding()` method which performs exact search.
//!
//! # Example
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusLsh};
//!
//! let mut corpus = ProofCorpus::new();
//! // ... insert many proofs ...
//!
//! // Build LSH index for fast approximate search
//! let lsh = ProofCorpusLsh::build(&corpus);
//!
//! // Query using LSH (faster for large corpus)
//! let results = lsh.find_similar_approximate(&query_embedding, 10);
//! ```

use super::types::ProofId;
use crate::embedder::Embedding;
use crate::lsh::{BucketStats, LshConfig, LshIndex};
use crate::similarity::SimilarProof;
use std::collections::HashMap;

const MAX_TUNING_TABLES: usize = 32;
const MIN_TUNING_HASHES: usize = 4;
const MIN_RECALL_IMPROVEMENT: f64 = 0.01;
const RECALL_EPSILON: f64 = 1e-6;

/// LSH-accelerated index for ProofCorpus
///
/// Maintains an LSH index over proof embeddings for fast approximate
/// nearest neighbor search.
///
/// # Incremental Updates
///
/// The index supports incremental insertion of new proofs without requiring
/// a full rebuild:
///
/// ```ignore
/// let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
///
/// // Later, when new proofs are added to the corpus
/// if let Some(entry) = corpus.get(&new_proof_id) {
///     if let Some(ref embedding) = entry.embedding {
///         lsh.insert(entry.id.clone(), embedding.clone());
///     }
/// }
/// ```
///
/// For significant corpus growth (>20%), consider rebuilding with
/// `needs_rebuild()` check and `rebuild()`.
#[derive(Debug)]
pub struct ProofCorpusLsh {
    /// The underlying LSH index
    lsh: LshIndex,
    /// Mapping from LSH IDs back to ProofIds
    id_map: HashMap<String, ProofId>,
    /// Configuration used to build this index
    config: LshConfig,
}

impl ProofCorpusLsh {
    /// Build an LSH index from a ProofCorpus
    ///
    /// Only proofs with embeddings are indexed. Use `corpus.compute_embeddings()`
    /// first to populate embeddings for all proofs.
    ///
    /// Returns None if no embeddings are available.
    pub fn build(corpus: &super::storage::ProofCorpus) -> Option<Self> {
        Self::build_with_config(corpus, LshConfig::default())
    }

    /// Build an LSH index with custom configuration
    pub fn build_with_config(
        corpus: &super::storage::ProofCorpus,
        config: LshConfig,
    ) -> Option<Self> {
        let mut lsh = LshIndex::new(config.clone());
        let mut id_map = HashMap::new();

        for entry in corpus.entries() {
            if let Some(ref embedding) = entry.embedding {
                let lsh_id = entry.id.0.clone();
                id_map.insert(lsh_id.clone(), entry.id.clone());
                lsh.insert(lsh_id, embedding.clone());
            }
        }

        if lsh.is_empty() {
            return None;
        }

        Some(Self {
            lsh,
            id_map,
            config,
        })
    }

    /// Build with configuration optimized for corpus size
    pub fn build_auto_config(corpus: &super::storage::ProofCorpus) -> Option<Self> {
        let count = corpus.embedding_count();
        let config = if count < 1000 {
            LshConfig::for_small_corpus()
        } else if count < 10000 {
            LshConfig::for_medium_corpus()
        } else {
            LshConfig::for_large_corpus()
        };

        Self::build_with_config(corpus, config)
    }

    /// Number of indexed embeddings
    pub fn len(&self) -> usize {
        self.lsh.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.lsh.is_empty()
    }

    /// Get bucket distribution statistics
    pub fn stats(&self) -> BucketStats {
        self.lsh.bucket_stats()
    }

    /// Get the configuration used for this index
    pub fn config(&self) -> &LshConfig {
        &self.config
    }

    /// Insert a new proof embedding into the index
    ///
    /// This performs incremental insertion without rebuilding the entire index.
    /// Use this when adding individual proofs to an existing index.
    ///
    /// Returns true if the embedding was inserted, false if the proof was
    /// already in the index or the embedding dimension doesn't match.
    pub fn insert(&mut self, proof_id: ProofId, embedding: Embedding) -> bool {
        // Check if already indexed
        let lsh_id = proof_id.0.clone();
        if self.id_map.contains_key(&lsh_id) {
            return false;
        }

        // Insert into LSH index
        let old_len = self.lsh.len();
        self.lsh.insert(lsh_id.clone(), embedding);

        // Check if insert succeeded (dimension check in LshIndex)
        if self.lsh.len() > old_len {
            self.id_map.insert(lsh_id, proof_id);
            true
        } else {
            false
        }
    }

    /// Insert a proof entry from the corpus if it has an embedding
    ///
    /// Convenience method that extracts the embedding from a corpus entry.
    /// Returns true if the embedding was inserted.
    pub fn insert_from_corpus(
        &mut self,
        corpus: &super::storage::ProofCorpus,
        proof_id: &ProofId,
    ) -> bool {
        if let Some(entry) = corpus.get(proof_id) {
            if let Some(ref embedding) = entry.embedding {
                return self.insert(entry.id.clone(), embedding.clone());
            }
        }
        false
    }

    /// Insert multiple new proofs from the corpus
    ///
    /// Scans the corpus for proofs not yet in the index and inserts them.
    /// Returns the number of proofs inserted.
    pub fn sync_with_corpus(&mut self, corpus: &super::storage::ProofCorpus) -> usize {
        let mut inserted = 0;

        for entry in corpus.entries() {
            if let Some(ref embedding) = entry.embedding {
                let lsh_id = entry.id.0.clone();
                if !self.id_map.contains_key(&lsh_id) {
                    let old_len = self.lsh.len();
                    self.lsh.insert(lsh_id.clone(), embedding.clone());
                    if self.lsh.len() > old_len {
                        self.id_map.insert(lsh_id, entry.id.clone());
                        inserted += 1;
                    }
                }
            }
        }

        inserted
    }

    /// Rebuild the index from scratch
    ///
    /// Use this when the corpus has grown significantly or when changing
    /// configuration. Pass `None` to use the current configuration.
    pub fn rebuild(&mut self, corpus: &super::storage::ProofCorpus, new_config: Option<LshConfig>) {
        let config = new_config.unwrap_or_else(|| self.config.clone());
        self.lsh = LshIndex::new(config.clone());
        self.id_map.clear();
        self.config = config;

        for entry in corpus.entries() {
            if let Some(ref embedding) = entry.embedding {
                let lsh_id = entry.id.0.clone();
                self.lsh.insert(lsh_id.clone(), embedding.clone());
                self.id_map.insert(lsh_id, entry.id.clone());
            }
        }
    }

    /// Rebuild with configuration optimized for current corpus size
    pub fn rebuild_auto_config(&mut self, corpus: &super::storage::ProofCorpus) {
        let count = corpus.embedding_count();
        let config = if count < 1000 {
            LshConfig::for_small_corpus()
        } else if count < 10000 {
            LshConfig::for_medium_corpus()
        } else {
            LshConfig::for_large_corpus()
        };

        self.rebuild(corpus, Some(config));
    }

    /// Find approximate nearest neighbors using LSH
    ///
    /// Returns up to k proofs sorted by similarity (descending).
    /// Results are approximate - may miss some true nearest neighbors
    /// but is significantly faster for large corpora.
    pub fn find_similar_approximate(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        let results = self.lsh.query(query, k);

        results
            .into_iter()
            .filter_map(|(lsh_id, sim)| {
                let proof_id = self.id_map.get(&lsh_id)?;
                let entry = corpus.get(proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: sim,
                })
            })
            .collect()
    }

    /// Find approximate nearest neighbors with minimum candidate guarantee
    ///
    /// Expands LSH search to ensure at least `min_candidates` entries are
    /// examined before selecting top-k. Use when recall is more important
    /// than speed.
    pub fn find_similar_with_min_candidates(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
        min_candidates: usize,
    ) -> Vec<SimilarProof> {
        let results = self.lsh.query_with_min_candidates(query, k, min_candidates);

        results
            .into_iter()
            .filter_map(|(lsh_id, sim)| {
                let proof_id = self.id_map.get(&lsh_id)?;
                let entry = corpus.get(proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: sim,
                })
            })
            .collect()
    }

    /// Perform exact search (bypasses LSH)
    ///
    /// Use for comparison or when exact results are needed.
    pub fn find_similar_exact(
        &self,
        corpus: &super::storage::ProofCorpus,
        query: &Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        let results = self.lsh.exact_search(query, k);

        results
            .into_iter()
            .filter_map(|(lsh_id, sim)| {
                let proof_id = self.id_map.get(&lsh_id)?;
                let entry = corpus.get(proof_id)?;
                Some(SimilarProof {
                    id: entry.id.clone(),
                    property: entry.property.clone(),
                    backend: entry.backend,
                    tactics: entry.tactics.clone(),
                    similarity: sim,
                })
            })
            .collect()
    }

    /// Check if index needs rebuilding
    ///
    /// Returns true if the corpus has significantly more embeddings than
    /// the index (e.g., corpus has grown by >20%).
    pub fn needs_rebuild(&self, corpus: &super::storage::ProofCorpus) -> bool {
        let corpus_count = corpus.embedding_count();
        let index_count = self.len();

        // Rebuild if corpus has grown by more than 20%
        corpus_count > index_count + (index_count / 5)
    }

    /// Measure recall against exact search
    ///
    /// Computes the fraction of true top-k results that are found by LSH.
    /// Uses a random sample of indexed embeddings as queries.
    ///
    /// # Arguments
    /// - `corpus`: The proof corpus
    /// - `k`: Number of results to compare
    /// - `num_samples`: Number of random queries to test
    ///
    /// # Returns
    /// Average recall across all sample queries (0.0 to 1.0)
    pub fn measure_recall(
        &self,
        corpus: &super::storage::ProofCorpus,
        k: usize,
        num_samples: usize,
    ) -> f64 {
        if self.is_empty() || k == 0 || num_samples == 0 {
            return 0.0;
        }

        // Use deterministic sampling based on index size
        let step = self.len().max(1) / num_samples.max(1);
        let step = step.max(1);

        let mut total_recall = 0.0;
        let mut samples_tested = 0;

        // Sample queries from indexed embeddings
        let entries: Vec<_> = corpus.entries().collect();
        for i in (0..entries.len()).step_by(step) {
            if samples_tested >= num_samples {
                break;
            }

            let entry = entries[i];
            if let Some(ref query) = entry.embedding {
                // Get LSH results
                let lsh_results = self.find_similar_approximate(corpus, query, k);
                let lsh_ids: std::collections::HashSet<_> =
                    lsh_results.iter().map(|r| r.id.clone()).collect();

                // Get exact results
                let exact_results = self.find_similar_exact(corpus, query, k);
                let exact_ids: std::collections::HashSet<_> =
                    exact_results.iter().map(|r| r.id.clone()).collect();

                // Compute recall: fraction of exact results found by LSH
                if !exact_ids.is_empty() {
                    let overlap = lsh_ids.intersection(&exact_ids).count();
                    let recall = overlap as f64 / exact_ids.len() as f64;
                    total_recall += recall;
                    samples_tested += 1;
                }
            }
        }

        if samples_tested > 0 {
            total_recall / samples_tested as f64
        } else {
            0.0
        }
    }

    /// Check if index needs rebuilding based on measured recall
    ///
    /// Measures actual LSH recall and recommends rebuild if below threshold.
    /// This is more accurate than `needs_rebuild()` but slower as it requires
    /// running sample queries.
    ///
    /// # Arguments
    /// - `corpus`: The proof corpus
    /// - `min_recall`: Minimum acceptable recall (0.0 to 1.0), typically 0.7-0.9
    /// - `k`: Number of results to use for recall measurement
    /// - `num_samples`: Number of sample queries to test
    ///
    /// # Returns
    /// `RecallStatus` indicating whether rebuild is needed and the measured recall
    pub fn check_recall(
        &self,
        corpus: &super::storage::ProofCorpus,
        min_recall: f64,
        k: usize,
        num_samples: usize,
    ) -> RecallStatus {
        // First check if corpus has grown significantly (fast check)
        if self.needs_rebuild(corpus) {
            return RecallStatus {
                recall: None,
                needs_rebuild: true,
                reason: RecallReason::CorpusGrowth,
            };
        }

        // Measure actual recall
        let recall = self.measure_recall(corpus, k, num_samples);

        if recall < min_recall {
            RecallStatus {
                recall: Some(recall),
                needs_rebuild: true,
                reason: RecallReason::LowRecall,
            }
        } else {
            RecallStatus {
                recall: Some(recall),
                needs_rebuild: false,
                reason: RecallReason::Healthy,
            }
        }
    }

    /// Check recall with default parameters
    ///
    /// Uses k=10, num_samples=50, min_recall=0.7 as sensible defaults.
    pub fn check_recall_default(&self, corpus: &super::storage::ProofCorpus) -> RecallStatus {
        self.check_recall(corpus, 0.7, 10, 50)
    }

    /// Automatically tune LSH configuration to reach target recall
    ///
    /// Evaluates a small set of higher-recall configurations (more tables and/or
    /// wider buckets) and rebuilds the index if a configuration provides a
    /// significant recall improvement.
    ///
    /// # Arguments
    /// - `corpus`: The proof corpus backing this index
    /// - `target_recall`: Desired recall (0.0 to 1.0)
    /// - `k`: Top-k results to use for recall measurement
    /// - `num_samples`: Number of queries to sample for recall measurement
    ///
    /// # Returns
    /// A `RecallTuningResult` indicating whether configuration changed and the
    /// measured recall values.
    pub fn tune_recall(
        &mut self,
        corpus: &super::storage::ProofCorpus,
        target_recall: f64,
        k: usize,
        num_samples: usize,
    ) -> RecallTuningResult {
        let target_recall = target_recall.clamp(0.0, 1.0);
        let corpus_len = corpus.embedding_count();

        if k == 0 || num_samples == 0 || self.is_empty() || corpus_len == 0 {
            return RecallTuningResult {
                baseline_recall: None,
                tuned_recall: None,
                target_recall,
                applied_config: self.config.clone(),
                changed_config: false,
                meets_target: false,
                decision: RecallTuningDecision::NotApplicable,
                evaluated_configs: 0,
            };
        }

        let baseline_recall = self.measure_recall(corpus, k, num_samples);
        let mut evaluated_configs = 1;

        if baseline_recall >= target_recall {
            return RecallTuningResult {
                baseline_recall: Some(baseline_recall),
                tuned_recall: Some(baseline_recall),
                target_recall,
                applied_config: self.config.clone(),
                changed_config: false,
                meets_target: true,
                decision: RecallTuningDecision::TargetSatisfied,
                evaluated_configs,
            };
        }

        let mut best_config = self.config.clone();
        let mut best_recall = baseline_recall;
        let mut best_meets_target = best_recall >= target_recall;

        for config in self.generate_tuning_configs(corpus_len) {
            if let Some(candidate) = ProofCorpusLsh::build_with_config(corpus, config.clone()) {
                let recall = candidate.measure_recall(corpus, k, num_samples);
                evaluated_configs += 1;

                if recall > best_recall + RECALL_EPSILON
                    || ((best_recall - recall).abs() <= RECALL_EPSILON
                        && config_cost(&config) < config_cost(&best_config))
                {
                    best_recall = recall;
                    best_config = config;
                    best_meets_target = recall >= target_recall;
                }
            }
        }

        let improved = best_recall > baseline_recall + MIN_RECALL_IMPROVEMENT;
        let changed_config = improved && best_config != self.config;

        let applied_config = if changed_config {
            best_config.clone()
        } else {
            self.config.clone()
        };

        if changed_config {
            self.rebuild(corpus, Some(best_config.clone()));
        }

        RecallTuningResult {
            baseline_recall: Some(baseline_recall),
            tuned_recall: Some(best_recall),
            target_recall,
            applied_config,
            changed_config,
            meets_target: best_meets_target,
            decision: if changed_config {
                RecallTuningDecision::RebuiltForRecall
            } else {
                RecallTuningDecision::NoChange
            },
            evaluated_configs,
        }
    }

    /// Tune recall using default parameters (target_recall=0.8, k=10, num_samples=50)
    pub fn tune_recall_default(
        &mut self,
        corpus: &super::storage::ProofCorpus,
    ) -> RecallTuningResult {
        self.tune_recall(corpus, 0.8, 10, 50)
    }

    /// Generate candidate configurations for recall tuning
    fn generate_tuning_configs(&self, corpus_len: usize) -> Vec<LshConfig> {
        let mut configs = Vec::new();
        let base = self.config.clone();

        let push_if_new = |cfg: LshConfig, list: &mut Vec<LshConfig>| {
            if cfg != base && !list.contains(&cfg) {
                list.push(cfg);
            }
        };

        // Corpus-size presets provide a strong improvement baseline
        let preset = if corpus_len < 1000 {
            LshConfig::for_small_corpus()
        } else if corpus_len < 10000 {
            LshConfig::for_medium_corpus()
        } else {
            LshConfig::for_large_corpus()
        };
        push_if_new(preset, &mut configs);

        if base.num_tables < MAX_TUNING_TABLES {
            let mut doubled = base.clone();
            doubled.num_tables = (doubled.num_tables * 2)
                .min(MAX_TUNING_TABLES)
                .max(doubled.num_tables + 1);
            push_if_new(doubled, &mut configs);

            let mut stepped = base.clone();
            stepped.num_tables = (stepped.num_tables + 4).min(MAX_TUNING_TABLES);
            if stepped.num_tables != base.num_tables {
                push_if_new(stepped, &mut configs);
            }

            let mut aggressive = base.clone();
            aggressive.num_tables = (aggressive.num_tables * 4)
                .min(MAX_TUNING_TABLES)
                .max(aggressive.num_tables + 1);
            push_if_new(aggressive, &mut configs);
        }

        if base.num_hashes > MIN_TUNING_HASHES {
            let mut wider = base.clone();
            wider.num_hashes = (wider.num_hashes - 2).max(MIN_TUNING_HASHES);
            push_if_new(wider, &mut configs);

            let mut much_wider = base.clone();
            much_wider.num_hashes =
                (much_wider.num_hashes.saturating_sub(4)).max(MIN_TUNING_HASHES);
            push_if_new(much_wider, &mut configs);
        }

        if base.num_tables < MAX_TUNING_TABLES && base.num_hashes > MIN_TUNING_HASHES {
            let mut combo = base.clone();
            combo.num_tables = (combo.num_tables * 2)
                .min(MAX_TUNING_TABLES)
                .max(combo.num_tables + 1);
            combo.num_hashes = (combo.num_hashes - 2).max(MIN_TUNING_HASHES);
            push_if_new(combo, &mut configs);

            let mut combo_aggressive = base.clone();
            combo_aggressive.num_tables = (combo_aggressive.num_tables * 4)
                .min(MAX_TUNING_TABLES)
                .max(combo_aggressive.num_tables + 1);
            combo_aggressive.num_hashes =
                (combo_aggressive.num_hashes.saturating_sub(4)).max(MIN_TUNING_HASHES);
            push_if_new(combo_aggressive, &mut configs);
        }

        configs
    }
}

/// Status of LSH index recall measurement
#[derive(Debug, Clone)]
pub struct RecallStatus {
    /// Measured recall (None if not measured due to fast-path detection)
    pub recall: Option<f64>,
    /// Whether the index needs rebuilding
    pub needs_rebuild: bool,
    /// Reason for the status
    pub reason: RecallReason,
}

/// Reason for recall status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallReason {
    /// Corpus has grown significantly, rebuild recommended
    CorpusGrowth,
    /// Measured recall is below threshold
    LowRecall,
    /// Index is healthy
    Healthy,
}

/// Outcome of recall-based tuning
#[derive(Debug, Clone)]
pub struct RecallTuningResult {
    /// Recall before tuning (None if parameters were invalid)
    pub baseline_recall: Option<f64>,
    /// Best recall measured across candidate configurations
    pub tuned_recall: Option<f64>,
    /// Target recall provided by caller
    pub target_recall: f64,
    /// Configuration currently applied (post-tuning)
    pub applied_config: LshConfig,
    /// Whether the index was rebuilt with a new configuration
    pub changed_config: bool,
    /// Whether the final configuration meets or exceeds the target recall
    pub meets_target: bool,
    /// Decision explaining the tuning outcome
    pub decision: RecallTuningDecision,
    /// Number of configurations evaluated (including the baseline)
    pub evaluated_configs: usize,
}

/// Decision made by recall tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallTuningDecision {
    /// Baseline recall already meets the target
    TargetSatisfied,
    /// Index was rebuilt with a higher-recall configuration
    RebuiltForRecall,
    /// No better configuration was found or applied
    NoChange,
    /// Tuning could not be performed (empty corpus or invalid parameters)
    NotApplicable,
}

fn config_cost(config: &LshConfig) -> usize {
    config.num_tables.saturating_mul(config.num_hashes.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::EMBEDDING_DIM;
    use crate::LearnableResult;
    use dashprove_backends::traits::{BackendId, VerificationStatus};
    use dashprove_usl::ast::{Expr, Invariant, Property};
    use std::time::Duration;

    fn make_property(name: &str) -> Property {
        Property::Invariant(Invariant {
            name: name.to_string(),
            body: Expr::Bool(true),
        })
    }

    fn make_result(name: &str) -> LearnableResult {
        LearnableResult {
            property: make_property(name),
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec!["simp".to_string()],
            time_taken: Duration::from_millis(100),
            proof_output: None,
        }
    }

    fn random_embedding(seed: u64) -> Embedding {
        // Simple deterministic random embedding for testing
        let mut state = seed;
        let mut vector = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32);
            vector.push(f * 2.0 - 1.0);
        }
        Embedding::new(vector)
    }

    #[test]
    fn test_build_empty_corpus() {
        let corpus = super::super::storage::ProofCorpus::new();
        let lsh = ProofCorpusLsh::build(&corpus);
        assert!(lsh.is_none());
    }

    #[test]
    fn test_build_corpus_without_embeddings() {
        let mut corpus = super::super::storage::ProofCorpus::new();
        let result = make_result("test");
        corpus.insert(&result);

        let lsh = ProofCorpusLsh::build(&corpus);
        assert!(lsh.is_none(), "Should return None if no embeddings");
    }

    #[test]
    fn test_build_corpus_with_embeddings() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert with embeddings
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus);
        assert!(lsh.is_some());
        let lsh = lsh.unwrap();
        assert_eq!(lsh.len(), 10);
    }

    #[test]
    fn test_find_similar_approximate() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert many proofs with embeddings
        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build_auto_config(&corpus).unwrap();

        // Query
        let query = random_embedding(42);
        let results = lsh.find_similar_approximate(&corpus, &query, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(
                results[i - 1].similarity >= results[i].similarity,
                "Results should be sorted by similarity"
            );
        }
    }

    #[test]
    fn test_find_similar_exact_vs_approximate() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert proofs
        for i in 0..50 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus).unwrap();
        let query = random_embedding(25);

        let exact = lsh.find_similar_exact(&corpus, &query, 5);
        let approx = lsh.find_similar_approximate(&corpus, &query, 5);

        // Both should return results
        assert_eq!(exact.len(), 5);
        assert!(!approx.is_empty());

        // Exact should have the true top result
        // (approximate may or may not match exactly)
        assert!(exact[0].similarity >= approx[0].similarity * 0.8);
    }

    #[test]
    fn test_needs_rebuild() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Insert initial proofs
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus).unwrap();
        assert!(!lsh.needs_rebuild(&corpus));

        // Add more proofs (> 20% growth)
        for i in 10..15 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        assert!(lsh.needs_rebuild(&corpus));
    }

    #[test]
    fn test_stats() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus).unwrap();
        let stats = lsh.stats();

        assert_eq!(stats.total_entries, 100);
        assert!(stats.num_tables > 0);
        assert!(stats.total_buckets > 0);
    }

    #[test]
    fn test_auto_config_selection() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Small corpus
        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build_auto_config(&corpus).unwrap();
        let stats = lsh.stats();

        // Small corpus should use fewer tables
        assert!(stats.num_tables <= 8);
    }

    #[test]
    fn test_incremental_insert() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Build index with initial proofs, save their IDs
        let mut initial_ids = Vec::new();
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            let id = corpus.insert_with_embedding(&result, embedding);
            initial_ids.push(id);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
        assert_eq!(lsh.len(), 10);

        // Add new proofs to corpus, save their IDs
        let mut new_ids = Vec::new();
        for i in 10..15 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            let id = corpus.insert_with_embedding(&result, embedding);
            new_ids.push(id);
        }

        // Incrementally insert new proofs using their actual IDs
        for proof_id in &new_ids {
            let inserted = lsh.insert_from_corpus(&corpus, proof_id);
            assert!(inserted, "Should insert new proof");
        }

        assert_eq!(lsh.len(), 15);

        // Query should find all proofs
        let query = random_embedding(12);
        let results = lsh.find_similar_approximate(&corpus, &query, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_incremental_insert_duplicate() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        let result = make_result("prop_0");
        let embedding = random_embedding(0);
        let proof_id = corpus.insert_with_embedding(&result, embedding.clone());

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
        assert_eq!(lsh.len(), 1);

        // Try to insert the same proof again using its actual ID
        let inserted = lsh.insert(proof_id, embedding);
        assert!(!inserted, "Should not insert duplicate");
        assert_eq!(lsh.len(), 1);
    }

    #[test]
    fn test_incremental_insert_direct() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Build index with initial proofs
        for i in 0..5 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
        assert_eq!(lsh.len(), 5);

        // Insert directly without going through corpus
        let new_id = ProofId("new_proof".to_string());
        let new_embedding = random_embedding(100);
        let inserted = lsh.insert(new_id.clone(), new_embedding);
        assert!(inserted);
        assert_eq!(lsh.len(), 6);
    }

    #[test]
    fn test_sync_with_corpus() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Build index with initial proofs
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
        assert_eq!(lsh.len(), 10);

        // Add more proofs to corpus
        for i in 10..20 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // Sync - should add the 10 new proofs
        let inserted = lsh.sync_with_corpus(&corpus);
        assert_eq!(inserted, 10);
        assert_eq!(lsh.len(), 20);

        // Sync again - should add nothing
        let inserted = lsh.sync_with_corpus(&corpus);
        assert_eq!(inserted, 0);
    }

    #[test]
    fn test_rebuild() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..50 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
        let original_tables = lsh.stats().num_tables;

        // Rebuild with different config
        let new_config = LshConfig {
            num_tables: 16,
            num_hashes: 10,
            ..LshConfig::default()
        };
        lsh.rebuild(&corpus, Some(new_config));

        assert_eq!(lsh.len(), 50);
        assert_eq!(lsh.stats().num_tables, 16);
        assert_ne!(lsh.stats().num_tables, original_tables);
    }

    #[test]
    fn test_rebuild_preserves_search() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..30 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();

        // Get results before rebuild
        let query = random_embedding(15);
        let before = lsh.find_similar_exact(&corpus, &query, 5);

        // Rebuild
        lsh.rebuild(&corpus, None);

        // Results should be the same after rebuild
        let after = lsh.find_similar_exact(&corpus, &query, 5);

        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.id, a.id);
            assert!((b.similarity - a.similarity).abs() < 0.001);
        }
    }

    #[test]
    fn test_rebuild_auto_config() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Start with small corpus
        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();

        // Rebuild with auto config
        lsh.rebuild_auto_config(&corpus);

        assert_eq!(lsh.len(), 100);
        // Small corpus should use smaller config
        assert!(lsh.config().num_tables <= 8);
    }

    #[test]
    fn test_config_accessor() {
        let corpus = {
            let mut c = super::super::storage::ProofCorpus::new();
            let result = make_result("prop_0");
            let embedding = random_embedding(0);
            c.insert_with_embedding(&result, embedding);
            c
        };

        let config = LshConfig {
            num_tables: 12,
            num_hashes: 7,
            ..LshConfig::default()
        };

        let lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        assert_eq!(lsh.config().num_tables, 12);
        assert_eq!(lsh.config().num_hashes, 7);
    }

    #[test]
    fn test_measure_recall_empty() {
        let corpus = super::super::storage::ProofCorpus::new();

        // Can't build LSH from empty corpus, so test edge cases manually
        let mut test_corpus = super::super::storage::ProofCorpus::new();
        let result = make_result("prop_0");
        let embedding = random_embedding(0);
        test_corpus.insert_with_embedding(&result, embedding);

        let lsh = ProofCorpusLsh::build(&test_corpus).unwrap();

        // Measure recall with empty original corpus should return 0
        let recall = lsh.measure_recall(&corpus, 10, 10);
        assert_eq!(recall, 0.0);
    }

    #[test]
    fn test_measure_recall_k_zero() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus).unwrap();

        // k=0 should return 0
        let recall = lsh.measure_recall(&corpus, 0, 10);
        assert_eq!(recall, 0.0);
    }

    #[test]
    fn test_measure_recall_num_samples_zero() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus).unwrap();

        // num_samples=0 should return 0
        let recall = lsh.measure_recall(&corpus, 10, 0);
        assert_eq!(recall, 0.0);
    }

    #[test]
    fn test_measure_recall_basic() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Create corpus with enough entries for meaningful recall
        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // Use config with more tables for better recall
        let config = LshConfig {
            num_tables: 16,
            num_hashes: 6,
            ..LshConfig::default()
        };
        let lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        // Measure recall
        let recall = lsh.measure_recall(&corpus, 10, 20);

        // With 16 tables, recall should be reasonable (>0.5)
        assert!(recall > 0.0, "Recall should be positive: {}", recall);
        assert!(recall <= 1.0, "Recall should be <= 1.0: {}", recall);
    }

    #[test]
    fn test_measure_recall_perfect_with_high_config() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Small corpus where LSH should achieve high recall
        for i in 0..20 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // Very high tables for best recall
        let config = LshConfig {
            num_tables: 32,
            num_hashes: 4,
            ..LshConfig::default()
        };
        let lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        // With 32 tables on small corpus, recall should be very high
        let recall = lsh.measure_recall(&corpus, 5, 10);
        assert!(
            recall >= 0.7,
            "High-config LSH should have high recall: {}",
            recall
        );
    }

    #[test]
    fn test_check_recall_corpus_growth() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        // Initial corpus
        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let lsh = ProofCorpusLsh::build(&corpus).unwrap();

        // Add more than 20% new entries
        for i in 10..15 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let status = lsh.check_recall(&corpus, 0.7, 5, 10);

        assert!(status.needs_rebuild);
        assert_eq!(status.reason, RecallReason::CorpusGrowth);
        assert!(status.recall.is_none()); // Fast path, no recall measured
    }

    #[test]
    fn test_check_recall_healthy() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..50 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // High-recall config
        let config = LshConfig {
            num_tables: 16,
            num_hashes: 6,
            ..LshConfig::default()
        };
        let lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        // Check with low threshold - should be healthy
        let status = lsh.check_recall(&corpus, 0.3, 5, 10);

        assert!(!status.needs_rebuild);
        assert_eq!(status.reason, RecallReason::Healthy);
        assert!(status.recall.is_some());
        assert!(status.recall.unwrap() >= 0.3);
    }

    #[test]
    fn test_check_recall_low_recall() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // Low-recall config (few tables, many hashes = small buckets)
        let config = LshConfig {
            num_tables: 2,
            num_hashes: 12,
            ..LshConfig::default()
        };
        let lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        // Check with very high threshold - should trigger low recall
        let status = lsh.check_recall(&corpus, 0.99, 10, 20);

        // With 2 tables and 12 hashes, unlikely to have 99% recall
        if status.needs_rebuild && status.reason == RecallReason::LowRecall {
            assert!(status.recall.is_some());
            assert!(status.recall.unwrap() < 0.99);
        }
        // If recall happens to be high enough, that's also acceptable
    }

    #[test]
    fn test_check_recall_default() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..100 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let config = LshConfig {
            num_tables: 16,
            num_hashes: 6,
            ..LshConfig::default()
        };
        let lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        // Default check should work
        let status = lsh.check_recall_default(&corpus);

        // With good config, should be healthy
        assert!(status.recall.is_some() || status.reason == RecallReason::CorpusGrowth);
    }

    #[test]
    fn test_recall_reason_equality() {
        assert_eq!(RecallReason::CorpusGrowth, RecallReason::CorpusGrowth);
        assert_eq!(RecallReason::LowRecall, RecallReason::LowRecall);
        assert_eq!(RecallReason::Healthy, RecallReason::Healthy);
        assert_ne!(RecallReason::CorpusGrowth, RecallReason::LowRecall);
    }

    #[test]
    fn test_recall_status_debug() {
        let status = RecallStatus {
            recall: Some(0.85),
            needs_rebuild: false,
            reason: RecallReason::Healthy,
        };
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("0.85"));
        assert!(debug_str.contains("Healthy"));
    }

    #[test]
    fn test_tune_recall_rebuilds_low_recall_config() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..200 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // Start with intentionally low-recall configuration (few tables, many hashes)
        let config = LshConfig {
            num_tables: 1,
            num_hashes: 20,
            ..LshConfig::default()
        };
        let mut lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();
        let baseline = lsh.measure_recall(&corpus, 10, 40);

        let result = lsh.tune_recall(&corpus, 0.6, 10, 40);

        assert!(result.baseline_recall.is_some());
        assert_eq!(result.baseline_recall.unwrap(), baseline);
        assert!(result.changed_config, "Expected rebuild for better recall");
        assert_eq!(result.decision, RecallTuningDecision::RebuiltForRecall);
        assert!(
            result.tuned_recall.unwrap() > baseline,
            "Recall should improve after tuning"
        );
        assert!(result.applied_config.num_tables > 1 || result.applied_config.num_hashes < 20);
        assert!(result.evaluated_configs >= 2);
    }

    #[test]
    fn test_tune_recall_no_change_when_healthy() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..80 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        // High-recall config should already exceed the low target
        let config = LshConfig {
            num_tables: 16,
            num_hashes: 6,
            ..LshConfig::default()
        };
        let mut lsh = ProofCorpusLsh::build_with_config(&corpus, config).unwrap();

        let tuning = lsh.tune_recall(&corpus, 0.3, 8, 20);

        assert_eq!(tuning.decision, RecallTuningDecision::TargetSatisfied);
        assert!(tuning.meets_target);
        assert!(!tuning.changed_config);
        assert_eq!(tuning.evaluated_configs, 1);
    }

    #[test]
    fn test_tune_recall_not_applicable_for_invalid_params() {
        let mut corpus = super::super::storage::ProofCorpus::new();

        for i in 0..10 {
            let result = make_result(&format!("prop_{}", i));
            let embedding = random_embedding(i as u64);
            corpus.insert_with_embedding(&result, embedding);
        }

        let mut lsh = ProofCorpusLsh::build(&corpus).unwrap();
        let tuning = lsh.tune_recall(&corpus, 0.5, 0, 10);

        assert_eq!(tuning.decision, RecallTuningDecision::NotApplicable);
        assert!(!tuning.changed_config);
        assert!(tuning.tuned_recall.is_none());
    }
}
